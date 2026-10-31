#!/usr/bin/env python3
"""Animated replays of the Raft tests, and of the nodes running on their own.

    python visualization.py                 # every test in test_raft.py, plus a free run of main.py's nodes
    python visualization.py -k divergent    # only tests whose name contains "divergent"
    python visualization.py --no-run        # tests only
    python visualization.py --nodes 5 --ms 8000 --seed 3
    python visualization.py --pace 1.5      # slower still (the page also has a speed menu)
    python visualization.py -o replay.html --no-open

Writes one self-contained HTML file (default ``raft_viz.html``) and opens it.

Each test really runs. While it does, this script

- follows the test function line by line (``sys.settrace``), so the replay can
  highlight the line being run and tick off each ``assert`` as it passes;
- taps every ``Cluster``'s trace for messages and timers, and snapshots every
  ``RaftNode``'s full state (role, term, vote, log, commit index, store) after
  each event;
- for unit tests that drive one node through a fake transport, draws that
  node's peers as "test driver" nodes so its replies have somewhere to go.

Nothing in main.py, test_raft.py or dslabs is changed, and the hooks are
removed once recording is done.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import functools
import heapq
import importlib
import inspect
import io
import json
import statistics
import sys
import textwrap
import traceback
import webbrowser
from bisect import bisect_left
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from dslabs import Cluster  # noqa: E402
from dslabs.network import Endpoint  # noqa: E402
from main import RaftNode  # noqa: E402

# Pacing, in milliseconds of animation at 1x speed. --pace multiplies all of them.
HOP_MS = 1100           # how long a typical message takes to fly
LINE_HOLD_MS = 600      # pause on each line of test code
POKE_HOLD_MS = 500      # pause after test code changes a node's state directly
CHECK_HOLD_MS = 450     # pause on a passing assert
FAIL_HOLD_MS = 2200     # pause on a failure
REACT_MS = 300          # gap between a message landing and the receiver's reaction
MAX_IDLE_MS = 1400      # nothing on the wire: fast-forward, so no quiet stretch lasts longer
TAIL_MS = 2000          # hold the final state at the end
RUN_ERROR_HOLD_MS = 250
PACE = 1.0
MAX_RUN_ERRORS = 100


# --- describing state and messages ---------------------------------------------


def _plain(value: Any) -> Any:
    """``value`` if JSON can carry it, else its repr."""
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return repr(value)


def _short(value: Any, limit: int = 24) -> str:
    text = value if isinstance(value, str) else json.dumps(_plain(value))
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _entry(e: Any) -> list:
    if isinstance(e, dict):
        return [e.get("term"), str(e.get("key")), _plain(e.get("value"))]
    return [getattr(e, "term", None), str(getattr(e, "key", "?")), _plain(getattr(e, "value", None))]


def node_state(node: Any) -> dict[str, Any]:
    """Everything the replay shows about one RaftNode."""
    return {
        "role": str(getattr(node, "role", "?")),
        "term": getattr(node, "term", 0),
        "voted_for": getattr(node, "voted_for", "") or "",
        "leader": getattr(node, "leader", "") or "",
        "commit": getattr(node, "commit_index", 0),
        "applied": getattr(node, "last_applied", 0),
        "log": [_entry(e) for e in getattr(node, "log", [])],
        "votes": sorted(getattr(node, "votes", ()) or ()),
        "next": dict(getattr(node, "next_index", {}) or {}),
        "match": dict(getattr(node, "match_index", {}) or {}),
        "store": {str(k): _plain(v) for k, v in (getattr(node, "store", {}) or {}).items()},
    }


def diff_states(prev: dict | None, cur: dict) -> list[str]:
    """What changed between two snapshots of one node, in words."""
    if prev is None:
        return []
    out = []
    if cur["role"] != prev["role"]:
        out.append(f"became {cur['role']}" + (f" (T{cur['term']})" if cur["role"] != "follower" else ""))
    if cur["term"] != prev["term"]:
        out.append(f"term {prev['term']} → {cur['term']}")
    if cur["voted_for"] != prev["voted_for"] and cur["voted_for"]:
        out.append(f"voted for {cur['voted_for']}")
    if cur["leader"] != prev["leader"] and cur["leader"] and cur["role"] != "leader":
        out.append(f"follows leader {cur['leader']}")
    for v in cur["votes"]:
        if v not in prev["votes"]:
            out.append(f"got vote from {v}")
    old, new = prev["log"], cur["log"]
    same = 0
    while same < min(len(old), len(new)) and old[same] == new[same]:
        same += 1
    if len(old) > same:
        n = len(old) - same
        out.append(f"dropped {n} log entr{'y' if n == 1 else 'ies'}")
    if len(new) > same:
        out.append("appended " + ", ".join(f"T{t} {k}={_short(v, 12)}" for t, k, v in new[same:]))
    if cur["commit"] != prev["commit"]:
        out.append(f"commit {prev['commit']} → {cur['commit']}")
    for k, v in cur["store"].items():
        if prev["store"].get(k, object()) != v:
            out.append(f"applied {k}={_short(v, 12)}")
    if not out and (cur["next"] != prev["next"] or cur["match"] != prev["match"]):
        out.append("updated next/match index")
    return out


def describe_msg(msg: dict) -> tuple[str, str, str]:
    """(colour class, short label, full description) for one message."""
    m = {k: v for k, v in msg.items() if k != "from"}
    kind, term = m.get("type"), m.get("term")
    if kind == "request_vote":
        return ("rv", f"RequestVote T{term}",
                f"RequestVote term={term} candidate={m.get('candidate_id')} "
                f"last_log={m.get('last_log_index')}/T{m.get('last_log_term')}")
    if kind == "request_vote_response":
        ok = bool(m.get("vote_granted"))
        return ("yes" if ok else "no", f"vote {'✓' if ok else '✗'} T{term}",
                f"RequestVoteResponse term={term} vote_granted={ok}")
    if kind == "append_entries":
        entries = [_entry(e) for e in (m.get("entries") or [])]
        full = (f"AppendEntries term={term} leader={m.get('leader_id')} "
                f"prev={m.get('prev_log_index')}/T{m.get('prev_log_term')} "
                f"entries=[{', '.join(f'T{t} {k}={_short(v, 12)}' for t, k, v in entries)}] "
                f"commit={m.get('leader_commit')}")
        if entries:
            body = ", ".join(f"{k}={_short(v, 8)}" for _, k, v in entries)
            return "ae", _short(f"Append {body}", 26), full
        return "hb", f"heartbeat c{m.get('leader_commit')}", full
    if kind == "append_entries_response":
        ok = bool(m.get("success"))
        return ("ok" if ok else "nack", f"{'ok' if ok else 'reject'} m{m.get('match_index')}",
                f"AppendEntriesResponse term={term} success={ok} match_index={m.get('match_index')}")
    rest = " ".join(f"{k}={_short(v)}" for k, v in m.items() if k != "type")
    return "other", _short(str(kind), 20), f"{kind} {rest}".strip()


def _user_frames(exc: BaseException, limit: int = 6) -> list[str]:
    """The traceback frames in the code under test (not this script, dslabs or Python itself), innermost last."""
    frames = traceback.extract_tb(exc.__traceback__)
    skip = [HERE / "dslabs", Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()]

    def ours(f) -> bool:
        path = Path(f.filename).resolve()
        return path != Path(__file__).resolve() and not any(path.is_relative_to(d) for d in skip)

    mine = [f for f in frames if ours(f)]
    return [f"{Path(f.filename).name}:{f.lineno} in {f.name}: {(f.line or '').strip()}"
            for f in (mine or frames)[-limit:]]


def _pretty(name: str) -> str:
    return name.removeprefix("test_").replace("_", " ")


# --- recording one test (or one free run) ---------------------------------------


class World:
    """Everything one test, or the free run, did, in the order it happened."""

    def __init__(self, name: str, kind: str) -> None:
        self.name = name
        self.kind = kind  # "test" or "run"
        self.title = _pretty(name)
        self.doc = ""
        self.items: list[dict] = []
        self.nodes: list[str] = []
        self.puppets: set[str] = set()
        self.live: dict[str, Any] = {}
        self.clusters: list[Cluster] = []
        self.latency: tuple[int, int] | None = None
        self.unit = False
        self.vclock = 0  # the clock when no simulated scheduler is attached (unit tests)
        self.status = "passed" if kind == "test" else "ok"
        self.failure: dict | None = None
        self.output = ""
        self.source: dict | None = None
        self.run_errors: Counter[str] = Counter()
        self.run_error_frames: dict[str, tuple[str, list[str]]] = {}  # key -> (first message, frames)
        self.last_node: str | None = None
        self._last_state: dict[str, tuple[str, dict]] = {}
        self._pending: list[int] = []
        self._msg_ids: dict[tuple[int, int], int] = {}
        self._msgs: dict[int, dict] = {}
        self._stmts: list[tuple[int, int, str | None]] = []  # (first line, last line, assert text)
        self._cur: tuple[int, int, str | None] | None = None

    # -- basics --

    def now(self) -> float:
        return self.clusters[-1].scheduler.now_ms() if self.clusters else self.vclock

    def add(self, k: str, sim: float | None = None, **fields: Any) -> None:
        self.items.append({"k": k, "sim": self.now() if sim is None else sim, **fields})

    def add_node(self, nid: str, puppet: bool = False) -> None:
        if nid not in self.nodes:
            self.nodes.append(nid)
        if puppet and nid not in self.live:
            self.puppets.add(nid)
        else:
            self.puppets.discard(nid)

    def snapshot(self, poke: bool = False) -> None:
        """Record every node whose state changed since it was last recorded."""
        for nid, node in self.live.items():
            st = node_state(node)
            key = json.dumps(st, sort_keys=True, default=repr)
            last = self._last_state.get(nid)
            if last is not None and last[0] == key:
                continue
            texts = diff_states(last[1] if last else None, st)
            self._last_state[nid] = (key, st)
            self.add("state", node=nid, st=st, texts=texts, poke=poke and last is not None)

    def _send(self, frm: str, to: str, msg: dict, sim: float | None = None) -> int:
        mid = len(self._msgs) + 1
        cls, lab, desc = describe_msg(msg)
        self._msgs[mid] = {"frm": frm, "to": to, "cls": cls, "lab": lab, "desc": desc}
        self.add("send", sim=sim, id=mid)
        return mid

    # -- simulated clusters --

    def attach_cluster(self, cluster: Cluster) -> None:
        idx = len(self.clusters)
        prefix = "" if idx == 0 else f"c{idx + 1}:"

        def name(nid: str | None) -> str | None:
            return nid if nid is None else prefix + nid

        self.clusters.append(cluster)
        self.latency = tuple(cluster.network.latency_ms)
        for nid, node in cluster.nodes.items():
            self.live[name(nid)] = node
            self.add_node(name(nid))
        trace = cluster.trace
        for ev in list(trace.events):  # what happened while the cluster was being built
            self._on_event(idx, name, ev)
        record, snapshot = trace.record, trace.snapshot

        def tapped_record(ev) -> None:
            record(ev)
            self._on_event(idx, name, ev)

        def tapped_snapshot(node) -> None:
            snapshot(node)
            self.snapshot()

        trace.record = tapped_record
        trace.snapshot = tapped_snapshot
        self.snapshot()

    def _on_event(self, idx: int, name: Callable, ev) -> None:
        k = ev.kind
        if k == "send":
            self.snapshot()  # the sender may have just changed (became leader, started a campaign, ...)
            self._msg_ids[(idx, ev.msg_id)] = self._send(name(ev.frm), name(ev.to), ev.msg, sim=ev.t_ms)
        elif k == "deliver":
            self.last_node = name(ev.to)
            self.add("deliver", sim=ev.t_ms, id=self._msg_ids[(idx, ev.msg_id)])
        elif k == "drop":
            self.add("drop", sim=ev.t_ms, id=self._msg_ids[(idx, ev.msg_id)], rule=ev.rule,
                     due=ev.t_ms + (ev.delay_ms or 0))
        elif k == "timer_set":
            self.add("tset", sim=ev.t_ms, node=name(ev.node), tid=f"{idx}:{ev.timer_id}", name=ev.text,
                     due=ev.due_ms)
        elif k in ("timer_fired", "timer_cancelled"):
            if k == "timer_fired":
                self.last_node = name(ev.node)
            self.add("tend", sim=ev.t_ms, node=name(ev.node), tid=f"{idx}:{ev.timer_id}", name=ev.text,
                     fired=k == "timer_fired")
        elif k in ("client", "note"):
            self.add("note", sim=ev.t_ms, node=name(ev.node), text=f"{k}: {ev.text}")

    # -- unit tests: one node, a fake transport, no network --

    def attach_unit(self, node: Any) -> None:
        nid = node.node_id
        self.unit = True
        self.live[nid] = node
        for peer in list(getattr(node, "peers", [])) or [nid]:
            self.add_node(peer, puppet=peer != nid)
        self.add_node(nid)
        transport = node.transport
        send, on_message = transport.send, node.on_message

        def tapped_send(to: str, msg: dict) -> None:
            send(to, msg)
            if to not in self.nodes:
                self.add_node(to, puppet=True)
            self.snapshot()
            self._pending.append(self._send(nid, to, {**msg, "from": nid}))

        def tapped_on_message(msg: dict) -> None:
            self._land()
            frm = str(msg.get("from", "test"))
            if frm not in self.nodes:
                self.add_node(frm, puppet=True)
            mid = self._send(frm, nid, msg)
            self.vclock += 1
            self.add("deliver", id=mid)
            self.last_node = nid
            try:
                on_message(msg)
            finally:
                self.snapshot()
                self._land()

        transport.send = tapped_send
        node.on_message = tapped_on_message
        self.snapshot()

    def _land(self) -> None:
        """Deliver whatever a unit-tested node sent to the test-driver nodes."""
        if self._pending:
            self.vclock += 1
            for mid in self._pending:
                self.add("deliver", id=mid)
            self._pending.clear()

    # -- following the test code --

    def watch(self, fn: Callable) -> None:
        lines, first = inspect.getsourcelines(fn)
        src = textwrap.dedent("".join(lines))
        self.source = {"file": Path(inspect.getsourcefile(fn) or "?").name, "first": first,
                       "lines": [line.rstrip("\n") for line in lines]}
        func = ast.parse(src).body[0]
        off = first - 1
        for node in ast.walk(func):
            if not isinstance(node, ast.stmt) or node is func:
                continue
            start = node.lineno + off
            body = getattr(node, "body", None)
            if isinstance(body, list) and body and isinstance(body[0], ast.stmt):
                end = max(start, body[0].lineno + off - 1)  # a compound statement: just its header
            else:
                end = (node.end_lineno or node.lineno) + off
            check = ast.get_source_segment(src, node.test) if isinstance(node, ast.Assert) else None
            self._stmts.append((start, end, check))

    def _stmt_at(self, line: int | None):
        best = None
        for stmt in self._stmts:
            if line is not None and stmt[0] <= line <= stmt[1] and (best is None or stmt[0] > best[0]):
                best = stmt
        return best

    def on_line(self, line: int) -> None:
        stmt = self._stmt_at(line)
        if stmt is None or stmt is self._cur:
            return
        self._land()
        self._leave()
        self.snapshot(poke=True)
        self._cur = stmt
        self.add("line", line=stmt[0], end=stmt[1])

    def _leave(self) -> None:
        if self._cur is not None and self._cur[2] is not None:
            self.add("check", line=self._cur[0], ok=True, text=self._cur[2])

    def finish(self) -> None:
        self._land()
        self._leave()
        self._cur = None
        self.snapshot(poke=True)

    def fail(self, exc: BaseException, code) -> None:
        self._land()
        self.snapshot(poke=True)
        line, tb = None, exc.__traceback__
        while tb is not None:
            if tb.tb_frame.f_code is code:
                line = tb.tb_lineno
            tb = tb.tb_next
        stmt = self._stmt_at(line)
        text = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
        if stmt is not None and stmt[2] is not None and isinstance(exc, AssertionError):
            self.add("check", line=stmt[0], ok=False, text=stmt[2], why=text)
            text = f"assert {stmt[2]} failed" + (f": {exc}" if str(exc) else "")
        else:
            self.add("error", line=stmt[0] if stmt else line, node=None, text=text)
        self.status = "failed"
        self.failure = {"text": text, "line": stmt[0] if stmt else line, "frames": _user_frames(exc)}

    def run_error(self, exc: BaseException) -> None:
        text = f"{type(exc).__name__}: {exc}"
        frames = _user_frames(exc)
        where = frames[-1].split(":", 2)[:2] if frames else ["?", "?"]
        key = f"{type(exc).__name__} at {':'.join(where)}"
        self.snapshot()
        self.add("error", node=self.last_node, text=text, key=key)
        self.run_errors[key] += 1
        self.run_error_frames.setdefault(key, (text, frames))
        self.status = "errors"

    # -- turning the recording into an animation --

    def _hold(self, it: dict, nxt: dict | None) -> float:
        return self._base_hold(it, nxt) * PACE

    def _base_hold(self, it: dict, nxt: dict | None) -> float:
        k = it["k"]
        if k == "line":
            return LINE_HOLD_MS
        if k == "state":
            return POKE_HOLD_MS if it.get("poke") else 0
        if k == "check":
            return CHECK_HOLD_MS if it["ok"] else FAIL_HOLD_MS
        if k == "error":
            return RUN_ERROR_HOLD_MS if self.kind == "run" else FAIL_HOLD_MS
        if k == "deliver" or (k == "tend" and it.get("fired")):
            if nxt is not None and nxt["sim"] == it["sim"] and nxt["k"] in ("send", "state", "error"):
                return REACT_MS
        return 0

    def export(self) -> dict:
        items = self.items
        sent_at: dict[int, float] = {}
        latencies = []
        last_seen: dict[int, float] = {}  # when each message is last on the wire
        for it in items:
            if it["k"] == "send":
                sent_at[it["id"]] = last_seen[it["id"]] = it["sim"]
            elif it["k"] == "deliver":
                latencies.append(it["sim"] - sent_at[it["id"]])
                last_seen[it["id"]] = max(last_seen[it["id"]], it["sim"])
            elif it["k"] == "drop":
                last_seen[it["id"]] = max(last_seen[it["id"]], it["due"])
        hop = statistics.median(latencies) if latencies else 1
        scale = HOP_MS * PACE / max(hop, 1e-6)  # animation ms per simulated ms
        max_idle = MAX_IDLE_MS * PACE

        # Animation time: simulated time scaled to HOP_MS per typical hop, plus pauses so
        # the test code can be followed, with quiet stretches fast-forwarded.
        a = 0.0
        prev = items[0]["sim"] if items else 0
        keys: list[list[float]] = [[0.0, prev]]
        skips: list[list[float]] = []
        on_wire: list[float] = []
        for i, it in enumerate(items):
            s = max(it["sim"], prev)
            if s > prev:
                while on_wire and on_wire[0] <= prev:
                    heapq.heappop(on_wire)
                # Real time while something is on the wire, then fast-forward the quiet rest.
                busy = max(prev, min(s, max(on_wire))) if on_wire else prev
                keys.append([a, prev])
                a += (busy - prev) * scale
                keys.append([a, busy])
                gap = (s - busy) * scale
                if gap > max_idle:
                    skips.append([round(a, 1), round(a + max_idle, 1), busy, s])
                    gap = max_idle
                a += gap
                keys.append([a, s])
                prev = s
            it["a"] = a
            if it["k"] == "send":
                heapq.heappush(on_wire, last_seen[it["id"]])
            hold = self._hold(it, items[i + 1] if i + 1 < len(items) else None)
            if hold:
                a += hold
                keys.append([a, s])
        end = a + TAIL_MS * PACE
        keys.append([end, prev])

        sims = [k[1] for k in keys]

        def anim_of(sim: float) -> float:
            j = bisect_left(sims, sim)
            if j >= len(keys):
                return keys[-1][0]
            if j == 0:
                return keys[0][0]
            (a0, s0), (a1, s1) = keys[j - 1], keys[j]
            return a1 if s1 == s0 else a0 + (a1 - a0) * (sim - s0) / (s1 - s0)

        r = lambda x: round(x, 1)  # noqa: E731
        msgs: dict[int, dict] = {}
        states: dict[str, list] = {n: [] for n in self.nodes}
        timers: dict[str, list] = {}
        open_timers: dict[str, list] = {}
        lines, checks, errors, log = [], [], [], []
        for it in items:
            k, at, sim = it["k"], r(it["a"]), it["sim"]
            if k == "send":
                info = self._msgs[it["id"]]
                msgs[it["id"]] = m = {"id": it["id"], **info, "a0": at, "s0": sim, "arr": [], "sarr": [], "drops": []}
                log.append([at, sim, m["cls"], f"{m['frm']} → {m['to']}", m["lab"], m["desc"]])
            elif k == "deliver":
                m = msgs[it["id"]]
                again = " (duplicate)" if m["arr"] else ""
                m["arr"].append(at)
                m["sarr"].append(sim)
                log.append([at, sim, m["cls"], f"{m['to']} ← {m['frm']}", m["lab"] + again, m["desc"]])
            elif k == "drop":
                m = msgs[it["id"]]
                m["drops"].append([r(max(anim_of(it["due"]), it["a"] + 200)), it["rule"], it["due"]])
                log.append([at, sim, "drop", f"{m['frm']} → {m['to']}", f"{m['lab']} dropped by {it['rule']}",
                            m["desc"]])
            elif k == "state":
                states.setdefault(it["node"], []).append([at, it["st"]])
                if it["texts"]:
                    texts = it["texts"]
                    cls = it["st"]["role"] if texts[0].startswith("became") else "state"
                    log.append([at, sim, cls, it["node"], "; ".join(texts), ""])
            elif k == "tset":
                rec = [at, None, sim, it["due"], it["name"]]
                timers.setdefault(it["node"], []).append(rec)
                open_timers[it["tid"]] = rec
            elif k == "tend":
                rec = open_timers.pop(it["tid"], None)
                if rec is not None:
                    rec[1] = at
                if it["fired"]:
                    log.append([at, sim, "timer", it["node"], f"timer {it['name']} fired", ""])
            elif k == "line":
                lines.append([at, it["line"], it["end"]])
            elif k == "check":
                checks.append([at, it["line"], it["ok"], it["text"]])
                what = ("✓ assert " if it["ok"] else "✗ assert ") + it["text"]
                log.append([at, sim, "pass" if it["ok"] else "fail", "test", what, it.get("why", "")])
            elif k == "error":
                errors.append([at, it.get("node"), it.get("line"), it["text"], it.get("key")])
                log.append([at, sim, "err", it.get("node") or "test", it["text"], ""])
            elif k == "note":
                log.append([at, sim, "note", it.get("node") or "", it["text"], ""])
        for recs in timers.values():
            for rec in recs:
                if rec[1] is None:
                    rec[1] = r(end)
        for m in msgs.values():
            m["open"] = not m["arr"] and not m["drops"]
        if self.failure is not None:
            fails = [c[0] for c in checks if not c[2]] + [e[0] for e in errors]
            self.failure["a"] = min(fails) if fails else r(end)

        n_real = len([n for n in self.nodes if n not in self.puppets])
        if not self.doc:
            if self.kind == "run":
                self.doc = "Nothing scripted: the nodes act on their own timers."
            elif self.unit and not self.clusters:
                driven = ", ".join(n for n in self.nodes if n not in self.puppets)
                drivers = ", ".join(sorted(self.puppets)) or "nobody"
                self.doc = (f"Unit test: the test calls {driven}'s handlers directly and plays "
                            f"{drivers} itself. No network, no clock.")
            elif self.clusters:
                lo, hi = self.latency or (0, 0)
                self.doc = f"Simulated cluster of {n_real} nodes, network latency {lo}–{hi} ms."
        return {
            "name": self.name, "title": self.title, "kind": self.kind, "doc": self.doc,
            "status": self.status, "failure": self.failure, "output": self.output,
            "unit": self.unit and not self.clusters,
            "nodes": self.nodes, "puppets": sorted(self.puppets),
            "end": r(end), "hop": hop, "keys": _dedupe([[r(x), y] for x, y in keys]), "skips": skips,
            "msgs": list(msgs.values()), "states": states, "timers": timers,
            "lines": lines, "checks": checks, "errors": errors, "log": log,
            "source": self.source,
            "run_errors": [[key, n, *self.run_error_frames[key]] for key, n in self.run_errors.most_common()],
            "stats": {"messages": len(msgs), "sim_ms": prev, "nodes": n_real},
        }


def _dedupe(keys: list[list[float]]) -> list[list[float]]:
    out: list[list[float]] = []
    for k in keys:
        if not out or out[-1] != k:
            out.append(k)
    return out


# --- hooks and runners -------------------------------------------------------------

_current: World | None = None


@contextlib.contextmanager
def hooks():
    """While active, every Cluster and every RaftNode built joins the current World."""
    cluster_init, node_post_init = Cluster.__init__, RaftNode.__post_init__

    @functools.wraps(cluster_init)
    def init(self, *args, **kwargs):
        cluster_init(self, *args, **kwargs)
        if _current is not None:
            _current.attach_cluster(self)

    @functools.wraps(node_post_init)
    def post_init(self):
        node_post_init(self)
        if _current is not None and not isinstance(getattr(self, "transport", None), Endpoint):
            _current.attach_unit(self)

    Cluster.__init__, RaftNode.__post_init__ = init, post_init
    try:
        yield
    finally:
        Cluster.__init__, RaftNode.__post_init__ = cluster_init, node_post_init


def _line_tracer(world: World, code) -> Callable:
    def local(frame, event, arg):
        if event == "line":
            world.on_line(frame.f_lineno)
        return local

    def global_(frame, event, arg):
        return local if frame.f_code is code else None

    return global_


def record_test(fn: Callable) -> World:
    global _current
    world = World(fn.__name__, "test")
    world.doc = inspect.getdoc(fn) or ""
    world.watch(fn)
    out = io.StringIO()
    _current = world
    try:
        with contextlib.redirect_stdout(out):
            sys.settrace(_line_tracer(world, fn.__code__))
            try:
                fn()
            finally:
                sys.settrace(None)
    except Exception as exc:
        world.fail(exc, fn.__code__)
    else:
        world.finish()
    finally:
        _current = None
    world.output = out.getvalue()
    return world


def record_run(n: int, ms: int, seed: int) -> World:
    """Let main.py's nodes run on their own for ``ms`` simulated ms, carrying on past errors."""
    global _current
    world = World("free_run", "run")
    world.title = f"free run · {n} nodes"
    _current = world
    try:
        try:
            cluster = Cluster(functools.partial(RaftNode, seed=seed), n, seed=seed)
        except Exception as exc:  # the nodes could not even be built: show why instead of crashing
            world.run_error(exc)
            return world
        while True:
            try:
                cluster.run_until(ms)
                break
            except Exception as exc:
                world.run_error(exc)
                if sum(world.run_errors.values()) >= MAX_RUN_ERRORS:
                    world.add("note", node=None, text=f"stopped after {MAX_RUN_ERRORS} errors")
                    break
        world.snapshot()
    finally:
        _current = None
    world.doc = (f"main.py's RaftNode on its own: {n} nodes, seed {seed}, {ms} ms of simulated time. "
                 "Nothing scripted; the nodes act on their own timers. Errors raised by a node are "
                 "shown and the run carries on.")
    return world


def discover(module_name: str, pattern: str | None) -> list[Callable]:
    module = importlib.import_module(module_name)
    tests = [
        fn for name, fn in vars(module).items()
        if name.startswith("test") and inspect.isfunction(fn) and fn.__module__ == module.__name__
        and not any(p.default is p.empty for p in inspect.signature(fn).parameters.values())
    ]
    tests.sort(key=lambda fn: fn.__code__.co_firstlineno)
    if pattern:
        tests = [fn for fn in tests if pattern.lower() in fn.__name__.lower()]
    return tests


def main() -> None:
    parser = argparse.ArgumentParser(description="Animate the Raft tests and a free run of the nodes.")
    parser.add_argument("-k", dest="pattern", help="only tests whose name contains this")
    parser.add_argument("--module", default="test_raft", help="module to take tests from (default: %(default)s)")
    parser.add_argument("--no-run", action="store_true", help="skip the free run of the nodes")
    parser.add_argument("--nodes", type=int, default=4, help="free run: cluster size (default: %(default)s)")
    parser.add_argument("--ms", type=int, default=10_000, help="free run: simulated ms (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=0, help="free run: random seed (default: %(default)s)")
    parser.add_argument("-o", "--out", default="raft_viz.html", help="output file (default: %(default)s)")
    parser.add_argument("--no-open", action="store_true", help="don't open the result in a browser")
    parser.add_argument("--pace", type=float, default=1.0,
                        help="stretch the animation: 2 is twice as slow, 0.5 twice as fast (default: %(default)s)")
    args = parser.parse_args()
    global PACE
    PACE = args.pace

    worlds: list[World] = []
    with hooks():
        for fn in discover(args.module, args.pattern):
            world = record_test(fn)
            worlds.append(world)
            mark = "✓" if world.status == "passed" else "✗"
            extra = f"  — {world.failure['text']}" if world.failure else ""
            print(f"  {mark} {fn.__name__}{extra}")
        if not args.no_run:
            world = record_run(args.nodes, args.ms, args.seed)
            worlds.append(world)
            n_err = sum(world.run_errors.values())
            summary = f"{n_err} error{'s' if n_err != 1 else ''}" if n_err else "no errors"
            first = f"  (first: {next(iter(world.run_error_frames.values()))[0]})" if n_err else ""
            print(f"  • free run, {args.nodes} nodes, {args.ms} ms: {summary}{first}")
    if not worlds:
        sys.exit(f"no tests in {args.module} match {args.pattern!r}")

    payload = {
        "module": args.module,
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "worlds": [w.export() for w in worlds],
    }
    data = json.dumps(payload, default=repr, separators=(",", ":"), ensure_ascii=False)
    data = data.replace("</", "<\\/")  # never let message text close the script tag
    out = Path(args.out)
    out.write_text(TEMPLATE.replace("__VIZ_DATA__", data), encoding="utf-8")
    print(f"wrote {out}")
    if not args.no_open:
        webbrowser.open(out.resolve().as_uri())


# --- the page ------------------------------------------------------------------------

TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Raft Replay</title>
<style>
:root {
  --bg: #0b0e15; --bg-2: #0e121b; --panel: #121722; --panel-2: #171d2a; --raise: #1d2433;
  --line: #20283a; --line-2: #2c364b;
  --text: #e7eaf1; --muted: #8e98ac; --dim: #5d677c;
  --follower: #8b98b3; --candidate: #f5a524; --leader: #2fd38f;
  --rv: #f5a524; --yes: #3fdc98; --no: #ff6b6b; --ae: #5aa9ff; --hb: #7c9cc8; --ok: #35d0e0;
  --nack: #ff7aa2; --other: #b9bfcc; --drop: #ff4d4f; --err: #ff5c5c; --pass: #3fdc98;
  --fail: #ff5c5c; --dup: #ff9f43; --timer: #b48cff; --state: #c9d1e0; --note: #8e98ac;
  --accent: #7aa2ff; --cur: rgba(122, 162, 255, .14);
  --mono: "JetBrains Mono", "SF Mono", ui-monospace, Menlo, Consolas, monospace;
  --sans: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", Helvetica, Arial, sans-serif;
}
* { box-sizing: border-box; }
html, body { height: 100%; }
body { margin: 0; background: var(--bg); color: var(--text); font: 13px/1.45 var(--sans); overflow: hidden; }
button, select { font: inherit; color: inherit; }
.app { display: grid; height: 100vh; grid-template-columns: 250px minmax(0, 1fr) 430px;
       grid-template-rows: 54px minmax(0, 1fr) 62px;
       grid-template-areas: "top top top" "side stage right" "side bar right"; }

/* top bar */
.top { grid-area: top; display: flex; align-items: center; gap: 14px; padding: 0 18px;
       border-bottom: 1px solid var(--line); background: linear-gradient(180deg, #10141e, var(--bg)); min-width: 0; }
.brand { display: flex; align-items: center; gap: 10px; white-space: nowrap; }
.brand b { font-size: 15px; font-weight: 650; letter-spacing: .01em; }
.brand span { color: var(--muted); font-size: 12px; }
.legend { margin-left: auto; display: flex; gap: 12px; flex-wrap: nowrap; overflow: hidden; font-size: 11.5px; color: var(--muted); }
.lg { display: inline-flex; align-items: center; gap: 5px; white-space: nowrap; }
.lg i { width: 9px; height: 9px; border-radius: 50%; background: var(--c); box-shadow: 0 0 8px var(--c); }
.lg.role i { border-radius: 3px; box-shadow: none; border: 2px solid var(--c); background: transparent; }
.lg.tm i { border-radius: 50%; border-right-color: transparent; transform: rotate(45deg); }
.lg-sep { width: 1px; background: var(--line-2); align-self: stretch; margin: 2px 2px; }

/* sidebar */
.side { grid-area: side; border-right: 1px solid var(--line); background: var(--bg-2); overflow: auto;
        padding: 14px 10px; display: flex; flex-direction: column; gap: 4px; }
.side h3 { margin: 4px 8px 8px; font-size: 11px; letter-spacing: .1em; text-transform: uppercase; color: var(--muted);
           display: flex; justify-content: space-between; font-weight: 600; }
.side h3 em { font-style: normal; color: var(--text); letter-spacing: 0; text-transform: none; }
.worlds { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 3px; }
.w { display: grid; grid-template-columns: 22px 1fr; gap: 9px; align-items: start; padding: 9px 10px; border-radius: 10px;
     cursor: pointer; border: 1px solid transparent; transition: background .15s; }
.w:hover { background: var(--panel); }
.w.on { background: var(--panel-2); border-color: var(--line-2); box-shadow: inset 3px 0 0 var(--accent); }
.w .st { width: 20px; height: 20px; border-radius: 50%; display: grid; place-items: center; font-size: 11px; font-weight: 700; margin-top: 1px; }
.w.passed .st, .w.ok .st { background: rgba(63, 220, 152, .14); color: var(--pass); }
.w.failed .st { background: rgba(255, 92, 92, .16); color: var(--fail); }
.w.errors .st { background: rgba(245, 165, 36, .16); color: var(--candidate); }
.w .nm { font-weight: 560; line-height: 1.3; }
.w .sub { color: var(--muted); font: 11px/1.4 var(--mono); margin-top: 2px; }
.side .sep { height: 1px; background: var(--line); margin: 10px 6px; }
.keys { margin-top: auto; padding: 12px 8px 2px; color: var(--dim); font-size: 11.5px; line-height: 1.9; }
kbd { font: 10.5px var(--mono); background: var(--panel-2); border: 1px solid var(--line-2); border-bottom-width: 2px;
      border-radius: 4px; padding: 0 5px; color: var(--muted); }

/* stage */
.stage { grid-area: stage; position: relative; overflow: hidden; min-width: 0;
         background: radial-gradient(ellipse at 50% 45%, #121928 0%, var(--bg) 68%); }
.stage svg { position: absolute; inset: 0; width: 100%; height: 100%; display: block; }
.hud { position: absolute; left: 22px; top: 16px; right: 210px; pointer-events: none; }
.hud .kind { display: inline-block; font: 600 10px var(--mono); letter-spacing: .12em; text-transform: uppercase;
             padding: 2px 7px; border-radius: 5px; background: var(--panel-2); border: 1px solid var(--line-2); color: var(--muted); vertical-align: 3px; margin-right: 8px; }
.hud h2 { display: inline; font-size: 19px; font-weight: 650; margin: 0; letter-spacing: -.005em; }
.hud p { margin: 6px 0 0; color: var(--muted); max-width: 620px; font-size: 12.5px; }
.clock { position: absolute; right: 20px; top: 14px; text-align: right; pointer-events: none; }
.clock .big { font: 600 20px var(--mono); letter-spacing: -.02em; }
.clock .lbl { color: var(--dim); font: 10.5px var(--mono); text-transform: uppercase; letter-spacing: .1em; }
.ff { display: inline-block; margin-top: 6px; font: 600 11px var(--mono); color: var(--timer); background: rgba(180, 140, 255, .12);
      border: 1px solid rgba(180, 140, 255, .3); padding: 2px 8px; border-radius: 99px; visibility: hidden; }
.ff.on { visibility: visible; }
.ticker { position: absolute; left: 22px; right: 22px; bottom: 14px; font: 12px/1.7 var(--mono); pointer-events: none; }
.tk { display: flex; gap: 12px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; transition: opacity .2s; }
.tk .tt { color: var(--dim); width: 70px; text-align: right; flex: none; }
.tk .tw { color: var(--muted); width: 84px; flex: none; }
.tk .tx { color: var(--c, var(--text)); overflow: hidden; text-overflow: ellipsis; }
.toast { display: inline-block; margin-top: 12px; max-width: 100%; padding: 7px 13px; border-radius: 10px;
         background: rgba(255, 92, 92, .14); border: 1px solid rgba(255, 92, 92, .45); color: #ffd5d5; font: 12px/1.45 var(--mono);
         opacity: 0; transition: opacity .25s; pointer-events: none; }
.toast.on { opacity: 1; }
.toast.pass { background: rgba(63, 220, 152, .12); border-color: rgba(63, 220, 152, .4); color: #c9f7e2; }

.edge { stroke: #1d2535; stroke-width: 1.5; }
.edge.pup { stroke-dasharray: 3 6; }
.route { fill: none; stroke-width: 1.2; stroke-opacity: .2; stroke-dasharray: 3 5; }
.trail { fill: none; stroke-width: 3.5; stroke-linecap: round; stroke-opacity: .4; }
.dot-open { fill: var(--bg); stroke-width: 2; stroke-dasharray: 2.5 2; }
.mlab { font: 600 11.5px var(--mono); paint-order: stroke; stroke: var(--bg); stroke-width: 4px; stroke-linejoin: round; }
.xlab { font: 600 11px var(--mono); paint-order: stroke; stroke: var(--bg); stroke-width: 4px; fill: var(--drop); }
.nbody { stroke-width: 2.5; }
.nid { font: 700 16px var(--sans); fill: var(--text); text-anchor: middle; }
.nrole { font: 700 7.5px var(--sans); letter-spacing: .06em; text-anchor: middle; }
.tbadge { font: 700 10.5px var(--mono); fill: #0b0e15; text-anchor: middle; }
.nsub { font: 500 11px var(--sans); fill: var(--muted); text-anchor: middle; paint-order: stroke; stroke: var(--bg); stroke-width: 3px; }
.ncap { font: 500 10.5px var(--mono); fill: var(--muted); text-anchor: middle; paint-order: stroke; stroke: var(--bg); stroke-width: 3px; }
.logbg { fill: #0c1018; stroke: var(--line-2); stroke-width: 1; }
.lt { font: 600 10.5px var(--mono); text-anchor: middle; }
.empty { font: 500 10px var(--mono); fill: var(--dim); text-anchor: middle; }
.pup-c { fill: rgba(255, 255, 255, .015); stroke: var(--dim); stroke-width: 1.5; stroke-dasharray: 4 4; }
.dimt { fill: var(--muted); }

/* right panel */
.right { grid-area: right; border-left: 1px solid var(--line); background: var(--bg-2); display: flex; flex-direction: column; min-height: 0; }
.tabs { display: flex; gap: 2px; padding: 8px 10px 0; border-bottom: 1px solid var(--line); }
.tabs button { background: none; border: 0; padding: 8px 12px 9px; color: var(--muted); cursor: pointer; font-weight: 560;
               border-bottom: 2px solid transparent; margin-bottom: -1px; }
.tabs button.on { color: var(--text); border-bottom-color: var(--accent); }
.tabs button .n { font: 10.5px var(--mono); color: var(--dim); margin-left: 4px; }
.pane { display: none; flex: 1; min-height: 0; overflow: auto; }
.pane.on { display: block; }
.code-h { white-space: nowrap; position: sticky; top: 0; z-index: 1; background: var(--bg-2); padding: 10px 14px 8px; border-bottom: 1px solid var(--line);
          font: 12px var(--mono); color: var(--muted); display: flex; justify-content: space-between; gap: 8px; }
.code-h b { color: var(--text); font-weight: 600; }
.code-h span:first-child { overflow: hidden; text-overflow: ellipsis; }
.code-h span:last-child { flex: none; color: var(--pass); }
.code { padding: 6px 0 14px; font: 12.5px/1.7 var(--mono); }
.row { display: grid; grid-template-columns: 42px 20px 1fr; white-space: pre; cursor: pointer; }
.row:hover { background: rgba(255, 255, 255, .025); }
.row .g { color: var(--dim); text-align: right; padding-right: 4px; user-select: none; opacity: .7; }
.row.seen .g { opacity: 1; }
.row .m { text-align: center; font-weight: 700; }
.row .m.ok { color: var(--pass); }
.row .m.bad { color: var(--fail); }
.row.cur { background: var(--cur); box-shadow: inset 3px 0 0 var(--accent); }
.row.cur .g { color: var(--accent); }
.row.failed-line { background: rgba(255, 92, 92, .12); box-shadow: inset 3px 0 0 var(--fail); }
.row .s { padding-right: 14px; color: #c8cfdc; }
.hk { color: #c792ea; } .hs { color: #c3e88d; } .hn { color: #f78c6c; } .hc { color: #5d677c; font-style: italic; } .hf { color: #82aaff; }
.fail { margin: 12px 14px; border: 1px solid rgba(255, 92, 92, .4); background: rgba(255, 92, 92, .07); border-radius: 10px; padding: 10px 12px; }
.fail h4 { margin: 0 0 6px; color: #ffb3b3; font-size: 12.5px; }
.fail pre, .runbox pre, details pre { margin: 6px 0 0; font: 11.5px/1.55 var(--mono); white-space: pre-wrap; word-break: break-word; color: #d6dbe6; }
details pre { white-space: pre; word-break: normal; overflow-x: auto; padding-bottom: 6px; }
details { margin: 12px 14px; color: var(--muted); }
details summary { cursor: pointer; font-size: 12px; }
.runbox { padding: 14px; color: var(--muted); }
.runbox h4 { color: var(--text); margin: 0 0 6px; font-size: 13px; }
.runbox .err { margin: 10px 0; border: 1px solid var(--line-2); border-radius: 10px; padding: 10px 12px; background: var(--panel); cursor: pointer; }
.runbox .err:hover { border-color: rgba(255, 92, 92, .5); }
.runbox .err b { color: #ffb3b3; font: 600 12px var(--mono); }
.runbox .err .cnt { float: right; color: var(--muted); font: 11px var(--mono); }
.events { list-style: none; margin: 0; padding: 6px 0 14px; font: 11.5px/1.75 var(--mono); }
.ev { display: grid; grid-template-columns: 62px 82px 1fr; gap: 8px; padding: 0 12px; color: var(--dim); cursor: pointer; white-space: nowrap; }
.ev span:last-child { overflow: hidden; text-overflow: ellipsis; }
.ev.past { color: var(--muted); }
.ev.past span:last-child { color: var(--c, var(--text)); }
.ev.now { background: var(--cur); }
.ev .t { text-align: right; }
.c-rv { --c: var(--rv); } .c-yes { --c: var(--yes); } .c-no { --c: var(--no); } .c-ae { --c: var(--ae); }
.c-hb { --c: var(--hb); } .c-ok { --c: var(--ok); } .c-nack { --c: var(--nack); } .c-other { --c: var(--other); }
.c-drop { --c: var(--drop); } .c-err, .c-fail { --c: var(--fail); } .c-pass { --c: var(--pass); } .c-timer { --c: var(--timer); }
.c-state { --c: var(--state); } .c-note { --c: var(--note); } .c-leader { --c: var(--leader); } .c-candidate { --c: var(--candidate); }
.c-follower { --c: var(--follower); }
.stable { width: 100%; border-collapse: collapse; font: 11.5px/1.5 var(--mono); }
.stable th { text-align: left; color: var(--dim); font-weight: 500; padding: 10px 12px 6px; border-bottom: 1px solid var(--line); position: sticky; top: 0; background: var(--bg-2); }
.stable td { padding: 9px 12px; border-bottom: 1px solid var(--line); vertical-align: top; }
.stable td.k { color: var(--muted); width: 92px; }
.nodecard { padding: 12px 14px; border-bottom: 1px solid var(--line); }
.nodecard h4 { margin: 0 0 8px; font-size: 13px; display: flex; align-items: center; gap: 8px; }
.nodecard h4 .role { font: 700 9.5px var(--sans); letter-spacing: .12em; text-transform: uppercase; padding: 2px 7px; border-radius: 5px; color: #0b0e15; }
.kv { display: grid; grid-template-columns: 96px 1fr; gap: 2px 10px; font: 11.5px/1.6 var(--mono); }
.kv .k { color: var(--dim); }
.chips { display: flex; flex-wrap: wrap; gap: 4px; }
.chip { font: 11px var(--mono); padding: 0 6px; border-radius: 4px; border: 1px solid var(--tc); color: var(--tc); }
.chip.c { background: var(--tc); color: #0b0e15; }

/* transport bar */
.bar { grid-area: bar; display: flex; align-items: center; gap: 8px; padding: 0 16px; border-top: 1px solid var(--line); background: var(--bg-2); min-width: 0; }
.ib { width: 34px; height: 34px; border-radius: 50%; border: 1px solid var(--line-2); background: var(--panel); display: grid; place-items: center; cursor: pointer; flex: none; }
.ib:hover { background: var(--raise); }
.ib svg { width: 14px; height: 14px; fill: var(--text); }
.ib.play { width: 42px; height: 42px; background: var(--accent); border-color: transparent; }
.ib.play svg { fill: #0b0e15; width: 16px; height: 16px; }
.scrub { flex: 1; position: relative; height: 36px; cursor: pointer; margin: 0 8px; min-width: 80px; touch-action: none; }
.track { position: absolute; left: 0; right: 0; top: 50%; height: 6px; transform: translateY(-50%); background: var(--line); border-radius: 3px; }
.fill { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 3px; background: linear-gradient(90deg, #5a7fe0, var(--accent)); }
.head { position: absolute; top: 50%; width: 14px; height: 14px; margin: -7px 0 0 -7px; border-radius: 50%; background: #fff; box-shadow: 0 0 0 4px rgba(122, 162, 255, .25); }
.mk { position: absolute; pointer-events: none; }
.mk.line { top: -5px; width: 1px; height: 16px; background: rgba(255, 255, 255, .1); }
.mk.ok, .mk.bad, .mk.err { top: -1px; width: 8px; height: 8px; margin-left: -4px; border-radius: 50%; }
.mk.ok { background: var(--pass); top: 9px; width: 5px; height: 5px; margin-left: -2.5px; }
.mk.bad, .mk.err { background: var(--fail); top: -9px; width: 5px; height: 5px; margin-left: -2.5px; }
.mk.bad { width: 8px; height: 8px; margin-left: -4px; top: -11px; box-shadow: 0 0 6px var(--fail); }
.mk.skip { top: 0; bottom: 0; background: repeating-linear-gradient(45deg, rgba(180, 140, 255, .55) 0 2px, transparent 2px 5px); }
.time { font: 12px var(--mono); color: var(--muted); white-space: nowrap; min-width: 112px; text-align: right; }
.time b { color: var(--text); font-weight: 600; }
.bar select { background: var(--panel); border: 1px solid var(--line-2); border-radius: 8px; padding: 5px 6px; font: 12px var(--mono); }
.bar label { display: flex; align-items: center; gap: 5px; color: var(--muted); font-size: 12px; white-space: nowrap; cursor: pointer; }
.bar input { accent-color: var(--accent); }

@media (max-width: 1280px) { .app { grid-template-columns: 210px minmax(0, 1fr) 360px; } .legend .lg.msg-x { display: none; } }
@media (max-width: 980px) {
  body { overflow: auto; }
  .app { height: auto; grid-template-columns: minmax(0, 1fr); grid-template-rows: auto auto 62vh auto 60vh;
         grid-template-areas: "top" "side" "stage" "bar" "right"; }
  .legend { display: none; }
  .side { border-right: 0; border-bottom: 1px solid var(--line); }
  .keys { display: none; }
  .hud { right: 16px; } .clock { top: auto; bottom: 120px; }
  .bar { flex-wrap: wrap; padding: 10px 16px; gap: 8px; }
  .scrub { flex-basis: 100%; order: -1; margin: 0; }
}
</style>
</head>
<body>
<div class="app">
  <header class="top">
    <div class="brand">
      <svg width="24" height="24" viewBox="0 0 24 24" aria-hidden="true"><g fill="none" stroke="#7aa2ff" stroke-width="1.6"><path d="M12 4 L4.5 17 L19.5 17 Z" opacity=".55"/></g><circle cx="12" cy="4" r="3" fill="#2fd38f"/><circle cx="4.5" cy="17" r="3" fill="#8b98b3"/><circle cx="19.5" cy="17" r="3" fill="#8b98b3"/></svg>
      <b>Raft replay</b><span id="sub"></span>
    </div>
    <div class="legend">
      <span class="lg role" style="--c:var(--follower)"><i></i>follower</span>
      <span class="lg role" style="--c:var(--candidate)"><i></i>candidate</span>
      <span class="lg role" style="--c:var(--leader)"><i></i>leader</span>
      <span class="lg role tm" style="--c:var(--timer)"><i></i>election timer</span>
      <span class="lg-sep"></span>
      <span class="lg" style="--c:var(--rv)"><i></i>RequestVote</span>
      <span class="lg" style="--c:var(--yes)"><i></i>vote ✓</span>
      <span class="lg msg-x" style="--c:var(--no)"><i></i>vote ✗</span>
      <span class="lg" style="--c:var(--ae)"><i></i>AppendEntries</span>
      <span class="lg msg-x" style="--c:var(--hb)"><i></i>heartbeat</span>
      <span class="lg" style="--c:var(--ok)"><i></i>ok</span>
      <span class="lg msg-x" style="--c:var(--nack)"><i></i>reject</span>
      <span class="lg msg-x" style="--c:var(--other)"><i></i>other</span>
    </div>
  </header>

  <aside class="side">
    <h3>Tests <em id="sum"></em></h3>
    <ul class="worlds" id="tests"></ul>
    <div class="sep" id="runsep" hidden></div>
    <h3 id="runh" hidden>Nodes on their own</h3>
    <ul class="worlds" id="runs"></ul>
    <div class="keys">
      <kbd>space</kbd> play / pause<br>
      <kbd>←</kbd> <kbd>→</kbd> previous / next event<br>
      <kbd>↑</kbd> <kbd>↓</kbd> previous / next test<br>
      <kbd>L</kbd> labels &nbsp; <kbd>+</kbd> <kbd>−</kbd> speed
    </div>
  </aside>

  <section class="stage" id="stage">
    <svg id="svg" viewBox="0 0 1000 620" preserveAspectRatio="xMidYMid meet">
      <defs>
        <filter id="glow" x="-100%" y="-100%" width="300%" height="300%">
          <feGaussianBlur stdDeviation="3.5" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
        <filter id="soft" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="8"/></filter>
      </defs>
      <g id="static"></g>
      <g id="dyn"></g>
    </svg>
    <div class="hud"><span class="kind" id="wkind"></span><h2 id="wtitle"></h2><p id="wdoc"></p><div class="toast" id="toast"></div></div>
    <div class="clock"><div class="lbl" id="clbl">simulated time</div><div class="big" id="sim"></div><div class="ff" id="ff">⏩ fast-forward</div></div>
    <div class="ticker" id="ticker"></div>
  </section>

  <section class="right">
    <div class="tabs">
      <button data-tab="code" class="on">Code</button>
      <button data-tab="events">Events<span class="n" id="nev"></span></button>
      <button data-tab="state">State</button>
    </div>
    <div class="pane on" id="pane-code"></div>
    <div class="pane" id="pane-events"><ol class="events" id="events"></ol></div>
    <div class="pane" id="pane-state"></div>
  </section>

  <footer class="bar">
    <button class="ib" id="restart" title="restart (Home)"><svg viewBox="0 0 16 16"><path d="M3 2h2v12H3zM14 2v12L6 8z"/></svg></button>
    <button class="ib" id="prev" title="previous event (←)"><svg viewBox="0 0 16 16"><path d="M9 3v10L2 8zM15 3v10L8 8z"/></svg></button>
    <button class="ib play" id="play" title="play / pause (space)"><svg viewBox="0 0 16 16" id="playicon"><path d="M4 2l10 6-10 6z"/></svg></button>
    <button class="ib" id="next" title="next event (→)"><svg viewBox="0 0 16 16"><path d="M7 3v10l7-5zM1 3v10l7-5z"/></svg></button>
    <div class="scrub" id="scrub"><div class="track"><div class="fill" id="fill"></div><div id="marks"></div><div class="head" id="head"></div></div></div>
    <div class="time" id="time"></div>
    <select id="speed" title="playback speed"></select>
    <label title="show message labels"><input type="checkbox" id="labels" checked> labels</label>
    <label title="go on to the next test when one ends"><input type="checkbox" id="chain" checked> play all</label>
  </footer>
</div>

<script id="viz-data" type="application/json">__VIZ_DATA__</script>
<script>
(() => {
"use strict";
const DATA = JSON.parse(document.getElementById("viz-data").textContent);
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (v) => (v !== null && typeof v === "object") ? JSON.stringify(v) : String(v);
const short = (s, n) => (s.length > n ? s.slice(0, n - 1) + "…" : s);
const cssv = getComputedStyle(document.documentElement);
const C = {};
for (const k of ["follower", "candidate", "leader", "rv", "yes", "no", "ae", "hb", "ok", "nack", "other", "drop", "err",
                 "pass", "dup", "muted", "dim", "text", "line-2", "timer", "bg"]) C[k] = cssv.getPropertyValue("--" + k).trim();
const TERM = ["#5aa9ff", "#c08cff", "#ff7ac6", "#ff9f43", "#3fdc98", "#ffd43b", "#2ee6c9", "#e879f9"];
const termColor = (t) => (t > 0 ? TERM[(t - 1) % TERM.length] : C.dim);
const R = 34;
const SPEEDS = [0.25, 0.5, 1, 2, 4, 8, 16];
const V = { i: -1, w: null, A: 0, playing: false, speed: 1, last: 0, labels: true, chain: true, dirty: true, advanceAt: 0, tab: "code" };

function lastLE(arr, x, key) {   // index of the last element with key <= x, or -1
  let lo = 0, hi = arr.length - 1, ans = -1;
  while (lo <= hi) { const m = (lo + hi) >> 1; if (key(arr[m]) <= x) { ans = m; lo = m + 1; } else hi = m - 1; }
  return ans;
}
const k0 = (x) => x[0], id_ = (x) => x;
const ease = (t) => (t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2);
const f1 = (x) => x.toFixed(1);

// ---- per-test preparation ------------------------------------------------------------------
function prep(w) {
  if (w._p) return w;
  w._p = true;
  const n = w.nodes.length, cx = 500, cy = 300;
  const rad = n <= 1 ? 0 : n === 2 ? 210 : Math.min(265, 180 + 17 * n);
  const start = -Math.PI / 2 + (n % 2 === 0 ? Math.PI / n : 0);
  w.pos = {};
  w.nodes.forEach((id, i) => {
    const t = start + (2 * Math.PI * i) / n;
    w.pos[id] = { x: cx + rad * Math.cos(t), y: cy + rad * Math.sin(t) * 0.82 };
  });
  w.pup = new Set(w.puppets);
  w.geo = {};
  w.roleAt = {};
  for (const id of w.nodes) {
    const list = w.states[id] || [], ch = [];
    for (let j = 1; j < list.length; j++) if (list[j][1].role !== list[j - 1][1].role) ch.push(list[j][0]);
    w.roleAt[id] = ch;
  }
  w.errAt = {};
  for (const e of w.errors) if (e[1]) (w.errAt[e[1]] = w.errAt[e[1]] || []).push(e[0]);
  for (const m of w.msgs) {
    let until = m.a0 + 900;
    for (const a of m.arr) until = Math.max(until, a + 900);
    for (const d of m.drops) until = Math.max(until, d[0] + 900);
    m.until = m.open ? Infinity : until;
  }
  const moments = new Set();
  for (const e of w.log) moments.add(e[0]);
  for (const l of w.lines) moments.add(l[0]);
  w.moments = [...moments].sort((a, b) => a - b);
  w.notable = w.checks.filter((c) => !c[2]).map((c) => [c[0], "✗ assert " + c[3]]).concat(w.errors.map((e) => [e[0], (e[1] ? e[1] + ": " : "") + e[3]])).sort((a, b) => a[0] - b[0]);
  return w;
}

function simAt(w, A) {
  const K = w.keys;
  if (!K.length) return 0;
  const i = lastLE(K, A, k0);
  if (i < 0) return K[0][1];
  if (i >= K.length - 1) return K[K.length - 1][1];
  const [a0, s0] = K[i], [a1, s1] = K[i + 1];
  return a1 > a0 ? s0 + ((s1 - s0) * (A - a0)) / (a1 - a0) : s1;
}

function geom(w, a, b) {
  const key = a + ">" + b;
  if (key in w.geo) return w.geo[key];
  const P = w.pos[a], Q = w.pos[b];
  let g = null;
  if (P && Q && a !== b) {
    const dx = Q.x - P.x, dy = Q.y - P.y, d = Math.hypot(dx, dy), nx = dy / d, ny = -dx / d;
    const c = { x: (P.x + Q.x) / 2 + nx * d * 0.13, y: (P.y + Q.y) / 2 + ny * d * 0.13 };
    const gap = R + 9;
    const toward = (from, to) => { const ex = to.x - from.x, ey = to.y - from.y, l = Math.hypot(ex, ey); return { x: from.x + (ex / l) * gap, y: from.y + (ey / l) * gap }; };
    const p0 = toward(P, c), p2 = toward(Q, c);
    g = { p0, c, p2, nx, ny, P, Q, d: `M${f1(p0.x)},${f1(p0.y)} Q${f1(c.x)},${f1(c.y)} ${f1(p2.x)},${f1(p2.y)}` };
  }
  return (w.geo[key] = g);
}
function bez(g, t) {
  const u = 1 - t;
  return { x: u * u * g.p0.x + 2 * u * t * g.c.x + t * t * g.p2.x, y: u * u * g.p0.y + 2 * u * t * g.c.y + t * t * g.p2.y };
}
const ring = (p, r, col, op, sw) => `<circle cx="${f1(p.x)}" cy="${f1(p.y)}" r="${f1(r)}" fill="none" stroke="${col}" stroke-width="${sw}" opacity="${op.toFixed(2)}"/>`;

// ---- drawing ----------------------------------------------------------------------------------
function drawStatic(w) {
  let s = "";
  const ids = w.nodes;
  for (let i = 0; i < ids.length; i++) for (let j = i + 1; j < ids.length; j++) {
    const a = ids[i], b = ids[j], pa = w.pup.has(a), pb = w.pup.has(b);
    if (pa && pb) continue;
    const P = w.pos[a], Q = w.pos[b];
    s += `<line x1="${f1(P.x)}" y1="${f1(P.y)}" x2="${f1(Q.x)}" y2="${f1(Q.y)}" class="edge${pa || pb ? " pup" : ""}"/>`;
  }
  $("static").innerHTML = s;
}

function fly(g, f, col, m, dup, open, out) {
  const e = ease(Math.max(0, Math.min(1, f))), p = bez(g, e);
  const near = (out.seen[g.d] = out.seen[g.d] || []);
  const slot = near.filter((x) => Math.abs(x - e) < 0.08).length;
  near.push(e);
  out.shapes += `<path d="${g.d}" class="route" stroke="${col}"/>`;
  const pts = [];
  for (let j = 0; j <= 8; j++) { const q = bez(g, Math.max(0, e - 0.17 * (1 - j / 8))); pts.push(f1(q.x) + "," + f1(q.y)); }
  out.shapes += `<polyline points="${pts.join(" ")}" class="trail" stroke="${col}"/>`;
  const r = m.cls === "hb" ? 5 : 7;
  if (open) out.shapes += `<circle cx="${f1(p.x)}" cy="${f1(p.y)}" r="${r}" class="dot-open" stroke="${col}"/>`;
  else out.shapes += `<circle cx="${f1(p.x)}" cy="${f1(p.y)}" r="${r}" fill="${col}" filter="url(#glow)"/>`;
  if (dup) out.shapes += `<circle cx="${f1(p.x)}" cy="${f1(p.y)}" r="${r + 4}" fill="none" stroke="${C.dup}" stroke-width="1.5" stroke-dasharray="2 2"/>`;
  if (V.labels) {
    const anchor = g.nx > 0.35 ? "start" : g.nx < -0.35 ? "end" : "middle";
    const text = m.lab + (dup ? " (dup)" : "") + (open ? " …" : "");
    const stack = slot * 14 * (g.ny < 0 ? -1 : 1);   // pile labels of messages bunched on one route
    out.labels += `<text x="${f1(p.x + g.nx * 14)}" y="${f1(p.y + g.ny * 14 + 4 + stack)}" class="mlab" fill="${col}" text-anchor="${anchor}" opacity="${open ? 0.6 : 1}">${esc(text)}</text>`;
  }
}
function cross(g, m, rule, op, out) {
  const p = bez(g, 0.5);
  out.shapes += `<g opacity="${op.toFixed(2)}" stroke="${C.drop}" stroke-width="3" stroke-linecap="round"><path d="M${f1(p.x - 7)},${f1(p.y - 7)} l14,14 M${f1(p.x + 7)},${f1(p.y - 7)} l-14,14"/></g>`;
  if (V.labels) out.labels += `<text x="${f1(p.x + g.nx * 16)}" y="${f1(p.y + g.ny * 16 + 4)}" class="xlab" text-anchor="middle" opacity="${op.toFixed(2)}">${esc(m.lab)} ✗ ${esc(rule)}</text>`;
}

// Messages move with simulated time, so they hold still while the test code runs (the network is
// paused then); a message is only shown as landed once the receiver actually handles it.
const along = (sim, s0, s1) => (s1 > s0 ? (sim - s0) / (s1 - s0) : 1);
function drawMessages(w, A, sim, out) {
  for (const m of w.msgs) {
    if (m.a0 > A) break;
    if (m.until < A) continue;
    const g = geom(w, m.frm, m.to);
    if (!g) continue;
    const col = C[m.cls] || C.other;
    if (A - m.a0 < 320) { const q = (A - m.a0) / 320; out.shapes += ring(g.P, R + 3 + 12 * q, col, 0.6 * (1 - q), 2); }
    m.arr.forEach((a1, i) => {
      if (A < a1) fly(g, Math.min(0.97, along(sim, m.s0, m.sarr[i])), col, m, i > 0, false, out);
      else if (A - a1 < 900) { const q = (A - a1) / 900; out.shapes += ring(g.Q, R + 3 + 26 * ease(q), col, 0.85 * (1 - q), 2.5); }
    });
    for (const [aEnd, rule, due] of m.drops) {
      if (A < aEnd) {
        const f = along(sim, m.s0, due);
        if (f < 0.5) fly(g, f, col, m, false, false, out); else cross(g, m, rule, 1, out);
      } else if (A - aEnd < 500) cross(g, m, rule, 1 - (A - aEnd) / 500, out);
    }
    if (m.open) fly(g, Math.min(0.8, along(sim, m.s0, m.s0 + w.hop)), col, m, false, true, out);
  }
}

function timerArc(w, id, A, sim, r) {
  const list = w.timers[id];
  if (!list || !list.length) return "";
  const i = lastLE(list, A, k0);
  if (i < 0) return "";
  const [, a1, s0, due] = list[i];
  if (A >= a1) return "";
  const frac = Math.max(0, Math.min(1, 1 - (sim - s0) / Math.max(1e-6, due - s0)));
  const rr = r + 7, circ = 2 * Math.PI * rr, col = frac < 0.2 ? C.candidate : C.timer;
  return `<circle r="${f1(rr)}" fill="none" stroke="${C.timer}" stroke-opacity=".12" stroke-width="2.5"/>` +
         `<circle r="${f1(rr)}" fill="none" stroke="${col}" stroke-opacity=".75" stroke-width="2.5" stroke-linecap="round" stroke-dasharray="${f1(circ * frac)} ${f1(circ)}" transform="rotate(-90)"/>`;
}

function crown(r, col) {
  const y = -r - 9;
  return `<path d="M-12,${y} L-14,${y - 13} L-6,${y - 6} L0,${y - 16} L6,${y - 6} L14,${y - 13} L12,${y} Z" fill="${col}" stroke="${C.bg}" stroke-width="1.5" stroke-linejoin="round"/>`;
}

function logStrip(st, prev, age, y) {
  const log = st.log || [], n = log.length, ch = 22, gap = 4;
  const texts = log.map(([, k, v]) => short(`${k}=${fmt(v)}`, 8));
  let widths = texts.map((t) => Math.max(26, t.length * 7 + 12));
  let total = widths.reduce((a, b) => a + b, 0) + gap * Math.max(0, n - 1);
  let compact = false;
  if (total > 290) { compact = true; const cw = Math.max(8, Math.min(18, Math.floor(290 / n) - 3)); widths = log.map(() => cw); total = n * cw + 3 * Math.max(0, n - 1); }
  const g2 = compact ? 3 : gap;
  if (!n) total = 64;
  let common = n, c0 = 0, c1 = 0;
  const glow = Math.max(0, 1 - age / 900);
  if (prev && glow > 0) {
    const pl = prev.log || [];
    common = 0;
    while (common < pl.length && common < n && JSON.stringify(pl[common]) === JSON.stringify(log[common])) common++;
    if (st.commit > prev.commit) { c0 = prev.commit; c1 = st.commit; }
  }
  let s = `<g transform="translate(${f1(-total / 2)},${y})"><rect x="-6" y="-5" width="${f1(total + 12)}" height="${ch + 10}" rx="8" class="logbg"/>`;
  if (!n) s += `<text x="${total / 2}" y="15" class="empty">empty log</text>`;
  let x = 0;
  log.forEach(([t], i) => {
    const w = widths[i], col = termColor(t), committed = i < st.commit;
    if (i >= common && glow > 0) s += `<rect x="${x - 3}" y="-3" width="${w + 6}" height="${ch + 6}" rx="6" fill="${col}" opacity="${(0.5 * glow).toFixed(2)}" filter="url(#soft)"/>`;
    s += `<rect x="${x}" width="${w}" height="${ch}" rx="4" fill="${committed ? col : "none"}" stroke="${col}" stroke-width="1.5"${committed ? "" : ' stroke-dasharray="3 2"'}/>`;
    if (!compact) s += `<text x="${x + w / 2}" y="15" class="lt" fill="${committed ? "#0b0e15" : col}">${esc(texts[i])}</text>`;
    if (i >= c0 && i < c1 && glow > 0) s += `<rect x="${x - 2}" y="-2" width="${w + 4}" height="${ch + 4}" rx="5" fill="none" stroke="#fff" stroke-width="2" opacity="${glow.toFixed(2)}"/>`;
    x += w + g2;
  });
  s += `</g>`;
  const store = Object.entries(st.store || {}).map(([k, v]) => `${k}=${fmt(v)}`).join(" ");
  s += `<text y="${y + ch + 22}" class="ncap">commit ${st.commit}${store ? " · " + esc(short(store, 30)) : ""}</text>`;
  return s;
}

function drawNodes(w, A, sim, out) {
  const need = Math.floor(w.nodes.length / 2) + 1;
  for (const id of w.nodes) {
    const P = w.pos[id];
    if (w.pup.has(id)) {
      out.shapes += `<g transform="translate(${f1(P.x)},${f1(P.y)})"><circle r="${R}" class="pup-c"/><text y="3" class="nid dimt">${esc(id)}</text><text y="18" class="nrole" fill="${C.dim}">TEST</text><text y="${R + 18}" class="nsub">played by the test</text></g>`;
      continue;
    }
    const list = w.states[id] || [];
    const si = lastLE(list, A, k0);
    const born = si >= 0;
    const rec = born ? list[si] : list[0];
    const st = rec ? rec[1] : { role: "follower", term: 0, log: [], commit: 0, votes: [], voted_for: "", leader: "", store: {} };
    const prev = si > 0 ? list[si - 1][1] : null;
    const age = born ? A - rec[0] : 1e9;
    const role = C[st.role] ? st.role : "follower";
    const col = C[role];
    const rc = w.roleAt[id], ri = lastLE(rc, A, id_), rq = ri >= 0 ? (A - rc[ri]) / 800 : 9;
    const r = R * (rq < 1 ? 1 + 0.2 * Math.sin(Math.PI * rq) : 1);
    const ea = w.errAt[id] || [], ei = lastLE(ea, A, id_), eq = ei >= 0 ? (A - ea[ei]) / 900 : 9;
    const shake = eq < 1 ? Math.sin(eq * 46) * 6 * (1 - eq) : 0;
    let s = `<g transform="translate(${f1(P.x + shake)},${f1(P.y)})" opacity="${born ? 1 : 0.25}">`;
    if (role === "leader") {
      const q = (A % 2000) / 2000;
      s += `<circle r="${r + 16}" fill="${col}" opacity=".10" filter="url(#soft)"/>`;
      s += `<circle r="${f1(r + 10 + 18 * q)}" fill="none" stroke="${col}" stroke-width="1.5" opacity="${(0.5 * (1 - q)).toFixed(2)}"/>`;
    }
    if (role === "candidate") s += `<circle r="${r + 13}" fill="none" stroke="${col}" stroke-width="1.5" stroke-dasharray="5 7" opacity=".7" transform="rotate(${f1((A / 12) % 360)})"/>`;
    s += timerArc(w, id, A, sim, r);
    s += `<circle r="${f1(r)}" class="nbody" fill="${col}" fill-opacity=".14" stroke="${col}"/>`;
    if (rq < 1) s += `<circle r="${f1(r + 4 + 22 * rq)}" fill="none" stroke="${col}" stroke-width="3" opacity="${(1 - rq).toFixed(2)}"/>`;
    if (eq < 1) s += `<circle r="${f1(r + 6 + 18 * eq)}" fill="none" stroke="${C.err}" stroke-width="3" opacity="${(1 - eq).toFixed(2)}"/><circle r="${f1(r)}" fill="${C.err}" opacity="${(0.25 * (1 - eq)).toFixed(2)}"/>`;
    s += `<text y="3" class="nid">${esc(id)}</text><text y="18" class="nrole" fill="${col}">${role.toUpperCase()}</text>`;
    s += `<g transform="translate(${f1(r * 0.76)},${f1(-r * 0.76)})"><rect x="-15" y="-10" width="30" height="20" rx="10" fill="${termColor(st.term)}" stroke="${C.bg}" stroke-width="2"/><text y="4" class="tbadge">T${st.term}</text></g>`;
    if (role === "leader") s += crown(r, col);
    const sub = [];
    if (role === "candidate") sub.push(`votes ${1 + (st.votes || []).length}/${need}`);
    if (role !== "leader" && st.voted_for && st.voted_for !== id) sub.push(`voted ${st.voted_for}`);
    if (role !== "leader" && st.leader) sub.push(`leader ${st.leader}`);
    if (role === "leader") sub.push(`leading term ${st.term}`);
    s += `<text y="${f1(R + 22)}" class="nsub">${esc(sub.join(" · "))}</text>`;
    s += logStrip(st, prev, age, R + 34);
    s += `</g>`;
    out.shapes += s;
  }
}

// ---- side panels -----------------------------------------------------------------------------
const KW = new Set("def return assert for in if else elif while not and or is None True False lambda import from with as class try except finally raise pass break continue yield global nonlocal del".split(" "));
function highlight(line) {
  const re = /(#.*$)|([rbfu]?"(?:[^"\\]|\\.)*"|[rbfu]?'(?:[^'\\]|\\.)*')|(\b\d[\d_]*(?:\.\d+)?\b)|([A-Za-z_]\w*)/g;
  let out = "", last = 0, m;
  while ((m = re.exec(line))) {
    out += esc(line.slice(last, m.index));
    if (m[1]) out += `<span class="hc">${esc(m[1])}</span>`;
    else if (m[2]) out += `<span class="hs">${esc(m[2])}</span>`;
    else if (m[3]) out += `<span class="hn">${esc(m[3])}</span>`;
    else if (KW.has(m[4])) out += `<span class="hk">${m[4]}</span>`;
    else if (/^\s*\(/.test(line.slice(re.lastIndex))) out += `<span class="hf">${esc(m[4])}</span>`;
    else out += esc(m[4]);
    last = re.lastIndex;
  }
  return out + esc(line.slice(last));
}

function buildCode(w) {
  const pane = $("pane-code");
  w._rows = {}; w._li = -2; w._ci = -2; w._failShown = null;
  let h = "";
  if (w.source) {
    const nOk = w.checks.filter((c) => c[2]).length, n = w.checks.length;
    const tally = n ? `✓ ${nOk}/${n}` : "";
    h += `<div class="code-h"><span>${esc(w.source.file)} · <b>${esc(w.name)}</b></span><span>${tally}</span></div><div class="code">`;
    w.source.lines.forEach((line, i) => {
      const ln = w.source.first + i;
      h += `<div class="row" data-ln="${ln}"><span class="g">${ln}</span><span class="m"></span><span class="s">${highlight(line) || " "}</span></div>`;
    });
    h += `</div>`;
  } else {
    h += `<div class="runbox"><h4>${esc(w.title)}</h4><p>${esc(w.doc)}</p>`;
    h += `<p>${w.stats.messages} messages · ${Math.round(w.stats.sim_ms)} ms simulated · ${w.errors.length} error${w.errors.length === 1 ? "" : "s"}</p>`;
    for (const [key, count, text, frames] of w.run_errors) {
      const first = w.errors.find((e) => e[4] === key);
      h += `<div class="err" data-a="${first ? first[0] : 0}" title="jump to the first one"><span class="cnt">×${count}</span><b>${esc(key)}</b><pre>e.g. ${esc(text)}\n\n${esc(frames.join("\n"))}</pre></div>`;
    }
    if (!w.run_errors.length) h += `<p>No errors.</p>`;
    h += `</div>`;
  }
  if (w.failure) {
    h += `<div class="fail" id="failbox" hidden><h4>✗ ${esc(w.failure.text)}</h4><pre>${esc((w.failure.frames || []).join("\n"))}</pre></div>`;
  }
  if (w.output && w.output.trim()) h += `<details${w.failure ? " open" : ""}><summary>printed by the test</summary><pre>${esc(w.output)}</pre></details>`;
  pane.innerHTML = h;
  for (const row of pane.querySelectorAll(".row")) {
    w._rows[+row.dataset.ln] = row;
    row.addEventListener("click", () => {
      const ln = +row.dataset.ln;
      const hits = w.lines.filter((l) => l[1] <= ln && ln <= l[2]);
      if (!hits.length) return;
      const next = hits.find((l) => l[0] > V.A + 1) || hits[0];
      pause(); seek(next[0]);
    });
  }
  for (const box of pane.querySelectorAll(".err[data-a]")) box.addEventListener("click", () => { pause(); seek(+box.dataset.a - 300); });
}

function updateCode(w, A) {
  if (!w.source) return;
  const li = lastLE(w.lines, A, k0);
  if (li !== w._li) {
    for (const row of Object.values(w._rows)) row.classList.remove("cur", "seen");
    for (let j = 0; j <= li; j++) { const [, s, e] = w.lines[j]; for (let l = s; l <= e; l++) w._rows[l] && w._rows[l].classList.add("seen"); }
    if (li >= 0) {
      const [, s, e] = w.lines[li];
      for (let l = s; l <= e; l++) w._rows[l] && w._rows[l].classList.add("cur");
      const row = w._rows[s];
      if (row && V.tab === "code") {
        const pane = $("pane-code"), top = row.offsetTop, h = pane.clientHeight;
        if (top < pane.scrollTop + 50 || top > pane.scrollTop + h - 80) pane.scrollTo({ top: top - h / 3, behavior: V.playing ? "smooth" : "auto" });
      }
    }
    w._li = li;
  }
  const ci = lastLE(w.checks, A, k0);
  if (ci !== w._ci) {
    for (const row of Object.values(w._rows)) { const m = row.children[1]; m.className = "m"; m.textContent = ""; row.classList.remove("failed-line"); }
    for (let j = 0; j <= ci; j++) {
      const [, line, ok] = w.checks[j], row = w._rows[line];
      if (!row) continue;
      row.children[1].className = "m " + (ok ? "ok" : "bad");
      row.children[1].textContent = ok ? "✓" : "✗";
      if (!ok) row.classList.add("failed-line");
    }
    w._ci = ci;
  }
  if (w.failure) {
    const shown = A >= w.failure.a;
    if (shown !== w._failShown) {
      $("failbox").hidden = !shown;
      const row = w._rows[w.failure.line];
      if (row && !w.checks.some((c) => !c[2])) row.classList.toggle("failed-line", shown);
      w._failShown = shown;
    }
  }
}

function buildEvents(w) {
  const html = w.log.map((e, i) => `<li class="ev c-${e[2]}" data-i="${i}" title="${esc(e[5] || e[4])}"><span class="t">${fmtSim(w, e[1])}</span><span>${esc(e[3])}</span><span>${esc(e[4])}</span></li>`);
  const ol = $("events");
  ol.innerHTML = html.join("");
  w._evEls = ol.children;
  w._ei = -1;
  $("nev").textContent = w.log.length;
  ol.onclick = (ev) => { const li = ev.target.closest("li"); if (li) { pause(); seek(w.log[+li.dataset.i][0]); } };
}
function updateEvents(w, A) {
  const ei = lastLE(w.log, A, k0);
  if (ei === w._ei) return;
  const els = w._evEls, lo = Math.min(ei, w._ei), hi = Math.max(ei, w._ei);
  for (let j = Math.max(0, lo + 1); j <= hi && j < els.length; j++) els[j].classList.toggle("past", j <= ei);
  if (w._ei >= 0 && els[w._ei]) els[w._ei].classList.remove("now");
  if (ei >= 0) {
    els[ei].classList.add("now");
    if (V.tab === "events") {
      const pane = $("pane-events"), top = els[ei].offsetTop, h = pane.clientHeight;
      if (top < pane.scrollTop || top > pane.scrollTop + h - 40) pane.scrollTop = top - h / 2;
    }
  }
  w._ei = ei;
  // the ticker under the stage: the last few things that happened
  let t = "";
  for (let j = Math.max(0, ei - 3); j <= ei; j++) {
    const e = w.log[j], op = [0.3, 0.5, 0.75, 1][3 - (ei - j)];
    t += `<div class="tk c-${e[2]}" style="opacity:${op}"><span class="tt">${fmtSim(w, e[1])}</span><span class="tw">${esc(e[3])}</span><span class="tx">${esc(e[4])}</span></div>`;
  }
  $("ticker").innerHTML = t;
}

function updateState(w, A, force) {
  const idx = w.nodes.map((id) => lastLE(w.states[id] || [], A, k0));
  const sig = idx.join(",");
  if (!force && sig === w._ssig) return;
  w._ssig = sig;
  let h = "";
  w.nodes.forEach((id, n) => {
    if (w.pup.has(id)) { h += `<div class="nodecard"><h4>${esc(id)} <span class="role" style="background:${C.dim}">test</span></h4><div class="kv"><span class="k">role</span><span>played by the test, no state</span></div></div>`; return; }
    const list = w.states[id] || [];
    const rec = idx[n] >= 0 ? list[idx[n]] : list[0];
    if (!rec) return;
    const st = rec[1], role = C[st.role] ? st.role : "follower";
    const chips = (st.log || []).map(([t, k, v], i) => `<span class="chip${i < st.commit ? " c" : ""}" style="--tc:${termColor(t)}" title="index ${i + 1}, term ${t}">${i + 1}·T${t} ${esc(k)}=${esc(short(fmt(v), 12))}</span>`).join("") || `<span style="color:var(--dim)">empty</span>`;
    const map = (o) => Object.entries(o || {}).map(([k, v]) => `${k}:${v}`).join(" ") || "—";
    const store = Object.entries(st.store || {}).map(([k, v]) => `${esc(k)}=${esc(short(fmt(v), 20))}`).join(" ") || "—";
    h += `<div class="nodecard"><h4>${esc(id)} <span class="role" style="background:${C[role]}">${esc(st.role)}</span> <span style="color:${termColor(st.term)};font:600 12px var(--mono)">term ${st.term}</span></h4><div class="kv">
      <span class="k">voted_for</span><span>${esc(st.voted_for || "—")}</span>
      <span class="k">leader</span><span>${esc(st.leader || "—")}</span>
      <span class="k">votes</span><span>${esc((st.votes || []).join(", ") || "—")}</span>
      <span class="k">commit/applied</span><span>${st.commit} / ${st.applied}</span>
      <span class="k">log</span><span class="chips">${chips}</span>
      <span class="k">next_index</span><span>${esc(map(st.next))}</span>
      <span class="k">match_index</span><span>${esc(map(st.match))}</span>
      <span class="k">store</span><span>${store}</span></div></div>`;
  });
  $("pane-state").innerHTML = h;
}

function buildMarks(w) {
  let h = "";
  const pct = (a) => ((100 * a) / w.end).toFixed(3) + "%";
  for (const [a0, a1] of w.skips) h += `<div class="mk skip" style="left:${pct(a0)};width:${(100 * (a1 - a0) / w.end).toFixed(3)}%"></div>`;
  for (const l of w.lines) h += `<div class="mk line" style="left:${pct(l[0])}"></div>`;
  for (const c of w.checks) h += `<div class="mk ${c[2] ? "ok" : "bad"}" style="left:${pct(c[0])}"></div>`;
  for (const e of w.errors) h += `<div class="mk err" style="left:${pct(e[0])}"></div>`;
  $("marks").innerHTML = h;
}

function fmtSim(w, s) {
  if (w.unit) return "";
  return (s < 100 ? (Math.round(s * 10) / 10).toString() : Math.round(s).toString()) + " ms";
}
const fmtA = (a) => (a / 1000).toFixed(1) + "s";

function updateHud(w, A, sim) {
  if (w.unit) { $("clbl").textContent = "unit test"; $("sim").textContent = "no clock"; }
  else { $("clbl").textContent = "simulated time"; $("sim").textContent = fmtSim(w, sim); }
  $("ff").classList.toggle("on", w.skips.some(([a0, a1]) => A >= a0 && A < a1));
  $("time").innerHTML = `<b>${fmtA(A)}</b> / ${fmtA(w.end)}`;
  const p = Math.min(100, (100 * A) / w.end);
  $("fill").style.width = p + "%";
  $("head").style.left = p + "%";
  // toast: a failure stays up once reached; free-run errors flash
  const toast = $("toast");
  let text = "", cls = "";
  if (w.failure && A >= w.failure.a) text = "✗ " + w.failure.text;
  else {
    const ni = lastLE(w.notable, A, k0);
    if (ni >= 0 && A - w.notable[ni][0] < 1600) text = w.notable[ni][1];
    else if (w.kind === "test" && !w.failure && w.status === "passed" && A >= w.end - 1300 && w.checks.length) { text = `✓ all ${w.checks.length} asserts passed`; cls = "pass"; }
  }
  if (toast.textContent !== text) toast.textContent = text;
  toast.className = "toast" + (text ? " on" : "") + (cls ? " " + cls : "");
}

function render() {
  const w = V.w;
  if (!w) return;
  const A = V.A, sim = simAt(w, A), out = { shapes: "", labels: "", seen: {} };
  drawMessages(w, A, sim, out);
  const msgShapes = out.shapes;
  out.shapes = "";
  drawNodes(w, A, sim, out);
  $("dyn").innerHTML = msgShapes + out.shapes + out.labels;
  updateHud(w, A, sim);
  updateCode(w, A);
  updateEvents(w, A);
  if (V.tab === "state") updateState(w, A);
}

// ---- choosing a test, playback --------------------------------------------------------------
function buildSidebar() {
  const tests = DATA.worlds.filter((w) => w.kind === "test");
  const passed = tests.filter((w) => w.status === "passed").length;
  $("sum").textContent = tests.length ? `${passed}/${tests.length} passed` : "";
  $("sub").textContent = `${DATA.module}.py · ${DATA.generated}`;
  const item = (w, i) => {
    const icon = w.status === "failed" ? "✗" : w.status === "errors" ? "!" : "✓";
    let sub;
    if (w.kind === "run") sub = `${w.stats.nodes} nodes · ${w.stats.messages} msgs · ${w.errors.length} errors`;
    else if (w.unit) sub = `unit · ${w.checks.length} asserts`;
    else sub = `${w.stats.nodes} nodes · ${w.stats.messages} msgs · ${fmtSim(w, w.stats.sim_ms)}`;
    return `<li class="w ${w.status}" data-i="${i}"><span class="st">${icon}</span><div><div class="nm">${esc(w.title)}</div><div class="sub">${esc(sub)}</div></div></li>`;
  };
  let th = "", rh = "";
  DATA.worlds.forEach((w, i) => { if (w.kind === "test") th += item(w, i); else rh += item(w, i); });
  $("tests").innerHTML = th || `<li style="color:var(--dim);padding:8px 10px">no tests</li>`;
  $("runs").innerHTML = rh;
  $("runsep").hidden = $("runh").hidden = !rh;
  for (const li of document.querySelectorAll(".w")) li.addEventListener("click", () => selectWorld(+li.dataset.i, V.playing || true));
}

function selectWorld(i, play) {
  i = Math.max(0, Math.min(DATA.worlds.length - 1, i));
  V.i = i;
  const w = (V.w = prep(DATA.worlds[i]));
  V.A = 0;
  V.advanceAt = 0;
  for (const li of document.querySelectorAll(".w")) li.classList.toggle("on", +li.dataset.i === i);
  $("wkind").textContent = w.kind === "run" ? "free run" : w.unit ? "unit test" : "cluster test";
  $("wtitle").textContent = w.title;
  $("wdoc").textContent = w.doc;
  drawStatic(w);
  buildCode(w);
  buildEvents(w);
  buildMarks(w);
  w._ssig = null;
  if (V.tab === "state") updateState(w, 0, true);
  $("pane-code").scrollTop = 0; $("pane-events").scrollTop = 0;
  try { history.replaceState(null, "", "#" + encodeURIComponent(w.name)); } catch (e) { /* file:// in some browsers */ }
  V.dirty = true;
  if (play) start(); else pause();
}

const PLAY = '<path d="M4 2l10 6-10 6z"/>', PAUSE = '<path d="M3 2h3.5v12H3zM9.5 2H13v12H9.5z"/>';
function start() { if (!V.w) return; if (V.A >= V.w.end) V.A = 0; V.playing = true; V.advanceAt = 0; $("playicon").innerHTML = PAUSE; }
function pause() { V.playing = false; V.advanceAt = 0; $("playicon").innerHTML = PLAY; }
function seek(A) { V.A = Math.max(0, Math.min(V.w.end, A)); V.dirty = true; }
function step(dir) {
  pause();
  const M = V.w.moments;
  if (dir > 0) { const n = M.find((a) => a > V.A + 0.5); seek(n === undefined ? V.w.end : n + 1); }
  else { let p; for (const a of M) { if (a < V.A - 1.5) p = a; else break; } seek(p === undefined ? 0 : p + 1); }
}

function loop(now) {
  const dt = Math.min(80, now - (V.last || now));
  V.last = now;
  if (V.playing && V.w) {
    V.A += dt * V.speed;
    if (V.A >= V.w.end) {
      V.A = V.w.end;
      if (V.chain && V.i < DATA.worlds.length - 1) {
        if (!V.advanceAt) V.advanceAt = now + 1200;
        else if (now >= V.advanceAt) selectWorld(V.i + 1, true);
      } else pause();
    }
    V.dirty = true;
  }
  if (V.dirty) { V.dirty = false; render(); }
  requestAnimationFrame(loop);
}

// ---- wiring ------------------------------------------------------------------------------------
for (const s of SPEEDS) { const o = document.createElement("option"); o.value = s; o.textContent = s + "×"; $("speed").appendChild(o); }
$("speed").value = "1";
$("speed").addEventListener("change", () => { V.speed = +$("speed").value; });
function bumpSpeed(d) { const i = Math.max(0, Math.min(SPEEDS.length - 1, SPEEDS.indexOf(V.speed) + d)); V.speed = SPEEDS[i]; $("speed").value = String(V.speed); }
$("play").addEventListener("click", () => (V.playing ? pause() : start()));
$("prev").addEventListener("click", () => step(-1));
$("next").addEventListener("click", () => step(1));
$("restart").addEventListener("click", () => { seek(0); });
$("labels").addEventListener("change", () => { V.labels = $("labels").checked; V.dirty = true; });
$("chain").addEventListener("change", () => { V.chain = $("chain").checked; });
for (const b of document.querySelectorAll(".tabs button")) b.addEventListener("click", () => {
  V.tab = b.dataset.tab;
  for (const x of document.querySelectorAll(".tabs button")) x.classList.toggle("on", x === b);
  for (const p of document.querySelectorAll(".pane")) p.classList.toggle("on", p.id === "pane-" + V.tab);
  if (V.w) { V.w._li = -2; V.w._ei = -2; if (V.tab === "state") updateState(V.w, V.A, true); V.dirty = true; }
});
const scrub = $("scrub");
let dragging = false;
const scrubTo = (ev) => { const r = scrub.getBoundingClientRect(); seek(((ev.clientX - r.left) / r.width) * V.w.end); };
scrub.addEventListener("pointerdown", (ev) => { dragging = true; pause(); scrub.setPointerCapture(ev.pointerId); scrubTo(ev); });
scrub.addEventListener("pointermove", (ev) => { if (dragging) scrubTo(ev); });
scrub.addEventListener("pointerup", () => { dragging = false; });
document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "SELECT" || e.target.tagName === "INPUT" || e.metaKey || e.ctrlKey) return;
  if (e.code === "Space") { e.preventDefault(); V.playing ? pause() : start(); }
  else if (e.key === "ArrowRight") { e.preventDefault(); step(1); }
  else if (e.key === "ArrowLeft") { e.preventDefault(); step(-1); }
  else if (e.key === "ArrowDown") { e.preventDefault(); selectWorld(V.i + 1, V.playing); }
  else if (e.key === "ArrowUp") { e.preventDefault(); selectWorld(V.i - 1, V.playing); }
  else if (e.key === "Home") seek(0);
  else if (e.key === "End") seek(V.w.end);
  else if (e.key === "l" || e.key === "L") { $("labels").checked = V.labels = !V.labels; V.dirty = true; }
  else if (e.key === "+" || e.key === "=") bumpSpeed(1);
  else if (e.key === "-" || e.key === "_") bumpSpeed(-1);
});

buildSidebar();
// #test_name plays that test; #test_name@4200 opens it paused at 4.2 s
const [want, at] = decodeURIComponent(location.hash.slice(1)).split("@");
const first = Math.max(0, DATA.worlds.findIndex((w) => w.name === want));
selectWorld(first, at === undefined);
if (at !== undefined) seek(+at);
requestAnimationFrame(loop);
})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
