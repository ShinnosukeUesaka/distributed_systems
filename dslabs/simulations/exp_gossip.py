"""Experiment: how reliable and how costly is total order gossip?

Runs ``SimSendMany`` (1000 writes, default network delay) with
``NodeTotalOrderGossip`` over a grid of drop rates and gossip parameters,
several trials each. For every setting it reports the fraction of runs that
failed and the mean bandwidth used, then per drop rate the cheapest setting
that never failed, and saves a plot.

A run fails unless every node ends up having delivered every write, in the same
order. Nodes merely agreeing on ``x`` is not enough: when everyone is stuck
behind the same lost event they all still hold the same stale value (often
``None``), which is consistent but useless. That weaker notion is reported too,
as P(disagree).

Bandwidth is measured two ways: messages sent (``SimNetwork.stats``), and event
copies sent, since one gossip message carries many events.

    python -m dslabs.simulations.exp_gossip
    python -m dslabs.simulations.exp_gossip --baseline      # add eager broadcast, much slower
"""

import argparse
import csv
import functools
import itertools
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass

from dslabs.nodes import NodeTotalOrderEagerBroadcast, NodeTotalOrderGossip
from dslabs.simulations.sim_send_many import SimSendMany


@dataclass
class Result:
    node: str
    drop_prob: float
    peers: int | None
    rounds: int | None
    trial: int
    complete: bool  # every node delivered every write, in the same order
    consistent: bool  # every node ends with the same x
    messages: int
    event_copies: int


def untraced(sim: SimSendMany) -> None:
    """Stop recording the trace, which the experiment does not need."""
    trace = sim.trace
    trace.send = lambda *args, **kwargs: 0
    trace.snapshot = trace.deliver = trace.drop = trace.client = lambda *args, **kwargs: None
    trace.timer_set = trace.timer_fired = trace.timer_cancelled = lambda *args, **kwargs: None


def run_one(job: tuple, num_nodes: int, num_messages: int) -> Result:
    node, drop_prob, peers, rounds, trial = job
    if node == "gossip":
        node_class = functools.partial(NodeTotalOrderGossip, num_gossip_peers=peers,
                                       num_gossip_rounds=rounds, seed=trial)
    else:
        node_class = NodeTotalOrderEagerBroadcast
    sim = SimSendMany(node_class, num_nodes=num_nodes, num_messages=num_messages,
                      drop_prob=drop_prob, seed=trial)
    untraced(sim)
    values = sim.run()
    logs = [n.log for n in sim.nodes.values()]
    messages = sim.network.stats["sent"]
    return Result(
        node=node, drop_prob=drop_prob, peers=peers, rounds=rounds, trial=trial,
        complete=all(log == logs[0] for log in logs) and len(logs[0]) == num_messages,
        consistent=len(set(values.values())) == 1,
        messages=messages,
        event_copies=sum(n.events_sent for n in sim.nodes.values()) if node == "gossip" else messages,
    )


@dataclass
class Summary:
    node: str
    drop_prob: float
    peers: int | None
    rounds: int | None
    trials: int
    p_fail: float
    p_disagree: float
    messages: float
    event_copies: float


def summarise(results: list[Result]) -> list[Summary]:
    groups: dict[tuple, list[Result]] = {}
    for r in results:
        groups.setdefault((r.node, r.drop_prob, r.peers, r.rounds), []).append(r)
    return sorted(
        (Summary(node, d, p, rd, len(rs),
                 p_fail=sum(not r.complete for r in rs) / len(rs),
                 p_disagree=sum(not r.consistent for r in rs) / len(rs),
                 messages=sum(r.messages for r in rs) / len(rs),
                 event_copies=sum(r.event_copies for r in rs) / len(rs))
         for (node, d, p, rd), rs in groups.items()),
        key=lambda s: (s.drop_prob, s.node, s.peers or 0, s.rounds or 0),
    )


def cheapest_reliable(summaries: list[Summary], drop_prob: float) -> Summary | None:
    ok = [s for s in summaries if s.node == "gossip" and s.drop_prob == drop_prob and s.p_fail == 0]
    return min(ok, key=lambda s: s.event_copies) if ok else None


def report(summaries: list[Summary]) -> None:
    print(f"\n{'drop':>5} {'node':>7} {'peers':>5} {'rounds':>6} {'P(fail)':>8} {'P(disagree)':>11} "
          f"{'messages':>10} {'event copies':>13}")
    for s in summaries:
        print(f"{s.drop_prob:>5} {s.node:>7} {s.peers or '-':>5} {s.rounds or '-':>6} {s.p_fail:>8.2f} "
              f"{s.p_disagree:>11.2f} {s.messages:>10,.0f} {s.event_copies:>13,.0f}")

    print("\nCheapest gossip setting that never failed, per drop rate:")
    for d in sorted({s.drop_prob for s in summaries}):
        best = cheapest_reliable(summaries, d)
        eager = [s for s in summaries if s.node == "eager" and s.drop_prob == d]
        note = (f"; eager broadcast: P(fail) {eager[0].p_fail:.2f}, {eager[0].messages:,.0f} messages"
                if eager else "")
        if best:
            print(f"  drop {d}: peers={best.peers}, rounds={best.rounds}: {best.messages:,.0f} messages, "
                  f"{best.event_copies:,.0f} event copies{note}")
        else:
            print(f"  drop {d}: none of the settings tried{note}")


def plot(summaries: list[Summary], path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    gossip = [s for s in summaries if s.node == "gossip"]
    drops = sorted({s.drop_prob for s in gossip})
    peers = sorted({s.peers for s in gossip})
    rounds = sorted({s.rounds for s in gossip})
    fig, axes = plt.subplots(1, len(drops) + 1, figsize=(4.2 * (len(drops) + 1), 4.6))

    for col, d in enumerate(drops):
        by_cell = {(s.peers, s.rounds): s for s in gossip if s.drop_prob == d}
        best = cheapest_reliable(summaries, d)

        # P(fail) for every (peers, rounds), with bandwidth written in each cell
        ax = axes[col]
        grid = np.array([[by_cell[(p, r)].p_fail if (p, r) in by_cell else np.nan for r in rounds]
                         for p in peers])
        ax.imshow(grid, cmap="RdYlGn_r", vmin=0, vmax=1, origin="lower", aspect="auto")
        for i, p in enumerate(peers):
            for j, r in enumerate(rounds):
                s = by_cell.get((p, r))
                if s:
                    ax.text(j, i, f"{s.p_fail:.2f}\n{s.event_copies / 1000:,.0f}k", ha="center", va="center",
                            fontsize=7, fontweight="bold" if s is best else "normal")
        if best:
            ax.add_patch(plt.Rectangle((rounds.index(best.rounds) - 0.5, peers.index(best.peers) - 0.5),
                                       1, 1, fill=False, edgecolor="blue", linewidth=2.5))
        ax.set_xticks(range(len(rounds)), rounds)
        ax.set_yticks(range(len(peers)), peers)
        ax.set_xlabel("num_gossip_rounds")
        ax.set_ylabel("num_gossip_peers")
        ax.set_title(f"drop {d}: P(fail) / event copies sent", fontsize=10)

    # The price of reliability: bandwidth of the cheapest setting that never failed
    ax = axes[-1]
    bests = [(d, cheapest_reliable(summaries, d)) for d in drops]
    bests = [(d, b) for d, b in bests if b]
    bars = ax.bar([str(d) for d, _ in bests], [b.event_copies / 1000 for _, b in bests], color="tab:blue")
    for bar, (_, b) in zip(bars, bests):
        ax.annotate(f"{b.peers} peers\nx {b.rounds} rounds\n{b.messages:,.0f} msgs",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()), ha="center", va="bottom", fontsize=7)
    ax.set_ylim(0, max(b.event_copies for _, b in bests) / 1000 * 1.35)
    ax.set_xlabel("drop probability")
    ax.set_ylabel("event copies sent (thousands)")
    ax.set_title("cheapest setting with no failures", fontsize=10)

    fig.suptitle("Total order gossip, 10 nodes, 1000 writes. Cells: P(fail) over trials, and event copies "
                 "sent. Blue box: cheapest setting with no failures", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--drops", type=float, nargs="+", default=[0.0, 0.1, 0.3, 0.5])
    parser.add_argument("--peers", type=int, nargs="+", default=[1, 2, 3, 4, 6, 9])
    parser.add_argument("--rounds", type=int, nargs="+", default=[1, 2, 3, 4, 6])
    parser.add_argument("--max-fanout", type=int, default=24,
                        help="skip settings where peers * rounds exceeds this; they never fail and "
                             "only cost more (default: %(default)s)")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--nodes", type=int, default=10)
    parser.add_argument("--messages", type=int, default=1000)
    parser.add_argument("--baseline", action="store_true", help="also run eager broadcast (slow)")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--csv", default="exp_gossip.csv", help="every run (default: %(default)s)")
    parser.add_argument("--plot", default="exp_gossip.png", help="plot (default: %(default)s)")
    args = parser.parse_args()

    jobs = [("gossip", d, p, r, t)
            for d, p, r, t in itertools.product(args.drops, args.peers, args.rounds, range(args.trials))
            if p * r <= args.max_fanout]
    if args.baseline:
        jobs += [("eager", d, None, None, t) for d in args.drops for t in range(args.trials)]
    # Biggest runs first, so the pool is not left waiting on one at the end
    jobs.sort(key=lambda j: -(j[2] * j[3] if j[2] else 1000))

    start = time.time()
    print(f"{len(jobs)} runs on {args.workers} workers...", flush=True)
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(functools.partial(run_one, num_nodes=args.nodes, num_messages=args.messages),
                                jobs, chunksize=4))
    print(f"done in {time.time() - start:.0f} s")

    with open(args.csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(Result.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(asdict(r) for r in results)

    summaries = summarise(results)
    report(summaries)
    plot(summaries, args.plot)
    print(f"\nevery run written to {args.csv}, plot saved to {args.plot}")


if __name__ == "__main__":
    main()
