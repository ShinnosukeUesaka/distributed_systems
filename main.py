import argparse
import functools
import random
from dataclasses import asdict, dataclass, field
from typing import Any

from dslabs import Cluster
from dslabs.protocols import Cancel, Message, Scheduler, Transport
from raft_types import (
    AppendEntries,
    AppendEntriesResponse,
    LogEntry,
    RequestVote,
    RequestVoteResponse,
    parse,
)


@dataclass
class RaftNode:
    node_id: str
    peers: list[str]
    transport: Transport
    scheduler: Scheduler
    seed: int = 0
    leader: str = ""
    log: list[LogEntry] = field(default_factory=list)
    term: int = 0
    rng: random.Random = field(init=False, repr=False)
    role: str = "follower"
    voted_for: str = ""
    commit_index: int = 0
    last_applied: int = 0
    votes: set[str] = field(default_factory=set)
    next_index: dict[str, int] = field(default_factory=dict)
    match_index: dict[str, int] = field(default_factory=dict)
    store: dict[str, Any] = field(default_factory=dict)
    _election: Cancel | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.rng = random.Random(f"{self.seed}:{self.node_id}")
        self.reset_timer_heartbeat_from_leader()

    def reset_timer_heartbeat_from_leader(self) -> None:
        # if follower recieves a heartbeat from leader, it will reset the timeout
        if self._election is not None:
            self._election()
        self._election = self.scheduler.call_later(self.rng.randint(1000, 3000), self.start_election)

    def send_heartbeat_to_followers(self) -> None:
        if self.role != "leader":
            return
        for peer in self.peers:
            if peer != self.node_id:
                self._replicate(peer)  # empty when the peer is caught up, the missing entries when it isn't
        self.scheduler.call_later(100, self.send_heartbeat_to_followers)

    def start_election(self) -> None:
        if self.role == "leader":
            return
        self.term += 1
        self.role = "candidate"
        self.voted_for = self.node_id
        self.votes.clear()
        self.reset_timer_heartbeat_from_leader()

        # request votes from all peers
        for peer in self.peers:
            if peer != self.node_id:
                self._send(peer, RequestVote(term=self.term, candidate_id=self.node_id, last_log_index=len(self.log), last_log_term=self.log[-1].term if self.log else 0))

    def on_message(self, msg: Message) -> None:
        frm = msg["from"]
        m = parse(msg)  # raises on an unknown type
        if isinstance(m, RequestVote):
            self.handle_request_vote(frm, m)
        elif isinstance(m, RequestVoteResponse):
            self.handle_request_vote_response(frm, m)
        elif isinstance(m, AppendEntries):
            self.handle_append_entries(frm, m)
        elif isinstance(m, AppendEntriesResponse):
            self.handle_append_entries_response(frm, m)

    def handle_request_vote(self, frm: str, message: RequestVote) -> None:
        if message.term > self.term:
            self._become_follower(message.term)

        grant = (
            message.term == self.term
            and self.voted_for in ("", message.candidate_id)
            and self._log_up_to_date(message.last_log_index, message.last_log_term)
        )
        if grant:
            self.voted_for = message.candidate_id
            self.reset_timer_heartbeat_from_leader()
        self._send(frm, RequestVoteResponse(term=self.term, vote_granted=grant))

    def handle_request_vote_response(self, frm: str, message: RequestVoteResponse) -> None:
        if message.term > self.term:
            self._become_follower(message.term)
            return
        if self.role != "candidate" or message.term != self.term or not message.vote_granted:
            return
        self.votes.add(frm)
        # Our own vote counts, and peers includes this node.
        if 1 + len(self.votes) >= len(self.peers) // 2 + 1:
            self._become_leader()

    def handle_append_entries(self, frm: str, message: AppendEntries) -> None:
        if message.term > self.term:
            self._become_follower(message.term)
        if message.term < self.term:
            self._send(frm, AppendEntriesResponse(term=self.term, success=False, match_index=0))
            return

        self.role = "follower"
        self.leader = message.leader_id
        self.votes.clear()
        self.reset_timer_heartbeat_from_leader()

        if not self._prev_matches(message.prev_log_index, message.prev_log_term):
            hint = self._conflict_hint(message.prev_log_index)
            self._send(frm, AppendEntriesResponse(term=self.term, success=False, match_index=hint))
            return

        self._merge(message.prev_log_index, message.entries)
        if message.leader_commit > self.commit_index:
            self.commit_index = min(message.leader_commit, len(self.log))
            self._apply_committed()
        match = message.prev_log_index + len(message.entries)
        self._send(frm, AppendEntriesResponse(term=self.term, success=True, match_index=match))

    def handle_append_entries_response(self, frm: str, message: AppendEntriesResponse) -> None:
        if message.term > self.term:
            self._become_follower(message.term)
            return
        if self.role != "leader" or message.term != self.term:
            return
        if message.success:
            if message.match_index >= self.match_index.get(frm, 0):
                self.match_index[frm] = message.match_index
                self.next_index[frm] = message.match_index + 1
                committed = self.commit_index
                self._advance_commit()
                if self.commit_index > committed:
                    for peer in self.next_index:
                        self._replicate(peer)
            return
        suggested = message.match_index + 1
        if suggested < self.next_index.get(frm, 1):
            self.next_index[frm] = max(1, suggested)
            self._replicate(frm)

    def _send(self, to: str, message: RequestVote | RequestVoteResponse | AppendEntries | AppendEntriesResponse) -> None:
        body = asdict(message)
        body["type"] = message.TYPE
        self.transport.send(to, body)

    def _become_follower(self, term: int) -> None:
        self.term = term
        self.role = "follower"
        self.leader = ""
        self.voted_for = ""
        self.votes.clear()

    def _become_leader(self) -> None:
        if self.role == "leader":
            return
        self.role = "leader"
        self.leader = self.node_id
        last = len(self.log) + 1
        self.next_index = {peer: last for peer in self.peers if peer != self.node_id}
        self.match_index = {peer: 0 for peer in self.peers if peer != self.node_id}
        for peer in self.next_index:
            self._replicate(peer)
        self.send_heartbeat_to_followers()

    def _log_up_to_date(self, last_index: int, last_term: int) -> bool:
        my_term = self.log[-1].term if self.log else 0
        if last_term != my_term:
            return last_term > my_term
        return last_index >= len(self.log)

    def _prev_matches(self, prev_index: int, prev_term: int) -> bool:
        if prev_index > len(self.log):
            return False
        if prev_index == 0:
            return True
        return self.log[prev_index - 1].term == prev_term

    def _conflict_hint(self, prev_index: int) -> int:
        """Last index that might still match, so the leader can skip ahead."""
        if prev_index > len(self.log):
            return len(self.log)
        term = self.log[prev_index - 1].term
        index = prev_index
        while index > 1 and self.log[index - 2].term == term:
            index -= 1
        return index - 1

    def _merge(self, prev_index: int, entries: list[LogEntry]) -> None:
        for offset, entry in enumerate(entries, start=1):
            index = prev_index + offset
            if index <= len(self.log):
                if self.log[index - 1].term == entry.term:
                    continue
                del self.log[index - 1 :]
            self.log.append(LogEntry(entry.term, entry.key, entry.value))

    def _replicate(self, peer: str) -> None:
        next_index = min(self.next_index[peer], len(self.log) + 1)
        prev_index = next_index - 1
        prev_term = self.log[prev_index - 1].term if prev_index else 0
        self._send(
            peer,
            AppendEntries(
                term=self.term,
                leader_id=self.node_id,
                prev_log_index=prev_index,
                prev_log_term=prev_term,
                entries=list(self.log[prev_index:]),
                leader_commit=self.commit_index,
            ),
        )

    def _advance_commit(self) -> None:
        needed = len(self.peers) // 2 + 1
        for n in range(len(self.log), self.commit_index, -1):
            if self.log[n - 1].term != self.term:
                continue
            replicated = 1 + sum(index >= n for index in self.match_index.values())
            if replicated >= needed:
                self.commit_index = n
                self._apply_committed()
                return

    def _apply_committed(self) -> None:
        while self.last_applied < self.commit_index:
            self.last_applied += 1
            entry = self.log[self.last_applied - 1]
            self.store[entry.key] = entry.value
    
    def client_put(self, key: str, value: Any) -> None:
        if self.role != "leader":
            return
        self.log.append(LogEntry(self.term, key, value))
        for peer in self.peers:
            if peer != self.node_id:
                self._replicate(peer)
    
    def client_get(self, key: str) -> Any:
        return self.store.get(key)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run every node in the dslabs simulator.")
    parser.add_argument("--nodes", type=int, default=4, help="cluster size (default: %(default)s)")
    parser.add_argument("--ms", type=int, default=10_000, help="simulated time to run (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=0, help="random seed (default: %(default)s)")
    parser.add_argument("--viewer", metavar="FILE", help="save an animated replay of the run to FILE")
    args = parser.parse_args()

    cluster = Cluster(functools.partial(RaftNode, seed=args.seed), args.nodes, seed=args.seed, verbose=True)
    cluster.run_until(args.ms)
    print(f"network: {cluster.stats}")
    if args.viewer:
        cluster.save_viewer(args.viewer)
        print(f"replay saved to {args.viewer}")


if __name__ == "__main__":
    main()
