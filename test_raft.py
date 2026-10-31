"""Check the Raft vote and append-entries handlers.

``tick`` does not start an election yet, so the scenario tests campaign by
sending RequestVote the way a candidate would, then let the simulator deliver
every message. Run with ``python test_raft.py``.
"""

import functools

from dslabs import Cluster
from raft_types import LogEntry, RequestVote
from main import RaftNode


class _Scheduler:
    def now_ms(self) -> int:
        return 0

    def call_later(self, ms: int, cb) -> object:
        return lambda: None


class _Transport:
    def __init__(self) -> None:
        self.sent: list[tuple[str, dict]] = []

    def send(self, to: str, msg: dict) -> None:
        self.sent.append((to, msg))

    def note(self, text: str) -> None:
        pass


def node(node_id: str = "n2", peers: list[str] | None = None) -> tuple[RaftNode, _Transport]:
    transport = _Transport()
    raft = RaftNode(node_id, peers or ["n1", "n2", "n3"], transport, _Scheduler(), seed=0)
    return raft, transport


def vote(frm: str, candidate: str, term: int, last_index: int = 0, last_term: int = 0) -> dict:
    return {
        "from": frm,
        "type": "request_vote",
        "term": term,
        "candidate_id": candidate,
        "last_log_index": last_index,
        "last_log_term": last_term,
    }


def test_vote_rules() -> None:
    raft, transport = node()

    raft.on_message(vote("n1", "n1", term=1))
    assert transport.sent[-1][1]["vote_granted"] is True
    assert raft.voted_for == "n1" and raft.term == 1

    # A retry of the same request is granted again. A second candidate is not.
    raft.on_message(vote("n1", "n1", term=1))
    assert transport.sent[-1][1]["vote_granted"] is True
    raft.on_message(vote("n3", "n3", term=1))
    assert transport.sent[-1][1]["vote_granted"] is False
    assert raft.voted_for == "n1"

    # An older term cannot win a vote or move this node backwards.
    raft.on_message(vote("n3", "n3", term=0, last_index=5, last_term=9))
    assert transport.sent[-1][1]["vote_granted"] is False
    assert transport.sent[-1][1]["term"] == 1
    assert raft.term == 1

    # A newer term is adopted, but a shorter log is still refused.
    raft.log.append(LogEntry(1, "k", "mine"))
    raft.on_message(vote("n3", "n3", term=2, last_index=0, last_term=0))
    assert raft.term == 2 and raft.voted_for == ""
    assert transport.sent[-1][1]["vote_granted"] is False

    # The same candidate wins once its log catches up.
    raft.on_message(vote("n3", "n3", term=3, last_index=1, last_term=1))
    assert transport.sent[-1][1]["vote_granted"] is True
    assert raft.voted_for == "n3"


def cluster(seed: int = 0) -> Cluster:
    return Cluster(functools.partial(RaftNode, seed=seed), 3, seed=seed, latency_ms=(1, 5))


def campaign(candidate: RaftNode) -> None:
    """Send RequestVote for a new term. The node votes for itself."""
    candidate.term += 1
    candidate.role = "candidate"
    candidate.voted_for = candidate.node_id
    candidate.leader = ""
    candidate.votes.clear()
    candidate.reset_timer_heartbeat_from_leader()
    last_index = len(candidate.log)
    last_term = candidate.log[-1].term if candidate.log else 0
    for peer in candidate.peers:
        if peer != candidate.node_id:
            candidate._send(peer, RequestVote(candidate.term, candidate.node_id, last_index, last_term))


def run_until(sim: Cluster, ready, limit_ms: int = 5_000) -> None:
    while sim.scheduler.now_ms() < limit_ms:
        if ready():
            return
        if not sim.step():
            break
    if not ready():
        print(sim.messages())
        raise AssertionError(f"not ready at {sim.scheduler.now_ms()} ms")


def leaders(sim: Cluster) -> list[str]:
    return [nid for nid, n in sim.nodes.items() if n.role == "leader"]


def test_election_commits_on_every_node() -> None:
    sim = cluster()
    campaign(sim.nodes["n1"])
    run_until(sim, lambda: leaders(sim) == ["n1"])

    leader = sim.nodes["n1"]
    leader.log.append(LogEntry(leader.term, "x", 1))
    for peer in leader.next_index:
        leader._replicate(peer)
    run_until(sim, lambda: sim.values("x") == {"n1": 1, "n2": 1, "n3": 1})

    assert {n.leader for n in sim.nodes.values()} == {"n1"}
    assert {n.term for n in sim.nodes.values()} == {leader.term}
    assert [n.commit_index for n in sim.nodes.values()] == [1, 1, 1]


def test_divergent_log_is_replaced() -> None:
    sim = cluster(seed=1)
    follower = sim.nodes["n2"]
    follower.term = 1
    follower.log.append(LogEntry(1, "old", 9))

    candidate = sim.nodes["n1"]
    candidate.term = 1
    campaign(candidate)
    run_until(sim, lambda: leaders(sim) == ["n1"])

    leader = sim.nodes["n1"]
    leader.log.append(LogEntry(leader.term, "x", 2))
    for peer in leader.next_index:
        leader._replicate(peer)
    run_until(sim, lambda: sim.values("x") == {"n1": 2, "n2": 2, "n3": 2})

    assert [(e.term, e.key, e.value) for e in follower.log] == [(leader.term, "x", 2)]
    assert follower.client_get("old") is None


def main() -> None:
    test_vote_rules()
    test_election_commits_on_every_node()
    test_divergent_log_is_replaced()
    print("passed: vote rules, election and commit, divergent log")


if __name__ == "__main__":
    main()
