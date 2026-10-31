import random
from dataclasses import dataclass, field

from dslabs.protocols import Cancel, Message

from .node_total_order_eager_broadcast import NodeTotalOrderEagerBroadcast

GOSSIP_INTERVAL_MS = 100


@dataclass
class NodeTotalOrderGossip(NodeTotalOrderEagerBroadcast):
    """``NodeTotalOrderEagerBroadcast`` with gossip in place of eager broadcast.

    Ordering is unchanged: Lamport timestamps and a buffer. What changes is how
    events spread. A node gossips in rounds, 100 ms apart. In each round it
    sends one message to ``num_gossip_peers`` peers picked at random, carrying
    every event it is still spreading. An event is spread for
    ``num_gossip_rounds`` rounds after the node first sees it.

    Acks are batched to match: instead of one ack per write, a node that has
    received writes since its last round starts the round with a single ack.
    Stamped after everything it has seen, that one ack does the job of all of
    them.

    A node that never hears of an event, or of some peer's ack, stays stuck
    behind it, so how often that happens depends on the gossip parameters and
    the drop rate.
    """

    num_gossip_peers: int = 3
    num_gossip_rounds: int = 2
    seed: int = 0
    rng: random.Random = field(init=False, repr=False)
    # Events still being spread, with the number of rounds each has left
    hot: dict[str, int] = field(default_factory=dict)
    gossip_timer: Cancel | None = None
    ack_due: bool = False
    # Event copies sent, summed over all gossip messages: the bandwidth used
    events_sent: int = 0

    def __post_init__(self) -> None:
        self.rng = random.Random(f"{self.seed}:{self.node_id}")

    def relay(self, evt: Message, sender: str | None) -> None:
        self.hot[f"{evt['origin']}:{evt['seq']}"] = self.num_gossip_rounds
        if self.gossip_timer is None:
            self.gossip_timer = self.scheduler.call_later(GOSSIP_INTERVAL_MS, self.gossip_round)

    def acknowledge(self, eid: str) -> None:
        self.ack_due = True

    def gossip_round(self) -> None:
        # Still set while the round runs, so the ack below does not arm another round
        if self.ack_due:
            self.ack_due = False
            self.broadcast({"kind": "ack"})
        self.gossip_timer = None
        events = [self.seen[eid] for eid in self.hot]
        others = [p for p in self.peers if p != self.node_id]
        for peer in self.rng.sample(others, min(self.num_gossip_peers, len(others))):
            self.transport.send(peer, {"type": "gossip", "events": events})
            self.events_sent += len(events)
        self.hot = {eid: left - 1 for eid, left in self.hot.items() if left > 1}
        if self.hot:
            self.gossip_timer = self.scheduler.call_later(GOSSIP_INTERVAL_MS, self.gossip_round)

    def on_message(self, msg: Message) -> None:
        if msg["type"] == "gossip":
            for evt in msg["events"]:
                self.receive(evt, sender=msg["from"])
        else:
            super().on_message(msg)
