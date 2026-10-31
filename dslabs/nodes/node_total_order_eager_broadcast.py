import heapq
from dataclasses import dataclass, field
from typing import Any

from dslabs.protocols import Message, Scheduler, Transport


def event_id(evt: Message) -> str:
    return f"{evt['origin']}:{evt['seq']}"


@dataclass
class NodeTotalOrderEagerBroadcast:
    """Total order broadcast from Lamport clocks, spread by eager broadcast.

    per-origin sequence number. Writes are delivered in ``(timestamp, origin)``
    Ordering. Every write is an event stamped with a Lamport timestamp and a
    order and wait in a buffer until nothing smaller can still arrive. An
    origin's timestamps only grow, so once we hold ``p``'s events without gaps
    up to one stamped ``T`` or later, nothing from ``p`` can still come in below
    ``T``. To keep every node's clock heard, each node answers every write it
    receives with an ack event of its own, stamped after the write.

    Spreading. Writes and acks alike spread by eager broadcast: the first time
    a node sees an event it relays it to every peer, so an event reaches a node
    as long as any one copy of it gets through. Events already seen are ignored.

    """
    node_id: str
    peers: list[str]
    transport: Transport
    scheduler: Scheduler
    # Key-value store
    store: dict[str, Any] = field(default_factory=dict)
    # Append-only log of everything delivered, in delivery order
    log: list[tuple[str, Any]] = field(default_factory=list)

    # Lamport clock
    clock: int = 0
    # Sequence number of this node's next event
    seq: int = 0
    # Every event seen, by event id
    seen: dict[str, Message] = field(default_factory=dict)
    # Per origin: the next sequence number not yet held, and the timestamp of the last one held
    next_seq: dict[str, int] = field(default_factory=dict)
    heard_clock: dict[str, int] = field(default_factory=dict)
    # Writes awaiting delivery, as a heap of (timestamp, origin, event id)
    buffer: list[tuple[int, str, str]] = field(default_factory=list)

    # Client-facing API
    def client_put(self, key: str, value: Any) -> None:
        self.broadcast({"kind": "put", "key": key, "value": value})

    def client_get(self, key: str) -> Any:
        return self.store.get(key)

    # Node internals
    def broadcast(self, payload: dict[str, Any]) -> None:
        """Originate a new event from this node."""
        self.clock += 1
        evt = {"type": "event", "origin": self.node_id, "seq": self.seq, "ts": self.clock, **payload}
        self.seq += 1
        self.receive(evt)

    def receive(self, evt: Message, sender: str | None = None) -> None:
        eid = event_id(evt)
        if eid in self.seen:
            return
        evt = {k: v for k, v in evt.items() if k != "from"}
        self.seen[eid] = evt
        self.clock = max(self.clock, evt["ts"])
        self.relay(evt, sender)

        self.advance(evt["origin"])
        if evt["kind"] == "put" and evt["origin"] != self.node_id:
            self.acknowledge(eid)
        self.try_deliver()

    def acknowledge(self, eid: str) -> None:
        self.broadcast({"kind": "ack", "ack_of": eid})

    def relay(self, evt: Message, sender: str | None) -> None:
        """Pass on an event seen for the first time: eagerly, to every peer."""
        for peer in self.peers:
            if peer not in (self.node_id, sender, evt["origin"]):
                self.transport.send(peer, evt)

    def advance(self, origin: str) -> None:
        """Take in the origin's events that now follow on without a gap."""
        n = self.next_seq.get(origin, 0)
        while (eid := f"{origin}:{n}") in self.seen:
            evt = self.seen[eid]
            self.heard_clock[origin] = evt["ts"]
            if evt["kind"] == "put":
                heapq.heappush(self.buffer, (evt["ts"], origin, eid))
            n += 1
        self.next_seq[origin] = n

    def heard_up_to(self, peer: str) -> int:
        """Every event from ``peer`` still to arrive has a timestamp above this."""
        return self.clock if peer == self.node_id else self.heard_clock.get(peer, 0)

    def try_deliver(self) -> None:
        while self.buffer:
            ts, _, eid = self.buffer[0]
            if any(self.heard_up_to(p) < ts for p in self.peers):
                return
            heapq.heappop(self.buffer)
            evt = self.seen[eid]
            self.deliver(evt["key"], evt["value"])

    def deliver(self, key: str, value: Any) -> None:
        """Apply a message to this node: update the store, append to the log."""
        self.store[key] = value
        self.log.append((key, value))

    # Network handler
    def on_message(self, msg: Message) -> None:
        if msg["type"] == "event":
            self.receive(msg, sender=msg["from"])
        else:
            raise ValueError(f"Unknown message type in {msg!r}")

    # What the trace and diagrams show as this node's state
    def brief_state(self) -> dict[str, Any]:
        return dict(self.store)
