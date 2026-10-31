from dataclasses import dataclass
from typing import Any, ClassVar

from dslabs.protocols import Message


@dataclass
class LogEntry:
    term: int
    key: str
    value: Any


@dataclass
class RequestVote:
    """Sent by a candidate to ask for a vote."""

    TYPE: ClassVar[str] = "request_vote"
    term: int
    candidate_id: str
    last_log_index: int
    last_log_term: int


@dataclass
class RequestVoteResponse:
    TYPE: ClassVar[str] = "request_vote_response"
    term: int
    vote_granted: bool


@dataclass
class AppendEntries:
    """Sent by the leader to replicate entries; with no entries, a heartbeat."""

    TYPE: ClassVar[str] = "append_entries"
    term: int
    leader_id: str
    prev_log_index: int
    prev_log_term: int
    entries: list[LogEntry]
    leader_commit: int


@dataclass
class AppendEntriesResponse:
    TYPE: ClassVar[str] = "append_entries_response"
    term: int
    success: bool
    # Not in the paper: the last index the follower now matches the leader on.
    # The network can reorder and duplicate replies, so the leader cannot work
    # this out from which request it thinks is being answered.
    match_index: int


def parse(msg: Message) -> RequestVote | RequestVoteResponse | AppendEntries | AppendEntriesResponse:
    """Rebuild the dataclass that a received message was made from."""
    kind = msg.get("type")
    args = {k: v for k, v in msg.items() if k not in ("type", "from")}
    if kind == RequestVote.TYPE:
        return RequestVote(**args)
    if kind == RequestVoteResponse.TYPE:
        return RequestVoteResponse(**args)
    if kind == AppendEntries.TYPE:
        args["entries"] = [LogEntry(**e) for e in args["entries"]]  # JSON leaves them as dicts
        return AppendEntries(**args)
    if kind == AppendEntriesResponse.TYPE:
        return AppendEntriesResponse(**args)
    raise ValueError(f"Unknown message type in {msg!r}")
