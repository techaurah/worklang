"""Deterministic event tokenizer for WorkLang (DESIGN.md, The tokenizer).

One event becomes one composite position of 7 token ids in fixed
order: family first (the coarse subtoken), then the fine attribute
subtokens in event field order: channel, score_band, latency_band,
stage, tool, outcome_class. Events are already discrete, so nothing is
learned and nothing is quantised; decode is exact by construction.

A whole trace encodes as: BOS, the vertical conditioning token
(emitted once, immediately after BOS), the 7 ids of each event in
order, then EOS.

PAD is never emitted; it exists for batching in later build steps.
UNK is never emitted for a schema-valid trace, because schema
validation rejects values outside the bucket sets before tokenisation
can see them. UNK is reserved for a possible future converter policy
(DESIGN.md risks section), not used by this step.

Lossy by design, named here as the design requires: message text,
names, contact details, exact timestamps, and any attribute outside
the bucket sets are dropped before tokenisation (see schema.py, the
lossy boundary). The round trip guarantee is at the event level,
exactly as the design states it: decode(encode(trace)) returns the
same vertical and the same events. Trace metadata (trace_id, source,
synthetic, agent, agent_version, generator, and the duplicated trace
level outcome) is not carried in the token stream; it lives in the
JSONL record beside the tokens.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import NamedTuple

from .errors import TokenError
from .schema import Event, Trace, validate_event_sequence
from .vocab import EVENT_STREAMS, STREAMS, token_at, token_id

TOKENS_PER_EVENT = len(EVENT_STREAMS)

_BOS = token_id("special", "BOS")
_EOS = token_id("special", "EOS")


class DecodedTrace(NamedTuple):
    """What a token stream carries: the vertical and the events."""

    vertical: str
    events: tuple[Event, ...]


def encode_event(event: Event) -> tuple[int, ...]:
    """Encode one event to its 7 token ids, in EVENT_STREAMS order."""
    if not isinstance(event, Event):
        raise TokenError(f"encode_event expects an Event, got {type(event).__name__}")
    return tuple(
        token_id(stream, getattr(event, stream)) for stream in EVENT_STREAMS
    )


def decode_event(ids: Sequence[int], seq: int = 0) -> Event:
    """Decode 7 token ids back to an Event with the given seq.

    Each position must hold a token from its own stream; a token from
    the wrong stream, an unassigned id, or the wrong number of ids
    raises TokenError.
    """
    if len(ids) != TOKENS_PER_EVENT:
        raise TokenError(
            f"an event is exactly {TOKENS_PER_EVENT} tokens, got {len(ids)}"
        )
    values = {}
    for position, (stream, token) in enumerate(zip(EVENT_STREAMS, ids, strict=True)):
        actual_stream, value = token_at(token)
        if actual_stream != stream:
            raise TokenError(
                f"position {position} of an event must be a {stream} token, "
                f"got {actual_stream}:{value} (id {token})"
            )
        values[stream] = value
    return Event(seq=seq, **values)


def encode_trace(trace: Trace) -> list[int]:
    """Encode a trace: BOS, vertical token, event tokens, EOS."""
    if not isinstance(trace, Trace):
        raise TokenError(f"encode_trace expects a Trace, got {type(trace).__name__}")
    ids = [_BOS, token_id("vertical", trace.vertical)]
    for event in trace.events:
        ids.extend(encode_event(event))
    ids.append(_EOS)
    return ids


def decode_trace(ids: Sequence[int]) -> DecodedTrace:
    """Decode a full trace token stream back to vertical and events.

    The stream must open with BOS then exactly one vertical token,
    carry a whole number of 7-token events, and close with EOS. The
    decoded events must satisfy the same sequence rules the schema
    enforces (positional seq, exactly one terminal outcome event).
    """
    ids = list(ids)
    if len(ids) < 2 + TOKENS_PER_EVENT + 1:
        raise TokenError(
            f"a trace stream needs BOS, a vertical token, at least one "
            f"event ({TOKENS_PER_EVENT} tokens), and EOS; got {len(ids)} tokens"
        )
    if ids[0] != _BOS:
        stream, value = token_at(ids[0])
        raise TokenError(
            f"a trace stream must start with BOS, got {stream}:{value} (id {ids[0]})"
        )
    vertical_stream, vertical = token_at(ids[1])
    if vertical_stream != "vertical":
        raise TokenError(
            f"the token after BOS must be the vertical conditioning token, "
            f"got {vertical_stream}:{vertical} (id {ids[1]})"
        )
    if ids[-1] != _EOS:
        stream, value = token_at(ids[-1])
        raise TokenError(
            f"a trace stream must end with EOS, got {stream}:{value} "
            f"(id {ids[-1]})"
        )
    body = ids[2:-1]
    if len(body) % TOKENS_PER_EVENT != 0:
        raise TokenError(
            f"the event body must be a multiple of {TOKENS_PER_EVENT} "
            f"tokens, got {len(body)}"
        )
    events = tuple(
        decode_event(body[start : start + TOKENS_PER_EVENT], seq=index)
        for index, start in enumerate(range(0, len(body), TOKENS_PER_EVENT))
    )
    validate_event_sequence(events)
    return DecodedTrace(vertical=vertical, events=events)


def score_to_band(score: float | int | None) -> str:
    """Map a 0 to 100 score to its score_band value.

    Thresholds are LeadPipe's own (DESIGN.md): 80 to 100 hot, 65 to 79
    warm, 50 to 64 nurture, below 50 low. None maps to "none" (no score
    recorded). A score outside 0 to 100, a non-number, or a bool raises
    TokenError.
    """
    if score is None:
        return "none"
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise TokenError(f"score must be a number or None, got {score!r}")
    if isinstance(score, float) and math.isnan(score):
        raise TokenError("score must not be NaN")
    if score < 0 or score > 100:
        raise TokenError(f"score {score} is outside the 0 to 100 scale")
    if score >= 80:
        return "hot"
    if score >= 65:
        return "warm"
    if score >= 50:
        return "nurture"
    return "low"


# Re-exported for callers that validate bucket membership via STREAMS.
__all__ = [
    "TOKENS_PER_EVENT",
    "DecodedTrace",
    "decode_event",
    "decode_trace",
    "encode_event",
    "encode_trace",
    "score_to_band",
    "STREAMS",
]
