"""Trace and event schema for WorkLang (DESIGN.md, Trace schema).

One JSON object per trace; corpus files are JSONL. This module defines
the dataclasses, their validation, and dict and JSON conversion. It
holds no token logic; see tokenizer.py.

Schema decisions taken from the design text:

- Every event carries all seven attribute fields (family, channel,
  score_band, latency_band, stage, tool, outcome_class) plus its seq.
  None and unknown are values, not omissions: a missing field is a
  schema error, never a silent default.
- seq is the event's 0-based position in the trace. It is positional
  metadata, not a token, and validation requires it to match the
  position so decode can rebuild it exactly.
- Exactly one event has family "outcome"; it is the last event, its
  outcome_class is not "none", every other event carries outcome_class
  "none", and the trace level outcome field duplicates the outcome
  event's outcome_class. These are the design's words ("the trace
  terminates (exactly one per trace)", outcome "duplicated at trace
  level") written as checks.
- agent names the producing agent and agent_version its version, as
  two trace level fields. (DESIGN.md writes generator as an explicit
  {name, version, seed} object but agent as "agent ... with
  agent_version", so agent follows the flat reading.)
- source is a non-empty string. The design names qualbench,
  leadpipe-demo, and capstone, and leaves the set open to converter
  names, so the schema does not close it.
- synthetic is mandatory and must be a bool, so no future real trace
  can silently mix into a synthetic corpus.

Lossy boundary, named not hidden (DESIGN.md round trip section):
encoding is exact at the event level for schema fields and lossy at
the raw-record level. from_dict drops keys that are not schema fields.
That drop is where raw-record data dies: message text, names, contact
details, exact timestamps, and any attribute outside the bucket sets
never enter the schema, so they can never be recovered from a trace or
from its tokens. Tests assert this drop explicitly.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping, Sequence

from .errors import SchemaError
from .vocab import (
    CHANNELS,
    FAMILIES,
    LATENCY_BANDS,
    OUTCOME_CLASSES,
    SCORE_BANDS,
    STAGES,
    TOOLS,
    VERTICALS,
)

EVENT_FIELDS = (
    "seq",
    "family",
    "channel",
    "score_band",
    "latency_band",
    "stage",
    "tool",
    "outcome_class",
)

TRACE_FIELDS = (
    "trace_id",
    "source",
    "synthetic",
    "vertical",
    "agent",
    "agent_version",
    "events",
    "outcome",
    "generator",
)

GENERATOR_FIELDS = ("name", "version", "seed")

# Sources named by the design. The set stays open to converter names.
KNOWN_SOURCES = ("qualbench", "leadpipe-demo", "capstone")

_EVENT_STREAM_VALUES = {
    "family": FAMILIES,
    "channel": CHANNELS,
    "score_band": SCORE_BANDS,
    "latency_band": LATENCY_BANDS,
    "stage": STAGES,
    "tool": TOOLS,
    "outcome_class": OUTCOME_CLASSES,
}


def _require_str(field: str, value: object, owner: str) -> str:
    if not isinstance(value, str) or not value:
        raise SchemaError(f"{owner} field {field!r} must be a non-empty string")
    return value


def _require_bucket(field: str, value: object, owner: str) -> str:
    allowed = _EVENT_STREAM_VALUES[field]
    if not isinstance(value, str) or value not in allowed:
        raise SchemaError(
            f"{owner} field {field!r} has value {value!r}; "
            f"expected one of: {', '.join(allowed)}"
        )
    return value


def validate_event_sequence(events: Sequence[Event]) -> None:
    """Check the trace level rules over an ordered event sequence.

    Rules: at least one event; seq matches position; exactly one
    outcome event and it is last; outcome_class is "none" everywhere
    except on the outcome event, where it must not be "none".
    Raises SchemaError on the first violation found.
    """
    if not events:
        raise SchemaError("a trace must contain at least one event")
    for index, event in enumerate(events):
        if not isinstance(event, Event):
            raise SchemaError(f"event at position {index} is not an Event: {event!r}")
        if event.seq != index:
            raise SchemaError(
                f"event at position {index} has seq {event.seq}; "
                "seq must equal the 0-based position in the trace"
            )
    outcome_positions = [i for i, e in enumerate(events) if e.family == "outcome"]
    if len(outcome_positions) != 1:
        raise SchemaError(
            f"a trace must contain exactly one outcome event, "
            f"found {len(outcome_positions)}"
        )
    if outcome_positions[0] != len(events) - 1:
        raise SchemaError("the outcome event must be the last event in the trace")
    for event in events[:-1]:
        if event.outcome_class != "none":
            raise SchemaError(
                f"event with seq {event.seq} has outcome_class "
                f"{event.outcome_class!r}; outcome_class is only meaningful "
                "on the outcome event and must be 'none' elsewhere"
            )
    if events[-1].outcome_class == "none":
        raise SchemaError(
            "the outcome event must carry a real outcome_class, not 'none'"
        )


@dataclasses.dataclass(frozen=True)
class Event:
    """One schema event: seq plus the seven tokenised attribute fields."""

    seq: int
    family: str
    channel: str
    score_band: str
    latency_band: str
    stage: str
    tool: str
    outcome_class: str

    def __post_init__(self) -> None:
        if isinstance(self.seq, bool) or not isinstance(self.seq, int) or self.seq < 0:
            raise SchemaError(
                f"event field 'seq' must be a non-negative int, got {self.seq!r}"
            )
        for field in EVENT_FIELDS[1:]:
            _require_bucket(field, getattr(self, field), "event")

    @classmethod
    def from_dict(cls, data: Mapping) -> Event:
        """Build an Event from a mapping.

        Keys outside the schema are dropped (the documented lossy
        boundary: message text, names, timestamps, and friends die
        here). Missing keys and out-of-bucket values raise SchemaError.
        """
        if not isinstance(data, Mapping):
            raise SchemaError(f"an event must be a mapping, got {type(data).__name__}")
        missing = [f for f in EVENT_FIELDS if f not in data]
        if missing:
            raise SchemaError(f"event is missing required fields: {', '.join(missing)}")
        return cls(**{field: data[field] for field in EVENT_FIELDS})

    def to_dict(self) -> dict:
        return {field: getattr(self, field) for field in EVENT_FIELDS}


@dataclasses.dataclass(frozen=True)
class Generator:
    """Provenance of a trace: generator name, version, and seed."""

    name: str
    version: str
    seed: int

    def __post_init__(self) -> None:
        _require_str("name", self.name, "generator")
        _require_str("version", self.version, "generator")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise SchemaError(
                f"generator field 'seed' must be an int, got {self.seed!r}"
            )

    @classmethod
    def from_dict(cls, data: Mapping) -> Generator:
        if not isinstance(data, Mapping):
            raise SchemaError(
                f"generator must be a mapping, got {type(data).__name__}"
            )
        missing = [f for f in GENERATOR_FIELDS if f not in data]
        if missing:
            raise SchemaError(
                f"generator is missing required fields: {', '.join(missing)}"
            )
        return cls(**{field: data[field] for field in GENERATOR_FIELDS})

    def to_dict(self) -> dict:
        return {field: getattr(self, field) for field in GENERATOR_FIELDS}


@dataclasses.dataclass(frozen=True)
class Trace:
    """One trace: metadata plus an ordered, validated event sequence."""

    trace_id: str
    source: str
    synthetic: bool
    vertical: str
    agent: str
    agent_version: str
    events: tuple[Event, ...]
    outcome: str
    generator: Generator

    def __post_init__(self) -> None:
        _require_str("trace_id", self.trace_id, "trace")
        _require_str("source", self.source, "trace")
        if not isinstance(self.synthetic, bool):
            raise SchemaError(
                f"trace field 'synthetic' is mandatory and must be a bool, "
                f"got {self.synthetic!r}"
            )
        if self.vertical not in VERTICALS:
            raise SchemaError(
                f"trace field 'vertical' has value {self.vertical!r}; "
                f"expected one of: {', '.join(VERTICALS)}"
            )
        _require_str("agent", self.agent, "trace")
        _require_str("agent_version", self.agent_version, "trace")
        if isinstance(self.events, (str, bytes)) or not isinstance(
            self.events, Sequence
        ):
            raise SchemaError("trace field 'events' must be a sequence of events")
        events = tuple(self.events)
        object.__setattr__(self, "events", events)
        validate_event_sequence(events)
        if not isinstance(self.generator, Generator):
            raise SchemaError("trace field 'generator' must be a Generator")
        terminal = events[-1].outcome_class
        if self.outcome != terminal:
            raise SchemaError(
                f"trace field 'outcome' is {self.outcome!r} but the outcome "
                f"event carries {terminal!r}; the trace level field must "
                "duplicate the terminal outcome_class"
            )

    @classmethod
    def from_dict(cls, data: Mapping) -> Trace:
        """Build a Trace from a mapping (one JSONL record).

        Keys outside the schema are dropped (see the lossy boundary in
        the module docstring). Missing keys, a non-bool synthetic, an
        unknown vertical, and any event sequence violation raise
        SchemaError.
        """
        if not isinstance(data, Mapping):
            raise SchemaError(f"a trace must be a mapping, got {type(data).__name__}")
        missing = [f for f in TRACE_FIELDS if f not in data]
        if missing:
            raise SchemaError(f"trace is missing required fields: {', '.join(missing)}")
        raw_events = data["events"]
        if isinstance(raw_events, (str, bytes)) or not isinstance(
            raw_events, Sequence
        ):
            raise SchemaError("trace field 'events' must be a list of event objects")
        events = tuple(Event.from_dict(item) for item in raw_events)
        return cls(
            trace_id=data["trace_id"],
            source=data["source"],
            synthetic=data["synthetic"],
            vertical=data["vertical"],
            agent=data["agent"],
            agent_version=data["agent_version"],
            events=events,
            outcome=data["outcome"],
            generator=Generator.from_dict(data["generator"]),
        )

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "source": self.source,
            "synthetic": self.synthetic,
            "vertical": self.vertical,
            "agent": self.agent,
            "agent_version": self.agent_version,
            "events": [event.to_dict() for event in self.events],
            "outcome": self.outcome,
            "generator": self.generator.to_dict(),
        }


def trace_to_json(trace: Trace) -> str:
    """Serialise one trace to a JSON string (one JSONL line)."""
    return json.dumps(trace.to_dict())


def trace_from_json(text: str) -> Trace:
    """Parse one trace from a JSON string, with full validation."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SchemaError(f"trace is not valid JSON: {exc}") from exc
    return Trace.from_dict(data)
