"""QualBench trace emitter: RunResult in, WorkLang Trace out.

Build step 2 of DESIGN.md (corpus source A). This module runs
QualBench scenarios through the unmodified QualBench runner with the
two shipped example agents and converts each run into a schema valid
WorkLang trace. The QualBench repository is not modified, and no agent
contract changes: the emitter consumes QualBench through its existing
Python API only (scenario.discover_packs / load_pack,
runner.run_scenario, agents.webhook_agent.load_agent_file).

Why a recording session wrapper: RunResult alone does not retain the
ordered tool executions. The transcript keeps user and agent text
turns, the calendar keeps final bookings, and the CRM keeps write
snapshots, but the sequence of tool calls and their results exists
only inside the runner while it drives the agent. The responses that
carry those tool calls pass through the agent session's send method,
and the results come back as the next payload's tool_results. So the
emitter wraps the agent session in a recording proxy that implements
the exact same start/send protocol, remembers every payload and
response in order, and hands the wrapped session to run_scenario
unchanged. After the run, the emitter replays the recording in the
runner's own order (reply absorbed, outcome absorbed, tool calls
executed, results delivered) alongside the final RunResult state
(bookings, final stage, score, handoff) and builds the event stream.
CLI outputs were not used: the published results JSONs carry assertion
outcomes only, no traces (DESIGN.md grounding note), and the CLI never
exposes per-call data.

Event mapping, one rule per family:

- enquiry: one per user turn (each payload that carries a message).
- reply: one per non-empty agent reply in a response.
- tool_call: one per executed tool call, emitted when its result is
  delivered, in execution order. A call whose tool name is outside
  the schema's tool bucket set raises EmitterError rather than being
  silently normalised; the two shipped agents only ever call the five
  ToolExecutor tools, which are all in the bucket set.
- booking: one per successful calendar.book result (result ok). The
  outcome class "booked" likewise requires a real booking on the
  calendar; an agent merely claiming a booking in its reported stage
  does not produce one. Failed booking attempts stay tool_call events.
- stage: one per recorded stage transition. Stage reports are absorbed
  from the same places the runner absorbs them (response outcome,
  crm.upsert_lead fields, report_outcome args), in the same order, and
  a stage event fires when the absorbed in-vocabulary stage changes.
  A final stage event fires at the end if the runner's final stage
  reporting (RunResult.stage) names a stage the stream has not
  reached. Every event also carries the current stage after the event
  on its stage attribute, or "none" before any stage is recorded.
- handoff: one, at the first point a handoff is reported true (or at
  the end if only RunResult.handoff shows it).
- follow_up: never emitted from this source. QualBench runs contain
  no scheduled-chase signal; inventing one would be fabrication.
- outcome: exactly one, last, derived from final run state by
  _derive_outcome below.

Attribute rules from the design, applied exactly:

- channel: taken from an optional scenario channel field (the
  QualBench Scenario object carries none today), validated against
  the channel bucket set, and defaulting to "unknown". A channel is
  never invented per scenario (DESIGN.md conflict 3).
- score_band: the current absorbed score after the event, mapped
  through tokenizer.score_to_band (LeadPipe thresholds). Scores the
  runner itself would reject (non-numeric, outside 0 to 100) never
  enter the state.
- latency_band: generator-declared, never measured (DESIGN.md
  conflict 1). The first event of a trace carries "none" (it has no
  predecessor). Every later event draws the declared gap before it
  from LATENCY_DISTRIBUTIONS for the trace's vertical, using the
  trace's seeded generator. The distributions are printed in the
  corpus card.
- synthetic: always True. source: "qualbench".
- generator: {name: EMITTER_NAME, version: EMITTER_VERSION, seed:
  the per-trace seed}. agent and agent_version: the shipped agent's
  name and the version QualBench's own published results files
  declare for it ("1.0.0" for both agents in results/*.json).

Recorded conflict, not smoothed over: QualBench agents report two
stage strings that the WorkLang stage bucket set (the ten LeadPipe
stages plus none, fixed by enumeration in DESIGN.md) does not contain:
HANDOFF and UNQUALIFIED. Following the design's own rule for tool
names (no silent normalisation; merging is a versioned schema
decision), neither string is written into the stage stream:
- HANDOFF reports coincide with the outcome's handoff flag, which the
  schema already models as the handoff family; the stage state is
  left unchanged.
- UNQUALIFIED reports leave the stage state unchanged, but a run whose
  final reported stage is UNQUALIFIED closes with outcome "lost": the
  agent actively disqualified the lead, which is a lost pipeline
  outcome, not a stall (nothing is pending) and not a drop (contact
  did not simply cease). This outcome level mapping is an emitter
  decision, stated here and in the corpus card; whether UNQUALIFIED
  and HANDOFF deserve stage tokens of their own is a schema v2
  question for the design, flagged in the step 2 progress log entry.
"""

from __future__ import annotations

import copy
import hashlib
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import WorkLangError
from .schema import Event, Generator, Trace
from .tokenizer import score_to_band
from .vocab import CHANNELS, STAGES, TOOLS, VERTICALS

EMITTER_NAME = "worklang-qualbench-emitter"
EMITTER_VERSION = "0.1.0"

SOURCE = "qualbench"

# The shipped example agents, by the names QualBench's published
# results files use, with the agent_version those files declare.
AGENT_FILES = {
    "reference-agent": "examples/reference-agent/agent.py",
    "naive-agent": "examples/naive-agent/agent.py",
}
AGENT_VERSIONS = {
    "reference-agent": "1.0.0",
    "naive-agent": "1.0.0",
}

# Generator-declared latency distributions (DESIGN.md conflict 1):
# weights over the five non-none latency bands, per vertical. These
# are declared by the generator, printed in the corpus card, and
# never described as measured. The first event of every trace carries
# "none" instead of a draw.
LATENCY_DISTRIBUTIONS: dict[str, tuple[tuple[str, float], ...]] = {
    "real-estate": (
        ("instant", 0.10),
        ("fast", 0.35),
        ("slow", 0.30),
        ("delayed", 0.15),
        ("overnight", 0.10),
    ),
    "med-spa": (
        ("instant", 0.15),
        ("fast", 0.40),
        ("slow", 0.25),
        ("delayed", 0.12),
        ("overnight", 0.08),
    ),
    "home-services": (
        ("instant", 0.10),
        ("fast", 0.30),
        ("slow", 0.30),
        ("delayed", 0.20),
        ("overnight", 0.10),
    ),
    "dental": (
        ("instant", 0.12),
        ("fast", 0.38),
        ("slow", 0.28),
        ("delayed", 0.14),
        ("overnight", 0.08),
    ),
    "agencies": (
        ("instant", 0.08),
        ("fast", 0.30),
        ("slow", 0.32),
        ("delayed", 0.18),
        ("overnight", 0.12),
    ),
    "general": (
        ("instant", 0.10),
        ("fast", 0.35),
        ("slow", 0.30),
        ("delayed", 0.15),
        ("overnight", 0.10),
    ),
}


class EmitterError(WorkLangError):
    """A run cannot be emitted as a schema valid trace."""


def default_qualbench_root() -> Path:
    """Locate the QualBench repository.

    The WORKLANG_QUALBENCH_ROOT environment variable wins. Otherwise
    the sibling checkout in the flagship workspace is used:
    github-profile/open-source-5/repos/qualbench, relative to this
    repo at github-profile/flagships/worklang/repo.
    """
    env = os.environ.get("WORKLANG_QUALBENCH_ROOT")
    if env:
        return Path(env)
    return (
        Path(__file__).resolve().parents[5]
        / "open-source-5"
        / "repos"
        / "qualbench"
    )


@dataclass
class _QualBench:
    """The slice of the QualBench Python API the emitter consumes."""

    run_scenario: Any
    discover_packs: Any
    load_pack: Any
    load_agent_file: Any


def load_qualbench(root: Path) -> _QualBench:
    """Import the QualBench API from a checkout, via its src layout."""
    src = root / "src"
    if not (src / "qualbench" / "runner.py").is_file():
        raise EmitterError(
            f"no QualBench checkout found at {root} "
            "(expected src/qualbench/runner.py there); pass the repo "
            "root explicitly or set WORKLANG_QUALBENCH_ROOT"
        )
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from qualbench.agents.webhook_agent import load_agent_file
    from qualbench.runner import run_scenario
    from qualbench.scenario import discover_packs, load_pack

    return _QualBench(
        run_scenario=run_scenario,
        discover_packs=discover_packs,
        load_pack=load_pack,
        load_agent_file=load_agent_file,
    )


def group_id_for(scenario: Any) -> str:
    """The split group of a scenario: all its variants share this id."""
    return f"{SOURCE}:{scenario.scenario_id}"


def trace_id_for(scenario: Any, agent_name: str, variant: int) -> str:
    """Stable trace id: source, scenario slug, agent, variant index."""
    return f"{group_id_for(scenario)}:{agent_name}:v{variant}"


def trace_seed_for(corpus_seed: int, group_id: str, agent_name: str,
                   variant: int) -> int:
    """Derive one trace's generator seed from the corpus seed."""
    key = f"{corpus_seed}|{group_id}|{agent_name}|{variant}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


class _RecordingSession:
    """Session proxy that records payloads and responses in order.

    Implements the same start/send protocol as the wrapped session;
    the runner and the agent both see an ordinary session. Recorded
    exchanges are dicts with the payload's message and tool_results
    and the response the agent returned.
    """

    def __init__(self, inner: Any):
        self._inner = inner
        self.exchanges: list[dict] = []

    def start(self, context: dict) -> None:
        self._inner.start(context)

    def send(self, payload: dict) -> dict:
        response = self._inner.send(payload)
        self.exchanges.append(
            {
                "message": payload.get("message"),
                "tool_results": copy.deepcopy(payload.get("tool_results") or []),
                "response": copy.deepcopy(response)
                if isinstance(response, dict)
                else response,
            }
        )
        return response


class _TraceBuilder:
    """Accumulates events, tracking stage, score, and handoff state.

    Mirrors the runner's absorption rules: outcome data is absorbed
    from response outcome dicts and from crm.upsert_lead and
    report_outcome arguments at execution time, in run order. Every
    event snapshots the stage and score band as they stand after that
    event's own effects.
    """

    def __init__(self, vertical: str, rng: random.Random):
        self.vertical = vertical
        self.rng = rng
        self.raw_events: list[dict] = []
        self.stage_state: str | None = None
        self.score_state: float | None = None
        self.handoff_state = False

    def _draw_latency(self) -> str:
        if not self.raw_events:
            return "none"
        bands = LATENCY_DISTRIBUTIONS[self.vertical]
        draw = self.rng.random()
        cumulative = 0.0
        for band, weight in bands:
            cumulative += weight
            if draw < cumulative:
                return band
        return bands[-1][0]

    def add(self, family: str, tool: str = "none",
            outcome_class: str = "none") -> None:
        if family == "tool_call" and tool not in TOOLS:
            raise EmitterError(
                f"tool {tool!r} is not in the schema's tool bucket set; "
                "refusing to silently normalise it into the corpus"
            )
        self.raw_events.append(
            {
                "family": family,
                "channel": "unknown",
                "score_band": score_to_band(self.score_state)
                if self.score_state is not None
                else "none",
                "latency_band": self._draw_latency(),
                "stage": self.stage_state or "none",
                "tool": tool,
                "outcome_class": outcome_class,
            }
        )

    def absorb(self, data: Any) -> None:
        """Absorb outcome data exactly where the runner absorbs it."""
        if not isinstance(data, dict):
            return
        stage = data.get("stage")
        if stage:
            self._absorb_stage(str(stage).upper())
        score = data.get("score")
        if score is not None:
            try:
                value = float(score)
            except (TypeError, ValueError):
                value = None
            if value is not None and 0 <= value <= 100:
                self.score_state = value
        if data.get("handoff"):
            self._absorb_handoff()

    def _absorb_stage(self, stage: str) -> None:
        if stage not in STAGES:
            # HANDOFF and UNQUALIFIED live here: see the module
            # docstring's recorded conflict. Neither enters the stage
            # stream; HANDOFF surfaces through the handoff flag in the
            # same outcome data, UNQUALIFIED through outcome "lost".
            return
        if stage != self.stage_state:
            self.stage_state = stage
            self.add("stage")

    def _absorb_handoff(self) -> None:
        if not self.handoff_state:
            self.handoff_state = True
            self.add("handoff")


def _normalise_calls(response: dict) -> list[dict]:
    """Normalise a response's tool calls the way the runner does."""
    calls = []
    for index, call in enumerate(response.get("tool_calls") or []):
        if not isinstance(call, dict):
            continue
        calls.append(
            {
                "id": call.get("id") or f"call-{index + 1}",
                "tool": str(call.get("tool", "")),
                "args": call.get("args") or {},
            }
        )
    return calls


def _absorb_tool_args(builder: _TraceBuilder, call: dict) -> None:
    """Mirror ToolExecutor: outcome data hides in two tools' args."""
    args = call["args"]
    if not isinstance(args, dict):
        return
    if call["tool"] == "crm.upsert_lead":
        fields = args.get("fields", args)
        builder.absorb(fields)
    elif call["tool"] == "report_outcome":
        builder.absorb(args)


def _derive_outcome(result: Any, builder: _TraceBuilder) -> str:
    """Terminal outcome class from final run state.

    Precedence: a real calendar booking beats any reported stage;
    then the final reported stage (WON, LOST, and UNQUALIFIED as a
    lost disqualification); then stalled, which per the design means
    the trace ends with work still pending, true both when a human
    handoff is waiting and when the agent reported a mid-pipeline
    outcome and the scripted conversation simply ended; otherwise
    dropped, contact ceased with nothing recorded.
    """
    if result.bookings:
        return "booked"
    stage = (result.stage or "").upper()
    if stage == "WON":
        return "won"
    if stage in ("LOST", "UNQUALIFIED"):
        return "lost"
    if result.handoff or builder.handoff_state:
        return "stalled"
    if result.outcome_reported:
        return "stalled"
    return "dropped"


def emit_trace(result: Any, exchanges: list[dict], *, trace_id: str,
               agent_name: str, trace_seed: int) -> Trace:
    """Convert one recorded run (RunResult plus exchanges) to a Trace."""
    scenario = result.scenario
    vertical = scenario.vertical if scenario.vertical in VERTICALS else "general"
    channel = getattr(scenario, "channel", None)
    if channel not in CHANNELS:
        channel = "unknown"
    builder = _TraceBuilder(vertical, random.Random(trace_seed))
    pending: list[dict] = []
    for exchange in exchanges:
        results = exchange["tool_results"]
        if results:
            by_id = {call["id"]: call for call in pending}
            remaining = list(pending)
            for item in results:
                call = by_id.get(item.get("id"))
                if call is None and remaining:
                    call = remaining[0]
                if call is None:
                    continue
                if call in remaining:
                    remaining.remove(call)
                builder.add("tool_call", tool=call["tool"])
                if call["tool"] == "calendar.book" and item.get("ok"):
                    builder.add("booking")
                _absorb_tool_args(builder, call)
            pending = []
        if exchange["message"] is not None:
            builder.add("enquiry")
        response = exchange["response"]
        if not isinstance(response, dict):
            response = {}
        if response.get("reply"):
            builder.add("reply")
        builder.absorb(response.get("outcome") or {})
        pending = _normalise_calls(response)
    # Calls whose results never arrived (the run errored mid-turn):
    # the invocations happened, so they are recorded without results.
    for call in pending:
        builder.add("tool_call", tool=call["tool"])
    # Final stage reporting and handoff from the RunResult itself.
    final_stage = (result.stage or "").upper()
    if final_stage in STAGES and final_stage != builder.stage_state:
        builder.stage_state = final_stage
        builder.add("stage")
    if result.handoff:
        builder._absorb_handoff()
    if result.score is not None and builder.score_state is None:
        try:
            value = float(result.score)
        except (TypeError, ValueError):
            value = None
        if value is not None and 0 <= value <= 100:
            builder.score_state = value
    outcome_class = _derive_outcome(result, builder)
    builder.add("outcome", outcome_class=outcome_class)
    events = tuple(
        Event(seq=index, **{**raw, "channel": channel})
        for index, raw in enumerate(builder.raw_events)
    )
    return Trace(
        trace_id=trace_id,
        source=SOURCE,
        synthetic=True,
        vertical=vertical,
        agent=agent_name,
        agent_version=AGENT_VERSIONS[agent_name],
        events=events,
        outcome=outcome_class,
        generator=Generator(
            name=EMITTER_NAME, version=EMITTER_VERSION, seed=trace_seed
        ),
    )


def run_and_emit(root: Path, scenario: Any, agent_name: str, *,
                 variant: int = 0, corpus_seed: int = 0) -> Trace:
    """Run one scenario through the unmodified runner and emit its trace."""
    if agent_name not in AGENT_FILES:
        raise EmitterError(
            f"unknown agent {agent_name!r}; expected one of: "
            + ", ".join(sorted(AGENT_FILES))
        )
    qb = load_qualbench(root)
    agent_file = root / AGENT_FILES[agent_name]
    if not agent_file.is_file():
        raise EmitterError(f"agent file not found: {agent_file}")
    holder: dict[str, _RecordingSession] = {}

    def factory() -> _RecordingSession:
        recording = _RecordingSession(qb.load_agent_file(agent_file))
        holder["recording"] = recording
        return recording

    result = qb.run_scenario(factory, scenario)
    trace_seed = trace_seed_for(
        corpus_seed, group_id_for(scenario), agent_name, variant
    )
    return emit_trace(
        result,
        holder["recording"].exchanges,
        trace_id=trace_id_for(scenario, agent_name, variant),
        agent_name=agent_name,
        trace_seed=trace_seed,
    )
