"""LeadPipe lifecycle synthesiser: seed world in, WorkLang traces out.

Build step 3 of DESIGN.md (corpus source B). LeadPipe's demo world is
a snapshot, not a sequence: 12 seeded leads with stage, source, score
components and band, next_action and relative timing, plus audit
entries for the seeded captures. This module turns that snapshot into
lifecycle traces two ways, consuming LeadPipe through its existing
Python API only (demo_data.seed_demo_dataset, stages.TRANSITIONS,
scoring, store.fake.FakeGoogleClient, tools.dispatch). The LeadPipe
repository is not modified.

Lifecycle traces (one family per seed lead, several seeded walks
each). A trace is the lead's lifecycle in two segments:

- History: the recorded beginning. An enquiry (capture) at NEW, a
  score_lead tool call when the seed carries score components (the
  seed's own audit log records exactly those two acts, at capture
  time), then stage events along a legal path over TRANSITIONS from
  NEW to the lead's seeded stage. The path is a shortest legal path;
  where several shortest paths exist, the trace's seeded generator
  chooses, preferring paths whose intermediate stages avoid BOOKED so
  the history does not imply a booking the seed never recorded. Every
  step is a legal TRANSITIONS step by construction.
- Future: a seeded random walk over TRANSITIONS from the seeded
  stage. The lead's seeded next_action is pending at the snapshot:
  its follow_up fires at the start of the future segment when the
  action is already due (next_action_in_days <= 0) and after the
  first future transition otherwise; terminal leads (WON, LOST) have
  no next_action and no future segment. Entering BOOKED emits a
  booking event (LeadPipe's book_appointment). After each future
  transition the lead stays engaged with the declared per-band
  probability in CONTINUE_PROBABILITIES (generator parameters, printed
  in the corpus card); if contact ceases, the outcome is booked when
  the walk stands at BOOKED, stalled when the seeded next_action is
  still unactioned, and dropped otherwise. A walk that reaches WON
  or LOST closes with that outcome; a walk truncated at
  MAX_FUTURE_STEPS with no resolution closes as stalled (the
  lifecycle had not resolved when the generator stopped, and
  follow-up work remains pending), or booked when it stands at
  BOOKED. Outcome precedence at any ending: WON, then LOST, then
  standing at BOOKED, then the cease rules above. This is an
  end-state rule, deliberately different from source A's (a
  QualBench run ends at its booking; a LeadPipe lifecycle continues
  past BOOKED toward PROPOSAL and WON, so the terminal state is what
  the outcome names).

Anchor traces (the demo walkthrough family). DESIGN.md includes the
walkthrough run as an anchor trace family. The walkthrough script
pins no seed moment (it seeds relative to the real clock), so the
emitter executes its exact dispatch sequence (seed, then the nine
numbered steps, approval mode) against a fake store seeded at the
fixed moment LeadPipe's own test suite uses, Monday 2026-11-23
09:00 UTC, which keeps the family reproducible. The family holds one
trace per lead the run touches: the 12 seeded leads, built from
their seeded audit entries only (capture, the one deduplicated
capture, score; no stage transitions are recorded in the audit log
and none are invented), closed with the outcome their post-run state
names (stage WON, LOST, or BOOKED, else stalled while a next_action
is pending, else dropped); and the walkthrough's own new lead,
recorded dispatch by dispatch, reads included. The results of each
dispatch are checked against the behaviour LeadPipe's own tests pin
(scored total, dedup, approval preview, booking created, idempotent
re-book); a mismatch raises SynthesiserError rather than emitting a
wrong trace.

Attribute rules:

- channel: the lead's real source value, on every event of its
  lifecycle traces. Recorded conflict, not smoothed over: DESIGN.md
  enumerates the channel bucket set as "the seven source values in
  LeadPipe's demo seed and Source enum", but the Source enum has
  nine values and the demo seed itself uses two with no bucket:
  walk_in (Grace Liu) and other (Noah Smith). The schema cannot hold
  them. Following step 2's precedent for out-of-vocabulary values,
  both degrade to the channel stream's absence value "unknown", and
  the degradation is stated here, in the corpus card, and in the
  design progress log; whether they deserve channel tokens is a
  schema v2 question. Every other lead keeps its exact source.
- vertical: "general" for every source B trace. LeadPipe carries no
  vertical field; assigning the five QualBench verticals from
  company names would be inventing data.
- score_band: the lead's score state, LeadPipe's own total mapped
  through tokenizer.score_to_band. The seed lead with no score
  components carries "none" throughout.
- latency_band: generator-declared, never measured (DESIGN.md
  conflict 1), drawn from the "general" row of the same declared
  distributions source A uses; the first event of every trace
  carries "none".
- synthetic: always True. source: "leadpipe-demo".
- generator: {name: SYNTHESISER_NAME, version: SYNTHESISER_VERSION,
  seed: the per-trace seed}. agent: "leadpipe-synthesiser" for
  lifecycle traces (agent_version SYNTHESISER_VERSION) and
  "demo-walkthrough" for the anchor family (agent_version: the
  LeadPipe version that produced the run).
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .errors import WorkLangError
from .qualbench_emitter import LATENCY_DISTRIBUTIONS
from .schema import Event, Generator, Trace
from .tokenizer import score_to_band

SYNTHESISER_NAME = "worklang-leadpipe-synthesiser"
SYNTHESISER_VERSION = "0.1.0"

SOURCE = "leadpipe-demo"
WALKTHROUGH_GROUP = f"{SOURCE}:walkthrough"

# The fixed seed moment LeadPipe's own test suite uses
# (tests/conftest.py FIXED_NOW): Monday 2026-11-23 09:00 UTC, whose
# seeded business days are 2026-11-24, 2026-11-25, 2026-11-26.
FIXED_NOW = datetime(2026, 11, 23, 9, 0, tzinfo=UTC)

# Source values with no channel bucket (see the module docstring's
# recorded conflict). They degrade to the stream's absence value.
UNBUCKETED_SOURCES = ("walk_in", "other")

# Declared generator parameters for the future walk: after each
# future transition, the probability the lead stays engaged, by the
# lead's score band ("none": never scored). Printed in the corpus
# card; these are generator declarations, not measured behaviour.
CONTINUE_PROBABILITIES = {
    "hot": 0.85,
    "warm": 0.75,
    "nurture": 0.60,
    "low": 0.45,
    "none": 0.60,
}

MAX_FUTURE_STEPS = 8

_WALKTHROUGH_SCORES = {
    "problem": 18,
    "fit": 19,
    "value": 13,
    "size": 8,
    "pay": 9,
    "access": 9,
    "urgency": 6,
    "activity": 3,
}


class SynthesiserError(WorkLangError):
    """A LeadPipe run cannot be emitted as a schema valid trace."""


def default_leadpipe_root() -> Path:
    """Locate the LeadPipe repository.

    The WORKLANG_LEADPIPE_ROOT environment variable wins. Otherwise
    the sibling checkout in the flagship workspace is used:
    github-profile/open-source-5/repos/leadpipe-mcp, relative to this
    repo at github-profile/flagships/worklang/repo.
    """
    env = os.environ.get("WORKLANG_LEADPIPE_ROOT")
    if env:
        return Path(env)
    return (
        Path(__file__).resolve().parents[5]
        / "open-source-5"
        / "repos"
        / "leadpipe-mcp"
    )


@dataclass
class _LeadPipe:
    """The slice of the LeadPipe Python API the synthesiser consumes."""

    transitions: dict[str, frozenset[str]]
    seed_demo_dataset: Any
    bootstrap_store: Any
    fake_client_cls: Any
    tool_context_cls: Any
    config_cls: Any
    write_mode: Any
    dispatch: Any
    components_from_row: Any
    compute_total: Any
    version: str


def load_leadpipe(root: Path) -> _LeadPipe:
    """Import the LeadPipe API from a checkout, via its src layout."""
    src = root / "src"
    if not (src / "leadpipe" / "stages.py").is_file():
        raise SynthesiserError(
            f"no LeadPipe checkout found at {root} "
            "(expected src/leadpipe/stages.py there); pass the repo "
            "root explicitly or set WORKLANG_LEADPIPE_ROOT"
        )
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    import leadpipe
    from leadpipe import scoring
    from leadpipe.config import Config
    from leadpipe.demo_data import seed_demo_dataset
    from leadpipe.models import WriteMode
    from leadpipe.stages import TRANSITIONS
    from leadpipe.store.fake import FakeGoogleClient
    from leadpipe.store.schema import bootstrap_store
    from leadpipe.tools import ToolContext, dispatch

    transitions = {
        stage.value: frozenset(target.value for target in targets)
        for stage, targets in TRANSITIONS.items()
    }
    return _LeadPipe(
        transitions=transitions,
        seed_demo_dataset=seed_demo_dataset,
        bootstrap_store=bootstrap_store,
        fake_client_cls=FakeGoogleClient,
        tool_context_cls=ToolContext,
        config_cls=Config,
        write_mode=WriteMode,
        dispatch=dispatch,
        components_from_row=scoring.components_from_row,
        compute_total=scoring.compute_total,
        version=leadpipe.__version__,
    )


@dataclass(frozen=True)
class SeedLead:
    """One seeded demo lead, as the synthesiser consumes it."""

    lead_id: str
    slug: str
    source: str
    stage: str
    score_total: int | None
    band: str | None
    next_action: str
    next_action_in_days: int | None

    @property
    def channel(self) -> str:
        """The lead's channel: its source, or the absence value."""
        return channel_for_source(self.source)


def channel_for_source(source: str) -> str:
    """Map a LeadPipe source value to a schema channel.

    Seven of the nine Source values are channel buckets verbatim.
    walk_in and other have no bucket and degrade to "unknown" (see
    the module docstring's recorded conflict).
    """
    if source in UNBUCKETED_SOURCES:
        return "unknown"
    return source


def slug_for_name(name: str) -> str:
    """Stable lead slug from a display name (data dies here by design:
    the schema holds no names; the slug is only an id)."""
    slug = name.lower().replace("'", "")
    return "".join(ch if ch.isalnum() else "-" for ch in slug).strip("-")


def _seeded_client(lp: _LeadPipe) -> tuple[Any, Any]:
    """A fake store bootstrapped and seeded at the fixed moment."""
    client = lp.fake_client_cls()
    lp.bootstrap_store(client, now=FIXED_NOW)
    seed = lp.seed_demo_dataset(client, now=FIXED_NOW)
    return client, seed


def load_seed_leads(root: Path) -> list[SeedLead]:
    """Read the 12 seeded leads from a freshly seeded fake store."""
    lp = load_leadpipe(root)
    client, _seed = _seeded_client(lp)
    today = FIXED_NOW.date()
    leads: list[SeedLead] = []
    for row in client.list_leads():
        components = lp.components_from_row(row)
        total = lp.compute_total(components) if components is not None else None
        raw_date = row.get("next_action_date") or ""
        in_days = None
        if raw_date:
            from datetime import date

            in_days = (date.fromisoformat(raw_date) - today).days
        leads.append(
            SeedLead(
                lead_id=row["lead_id"],
                slug=slug_for_name(row["name"]),
                source=row["source"],
                stage=row["stage"],
                score_total=total,
                band=row.get("band") or None,
                next_action=row.get("next_action") or "",
                next_action_in_days=in_days,
            )
        )
    return leads


def group_id_for_lead(lead: SeedLead) -> str:
    """The split group of a lifecycle family: one seed lead."""
    return f"{SOURCE}:{lead.slug}"


def trace_id_for_walk(lead: SeedLead, walk: int) -> str:
    return f"{group_id_for_lead(lead)}:w{walk}"


def trace_seed_for(corpus_seed: int, group_id: str, variant_key: str) -> int:
    """Derive one trace's generator seed from the corpus seed."""
    key = f"{corpus_seed}|{group_id}|{variant_key}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


class _TraceBuilder:
    """Accumulates events with stage, score, and latency state.

    Mirrors the QualBench emitter's builder: every event snapshots
    the stage and score band as they stand after that event's own
    effects; latency is declared by draw, first event "none".
    """

    def __init__(self, channel: str, rng: random.Random):
        self.channel = channel
        self.rng = rng
        self.raw_events: list[dict] = []
        self.stage_state: str | None = None
        self.score_state: int | None = None

    def _draw_latency(self) -> str:
        if not self.raw_events:
            return "none"
        bands = LATENCY_DISTRIBUTIONS["general"]
        draw = self.rng.random()
        cumulative = 0.0
        for band, weight in bands:
            cumulative += weight
            if draw < cumulative:
                return band
        return bands[-1][0]

    def add(
        self,
        family: str,
        tool: str = "none",
        outcome_class: str = "none",
        channel: str | None = None,
    ) -> None:
        self.raw_events.append(
            {
                "family": family,
                "channel": self.channel if channel is None else channel,
                "score_band": score_to_band(self.score_state)
                if self.score_state is not None
                else "none",
                "latency_band": self._draw_latency(),
                "stage": self.stage_state or "none",
                "tool": tool,
                "outcome_class": outcome_class,
            }
        )

    def build_events(self) -> tuple[Event, ...]:
        return tuple(
            Event(seq=index, **raw)
            for index, raw in enumerate(self.raw_events)
        )


def _distances_to(
    transitions: dict[str, frozenset[str]], target: str
) -> dict[str, int]:
    """Shortest-path distances to target over the reversed graph."""
    reverse: dict[str, set[str]] = {stage: set() for stage in transitions}
    for stage, targets in transitions.items():
        for nxt in targets:
            reverse[nxt].add(stage)
    dist = {target: 0}
    queue: deque[str] = deque([target])
    while queue:
        current = queue.popleft()
        for prev in reverse[current]:
            if prev not in dist:
                dist[prev] = dist[current] + 1
                queue.append(prev)
    return dist


def legal_path(
    transitions: dict[str, frozenset[str]],
    start: str,
    target: str,
    rng: random.Random,
) -> list[str]:
    """A shortest legal path from start to target, as stage names
    after start (empty when start is target).

    Seeded choice among shortest continuations, preferring paths
    whose intermediate stages avoid BOOKED unless BOOKED is the
    target, so the history does not imply an unrecorded booking.
    """
    if start == target:
        return []
    dist = _distances_to(transitions, target)
    if start not in dist:
        raise SynthesiserError(
            f"stage {target} is not reachable from {start} in TRANSITIONS"
        )
    path: list[str] = []
    current = start
    while current != target:
        candidates = [
            nxt
            for nxt in transitions[current]
            if dist.get(nxt) == dist[current] - 1
        ]
        if target != "BOOKED":
            non_booked = [nxt for nxt in candidates if nxt != "BOOKED"]
            if non_booked:
                candidates = non_booked
        current = rng.choice(sorted(candidates))
        path.append(current)
    return path


def synthesise_walk(
    lp: _LeadPipe, lead: SeedLead, walk: int, corpus_seed: int
) -> Trace:
    """One lifecycle trace for a seed lead (see the module docstring)."""
    group_id = group_id_for_lead(lead)
    trace_seed = trace_seed_for(corpus_seed, group_id, f"w{walk}")
    rng = random.Random(trace_seed)
    builder = _TraceBuilder(lead.channel, rng)

    # History: capture, score, then the legal path to the seeded stage.
    builder.stage_state = "NEW"
    builder.add("enquiry", tool="capture_lead")
    if lead.score_total is not None:
        builder.score_state = lead.score_total
        builder.add("tool_call", tool="score_lead")
    for stage in legal_path(lp.transitions, "NEW", lead.stage, rng):
        builder.stage_state = stage
        builder.add("stage", tool="update_stage")
        if stage == "BOOKED":
            builder.add("booking", tool="book_appointment")

    outcome = _walk_future(lp, builder, lead, rng)
    builder.add("outcome", outcome_class=outcome)
    return Trace(
        trace_id=trace_id_for_walk(lead, walk),
        source=SOURCE,
        synthetic=True,
        vertical="general",
        agent="leadpipe-synthesiser",
        agent_version=SYNTHESISER_VERSION,
        events=builder.build_events(),
        outcome=outcome,
        generator=Generator(
            name=SYNTHESISER_NAME,
            version=SYNTHESISER_VERSION,
            seed=trace_seed,
        ),
    )


def _walk_future(
    lp: _LeadPipe, builder: _TraceBuilder, lead: SeedLead, rng: random.Random
) -> str:
    """The future segment: from the seeded stage to the outcome."""
    seeded = lead.stage
    if seeded == "WON":
        return "won"
    if seeded == "LOST":
        return "lost"
    pending = bool(lead.next_action)
    due_at_start = pending and (lead.next_action_in_days or 0) <= 0
    fired = False
    if due_at_start:
        builder.add("follow_up")
        fired = True
    band = score_to_band(lead.score_total) if lead.score_total is not None else "none"
    p_continue = CONTINUE_PROBABILITIES[band]
    current = seeded
    for _step in range(MAX_FUTURE_STEPS):
        nxt = rng.choice(sorted(lp.transitions[current]))
        builder.stage_state = nxt
        builder.add("stage", tool="update_stage")
        current = nxt
        if nxt == "BOOKED":
            builder.add("booking", tool="book_appointment")
        if pending and not fired and nxt not in ("WON", "LOST"):
            builder.add("follow_up")
            fired = True
        if nxt == "WON":
            return "won"
        if nxt == "LOST":
            return "lost"
        if rng.random() >= p_continue:
            # Contact ceases here.
            if current == "BOOKED":
                return "booked"
            if pending and not fired:
                return "stalled"
            return "dropped"
    return "booked" if current == "BOOKED" else "stalled"


def _anchor_outcome(lead: SeedLead) -> str:
    """Terminal label of a seeded lead's anchor trace, from its
    post-run state (stage, then pending next_action)."""
    if lead.stage == "WON":
        return "won"
    if lead.stage == "LOST":
        return "lost"
    if lead.stage == "BOOKED":
        return "booked"
    if lead.next_action:
        return "stalled"
    return "dropped"


def _seed_anchor_trace(
    lead: SeedLead,
    audit_entries: list[dict],
    corpus_seed: int,
    leadpipe_version: str,
) -> Trace:
    """One seeded lead's anchor trace, from its audit entries only."""
    trace_seed = trace_seed_for(corpus_seed, WALKTHROUGH_GROUP, lead.slug)
    builder = _TraceBuilder(lead.channel, random.Random(trace_seed))
    entries = [
        entry for entry in audit_entries if entry["lead_id"] == lead.lead_id
    ]
    if not entries:
        raise SynthesiserError(
            f"seed lead {lead.lead_id} has no audit entries to anchor on"
        )
    for entry in entries:
        detail = json.loads(entry["detail"])
        if entry["action"] == "capture_lead":
            if builder.stage_state is None:
                builder.stage_state = "NEW"
            builder.add("enquiry", tool="capture_lead")
        elif entry["action"] == "score_lead":
            builder.score_state = int(detail["new_total"])
            builder.add("tool_call", tool="score_lead")
        else:  # pragma: no cover, the seed writes only these two
            raise SynthesiserError(
                f"unexpected seeded audit action {entry['action']!r}"
            )
    outcome = _anchor_outcome(lead)
    builder.add("outcome", outcome_class=outcome)
    return Trace(
        trace_id=f"{WALKTHROUGH_GROUP}:{lead.slug}",
        source=SOURCE,
        synthetic=True,
        vertical="general",
        agent="demo-walkthrough",
        agent_version=leadpipe_version,
        events=builder.build_events(),
        outcome=outcome,
        generator=Generator(
            name=SYNTHESISER_NAME,
            version=SYNTHESISER_VERSION,
            seed=trace_seed,
        ),
    )


def _walkthrough_trace(
    lp: _LeadPipe, records: list[tuple[str, dict]], corpus_seed: int
) -> Trace:
    """The walkthrough lead's anchor trace, dispatch by dispatch.

    records is the ordered (tool, result) list of the nine steps.
    The mapping is positional because the sequence is fixed by the
    walkthrough script; the result checks done by the caller pin the
    behaviour each position must show.
    """
    trace_seed = trace_seed_for(corpus_seed, WALKTHROUGH_GROUP, "walkthrough-lead")
    builder = _TraceBuilder("website", random.Random(trace_seed))
    builder.stage_state = "NEW"
    for tool, result in records:
        if tool == "capture_lead" and result.get("created"):
            builder.add("enquiry", tool=tool)
        elif tool == "capture_lead" and result.get("deduped"):
            # The second capture arrived via instagram (the call's
            # own source), not the lead's first-contact channel.
            builder.add("enquiry", tool=tool, channel="instagram")
        elif tool == "score_lead":
            builder.score_state = int(result["score_total"])
            builder.add("tool_call", tool=tool)
        elif tool == "book_appointment" and result.get("created"):
            builder.add("booking", tool=tool)
        else:
            # list_hot_leads, check_availability, the approval
            # preview, and the idempotent re-book: invocations with
            # no new state of their own.
            builder.add("tool_call", tool=tool)
    builder.add("outcome", outcome_class="booked")
    return Trace(
        trace_id=f"{WALKTHROUGH_GROUP}:walkthrough-lead",
        source=SOURCE,
        synthetic=True,
        vertical="general",
        agent="demo-walkthrough",
        agent_version=lp.version,
        events=builder.build_events(),
        outcome="booked",
        generator=Generator(
            name=SYNTHESISER_NAME,
            version=SYNTHESISER_VERSION,
            seed=trace_seed,
        ),
    )


def run_walkthrough_anchors(root: Path, corpus_seed: int) -> list[Trace]:
    """Execute the demo walkthrough sequence and emit its family.

    The sequence is examples/demo_walkthrough.py step for step
    (seed, capture, dedup capture, score, hot list, availability,
    approval preview, confirmed booking, idempotent re-book, daily
    summary), against the fixed seed moment so the family is
    reproducible. Results are checked against the behaviour
    LeadPipe's own tests pin before any trace is built.
    """
    lp = load_leadpipe(root)
    client, seed = _seeded_client(lp)
    leads = load_seed_leads(root)
    seed_audit = client.list_audit()
    ctx = lp.tool_context_cls(
        store=client,
        calendar=client,
        config=lp.config_cls(demo=True, write_mode=lp.write_mode.APPROVAL),
    )
    records: list[tuple[str, dict]] = []

    def run(tool: str, **kwargs: Any) -> dict:
        result = lp.dispatch(ctx, tool, **kwargs)
        records.append((tool, result))
        return result

    created = run(
        "capture_lead",
        name="Walkthrough Wendy",
        email="wendy@wendysalon.example",
        phone="+1 415-555-0199",
        company="Wendy's Salon",
        source="website",
        service_interest="Booking automation",
        confirm=True,
    )
    if not created.get("created"):
        raise SynthesiserError(f"walkthrough capture did not create: {created}")
    lead_id = created["lead_id"]
    dupe = run(
        "capture_lead",
        name="Walkthrough Wendy",
        email="WENDY@wendysalon.example",
        phone="+14155550199",
        source="instagram",
        confirm=True,
    )
    if not dupe.get("deduped") or dupe.get("lead_id") != lead_id:
        raise SynthesiserError(f"walkthrough dedup changed shape: {dupe}")
    scored = run(
        "score_lead", lead_id=lead_id, scores=_WALKTHROUGH_SCORES, confirm=True
    )
    if scored.get("score_total") != 85 or scored.get("band") != "HOT":
        raise SynthesiserError(f"walkthrough score changed shape: {scored}")
    run("list_hot_leads", limit=5)
    day = seed["business_days"][0]
    avail = run(
        "check_availability",
        start=f"{day}T00:00:00Z",
        end=f"{day}T23:59:59Z",
        slot_minutes=30,
    )
    slot = next(s for s in avail["slots"] if s["start"] >= f"{day}T10:30")
    start = slot["start"]
    from datetime import timedelta

    end_dt = datetime.fromisoformat(start.replace("Z", "+00:00")) + timedelta(
        minutes=30
    )
    end = end_dt.isoformat().replace("+00:00", "Z")
    preview = run("book_appointment", lead_id=lead_id, start=start, end=end)
    if preview.get("ok") or preview.get("error", {}).get("code") != "APPROVAL_REQUIRED":
        raise SynthesiserError(f"walkthrough preview changed shape: {preview}")
    booked = run(
        "book_appointment", lead_id=lead_id, start=start, end=end, confirm=True
    )
    if not booked.get("created"):
        raise SynthesiserError(f"walkthrough booking did not create: {booked}")
    again = run(
        "book_appointment", lead_id=lead_id, start=start, end=end, confirm=True
    )
    if again.get("created") or again.get("appointment_id") != booked["appointment_id"]:
        raise SynthesiserError(f"walkthrough re-book changed shape: {again}")
    run("daily_pipeline_summary")

    traces = [
        _seed_anchor_trace(lead, seed_audit, corpus_seed, lp.version)
        for lead in leads
    ]
    traces.append(_walkthrough_trace(lp, records, corpus_seed))
    return traces
