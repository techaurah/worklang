"""Tests for the LeadPipe lifecycle synthesiser (corpus source B).

The core guarantee is legality: every stage step in every generated
trace is a legal LeadPipe TRANSITIONS step, by construction, and the
events stay consistent with the seed lead they came from (channel,
band, next_action). The walkthrough anchor family is checked against
the dispatch sequence and outcomes the walkthrough itself produces.
"""

import pytest

from worklang.leadpipe_synthesiser import (
    WALKTHROUGH_GROUP,
    SeedLead,
    channel_for_source,
    default_leadpipe_root,
    load_leadpipe,
    load_seed_leads,
    run_walkthrough_anchors,
    synthesise_walk,
)
from worklang.schema import Trace
from worklang.tokenizer import score_to_band

ROOT = default_leadpipe_root()
ALL_SOURCES = {
    "website",
    "instagram",
    "facebook",
    "whatsapp",
    "email",
    "referral",
    "call",
    "walk_in",
    "other",
}


@pytest.fixture(scope="module")
def lp():
    return load_leadpipe(ROOT)


@pytest.fixture(scope="module")
def leads() -> list[SeedLead]:
    return load_seed_leads(ROOT)


def _walk(lp, lead: SeedLead, walk: int = 0) -> Trace:
    return synthesise_walk(lp, lead, walk, 20261104)


def _stage_sequence(trace: Trace) -> list[str]:
    """The trace's stage path: event stage values, deduplicated."""
    stages: list[str] = []
    for event in trace.events:
        if not stages or stages[-1] != event.stage:
            stages.append(event.stage)
    return stages


def test_seed_world_shape(leads):
    assert len(leads) == 12
    # The seed covers every stage in the TRANSITIONS map exactly.
    assert {lead.stage for lead in leads} == {
        "NEW",
        "CONTACTED",
        "QUALIFIED",
        "DISCOVERY",
        "BOOKED",
        "PROPOSAL",
        "NEGOTIATION",
        "WON",
        "LOST",
        "NURTURE",
    }
    assert {lead.source for lead in leads} <= ALL_SOURCES
    # The two unbucketed sources are really in the seed (the recorded
    # channel conflict), so the degradation path is exercised.
    assert {"walk_in", "other"} <= {lead.source for lead in leads}


def test_channel_mapping():
    for source in ALL_SOURCES - {"walk_in", "other"}:
        assert channel_for_source(source) == source
    assert channel_for_source("walk_in") == "unknown"
    assert channel_for_source("other") == "unknown"


def test_every_stage_step_is_a_legal_transition(lp, leads):
    for lead in leads:
        for walk in range(3):
            trace = _walk(lp, lead, walk)
            stages = _stage_sequence(trace)
            assert stages[0] == "NEW"
            for before, after in zip(stages, stages[1:], strict=False):
                assert after in lp.transitions[before], (
                    f"{trace.trace_id}: {before} -> {after} is not a "
                    "legal TRANSITIONS step"
                )
            # The seeded stage is on the path: history reaches it.
            assert lead.stage in stages


def test_booking_events_sit_at_booked_stage(lp, leads):
    for lead in leads:
        trace = _walk(lp, lead)
        for event in trace.events:
            if event.family == "booking":
                assert event.stage == "BOOKED"
                assert event.tool == "book_appointment"


def test_events_match_lead_channel_and_band(lp, leads):
    for lead in leads:
        trace = _walk(lp, lead)
        assert {event.channel for event in trace.events} == {lead.channel}
        scored = [
            event
            for event in trace.events
            if event.family == "tool_call" and event.tool == "score_lead"
        ]
        if lead.score_total is None:
            assert scored == []
            assert {event.score_band for event in trace.events} == {"none"}
        else:
            assert len(scored) == 1
            assert scored[0].score_band == score_to_band(lead.score_total)
            # Events before the score carry no band.
            first_score = trace.events.index(scored[0])
            assert {
                event.score_band for event in trace.events[:first_score]
            } == {"none"}


def test_follow_up_fires_once_for_due_action_leads(lp, leads):
    due = [
        lead
        for lead in leads
        if lead.next_action
        and (lead.next_action_in_days or 0) <= 0
        and lead.stage not in ("WON", "LOST")
    ]
    assert due  # the seed has due-action leads (Tom Becker, in 0 days)
    for lead in due:
        for walk in range(3):
            trace = _walk(lp, lead, walk)
            follow_ups = [
                event for event in trace.events if event.family == "follow_up"
            ]
            assert len(follow_ups) == 1


def test_terminal_seeded_leads_have_no_future(lp, leads):
    for lead in leads:
        if lead.stage not in ("WON", "LOST"):
            continue
        trace = _walk(lp, lead)
        assert trace.outcome == lead.stage.lower()
        assert not [
            event for event in trace.events if event.family == "follow_up"
        ]
        # History reaches the terminal stage and stops there.
        assert _stage_sequence(trace)[-1] == lead.stage


def test_trace_identity_and_determinism(lp, leads):
    lead = leads[0]
    first = _walk(lp, lead, 2)
    second = _walk(lp, lead, 2)
    assert first == second
    assert first.trace_id == f"leadpipe-demo:{lead.slug}:w2"
    assert first.source == "leadpipe-demo"
    assert first.synthetic is True
    assert first.vertical == "general"
    assert first.generator.name == "worklang-leadpipe-synthesiser"


@pytest.fixture(scope="module")
def anchors() -> list[Trace]:
    return run_walkthrough_anchors(ROOT, 20261104)


def test_anchor_family_shape(anchors, leads):
    assert len(anchors) == len(leads) + 1
    for trace in anchors:
        assert trace.agent == "demo-walkthrough"
        assert trace.trace_id.startswith(WALKTHROUGH_GROUP + ":")


def test_anchor_walkthrough_lead_dispatch_sequence(anchors):
    wendy = next(
        t for t in anchors if t.trace_id.endswith(":walkthrough-lead")
    )
    assert [event.family for event in wendy.events] == [
        "enquiry",
        "enquiry",
        "tool_call",
        "tool_call",
        "tool_call",
        "tool_call",
        "booking",
        "tool_call",
        "tool_call",
        "outcome",
    ]
    assert [event.tool for event in wendy.events] == [
        "capture_lead",
        "capture_lead",
        "score_lead",
        "list_hot_leads",
        "check_availability",
        "book_appointment",
        "book_appointment",
        "book_appointment",
        "daily_pipeline_summary",
        "none",
    ]
    # The deduplicated capture arrived by instagram; the rest of the
    # trace rides the lead's first-contact channel.
    assert wendy.events[1].channel == "instagram"
    assert {event.channel for event in wendy.events} == {
        "website",
        "instagram",
    }
    # Scored 85 by the walkthrough: HOT from the score event on.
    assert wendy.events[2].score_band == "hot"
    assert wendy.outcome == "booked"


def test_anchor_seeded_leads_use_audit_only(anchors, leads):
    by_id = {trace.trace_id: trace for trace in anchors}
    for lead in leads:
        trace = by_id[f"{WALKTHROUGH_GROUP}:{lead.slug}"]
        # No stage transitions are recorded in the seed audit log, so
        # anchor traces never carry a stage event.
        assert not [
            event for event in trace.events if event.family == "stage"
        ]
        assert {event.channel for event in trace.events} == {lead.channel}
        scored = [
            event
            for event in trace.events
            if event.family == "tool_call" and event.tool == "score_lead"
        ]
        assert len(scored) == (0 if lead.score_total is None else 1)
        if lead.stage == "WON":
            assert trace.outcome == "won"
        elif lead.stage == "LOST":
            assert trace.outcome == "lost"
        elif lead.stage == "BOOKED":
            assert trace.outcome == "booked"
        elif lead.next_action:
            assert trace.outcome == "stalled"
        else:
            assert trace.outcome == "dropped"


def test_anchor_dedup_capture_present(anchors, leads):
    tom = next(lead for lead in leads if lead.next_action_in_days == 0)
    trace = next(
        t for t in anchors if t.trace_id == f"{WALKTHROUGH_GROUP}:{tom.slug}"
    )
    # capture, score, then the seed's one deduplicated capture.
    assert [event.family for event in trace.events] == [
        "enquiry",
        "tool_call",
        "enquiry",
        "outcome",
    ]
