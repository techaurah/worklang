"""QualBench emitter tests (build step 2).

Runs a small set of real scenarios, at least one per pack including
safety scenarios, through the unmodified QualBench runner with the
shipped agents, and asserts the emitted traces validate against the
schema (Trace construction itself validates), tokenize, and round
trip exactly. These are integration tests against the sibling
QualBench checkout; they skip only if that checkout is absent.
"""

import pytest

from worklang.qualbench_emitter import (
    AGENT_VERSIONS,
    EMITTER_NAME,
    LATENCY_DISTRIBUTIONS,
    default_qualbench_root,
    group_id_for,
    load_qualbench,
    run_and_emit,
    trace_id_for,
)
from worklang.schema import Trace
from worklang.tokenizer import decode_trace, encode_trace
from worklang.vocab import LATENCY_BANDS

ROOT = default_qualbench_root()
SEED = 20261104

pytestmark = pytest.mark.skipif(
    not (ROOT / "src" / "qualbench" / "runner.py").is_file(),
    reason="QualBench checkout not found next to the flagship workspace",
)

# One scenario per pack, plus extra safety and handoff coverage.
SCENARIOS = [
    "real-estate/standard-inquiry",  # core, books
    "med-spa/contraindication-handoff",  # safety, hands off
    "home-services/gas-safety-handoff",  # safety, hands off
    "dental/hygiene-booking",  # core, books
    "agencies/discovery-call-booking",  # core, books
    "agencies/nda-legal-handoff",  # edge, hands off
]


@pytest.fixture(scope="module")
def scenarios():
    qb = load_qualbench(ROOT)
    found = {}
    for pack in qb.discover_packs(ROOT / "packs"):
        for scenario in pack.scenarios:
            if scenario.scenario_id in SCENARIOS:
                found[scenario.scenario_id] = scenario
    assert sorted(found) == sorted(SCENARIOS)
    return found


@pytest.fixture(scope="module")
def reference_traces(scenarios):
    return {
        scenario_id: run_and_emit(
            ROOT, scenario, "reference-agent", variant=0, corpus_seed=SEED
        )
        for scenario_id, scenario in scenarios.items()
    }


def test_every_emitted_trace_validates_and_tokenizes(reference_traces):
    for trace in reference_traces.values():
        assert isinstance(trace, Trace)  # construction is validation
        ids = encode_trace(trace)
        decoded = decode_trace(ids)
        assert decoded.vertical == trace.vertical
        assert decoded.events == trace.events
        assert len(ids) == 3 + 7 * len(trace.events)


def test_trace_metadata_follows_the_design(reference_traces):
    for scenario_id, trace in reference_traces.items():
        assert trace.source == "qualbench"
        assert trace.synthetic is True
        assert trace.agent == "reference-agent"
        assert trace.agent_version == AGENT_VERSIONS["reference-agent"]
        assert trace.generator.name == EMITTER_NAME
        assert trace.trace_id == trace_id_for(
            trace_id_scenario(scenario_id), "reference-agent", 0
        )
        assert trace.events[-1].family == "outcome"
        assert trace.events[-1].outcome_class == trace.outcome
        assert all(
            event.outcome_class == "none" for event in trace.events[:-1]
        )


def trace_id_scenario(scenario_id):
    class _Stub:
        pass

    stub = _Stub()
    stub.pack_name, stub.name = scenario_id.split("/", 1)
    stub.scenario_id = scenario_id
    return stub


def test_channel_is_never_invented(reference_traces):
    # QualBench scenarios declare no channel; the design says the
    # emitter defaults to unknown rather than inventing one.
    for trace in reference_traces.values():
        assert {event.channel for event in trace.events} == {"unknown"}


def test_latency_is_declared_from_the_vertical_distribution(reference_traces):
    for trace in reference_traces.values():
        allowed = {
            band for band, weight in LATENCY_DISTRIBUTIONS[trace.vertical]
        }
        assert trace.events[0].latency_band == "none"
        for event in trace.events[1:]:
            assert event.latency_band in allowed
            assert event.latency_band in LATENCY_BANDS


def test_group_id_keeps_variants_of_one_scenario_together(scenarios):
    scenario = scenarios["real-estate/standard-inquiry"]
    ids = {
        run_and_emit(
            ROOT, scenario, agent, variant=variant, corpus_seed=SEED
        ).trace_id
        for agent in ("reference-agent", "naive-agent")
        for variant in (0, 1)
    }
    groups = {trace_id.rsplit(":", 2)[0] for trace_id in ids}
    assert groups == {group_id_for(scenario)}


def test_booking_scenario_books_for_real(reference_traces):
    trace = reference_traces["real-estate/standard-inquiry"]
    families = [event.family for event in trace.events]
    assert trace.outcome == "booked"
    assert "booking" in families
    tools = [event.tool for event in trace.events if event.family == "tool_call"]
    assert "calendar.book" in tools
    assert "crm.upsert_lead" in tools


def test_safety_scenarios_hand_off_and_stall(reference_traces):
    for scenario_id in (
        "med-spa/contraindication-handoff",
        "home-services/gas-safety-handoff",
    ):
        trace = reference_traces[scenario_id]
        families = [event.family for event in trace.events]
        assert "handoff" in families
        assert trace.outcome == "stalled"


def test_naive_agent_traces_validate_and_differ(scenarios):
    scenario = scenarios["med-spa/contraindication-handoff"]
    trace = run_and_emit(
        ROOT, scenario, "naive-agent", variant=0, corpus_seed=SEED
    )
    assert trace.agent == "naive-agent"
    decoded = decode_trace(encode_trace(trace))
    assert decoded.events == trace.events
    # The naive agent never hands off, even on a safety scenario; the
    # trace must show that honestly rather than borrowing the
    # scenario's expected outcome.
    assert "handoff" not in [event.family for event in trace.events]


def test_unqualified_close_maps_to_lost_outcome(scenarios):
    qb = load_qualbench(ROOT)
    pack = qb.load_pack(ROOT / "packs" / "real-estate")
    scenario = next(s for s in pack.scenarios if s.name == "out-of-area")
    trace = run_and_emit(
        ROOT, scenario, "reference-agent", variant=0, corpus_seed=SEED
    )
    assert trace.outcome == "lost"
    # UNQUALIFIED has no stage token in schema v0.1; it must not leak
    # into the stage stream under any spelling.
    assert {event.stage for event in trace.events} <= {
        "NEW",
        "CONTACTED",
        "QUALIFIED",
        "NURTURE",
        "DISCOVERY",
        "BOOKED",
        "PROPOSAL",
        "NEGOTIATION",
        "WON",
        "LOST",
        "none",
    }
