"""Tests for the two v0 baselines, on tiny hand-checkable fixtures.

Every expectation below is counted by hand from the fixture events in
the test itself. The fixtures vary the family stream and, in one
place, channel and tool; every other stream is held at its default so
the composite symbols stay readable.
"""

import pytest

from worklang.baselines import MajorityBaseline, NGramBaseline, event_symbol
from worklang.errors import BaselineError
from worklang.schema import Event, Generator, Trace

GENERATOR = Generator(name="fixture", version="0", seed=1)


def make_event(seq: int, family: str, **overrides) -> Event:
    values = {
        "seq": seq,
        "family": family,
        "channel": "unknown",
        "score_band": "none",
        "latency_band": "none",
        "stage": "none",
        "tool": "none",
        "outcome_class": "booked" if family == "outcome" else "none",
    }
    values.update(overrides)
    return Event(**values)


def make_trace(trace_id: str, events: list[Event]) -> Trace:
    return Trace(
        trace_id=trace_id,
        source="qualbench",
        synthetic=True,
        vertical="general",
        agent="fixture-agent",
        agent_version="0",
        events=tuple(events),
        outcome="booked",
        generator=GENERATOR,
    )


def family_trace(trace_id: str, families: list[str]) -> Trace:
    return make_trace(
        trace_id, [make_event(index, family) for index, family in enumerate(families)]
    )


# Majority fixture, counted by hand:
#   families: enquiry 3, reply 5, tool_call 1, outcome 3 -> reply
#   channels: unknown 11, website 1 -> unknown
#   tools: none 11, crm.get_lead 1 -> none
#   outcome_class: none 9, booked 3 -> none; every other stream is
#   uniformly none across all 12 events.
MAJORITY_PREDICTION = {
    "family": "reply",
    "channel": "unknown",
    "score_band": "none",
    "latency_band": "none",
    "stage": "none",
    "tool": "none",
    "outcome_class": "none",
}


def majority_fixture() -> list[Trace]:
    return [
        family_trace("m1", ["enquiry", "reply", "reply", "outcome"]),
        make_trace(
            "m2",
            [
                make_event(0, "enquiry"),
                make_event(1, "reply"),
                make_event(2, "tool_call", channel="website", tool="crm.get_lead"),
                make_event(3, "outcome"),
            ],
        ),
        family_trace("m3", ["enquiry", "reply", "reply", "outcome"]),
    ]


def test_majority_picks_the_known_modes():
    baseline = MajorityBaseline().fit(majority_fixture())
    assert baseline.predict(()) == MAJORITY_PREDICTION


def test_majority_ignores_the_prefix():
    baseline = MajorityBaseline().fit(majority_fixture())
    prefix = family_trace("p", ["enquiry", "booking", "outcome"]).events[:2]
    assert baseline.predict(prefix) == MAJORITY_PREDICTION


# N-gram fixture, counted by hand. Symbols are named by family (all
# other streams are the shared defaults):
#   a: P Q R T      b: P Q R T      c: P Q S T
# P = enquiry, Q = reply, R = tool_call, S = booking, T = outcome.
# Unigram counts: P 3, Q 3, R 2, S 1, T 3.
# Continuations of (P, Q): R twice, S once. Continuations of Q: same.
def ngram_fixture() -> list[Trace]:
    return [
        family_trace("a", ["enquiry", "reply", "tool_call", "outcome"]),
        family_trace("b", ["enquiry", "reply", "tool_call", "outcome"]),
        family_trace("c", ["enquiry", "reply", "booking", "outcome"]),
    ]


def _events(families: list[str]) -> tuple[Event, ...]:
    return family_trace("prefix", [*families, "outcome"]).events[: len(families)]


def test_ngram_order2_uses_the_two_symbol_context():
    baseline = NGramBaseline().fit(ngram_fixture())
    predicted = baseline.predict(_events(["enquiry", "reply"]))
    # (enquiry, reply) continues to tool_call twice, booking once.
    assert predicted["family"] == "tool_call"
    assert predicted == {
        "family": "tool_call",
        "channel": "unknown",
        "score_band": "none",
        "latency_band": "none",
        "stage": "none",
        "tool": "none",
        "outcome_class": "none",
    }


def test_ngram_reproduces_a_memorised_sequence():
    baseline = NGramBaseline().fit(ngram_fixture())
    predicted = baseline.predict(_events(["enquiry", "reply", "tool_call"]))
    assert predicted["family"] == "outcome"
    assert predicted["outcome_class"] == "booked"


def test_ngram_backs_off_to_order1():
    baseline = NGramBaseline().fit(ngram_fixture())
    # handoff never appears in training, so the order-2 context
    # (handoff, reply) is unseen; the order-1 context (reply) gives
    # tool_call twice and booking once. The answer cannot be the
    # unigram mode: tool_call's unigram count is 2, below the 3 of
    # enquiry, reply, and outcome, so only the backoff path yields it.
    predicted = baseline.predict(_events(["handoff", "reply"]))
    assert predicted["family"] == "tool_call"


def test_ngram_backs_off_to_unigram():
    baseline = NGramBaseline().fit(ngram_fixture())
    # A lone unseen symbol has no order-1 continuations either, so the
    # unigram distribution answers. Its tie (enquiry, reply, outcome
    # at 3 each) resolves to enquiry by vocabulary order.
    assert baseline.predict(_events(["handoff"]))["family"] == "enquiry"
    assert baseline.predict(())["family"] == "enquiry"


def test_event_symbol_drops_seq():
    first = make_event(0, "reply")
    later = make_event(5, "reply")
    assert event_symbol(first) == event_symbol(later)


def test_fit_on_no_traces_raises():
    with pytest.raises(BaselineError):
        MajorityBaseline().fit([])
    with pytest.raises(BaselineError):
        NGramBaseline().fit([])


def test_predict_before_fit_raises():
    with pytest.raises(BaselineError):
        MajorityBaseline().predict(())
    with pytest.raises(BaselineError):
        NGramBaseline().predict(())
