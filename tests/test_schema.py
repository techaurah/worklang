"""Schema validation tests, including the documented lossy boundary."""

import pytest

from worklang import errors
from worklang.schema import (
    Event,
    Generator,
    Trace,
    trace_from_json,
    trace_to_json,
)


def make_event(seq=0, **overrides):
    fields = {
        "seq": seq,
        "family": "enquiry",
        "channel": "instagram",
        "score_band": "none",
        "latency_band": "none",
        "stage": "NEW",
        "tool": "none",
        "outcome_class": "none",
    }
    fields.update(overrides)
    return Event(**fields)


def make_trace(**overrides):
    fields = {
        "trace_id": "qualbench:med-spa/after-hours-booking:reference-agent",
        "source": "qualbench",
        "synthetic": True,
        "vertical": "med-spa",
        "agent": "reference-agent",
        "agent_version": "0.2.0",
        "events": (
            make_event(0),
            make_event(
                1,
                family="booking",
                channel="unknown",
                latency_band="instant",
                stage="BOOKED",
                tool="calendar.book",
            ),
            make_event(
                2, family="outcome", channel="unknown", outcome_class="booked"
            ),
        ),
        "outcome": "booked",
        "generator": Generator(name="qualbench-emitter", version="0.1", seed=42),
    }
    fields.update(overrides)
    return Trace(**fields)


def test_valid_trace_constructs_and_round_trips_through_dict():
    trace = make_trace()
    assert Trace.from_dict(trace.to_dict()) == trace


def test_valid_trace_round_trips_through_json():
    trace = make_trace()
    assert trace_from_json(trace_to_json(trace)) == trace


def test_all_six_verticals_accepted():
    for vertical in (
        "real-estate",
        "med-spa",
        "home-services",
        "dental",
        "agencies",
        "general",
    ):
        assert make_trace(vertical=vertical).vertical == vertical


def test_lossy_boundary_drops_non_schema_keys():
    # Raw-record fields the design names as dropped: message text,
    # names, contact details, exact timestamps. They must not survive
    # from_dict, and must not reappear in to_dict.
    record = make_trace().to_dict()
    record["notes"] = "internal only"
    record["events"][0]["message"] = "Hi, do you have Saturday free?"
    record["events"][0]["contact_name"] = "A Person"
    record["events"][0]["contact_email"] = "person@example.com"
    record["events"][0]["timestamp"] = "2026-11-04T21:47:00"
    trace = Trace.from_dict(record)
    out = trace.to_dict()
    assert "notes" not in out
    for event in out["events"]:
        assert set(event) == {
            "seq",
            "family",
            "channel",
            "score_band",
            "latency_band",
            "stage",
            "tool",
            "outcome_class",
        }


def test_missing_fields_raise():
    record = make_trace().to_dict()
    del record["events"][0]["channel"]
    with pytest.raises(errors.SchemaError, match="missing required fields"):
        Trace.from_dict(record)
    record = make_trace().to_dict()
    del record["trace_id"]
    with pytest.raises(errors.SchemaError, match="missing required fields"):
        Trace.from_dict(record)


def test_synthetic_is_mandatory_and_must_be_bool():
    record = make_trace().to_dict()
    del record["synthetic"]
    with pytest.raises(errors.SchemaError, match="missing required fields"):
        Trace.from_dict(record)
    with pytest.raises(errors.SchemaError, match="synthetic"):
        make_trace(synthetic="yes")


def test_unknown_vertical_raises():
    with pytest.raises(errors.SchemaError, match="vertical"):
        make_trace(vertical="space-tourism")


def test_outcome_must_duplicate_terminal_outcome_event():
    with pytest.raises(errors.SchemaError, match="duplicate"):
        make_trace(outcome="won")


def test_exactly_one_outcome_event_and_it_is_last():
    events = (
        make_event(0, family="outcome", outcome_class="booked"),
        make_event(1, family="reply"),
    )
    with pytest.raises(errors.SchemaError, match="last"):
        make_trace(events=events, outcome="booked")
    events = (make_event(0), make_event(1, family="reply"))
    with pytest.raises(errors.SchemaError, match="exactly one outcome"):
        make_trace(events=events, outcome="booked")


def test_outcome_class_none_rules():
    events = (
        make_event(0, outcome_class="booked"),
        make_event(1, family="outcome", outcome_class="booked"),
    )
    with pytest.raises(errors.SchemaError, match="only meaningful"):
        make_trace(events=events, outcome="booked")
    events = (make_event(0), make_event(1, family="outcome"))
    with pytest.raises(errors.SchemaError, match="real outcome_class"):
        make_trace(events=events, outcome="booked")


def test_seq_must_match_position():
    events = (
        make_event(0),
        make_event(5, family="reply"),
        make_event(2, family="outcome", outcome_class="dropped"),
    )
    with pytest.raises(errors.SchemaError, match="seq"):
        make_trace(events=events, outcome="dropped")


def test_malformed_traces_raise():
    with pytest.raises(errors.SchemaError, match="mapping"):
        Trace.from_dict(["not", "a", "mapping"])
    record = make_trace().to_dict()
    record["events"] = "not-a-list"
    with pytest.raises(errors.SchemaError, match="events"):
        Trace.from_dict(record)
    record = make_trace().to_dict()
    record["events"] = ["not-an-event"]
    with pytest.raises(errors.SchemaError, match="mapping"):
        Trace.from_dict(record)
    record = make_trace().to_dict()
    del record["generator"]["seed"]
    with pytest.raises(errors.SchemaError, match="generator"):
        Trace.from_dict(record)
    with pytest.raises(errors.SchemaError, match="JSON"):
        trace_from_json("{not json")


def test_event_level_invalid_values_raise():
    with pytest.raises(errors.SchemaError, match="family"):
        make_event(0, family="explosion")
    with pytest.raises(errors.SchemaError, match="score_band"):
        make_event(0, score_band="scorching")
    with pytest.raises(errors.SchemaError, match="seq"):
        make_event(-1)
    with pytest.raises(errors.SchemaError, match="seq"):
        make_event(True)
