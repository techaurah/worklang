"""Tokenizer round trip and error tests.

The matrix below cycles every attribute stream against every family,
so every bucket value in the vocabulary appears in encoded events.
The design's round trip requirement is exact equality at the event
level, and these tests assert it for the whole matrix, not a sample.
"""

import pytest

from worklang import errors
from worklang.schema import Event, Generator, Trace
from worklang.tokenizer import (
    decode_event,
    decode_trace,
    encode_event,
    encode_trace,
    score_to_band,
)
from worklang.vocab import (
    EVENT_STREAMS,
    STREAMS,
    TOKEN_TO_ID,
    token_at,
    token_id,
)

NON_NONE_OUTCOMES = ("booked", "won", "lost", "stalled", "dropped")


def matrix_events():
    """Events covering every family and every attribute bucket value."""
    events = []
    steps = max(len(STREAMS[stream]) for stream in EVENT_STREAMS[1:])
    for family in STREAMS["family"]:
        for i in range(steps):
            values = {"family": family}
            for stream in EVENT_STREAMS[1:]:
                bucket = STREAMS[stream]
                values[stream] = bucket[i % len(bucket)]
            if family == "outcome":
                values["outcome_class"] = NON_NONE_OUTCOMES[
                    i % len(NON_NONE_OUTCOMES)
                ]
            else:
                values["outcome_class"] = "none"
            events.append(Event(seq=0, **values))
    return events


def test_matrix_covers_every_family_and_bucket_value():
    events = matrix_events()
    assert {e.family for e in events} == set(STREAMS["family"])
    for stream in EVENT_STREAMS[1:]:
        assert {getattr(e, stream) for e in events} == set(STREAMS[stream])


def test_event_round_trip_exact_across_matrix():
    for event in matrix_events():
        ids = encode_event(event)
        assert len(ids) == 7
        assert decode_event(ids, seq=event.seq) == event


def test_event_token_positions_carry_the_right_streams():
    for event in matrix_events():
        ids = encode_event(event)
        assert [token_at(token)[0] for token in ids] == list(EVENT_STREAMS)
        assert [token_at(token)[1] for token in ids] == [
            getattr(event, stream) for stream in EVENT_STREAMS
        ]


def build_trace(vertical, outcome_class="stalled"):
    events = (
        Event(
            seq=0,
            family="enquiry",
            channel="instagram",
            score_band="none",
            latency_band="none",
            stage="NEW",
            tool="none",
            outcome_class="none",
        ),
        Event(
            seq=1,
            family="tool_call",
            channel="unknown",
            score_band="hot",
            latency_band="fast",
            stage="QUALIFIED",
            tool="score_lead",
            outcome_class="none",
        ),
        Event(
            seq=2,
            family="outcome",
            channel="unknown",
            score_band="hot",
            latency_band="overnight",
            stage="QUALIFIED",
            tool="none",
            outcome_class=outcome_class,
        ),
    )
    return Trace(
        trace_id=f"leadpipe-demo:{vertical}:walk-1",
        source="leadpipe-demo",
        synthetic=True,
        vertical=vertical,
        agent="lifecycle-synthesiser",
        agent_version="0.1",
        events=events,
        outcome=outcome_class,
        generator=Generator(name="transitions-walk", version="0.1", seed=7),
    )


def test_trace_round_trip_exact_for_every_vertical():
    for vertical in STREAMS["vertical"]:
        trace = build_trace(vertical)
        ids = encode_trace(trace)
        assert ids[0] == TOKEN_TO_ID["special:BOS"]
        assert ids[1] == token_id("vertical", vertical)
        assert ids[-1] == TOKEN_TO_ID["special:EOS"]
        assert len(ids) == 2 + 7 * len(trace.events) + 1
        decoded = decode_trace(ids)
        assert decoded.vertical == trace.vertical
        assert decoded.events == trace.events


def test_lossy_fields_never_reach_the_tokens():
    # A raw record carrying the design's named lossy fields decodes
    # back to schema fields only; nothing of the raw values survives.
    record = build_trace("dental").to_dict()
    record["events"][0]["message"] = "My tooth hurts, any openings?"
    record["events"][0]["contact_name"] = "A Person"
    record["events"][0]["timestamp"] = "2026-11-04T09:15:00"
    trace = Trace.from_dict(record)
    decoded = decode_trace(encode_trace(trace))
    assert decoded.events == trace.events
    for event in decoded.events:
        assert set(event.to_dict()) == {
            "seq",
            "family",
            "channel",
            "score_band",
            "latency_band",
            "stage",
            "tool",
            "outcome_class",
        }


def test_decode_event_rejects_wrong_stream_at_a_position():
    ids = list(encode_event(matrix_events()[0]))
    ids[1] = token_id("stage", "WON")  # channel position, stage token
    with pytest.raises(errors.TokenError, match="channel"):
        decode_event(ids)


def test_decode_event_rejects_wrong_length_and_unassigned_ids():
    ids = list(encode_event(matrix_events()[0]))
    with pytest.raises(errors.TokenError, match="exactly 7"):
        decode_event(ids[:6])
    ids[3] = 100  # reserved id space, not assigned
    with pytest.raises(errors.TokenError, match="not assigned"):
        decode_event(ids)


def test_decode_trace_rejects_bad_framing():
    ids = encode_trace(build_trace("agencies"))
    with pytest.raises(errors.TokenError, match="BOS"):
        decode_trace([token_id("special", "EOS"), *ids[1:]])
    with pytest.raises(errors.TokenError, match="vertical"):
        decode_trace([ids[0], token_id("family", "reply"), *ids[2:]])
    with pytest.raises(errors.TokenError, match="EOS"):
        decode_trace([*ids[:-1], token_id("special", "BOS")])
    with pytest.raises(errors.TokenError, match="multiple of 7"):
        decode_trace([*ids[:-2], ids[-1]])


def test_decode_trace_rejects_stream_without_outcome_event():
    trace = build_trace("general")
    ids = encode_trace(trace)
    body = ids[2:-1]
    no_outcome = [ids[0], ids[1], *body[:14], ids[-1]]
    with pytest.raises(errors.WorkLangError, match="outcome"):
        decode_trace(no_outcome)


def test_encode_rejects_non_event_and_non_trace():
    with pytest.raises(errors.TokenError, match="Event"):
        encode_event({"family": "enquiry"})
    with pytest.raises(errors.TokenError, match="Trace"):
        encode_trace("not a trace")


@pytest.mark.parametrize(
    ("score", "band"),
    [
        (100, "hot"),
        (80, "hot"),
        (79.9, "warm"),
        (65, "warm"),
        (64.9, "nurture"),
        (50, "nurture"),
        (49.9, "low"),
        (0, "low"),
        (None, "none"),
    ],
)
def test_score_to_band_thresholds(score, band):
    assert score_to_band(score) == band


@pytest.mark.parametrize("bad", [-1, 100.1, "80", True, float("nan")])
def test_score_to_band_rejects_out_of_range_and_non_numbers(bad):
    with pytest.raises(errors.TokenError):
        score_to_band(bad)
