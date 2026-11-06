"""Vocabulary enumeration tests.

The design fixes the vocabulary by enumerating bucket sets, and its
size calculation is 4 + 6 + 8 + 9 + 5 + 6 + 11 + 14 + 6 = 69. These
tests rebuild the vocabulary from the source constants and assert it
matches the design, so a silent bucket addition or removal fails CI.
"""

import json
from pathlib import Path

import pytest

from worklang import errors
from worklang.vocab import (
    ID_TO_TOKEN,
    RESERVED_VOCAB_SIZE,
    STREAMS,
    TOKEN_TO_ID,
    VOCAB_SIZE,
    token_at,
    token_id,
)

DESIGN_STREAM_COUNTS = {
    "special": 4,
    "vertical": 6,
    "family": 8,
    "channel": 9,
    "score_band": 5,
    "latency_band": 6,
    "stage": 11,
    "tool": 14,
    "outcome_class": 6,
}

VOCAB_JSON = Path(__file__).resolve().parents[1] / "src" / "worklang" / "vocab.json"


def test_stream_counts_match_design():
    assert {name: len(values) for name, values in STREAMS.items()} == (
        DESIGN_STREAM_COUNTS
    )


def test_vocab_size_is_69_by_enumeration():
    rebuilt = sum(len(values) for values in STREAMS.values())
    assert rebuilt == 69
    assert VOCAB_SIZE == 69
    assert len(TOKEN_TO_ID) == 69
    assert len(ID_TO_TOKEN) == 69


def test_vocab_rebuilt_from_source_constants_matches():
    rebuilt_to_id = {}
    rebuilt_id_to = {}
    next_id = 0
    for stream, values in STREAMS.items():
        for value in values:
            rebuilt_to_id[f"{stream}:{value}"] = next_id
            rebuilt_id_to[next_id] = (stream, value)
            next_id += 1
    assert rebuilt_to_id == TOKEN_TO_ID
    assert rebuilt_id_to == ID_TO_TOKEN


def test_ids_are_contiguous_from_zero():
    assert sorted(ID_TO_TOKEN) == list(range(69))


def test_reserved_id_space_up_to_128():
    assert RESERVED_VOCAB_SIZE == 128
    assert VOCAB_SIZE < RESERVED_VOCAB_SIZE


def test_special_token_ids():
    assert token_id("special", "PAD") == 0
    assert token_id("special", "BOS") == 1
    assert token_id("special", "EOS") == 2
    assert token_id("special", "UNK") == 3


def test_stream_boundary_ids():
    # 4 specials + 6 verticals put the first family token at id 10,
    # and the last token of the enumeration is outcome_class:none.
    assert token_id("vertical", "real-estate") == 4
    assert token_id("family", "enquiry") == 10
    assert token_id("outcome_class", "none") == 68
    assert token_at(68) == ("outcome_class", "none")


def test_id_lookup_round_trips_for_every_token():
    for token in range(VOCAB_SIZE):
        stream, value = token_at(token)
        assert token_id(stream, value) == token


def test_unassigned_and_out_of_range_ids_raise():
    for bad in (69, 100, 127, 128, -1):
        with pytest.raises(errors.TokenError):
            token_at(bad)


def test_unknown_stream_and_value_raise_clear_errors():
    with pytest.raises(errors.TokenError, match="unknown stream"):
        token_id("mood", "happy")
    with pytest.raises(errors.TokenError, match="bucket set"):
        token_id("channel", "smoke-signal")


def test_tool_stream_keeps_source_faithful_names_unmerged():
    # The design forbids silently merging near-duplicate tool names.
    for name in (
        "calendar.book",
        "book_appointment",
        "crm.get_lead",
        "get_lead",
        "calendar.check_availability",
        "check_availability",
        "crm.upsert_lead",
        "capture_lead",
        "report_outcome",
    ):
        assert name in STREAMS["tool"]
    assert len(STREAMS["tool"]) == 14


def test_vocab_json_artefact_matches_code():
    data = json.loads(VOCAB_JSON.read_text(encoding="utf-8"))
    assert data["vocab_size"] == 69
    assert data["reserved_vocab_size"] == 128
    assert data["token_to_id"] == TOKEN_TO_ID
    assert data["streams"] == {
        name: list(values) for name, values in STREAMS.items()
    }
