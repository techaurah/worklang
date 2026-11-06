"""Tests for corpus source B: build, split, card, determinism.

Mirrors the source A test standard: every generated trace validates
and tokenizes with an exact round trip, the split is group-aware
with no group in two splits, and the whole build (traces, splits,
card, written bytes) is deterministic under the corpus seed.
"""

from pathlib import Path

import pytest

from worklang.corpus_a import (
    DEFAULT_CORPUS_SEED,
    SPLIT_NAMES,
    SYNTHETIC_SENTENCE,
    load_corpus_file,
)
from worklang.corpus_b import (
    DEFAULT_WALKS_PER_LEAD,
    build_corpus_b,
    corpus_card,
    group_of,
    split_traces,
    write_corpus,
)
from worklang.schema import Trace, trace_from_json, trace_to_json
from worklang.tokenizer import decode_trace, encode_trace

CORPUS_DIR = Path(__file__).resolve().parents[1] / "corpus" / "source-b"


@pytest.fixture(scope="module")
def traces():
    return build_corpus_b()


@pytest.fixture(scope="module")
def splits(traces):
    return split_traces(traces)


def test_corpus_shape(traces):
    assert len(traces) == 12 * DEFAULT_WALKS_PER_LEAD + 13
    assert {trace.source for trace in traces} == {"leadpipe-demo"}
    assert {trace.vertical for trace in traces} == {"general"}
    assert all(trace.synthetic for trace in traces)
    ids = [trace.trace_id for trace in traces]
    assert len(ids) == len(set(ids))


def test_every_trace_validates_and_tokenizes(traces):
    for trace in traces:
        # Construction validates: a dict round trip re-runs every
        # schema check, and a JSON round trip must be exact.
        assert Trace.from_dict(trace.to_dict()) == trace
        assert trace_from_json(trace_to_json(trace)) == trace
        decoded = decode_trace(encode_trace(trace))
        assert decoded.events == trace.events
        assert decoded.vertical == trace.vertical


def test_first_event_latency_is_none(traces):
    for trace in traces:
        assert trace.events[0].latency_band == "none"


def test_split_is_group_aware(splits, traces):
    groups = {
        name: {group_of(trace) for trace in splits[name]}
        for name in SPLIT_NAMES
    }
    assert groups["train"].isdisjoint(groups["validation"])
    assert groups["train"].isdisjoint(groups["test"])
    assert groups["validation"].isdisjoint(groups["test"])
    # 13 groups in total: 12 seed leads plus the walkthrough family.
    assert len(groups["train"] | groups["validation"] | groups["test"]) == 13
    merged_ids = sorted(
        trace.trace_id for name in SPLIT_NAMES for trace in splits[name]
    )
    assert merged_ids == sorted(trace.trace_id for trace in traces)


def test_group_of_shapes():
    traces = build_corpus_b()
    walk = next(t for t in traces if t.agent == "leadpipe-synthesiser")
    anchor = next(t for t in traces if t.agent == "demo-walkthrough")
    assert group_of(walk) == ":".join(walk.trace_id.split(":")[:2])
    assert group_of(anchor) == "leadpipe-demo:walkthrough"


def test_build_is_deterministic(traces, splits):
    rebuilt = build_corpus_b()
    assert [t.to_dict() for t in rebuilt] == [t.to_dict() for t in traces]
    rebuilt_splits = split_traces(rebuilt)
    for name in SPLIT_NAMES:
        assert [t.trace_id for t in rebuilt_splits[name]] == [
            t.trace_id for t in splits[name]
        ]


def test_card_is_deterministic_and_counts_are_real(traces, splits):
    card = corpus_card(traces, splits)
    rebuilt = corpus_card(build_corpus_b(), split_traces(build_corpus_b()))
    assert card == rebuilt
    assert SYNTHETIC_SENTENCE in card
    assert f"- Traces: {len(traces)}" in card
    assert f"- Split seed: {DEFAULT_CORPUS_SEED}" in card
    assert "generator-declared" in card
    # The recorded channel conflict is on the card, not smoothed over.
    assert "walk_in" in card and '"unknown"' in card


def test_written_corpus_matches_build(tmp_path, traces, splits):
    card = corpus_card(traces, splits)
    paths = write_corpus(tmp_path, traces, splits, card)
    assert load_corpus_file(paths["corpus"]) == traces
    for name in SPLIT_NAMES:
        assert load_corpus_file(paths[name]) == splits[name]
    assert paths["card"].read_text(encoding="utf-8") == card


def test_committed_corpus_matches_build(traces, splits):
    committed = load_corpus_file(CORPUS_DIR / "corpus-b.jsonl")
    assert committed == traces
    for name in SPLIT_NAMES:
        committed_split = load_corpus_file(
            CORPUS_DIR / f"corpus-b-{name}.jsonl"
        )
        assert committed_split == splits[name]
    committed_card = (CORPUS_DIR / "CORPUS_CARD.md").read_text(
        encoding="utf-8"
    )
    assert committed_card == corpus_card(traces, splits)
