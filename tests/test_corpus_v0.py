"""Tests for the merged corpus v0.

The merge invariants are the point: group-aware splitting survives
the merge across BOTH sources (no group id in more than one merged
split), every trace of both full corpora appears exactly once, and
the combined card's numbers are the counted numbers of the merged
traces, deterministically.
"""

from pathlib import Path

import pytest

from worklang.corpus_a import SPLIT_NAMES, SYNTHETIC_SENTENCE, load_corpus_file
from worklang.corpus_v0 import (
    MergeError,
    build_merged,
    corpus_card,
    group_of,
    load_source_splits,
    merge_splits,
)

CORPUS_ROOT = Path(__file__).resolve().parents[1] / "corpus"


@pytest.fixture(scope="module")
def merged():
    splits, _all = build_merged(CORPUS_ROOT)
    return splits


def test_no_group_straddles_across_both_sources(merged):
    groups = {
        name: {group_of(trace) for trace in merged[name]}
        for name in SPLIT_NAMES
    }
    assert groups["train"].isdisjoint(groups["validation"])
    assert groups["train"].isdisjoint(groups["test"])
    assert groups["validation"].isdisjoint(groups["test"])
    # Group ids are namespaced by source.
    assert all(
        group.startswith(("qualbench:", "leadpipe-demo:"))
        for name in SPLIT_NAMES
        for group in groups[name]
    )


def test_merge_covers_both_corpora_exactly_once(merged):
    _a_splits, a_full = load_source_splits(CORPUS_ROOT / "source-a", "corpus-a")
    _b_splits, b_full = load_source_splits(CORPUS_ROOT / "source-b", "corpus-b")
    merged_ids = [
        trace.trace_id for name in SPLIT_NAMES for trace in merged[name]
    ]
    assert len(merged_ids) == len(set(merged_ids))
    assert set(merged_ids) == {t.trace_id for t in a_full} | {
        t.trace_id for t in b_full
    }
    assert {trace.source for name in SPLIT_NAMES for trace in merged[name]} == {
        "qualbench",
        "leadpipe-demo",
    }


def test_committed_v0_matches_merge(merged):
    for name in SPLIT_NAMES:
        committed = load_corpus_file(CORPUS_ROOT / "v0" / f"corpus-v0-{name}.jsonl")
        assert [t.trace_id for t in committed] == [
            t.trace_id for t in merged[name]
        ]
    committed_card = (CORPUS_ROOT / "v0" / "CORPUS_CARD.md").read_text(
        encoding="utf-8"
    )
    assert committed_card == corpus_card(merged)


def test_card_numbers_are_the_counted_numbers(merged):
    card = corpus_card(merged)
    total = sum(len(merged[name]) for name in SPLIT_NAMES)
    assert SYNTHETIC_SENTENCE in card
    assert f"- Traces: {total}" in card
    for name in SPLIT_NAMES:
        assert f"- {name}: {len(merged[name])} traces" in card
    # Determinism: the same merge produces the identical card.
    again, _all = build_merged(CORPUS_ROOT)
    assert corpus_card(again) == card


def test_merge_rejects_a_straddling_split():
    a_splits, a_full = load_source_splits(CORPUS_ROOT / "source-a", "corpus-a")
    b_splits, b_full = load_source_splits(CORPUS_ROOT / "source-b", "corpus-b")
    moved = {name: list(rows) for name, rows in b_splits.items()}
    moved["test"] = moved["test"] + [moved["train"][0]]
    moved["train"] = moved["train"][1:]
    with pytest.raises(MergeError):
        merge_splits(a_splits, moved, a_full, b_full)


def test_merge_rejects_a_duplicated_trace():
    a_splits, a_full = load_source_splits(CORPUS_ROOT / "source-a", "corpus-a")
    b_splits, b_full = load_source_splits(CORPUS_ROOT / "source-b", "corpus-b")
    dup = {name: list(rows) for name, rows in b_splits.items()}
    dup["test"] = dup["test"] + [dup["test"][0]]
    with pytest.raises(MergeError):
        merge_splits(a_splits, dup, a_full, b_full)
