"""Corpus source A tests: split integrity and determinism (step 2).

Uses one pack end to end so the suite stays fast: the group-aware
split must keep every scenario's traces (both agents, all variants)
in exactly one split, and a rebuild with the same seed must produce
the same corpus card numbers and the same corpus bytes.
"""

import random

import pytest

from worklang import corpus_a
from worklang.corpus_a import (
    SYNTHETIC_SENTENCE,
    corpus_card,
    group_of,
    perturb_scenario,
    split_traces,
)
from worklang.qualbench_emitter import (
    default_qualbench_root,
    load_qualbench,
    run_and_emit,
    trace_seed_for,
)

ROOT = default_qualbench_root()
SEED = 99

pytestmark = pytest.mark.skipif(
    not (ROOT / "src" / "qualbench" / "runner.py").is_file(),
    reason="QualBench checkout not found next to the flagship workspace",
)


@pytest.fixture(scope="module")
def one_pack_traces():
    """All traces of the dental pack: 10 scenarios, 2 agents, 2 variants."""
    qb = load_qualbench(ROOT)
    pack = qb.load_pack(ROOT / "packs" / "dental")
    traces = []
    for scenario in pack.scenarios:
        for agent in ("reference-agent", "naive-agent"):
            for variant in (0, 1):
                rng = random.Random(
                    trace_seed_for(
                        SEED,
                        f"qualbench:{scenario.scenario_id}",
                        agent,
                        variant,
                    )
                )
                perturbed = perturb_scenario(scenario, variant, rng)
                traces.append(
                    run_and_emit(
                        ROOT,
                        perturbed,
                        agent,
                        variant=variant,
                        corpus_seed=SEED,
                    )
                )
    return traces


def test_split_never_straddles_a_group(one_pack_traces):
    splits = split_traces(one_pack_traces, seed=SEED)
    groups_by_split = {
        name: {group_of(trace) for trace in rows}
        for name, rows in splits.items()
    }
    names = list(groups_by_split)
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            assert groups_by_split[first].isdisjoint(groups_by_split[second])
    # Every trace is accounted for exactly once.
    assert sum(len(rows) for rows in splits.values()) == len(one_pack_traces)
    # 10 groups cut 80/10/10 by group count.
    assert {
        name: len(groups_by_split[name]) for name in names
    } == {"train": 8, "validation": 1, "test": 1}


def test_perturbation_never_mutates_the_scenario():
    qb = load_qualbench(ROOT)
    pack = qb.load_pack(ROOT / "packs" / "dental")
    scenario = next(
        s for s in pack.scenarios if (s.seed_calendar or {}).get("busy")
    )
    before = [dict(block) for block in scenario.seed_calendar["busy"]]
    variant = perturb_scenario(scenario, 1, random.Random(1))
    assert scenario.seed_calendar["busy"] == before
    from datetime import datetime

    deltas = {
        datetime.fromisoformat(new["start"])
        - datetime.fromisoformat(old["start"])
        for old, new in zip(
            before, variant.seed_calendar["busy"], strict=True
        )
    }
    # Every block shifted by exactly one common whole-hour offset.
    assert len(deltas) == 1
    (delta,) = deltas
    assert delta.total_seconds() % 3600 == 0
    assert delta.total_seconds() != 0


def test_determinism_same_seed_same_card_and_bytes(one_pack_traces, tmp_path):
    splits = split_traces(one_pack_traces, seed=SEED)
    card_one = corpus_card(one_pack_traces, splits, seed=SEED, variants=2)
    paths_one = corpus_a.write_corpus(
        tmp_path / "run1", one_pack_traces, splits, card_one
    )
    # Rebuild the identical corpus from scratch and compare everything.
    rebuilt = one_pack_traces.__class__(one_pack_traces)
    splits_two = split_traces(rebuilt, seed=SEED)
    card_two = corpus_card(rebuilt, splits_two, seed=SEED, variants=2)
    paths_two = corpus_a.write_corpus(
        tmp_path / "run2", rebuilt, splits_two, card_two
    )
    assert card_one == card_two
    for name in paths_one:
        assert paths_one[name].read_bytes() == paths_two[name].read_bytes()


def test_full_rebuild_is_deterministic():
    """A second full build of one pack reproduces identical trace ids
    and identical card numbers (the real determinism check)."""
    qb = load_qualbench(ROOT)
    pack = qb.load_pack(ROOT / "packs" / "dental")

    def build():
        traces = []
        for scenario in pack.scenarios:
            rng = random.Random(1)
            perturbed = perturb_scenario(scenario, 1, rng)
            traces.append(
                run_and_emit(
                    ROOT,
                    perturbed,
                    "reference-agent",
                    variant=1,
                    corpus_seed=SEED,
                )
            )
        return traces

    first, second = build(), build()
    assert [t.trace_id for t in first] == [t.trace_id for t in second]
    assert [t.to_dict() for t in first] == [t.to_dict() for t in second]


def test_card_carries_the_required_sentences_and_counts(one_pack_traces):
    splits = split_traces(one_pack_traces, seed=SEED)
    card = corpus_card(one_pack_traces, splits, seed=SEED, variants=2)
    assert SYNTHETIC_SENTENCE in card
    assert "generator-declared, not measured" in card
    assert f"Traces: {len(one_pack_traces)}" in card
    assert f"Split seed: {SEED}" in card
