"""Tests for the evaluation harness.

Three things are pinned here:

- The scoring arithmetic, on a fixture small enough to score by hand.
- The split discipline, at the file level: fitting reads only the
  train file, scoring reads only the test file, and a poisoned
  validation file sitting in the same directory changes nothing.
- Determinism: the same corpus evaluated twice gives identical
  results, down to the written bytes.
"""

import json
from pathlib import Path

from worklang.corpus_a import load_corpus_file
from worklang.evaluation import (
    evaluate_corpus,
    extract_positions,
    run_evaluation,
    write_results,
)
from worklang.schema import Event, Generator, Trace, trace_to_json

CORPUS_V0_DIR = Path(__file__).resolve().parents[1] / "corpus" / "v0"


def family_trace(trace_id: str, families: list[str]) -> Trace:
    """A trace over the given families; all other streams at defaults."""
    events = tuple(
        Event(
            seq=index,
            family=family,
            channel="unknown",
            score_band="none",
            latency_band="none",
            stage="none",
            tool="none",
            outcome_class="booked" if family == "outcome" else "none",
        )
        for index, family in enumerate(families)
    )
    return Trace(
        trace_id=trace_id,
        source="qualbench",
        synthetic=True,
        vertical="general",
        agent="fixture-agent",
        agent_version="0",
        events=events,
        outcome="booked",
        generator=Generator(name="fixture", version="0", seed=1),
    )


def test_extract_positions_covers_every_length():
    trace = family_trace("t", ["enquiry", "reply", "outcome"])
    positions = list(extract_positions([trace]))
    assert [len(prefix) for prefix, _target in positions] == [0, 1, 2]
    assert [target.family for _prefix, target in positions] == [
        "enquiry",
        "reply",
        "outcome",
    ]


def test_majority_scoring_matches_hand_count():
    # Train: two copies of enquiry, reply, outcome. Family counts tie
    # at 2 each; vocabulary order makes the majority family enquiry,
    # and every other stream's mode is its uniform or majority value
    # (outcome_class: none 4, booked 2, so none).
    train = [
        family_trace("a", ["enquiry", "reply", "outcome"]),
        family_trace("b", ["enquiry", "reply", "outcome"]),
    ]
    # Test: one copy. The constant majority prediction (enquiry,
    # unknown, none, none, none, none, none) scores, position by
    # position: all 7 streams right on the enquiry event; family
    # wrong only on the reply event; family and outcome_class wrong
    # on the outcome event.
    test = [family_trace("c", ["enquiry", "reply", "outcome"])]
    block = run_evaluation(train, test)["majority"]
    assert block["positions"] == 3
    assert block["per_stream_correct"] == {
        "family": 1,
        "channel": 3,
        "score_band": 3,
        "latency_band": 3,
        "stage": 3,
        "tool": 3,
        "outcome_class": 2,
    }
    assert block["composite_correct"] == 1
    assert block["family_accuracy"] == round(1 / 3, 6)
    assert block["composite_exact_match"] == round(1 / 3, 6)
    assert block["stream_accuracies"]["outcome_class"] == round(2 / 3, 6)
    assert block["stream_accuracies"]["channel"] == 1.0


def _write_split(directory: Path, name: str, traces) -> None:
    text = "".join(trace_to_json(trace) + "\n" for trace in traces)
    (directory / f"corpus-v0-{name}.jsonl").write_text(text, encoding="utf-8")


def test_harness_fits_train_only_and_scores_test_only(tmp_path):
    # Train majority family is reply (3 reply, 2 outcome).
    train = [
        family_trace("train-1", ["reply", "reply", "outcome"]),
        family_trace("train-2", ["reply", "outcome"]),
    ]
    # Test families: enquiry 3, outcome 3, reply 1 over 7 positions.
    # The three candidate fitting sets give three different majority
    # family accuracies on these targets, so the number identifies
    # the set: train only predicts reply (1/7 right); train plus test
    # would predict outcome (3/7); the poisoned validation file alone
    # or added would predict booking (0/7).
    test = [
        family_trace("test-1", ["enquiry", "outcome"]),
        family_trace("test-2", ["enquiry", "outcome"]),
        family_trace("test-3", ["enquiry", "reply", "outcome"]),
    ]
    poison = [
        family_trace(f"val-{index}", ["booking", "booking", "outcome"])
        for index in range(5)
    ]
    _write_split(tmp_path, "train", train)
    _write_split(tmp_path, "test", test)
    _write_split(tmp_path, "validation", poison)
    results = evaluate_corpus(tmp_path, "v0")
    assert results["train_traces"] == 2
    assert results["test_traces"] == 3
    assert results["positions_scored"] == 7
    majority = results["baselines"]["majority"]
    assert majority["positions"] == 7
    assert majority["family_accuracy"] == round(1 / 7, 6)
    assert results["baselines"]["ngram_order2"]["positions"] == 7


def test_evaluation_on_frozen_corpus_v0_is_deterministic(tmp_path):
    first = evaluate_corpus(CORPUS_V0_DIR, "v0")
    second = evaluate_corpus(CORPUS_V0_DIR, "v0")
    assert first == second
    first_path = write_results(first, tmp_path / "one.json")
    second_path = write_results(second, tmp_path / "two.json")
    assert first_path.read_bytes() == second_path.read_bytes()
    # The scored positions are exactly the test split's events,
    # counted from the frozen file itself.
    test_traces = load_corpus_file(CORPUS_V0_DIR / "corpus-v0-test.jsonl")
    assert first["positions_scored"] == sum(
        len(trace.events) for trace in test_traces
    )
    assert first["test_traces"] == len(test_traces)
    for block in first["baselines"].values():
        assert 0.0 <= block["family_accuracy"] <= 1.0
        assert 0.0 <= block["composite_exact_match"] <= 1.0
        assert set(block["stream_accuracies"]) == set(
            block["per_stream_correct"]
        )
    # The written file parses back to the same document.
    assert json.loads(first_path.read_text(encoding="utf-8")) == first
