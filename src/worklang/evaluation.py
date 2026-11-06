"""Evaluation harness for the WorkLang baselines (DESIGN.md, Evaluation).

Build step 4. The protocol, fixed by the design:

- Held-out next-event accuracy on the test split. Prefixes are
  extracted at every length from each test trace: every event
  position is scored once, with the prefix being all prior events of
  that trace, including the empty prefix before the first event (the
  standard PPM prefix extraction the design names). The target is the
  next event at that position.
- Metrics per predictor: family accuracy, accuracy for each of the
  other six streams, and full-composite exact match (all 7 tokens of
  the next event correct).
- Both baselines are fitted on the train split only and scored on the
  test split only. The split discipline lives here: fit receives the
  train traces, scoring receives the test traces, and nothing else is
  read. The validation split is not touched by this harness at all.

Every number in the results document is counted by run_evaluation
from the traces it is handed; the document carries counts alongside
accuracies so any figure can be re-derived by hand. It contains no
timestamps and no absolute paths, so a re-run on the same corpus
produces a byte-identical file.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

from .baselines import MajorityBaseline, NGramBaseline
from .corpus_a import load_corpus_file
from .schema import Event, Trace
from .vocab import EVENT_STREAMS

Predictor = Callable[[Sequence[Event]], dict[str, str]]

PROTOCOL = (
    "Next-event prediction at every position of every test trace "
    "(prefix = all prior events of the trace, including the empty "
    "prefix); baselines fitted on the train split only, scored on "
    "the test split only; ties broken by highest count, then "
    "vocabulary order."
)


def extract_positions(
    traces: Sequence[Trace],
) -> Iterator[tuple[tuple[Event, ...], Event]]:
    """Yield (prefix, target) for every position of every trace.

    Prefixes at every length: for a trace of n events this yields n
    pairs, from the empty prefix targeting the first event to the
    n-1 event prefix targeting the last (outcome) event.
    """
    for trace in traces:
        events = trace.events
        for index, target in enumerate(events):
            yield events[:index], target


def score_predictor(
    predict: Predictor, test_traces: Sequence[Trace]
) -> dict:
    """Score one fitted predictor over the test split.

    Returns counted results: positions scored, per-stream correct
    counts, composite correct count, and the accuracies derived from
    them (family called out separately, as the design does).
    """
    per_stream_correct = {stream: 0 for stream in EVENT_STREAMS}
    composite_correct = 0
    positions = 0
    for prefix, target in extract_positions(test_traces):
        predicted = predict(prefix)
        positions += 1
        all_correct = True
        for stream in EVENT_STREAMS:
            if predicted[stream] == getattr(target, stream):
                per_stream_correct[stream] += 1
            else:
                all_correct = False
        if all_correct:
            composite_correct += 1
    if positions == 0:
        raise ValueError("no test positions to score (empty test split)")

    def accuracy(correct: int) -> float:
        return round(correct / positions, 6)

    return {
        "positions": positions,
        "per_stream_correct": per_stream_correct,
        "composite_correct": composite_correct,
        "family_accuracy": accuracy(per_stream_correct["family"]),
        "stream_accuracies": {
            stream: accuracy(per_stream_correct[stream]) for stream in EVENT_STREAMS
        },
        "composite_exact_match": accuracy(composite_correct),
    }


def run_evaluation(
    train_traces: Sequence[Trace], test_traces: Sequence[Trace]
) -> dict:
    """Fit both baselines on train_traces and score them on test_traces."""
    majority = MajorityBaseline().fit(train_traces)
    ngram = NGramBaseline().fit(train_traces)
    return {
        "majority": {
            "description": (
                "Most frequent family and most frequent value per "
                "attribute stream, counted over the train split."
            ),
            **score_predictor(majority.predict, test_traces),
        },
        "ngram_order2": {
            "description": (
                "Order-2 n-gram over composite event symbols with "
                "backoff to order-1 and unigram, trained on the "
                "train split."
            ),
            **score_predictor(ngram.predict, test_traces),
        },
    }


def evaluate_corpus(corpus_dir: Path, corpus_name: str = "v0") -> dict:
    """Run the full evaluation over a frozen corpus directory.

    Reads exactly two files: corpus-<name>-train.jsonl (fitting) and
    corpus-<name>-test.jsonl (scoring) inside corpus_dir.
    """
    stem = f"corpus-{corpus_name}"
    train_path = corpus_dir / f"{stem}-train.jsonl"
    test_path = corpus_dir / f"{stem}-test.jsonl"
    train_traces = load_corpus_file(train_path)
    test_traces = load_corpus_file(test_path)
    baselines = run_evaluation(train_traces, test_traces)
    positions = baselines["majority"]["positions"]
    return {
        "corpus": corpus_name,
        "files": {"train": train_path.name, "test": test_path.name},
        "train_traces": len(train_traces),
        "test_traces": len(test_traces),
        "positions_scored": positions,
        "protocol": PROTOCOL,
        "baselines": baselines,
    }


def results_to_json(results: dict) -> str:
    """Serialise a results document deterministically."""
    return json.dumps(results, indent=2, sort_keys=True) + "\n"


def write_results(results: dict, path: Path) -> Path:
    """Write the results document to path, creating parents as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(results_to_json(results), encoding="utf-8")
    return path
