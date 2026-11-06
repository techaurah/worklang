#!/usr/bin/env python3
"""Evaluate the WorkLang baselines on a frozen corpus and write results.

Build step 4 (DESIGN.md): fits the majority-class and order-2 n-gram
baselines on the corpus train split, scores them on the test split,
writes the results JSON produced by the run (never hand-written
numbers), and prints a summary table. The test split is the bar the
step 5 mini model has to beat, and the results file says so either
way.

Usage:
    python scripts/evaluate_baselines.py [--corpus-dir DIR]
        [--corpus-name NAME] [--out FILE]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from worklang.evaluation import evaluate_corpus, write_results  # noqa: E402
from worklang.vocab import EVENT_STREAMS  # noqa: E402


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-name",
        default="v0",
        help="Corpus version to evaluate (default: v0)",
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=None,
        help="Corpus directory (default: the repo's corpus/<name> directory)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Results JSON path (default: results/baselines-<name>.json)",
    )
    args = parser.parse_args()
    corpus_dir = args.corpus_dir or repo_root / "corpus" / args.corpus_name
    out = args.out or repo_root / "results" / f"baselines-{args.corpus_name}.json"
    results = evaluate_corpus(corpus_dir, args.corpus_name)
    write_results(results, out)
    print(
        f"Corpus {results['corpus']}: {results['train_traces']} train traces, "
        f"{results['test_traces']} test traces, "
        f"{results['positions_scored']} positions scored."
    )
    for name, block in results["baselines"].items():
        print(
            f"  {name}: family {block['family_accuracy']:.4f}, "
            f"composite exact match {block['composite_exact_match']:.4f}"
        )
        for stream in EVENT_STREAMS:
            print(f"    {stream}: {block['stream_accuracies'][stream]:.4f}")
    print(f"Results written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
