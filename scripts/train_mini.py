#!/usr/bin/env python3
"""Train worklang-mini on a frozen corpus, evaluate it, write results.

Build step 5 (DESIGN.md): trains the mini on the corpus train split
only (CPU only, fixed seed), scores it on the test split through the
step 4 harness protocol (the same 544 positions the baselines were
scored on), measures the futures calibration, and writes the results
JSON produced by the run (never hand-written numbers), with the step
4 baseline numbers embedded alongside for direct comparison and the
n-gram bar verdict stated either way.

Reproducibility is measured, not asserted: with --runs 2 (the
default) the whole train, score, and calibrate sequence runs twice
with the same seed, and the results document records both runs and
the largest absolute delta across their shared metrics.

Usage:
    python scripts/train_mini.py [--corpus-dir DIR]
        [--corpus-name NAME] [--out FILE] [--epochs N] [--lr X]
        [--batch-size N] [--seed N] [--runs N] [--samples N]
        [--temperature X] [--threads N]
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from worklang.corpus_a import load_corpus_file  # noqa: E402
from worklang.evaluation import PROTOCOL, write_results  # noqa: E402
from worklang.mini import (  # noqa: E402
    MiniConfig,
    TrainConfig,
    count_parameters,
    futures_calibration,
    score_mini,
    train_model,
)
from worklang.vocab import EVENT_STREAMS, RESERVED_VOCAB_SIZE  # noqa: E402


def _metric_deltas(run_a: dict, run_b: dict) -> dict[str, float]:
    """Absolute deltas across the metrics two same-seed runs share."""
    deltas: dict[str, float] = {}

    def add(name: str, a: float, b: float) -> None:
        deltas[name] = round(abs(a - b), 6)

    add("family_accuracy", run_a["family_accuracy"], run_b["family_accuracy"])
    add(
        "composite_exact_match",
        run_a["composite_exact_match"],
        run_b["composite_exact_match"],
    )
    for stream in EVENT_STREAMS:
        add(
            f"stream:{stream}",
            run_a["stream_accuracies"][stream],
            run_b["stream_accuracies"][stream],
        )
    add(
        "calibration:mean_group_total_variation",
        run_a["futures_calibration"]["mean_group_total_variation"],
        run_b["futures_calibration"]["mean_group_total_variation"],
    )
    add(
        "calibration:pooled_total_variation",
        run_a["futures_calibration"]["pooled_total_variation"],
        run_b["futures_calibration"]["pooled_total_variation"],
    )
    add(
        "final_epoch_loss",
        run_a["training"]["final_epoch_loss"],
        run_b["training"]["final_epoch_loss"],
    )
    return deltas


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-name", default="v0")
    parser.add_argument("--corpus-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=48)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    corpus_dir = args.corpus_dir or repo_root / "corpus" / args.corpus_name
    out = args.out or repo_root / "results" / f"mini-{args.corpus_name}.json"
    baselines_path = repo_root / "results" / f"baselines-{args.corpus_name}.json"
    stem = f"corpus-{args.corpus_name}"
    train_traces = load_corpus_file(corpus_dir / f"{stem}-train.jsonl")
    test_traces = load_corpus_file(corpus_dir / f"{stem}-test.jsonl")
    baselines_doc = json.loads(baselines_path.read_text(encoding="utf-8"))
    baselines = baselines_doc["baselines"]

    config = MiniConfig()
    parameter_count: int | None = None
    runs: list[dict] = []
    for run_index in range(1, args.runs + 1):
        started = time.perf_counter()
        model, info = train_model(
            train_traces,
            config=config,
            train_config=TrainConfig(
                epochs=args.epochs,
                lr=args.lr,
                batch_size=args.batch_size,
                seed=args.seed,
                threads=args.threads,
            ),
        )
        if parameter_count is None:
            parameter_count = count_parameters(model)
        scored = score_mini(model, test_traces)
        calibration = futures_calibration(
            model,
            test_traces,
            samples_per_prefix=args.samples,
            temperature=args.temperature,
            seed=args.seed,
        )
        wall = round(time.perf_counter() - started, 1)
        runs.append(
            {
                "run": run_index,
                "seed": args.seed,
                "wall_time_seconds": wall,
                "training": info,
                **scored,
                "futures_calibration": calibration,
            }
        )
        print(
            f"Run {run_index}: final epoch loss "
            f"{info['final_epoch_loss']:.4f}, family "
            f"{scored['family_accuracy']:.4f}, composite "
            f"{scored['composite_exact_match']:.4f}, calibration mean "
            f"group TV "
            f"{calibration['mean_group_total_variation']:.4f} "
            f"({wall:.1f}s)"
        )

    headline = runs[0]
    if headline["positions"] != baselines_doc["positions_scored"]:
        raise SystemExit(
            f"protocol mismatch: mini scored {headline['positions']} "
            f"positions, baselines scored "
            f"{baselines_doc['positions_scored']}"
        )
    ngram = baselines["ngram_order2"]
    beats_family = headline["family_accuracy"] > ngram["family_accuracy"]
    beats_composite = (
        headline["composite_exact_match"] > ngram["composite_exact_match"]
    )
    deltas = (
        _metric_deltas(runs[0], runs[1]) if len(runs) > 1 else {}
    )
    results = {
        "corpus": args.corpus_name,
        "files": {
            "train": f"{stem}-train.jsonl",
            "test": f"{stem}-test.jsonl",
        },
        "train_traces": len(train_traces),
        "test_traces": len(test_traces),
        "positions_scored": headline["positions"],
        "protocol": PROTOCOL,
        "model": {
            "name": "worklang-mini",
            "architecture": (
                "Decoder-only transformer over the flat token stream "
                "(BOS, vertical token, 7 tokens per event); family "
                "head first, attribute heads conditioned on the "
                "family via its token embedding; loss is the sum of "
                "per-stream cross-entropies."
            ),
            "d_model": config.d_model,
            "n_layer": config.n_layer,
            "n_head": config.n_head,
            "max_events": config.max_events,
            "embedding_rows": RESERVED_VOCAB_SIZE,
            "parameter_count": parameter_count,
            "parameter_count_note": (
                "Counted from the built module by count_parameters. "
                "DESIGN.md estimated 5 to 6M before the code existed."
            ),
        },
        "training": {
            "device": "cpu",
            "torch_version": torch.__version__,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "threads": args.threads,
            "runs": runs,
        },
        "baselines": baselines,
        "bar": {
            "source": "results/baselines-v0.json (build step 4)",
            "ngram_family_accuracy": ngram["family_accuracy"],
            "ngram_composite_exact_match": ngram["composite_exact_match"],
            "mini_family_accuracy": headline["family_accuracy"],
            "mini_composite_exact_match": headline["composite_exact_match"],
            "beats_ngram_on_family": beats_family,
            "beats_ngram_on_composite": beats_composite,
            "bar_cleared": beats_family and beats_composite,
            "rule": (
                "DESIGN.md: if the n-gram wins, the results file says "
                "so and the mini does not ship as a model claim until "
                "it beats it or the corpus grows."
            ),
        },
        "reproducibility": {
            "runs_compared": len(runs),
            "same_seed": True,
            "metric_deltas": deltas,
            "max_abs_delta": max(deltas.values()) if deltas else 0.0,
            "exact_match": bool(deltas)
            and all(value == 0.0 for value in deltas.values()),
        },
    }
    write_results(results, out)
    print(
        f"Bar: n-gram family {ngram['family_accuracy']:.4f} / composite "
        f"{ngram['composite_exact_match']:.4f}; mini family "
        f"{headline['family_accuracy']:.4f} / composite "
        f"{headline['composite_exact_match']:.4f}; cleared: "
        f"{results['bar']['bar_cleared']}"
    )
    print(f"Results written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
