#!/usr/bin/env python3
"""Generate WorkLang corpus source A (QualBench) and print its card.

Runs every QualBench scenario with both shipped agents plus seeded
variants through the trace emitter, writes corpus-a.jsonl, the
group-aware 80/10/10 split files, and CORPUS_CARD.md into the output
directory, and prints the corpus card to stdout. Every number on the
card is counted from the generated traces.

Usage:
    python scripts/build_corpus_a.py [--out-dir DIR] [--seed N]
        [--variants N] [--qualbench-root DIR]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from worklang.corpus_a import (  # noqa: E402
    DEFAULT_CORPUS_SEED,
    DEFAULT_VARIANTS,
    build_corpus_a,
    corpus_card,
    split_traces,
    write_corpus,
)
from worklang.qualbench_emitter import default_qualbench_root  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "corpus" / "source-a",
        help="Where to write the corpus files (default: corpus/source-a)",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_CORPUS_SEED)
    parser.add_argument("--variants", type=int, default=DEFAULT_VARIANTS)
    parser.add_argument(
        "--qualbench-root",
        type=Path,
        default=None,
        help="QualBench checkout (default: sibling workspace checkout)",
    )
    args = parser.parse_args()
    root = args.qualbench_root or default_qualbench_root()
    traces = build_corpus_a(root, seed=args.seed, variants=args.variants)
    splits = split_traces(traces, seed=args.seed)
    card = corpus_card(traces, splits, seed=args.seed, variants=args.variants)
    paths = write_corpus(args.out_dir, traces, splits, card)
    print(card)
    print("Files written:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
