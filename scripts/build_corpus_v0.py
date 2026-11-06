#!/usr/bin/env python3
"""Freeze WorkLang corpus v0: merge sources A and B and print its card.

Loads the per-source corpora and split files from corpus/source-a
and corpus/source-b, verifies the merge invariants (no group id in
more than one merged split; every trace of both full corpora present
exactly once), writes corpus-v0.jsonl, the merged split files, and
the combined CORPUS_CARD.md into corpus/v0, and prints the card to
stdout. Run scripts/build_corpus_a.py and scripts/build_corpus_b.py
first.

Usage:
    python scripts/build_corpus_v0.py [--corpus-root DIR] [--out-dir DIR]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from worklang.corpus_v0 import build_merged, corpus_card, write_corpus  # noqa: E402


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-root",
        type=Path,
        default=repo_root / "corpus",
        help="Corpus tree holding source-a and source-b "
        "(default: the repo's corpus directory)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=repo_root / "corpus" / "v0",
        help="Where to write the merged corpus (default: corpus/v0)",
    )
    args = parser.parse_args()
    merged, _all_traces = build_merged(args.corpus_root)
    card = corpus_card(merged)
    paths = write_corpus(args.out_dir, merged, card)
    print(card)
    print("Files written:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
