#!/usr/bin/env python3
"""Generate WorkLang corpus source B (LeadPipe) and print its card.

Synthesises lifecycle walks over LeadPipe's TRANSITIONS map for each
of the 12 seed leads plus the demo walkthrough anchor family, writes
corpus-b.jsonl, the group-aware 80/10/10 split files, and
CORPUS_CARD.md into the output directory, and prints the corpus card
to stdout. Every number on the card is counted from the generated
traces.

Usage:
    python scripts/build_corpus_b.py [--out-dir DIR] [--seed N]
        [--walks N] [--leadpipe-root DIR]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from worklang.corpus_a import DEFAULT_CORPUS_SEED  # noqa: E402
from worklang.corpus_b import (  # noqa: E402
    DEFAULT_WALKS_PER_LEAD,
    build_corpus_b,
    corpus_card,
    split_traces,
    write_corpus,
)
from worklang.leadpipe_synthesiser import default_leadpipe_root  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "corpus" / "source-b",
        help="Where to write the corpus files (default: corpus/source-b)",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_CORPUS_SEED)
    parser.add_argument(
        "--walks", type=int, default=DEFAULT_WALKS_PER_LEAD
    )
    parser.add_argument(
        "--leadpipe-root",
        type=Path,
        default=None,
        help="LeadPipe checkout (default: sibling workspace checkout)",
    )
    args = parser.parse_args()
    root = args.leadpipe_root or default_leadpipe_root()
    traces = build_corpus_b(root, seed=args.seed, walks=args.walks)
    splits = split_traces(traces, seed=args.seed)
    card = corpus_card(traces, splits, seed=args.seed, walks=args.walks)
    paths = write_corpus(args.out_dir, traces, splits, card)
    print(card)
    print("Files written:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
