#!/usr/bin/env python3
"""Build the step 6 futures demo: sample, write data, render page.

DESIGN.md build step 6: the after-hours missed-lead demo is a
static page with precomputed outputs (no hosted inference at v0).
This script loads the trained worklang-mini (training and saving it
first if the weights file does not exist yet, with exactly the step
5 configuration scripts/train_mini.py evaluated), samples futures
from the two authored med-spa prefixes at the configuration recorded
in worklang.demo, and writes:

- demo/demo-data.json: the sampled outcome distributions, the full
  per-sample outcome list, and three example trajectories per prefix.
- demo/index.html: the single self-contained page, rendered from
  demo/index.template.html with the data inlined. No external
  requests, no tracking, no server needed.

Every number on the page comes from this run's sampling; nothing is
hand-written.

Usage:
    python scripts/sample_futures.py [--demo-dir DIR]
        [--samples N] [--temperature X] [--seed N]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from worklang.corpus_a import load_corpus_file  # noqa: E402
from worklang.demo import (  # noqa: E402
    DATA_FILENAME,
    PAGE_FILENAME,
    SAMPLES_PER_PREFIX,
    SEED,
    TEMPERATURE,
    TEMPLATE_FILENAME,
    WEIGHTS_FILENAME,
    build_demo_data,
    ensure_trained_model,
    render_page,
    write_demo_data,
)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--demo-dir", type=Path, default=repo_root / "demo"
    )
    parser.add_argument("--samples", type=int, default=SAMPLES_PER_PREFIX)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    demo_dir = args.demo_dir
    demo_dir.mkdir(parents=True, exist_ok=True)
    weights_path = demo_dir / WEIGHTS_FILENAME
    train_traces = None
    if not weights_path.exists():
        train_path = repo_root / "corpus" / "v0" / "corpus-v0-train.jsonl"
        train_traces = load_corpus_file(train_path)
        print(
            f"No weights file yet; training worklang-mini on "
            f"{len(train_traces)} train traces with the step 5 "
            "configuration (this takes a while on CPU)."
        )
    model, metadata = ensure_trained_model(
        weights_path, train_traces=train_traces
    )
    print(f"Model: {metadata['weights_source']}")
    print(f"Parameters: {metadata['parameter_count']}")

    data = build_demo_data(
        model,
        metadata,
        samples=args.samples,
        temperature=args.temperature,
        seed=args.seed,
    )
    data_path = demo_dir / DATA_FILENAME
    write_demo_data(data, data_path)
    template = (demo_dir / TEMPLATE_FILENAME).read_text(encoding="utf-8")
    page = render_page(data, template)
    page_path = demo_dir / PAGE_FILENAME
    page_path.write_text(page, encoding="utf-8")

    for block in data["prefixes"]:
        classes = block["distribution"]["classes"]
        shares = ", ".join(
            f"{name} {entry['share_percent']}%"
            for name, entry in classes.items()
            if entry["count"]
        )
        print(
            f"{block['id']}: {block['samples']} sampled, "
            f"{block['distribution']['completed']} completed, "
            f"{block['truncated']} truncated; {shares}"
        )
    print(f"Data written: {data_path}")
    print(f"Page written: {page_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
