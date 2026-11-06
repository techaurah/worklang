"""Corpus v0: the frozen merge of sources A and B.

Build step 3 of DESIGN.md (corpus plan). Corpus v0 is the union of
the two per-source splits: each source froze its own group-aware
80/10/10 split with its own card, and the merge concatenates the
split files source by source (A first, then B, each in file order).
Group ids are namespaced by source (qualbench:... and
leadpipe-demo:...), and the merge verifies, rather than assumes,
that no group id appears in more than one merged split and that
every trace of both full corpora appears exactly once across the
merged splits. A violation raises MergeError and nothing is
written by the caller.

The combined card is generated here from the merged traces; every
number in it is counted from the traces themselves. Both sources'
latency declarations and recorded conflicts are carried into it, so
the frozen corpus keeps the honesty record of its parts.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from . import corpus_a, corpus_b
from .corpus_a import SPLIT_NAMES, SYNTHETIC_SENTENCE, load_corpus_file
from .errors import WorkLangError
from .qualbench_emitter import LATENCY_DISTRIBUTIONS
from .schema import Trace, trace_to_json
from .tokenizer import encode_trace


class MergeError(WorkLangError):
    """The per-source corpora cannot be merged without breaking an
    invariant (group straddle, missing or duplicated traces)."""


def group_of(trace: Trace) -> str:
    """A merged-corpus group id, dispatched on the trace's source."""
    if trace.source == "qualbench":
        return corpus_a.group_of(trace)
    if trace.source == "leadpipe-demo":
        return corpus_b.group_of(trace)
    raise MergeError(
        f"trace {trace.trace_id} has unknown source {trace.source!r}; "
        "corpus v0 merges only the qualbench and leadpipe-demo sources"
    )


def merge_splits(
    source_a_splits: dict[str, list[Trace]],
    source_b_splits: dict[str, list[Trace]],
    source_a_full: list[Trace],
    source_b_full: list[Trace],
) -> dict[str, list[Trace]]:
    """Merge per-source splits and verify the merge invariants."""
    merged: dict[str, list[Trace]] = {
        name: list(source_a_splits[name]) + list(source_b_splits[name])
        for name in SPLIT_NAMES
    }
    groups_by_split = {
        name: {group_of(trace) for trace in merged[name]} for name in SPLIT_NAMES
    }
    names = list(SPLIT_NAMES)
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            overlap = groups_by_split[first] & groups_by_split[second]
            if overlap:
                raise MergeError(
                    f"group(s) straddle the {first} and {second} "
                    f"splits: {sorted(overlap)}"
                )
    merged_ids = [trace.trace_id for name in names for trace in merged[name]]
    if len(merged_ids) != len(set(merged_ids)):
        raise MergeError("a trace appears more than once across the merged splits")
    expected = {t.trace_id for t in source_a_full} | {
        t.trace_id for t in source_b_full
    }
    if set(merged_ids) != expected:
        missing = sorted(expected - set(merged_ids))
        extra = sorted(set(merged_ids) - expected)
        raise MergeError(
            f"merged splits do not cover the full corpora exactly: "
            f"missing {missing}, extra {extra}"
        )
    return merged


def load_source_splits(
    source_dir: Path, stem: str
) -> tuple[dict[str, list[Trace]], list[Trace]]:
    """Load one source's full corpus and its split files."""
    full = load_corpus_file(source_dir / f"{stem}.jsonl")
    splits = {
        name: load_corpus_file(source_dir / f"{stem}-{name}.jsonl")
        for name in SPLIT_NAMES
    }
    return splits, full


def build_merged(corpus_root: Path) -> tuple[dict[str, list[Trace]], list[Trace]]:
    """Load both sources from the corpus tree and merge their splits.

    Returns (merged splits, all traces in merged order). Raises
    MergeError on any invariant violation.
    """
    a_splits, a_full = load_source_splits(corpus_root / "source-a", "corpus-a")
    b_splits, b_full = load_source_splits(corpus_root / "source-b", "corpus-b")
    merged = merge_splits(a_splits, b_splits, a_full, b_full)
    all_traces = [trace for name in SPLIT_NAMES for trace in merged[name]]
    return merged, all_traces


def _latency_table() -> list[str]:
    lines = [
        "| Vertical | instant | fast | slow | delayed | overnight |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for vertical, bands in LATENCY_DISTRIBUTIONS.items():
        weights = dict(bands)
        row = " | ".join(
            f"{weights[band]:.0%}"
            for band in ("instant", "fast", "slow", "delayed", "overnight")
        )
        lines.append(f"| {vertical} | {row} |")
    return lines


def corpus_card(
    merged: dict[str, list[Trace]],
    *,
    seed_a: int = corpus_a.DEFAULT_CORPUS_SEED,
    seed_b: int = corpus_a.DEFAULT_CORPUS_SEED,
) -> str:
    """Build the combined corpus v0 card from the merged traces."""
    all_traces = [trace for name in SPLIT_NAMES for trace in merged[name]]
    event_count = sum(len(trace.events) for trace in all_traces)
    token_count = sum(len(encode_trace(trace)) for trace in all_traces)
    per_source = Counter(trace.source for trace in all_traces)
    per_vertical = Counter(trace.vertical for trace in all_traces)
    per_agent = Counter(trace.agent for trace in all_traces)
    outcomes = Counter(trace.outcome for trace in all_traces)
    split_groups = {
        name: len({group_of(trace) for trace in merged[name]})
        for name in SPLIT_NAMES
    }
    lines = [
        "# Corpus card: WorkLang corpus v0 (merged, frozen)",
        "",
        f"This corpus is {SYNTHETIC_SENTENCE}.",
        "",
        "Corpus v0 is the frozen merge of corpus source A (QualBench "
        "runs through the trace emitter) and corpus source B "
        "(LeadPipe demo lifecycle synthesis plus the demo walkthrough "
        "anchor family). Each source froze its own group-aware "
        "80/10/10 split under its own card (corpus/source-a/ and "
        "corpus/source-b/); this merge unions the split files and "
        "verifies that no group id appears in more than one merged "
        "split and that every trace appears exactly once.",
        "",
        "## Counts",
        "",
        f"- Traces: {len(all_traces)}",
        f"- Events: {event_count}",
        f"- Tokens: {token_count} (encoded trace length: BOS, one "
        "vertical token, 7 tokens per event, EOS)",
        "",
        "Per source:",
        "",
    ]
    for name in sorted(per_source):
        lines.append(f"- {name}: {per_source[name]} traces")
    lines += ["", "Per vertical:", ""]
    for name in sorted(per_vertical):
        lines.append(f"- {name}: {per_vertical[name]} traces")
    lines += ["", "Per agent:", ""]
    for name in sorted(per_agent):
        lines.append(f"- {name}: {per_agent[name]} traces")
    lines += ["", "Outcome distribution (trace level outcome class):", ""]
    for name in sorted(outcomes):
        lines.append(f"- {name}: {outcomes[name]} traces")
    lines += [
        "",
        "## Splits",
        "",
        "Group-aware split by trace, 80/10/10 train/validation/test, "
        "inherited from the per-source splits. Groups are namespaced "
        "by source: one QualBench scenario (both agents, all "
        "variants), one LeadPipe seed lead (all its walks), or the "
        "LeadPipe walkthrough anchor family. No group appears in two "
        "splits of this merged corpus; the merge checks it.",
        "",
        f"- Split seeds: source A {seed_a}, source B {seed_b}",
    ]
    for name in SPLIT_NAMES:
        lines.append(
            f"- {name}: {len(merged[name])} traces "
            f"({split_groups[name]} groups)"
        )
    lines += [
        "",
        "## Latency declarations",
        "",
        "Latency values in this corpus are generator-declared, not "
        "measured, in both sources. Neither simulator emits "
        "inter-event timing. Source A declares the gap before each "
        "event from the per-vertical distributions below; source B "
        "draws from the general row (its traces are all vertical "
        '"general"). The first event of every trace carries latency '
        '"none" (it has no predecessor).',
        "",
        *_latency_table(),
        "",
        "## Channels",
        "",
        "Source A carries channel \"unknown\" on every event: "
        "QualBench scenarios declare no channel and none was "
        "invented. Source B carries each LeadPipe lead's real source "
        "as the channel, except walk_in and other, which have no "
        "channel bucket and degrade to \"unknown\" (see the "
        "conflicts section).",
        "",
        "## Known conflicts and emitter decisions",
        "",
        "From source A: QualBench agents report two stage strings "
        "with no token in the stage vocabulary, HANDOFF and "
        "UNQUALIFIED. Neither is silently mapped into the stage "
        "stream: HANDOFF surfaces as handoff events, and a run whose "
        "final stage is UNQUALIFIED closes with outcome \"lost\". "
        "From source B: DESIGN.md enumerates the channel buckets as "
        "the seven LeadPipe source values, but the Source enum has "
        "nine and the demo seed uses walk_in and other, which "
        "degrade to \"unknown\". Both are schema v2 questions, "
        "recorded in DESIGN.md.",
        "",
        "## Provenance",
        "",
        "- Source A: worklang-qualbench-emitter 0.1.0, "
        f"corpus seed {seed_a} (corpus/source-a/CORPUS_CARD.md)",
        "- Source B: worklang-leadpipe-synthesiser 0.1.0, "
        f"corpus seed {seed_b} (corpus/source-b/CORPUS_CARD.md)",
        f"- Sentence required wherever this corpus is described: "
        f"\"{SYNTHETIC_SENTENCE}\"",
        "",
    ]
    return "\n".join(lines)


def write_corpus(
    out_dir: Path,
    merged: dict[str, list[Trace]],
    card: str,
) -> dict[str, Path]:
    """Write the merged corpus, the split files, and the card."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    def _write_jsonl(name: str, rows: list[Trace]) -> Path:
        path = out_dir / name
        path.write_text(
            "".join(trace_to_json(trace) + "\n" for trace in rows),
            encoding="utf-8",
        )
        return path

    all_traces = [trace for name in SPLIT_NAMES for trace in merged[name]]
    paths["corpus"] = _write_jsonl("corpus-v0.jsonl", all_traces)
    for name in SPLIT_NAMES:
        paths[name] = _write_jsonl(f"corpus-v0-{name}.jsonl", merged[name])
    card_path = out_dir / "CORPUS_CARD.md"
    card_path.write_text(card, encoding="utf-8")
    paths["card"] = card_path
    return paths
