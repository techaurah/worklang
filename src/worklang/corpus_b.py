"""Corpus source B: generation, group-aware split, and corpus card.

Build step 3 of DESIGN.md (corpus plan, LeadPipe source). Generation
method, exactly as the design fixes it: for each of the 12 seed
leads, lifecycle traces by seeded random walks over LeadPipe's
TRANSITIONS map from the lead's seeded stage (see
leadpipe_synthesiser.py for the walk semantics and its recorded
decisions), plus the demo walkthrough executed and emitted as an
anchor trace family.

Splits: by trace, 80/10/10 train/validation/test, group-aware, the
same algorithm and the same corpus seed as source A. A group is one
seed lead (all its walks) or the walkthrough family (all its anchor
traces); every trace of a group lands in exactly one split, so the
test split measures generalisation to unseen leads, not memorised
walks. Groups are shuffled with the split seed and cut 80/10/10 by
group count; trace counts per split follow from group sizes and are
published on the card as counted.

The corpus card is generated here from the built corpus; every
number in it is counted from the traces themselves.
"""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

from .corpus_a import DEFAULT_CORPUS_SEED, SPLIT_NAMES, SYNTHETIC_SENTENCE
from .corpus_a import load_corpus_file as _load_corpus_file
from .leadpipe_synthesiser import (
    CONTINUE_PROBABILITIES,
    MAX_FUTURE_STEPS,
    SYNTHESISER_NAME,
    SYNTHESISER_VERSION,
    WALKTHROUGH_GROUP,
    default_leadpipe_root,
    load_seed_leads,
    run_walkthrough_anchors,
    synthesise_walk,
)
from .leadpipe_synthesiser import load_leadpipe as _load_leadpipe
from .qualbench_emitter import LATENCY_DISTRIBUTIONS
from .schema import Trace, trace_to_json
from .tokenizer import encode_trace

DEFAULT_WALKS_PER_LEAD = 8

load_corpus_file = _load_corpus_file


def build_corpus_b(
    root: Path | None = None,
    *,
    seed: int = DEFAULT_CORPUS_SEED,
    walks: int = DEFAULT_WALKS_PER_LEAD,
) -> list[Trace]:
    """Synthesise source B: lifecycle walks, then the anchor family.

    Order is deterministic: seed leads in seed order, walks in index
    order, then the walkthrough anchor traces (seeded leads in seed
    order, the walkthrough lead last).
    """
    root = root if root is not None else default_leadpipe_root()
    lp = _load_leadpipe(root)
    traces: list[Trace] = []
    for lead in load_seed_leads(root):
        for walk in range(walks):
            traces.append(synthesise_walk(lp, lead, walk, seed))
    traces.extend(run_walkthrough_anchors(root, seed))
    return traces


def group_of(trace: Trace) -> str:
    """Recover a trace's split group from its stable trace id.

    Lifecycle ids are leadpipe-demo:<slug>:w<n>; anchor ids are
    leadpipe-demo:walkthrough:<slug>. Both reduce to the first two
    colon-separated segments when the second names the walkthrough
    group, and to source plus lead slug otherwise; the lifecycle form
    needs its walk segment dropped, which the shape check does.
    """
    parts = trace.trace_id.split(":")
    if parts[1] == "walkthrough":
        return WALKTHROUGH_GROUP
    return ":".join(parts[:2])


def split_traces(
    traces: list[Trace], *, seed: int = DEFAULT_CORPUS_SEED
) -> dict[str, list[Trace]]:
    """Group-aware 80/10/10 split by trace, with a fixed seed.

    Groups are shuffled by the seed and cut 80/10/10 by group count;
    every trace follows its group. Returns {train, validation, test}
    lists in corpus order.
    """
    groups = sorted({group_of(trace) for trace in traces})
    shuffled = list(groups)
    random.Random(seed).shuffle(shuffled)
    n_train = int(0.8 * len(shuffled))
    n_validation = int(0.1 * len(shuffled))
    assignment: dict[str, str] = {}
    for index, group in enumerate(shuffled):
        if index < n_train:
            assignment[group] = "train"
        elif index < n_train + n_validation:
            assignment[group] = "validation"
        else:
            assignment[group] = "test"
    splits: dict[str, list[Trace]] = {name: [] for name in SPLIT_NAMES}
    for trace in traces:
        splits[assignment[group_of(trace)]].append(trace)
    return splits


def _counts_section(traces: list[Trace]) -> list[str]:
    event_count = sum(len(trace.events) for trace in traces)
    token_count = sum(len(encode_trace(trace)) for trace in traces)
    per_source = Counter(trace.source for trace in traces)
    per_vertical = Counter(trace.vertical for trace in traces)
    per_agent = Counter(trace.agent for trace in traces)
    outcomes = Counter(trace.outcome for trace in traces)
    lines = [
        "## Counts",
        "",
        f"- Traces: {len(traces)}",
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
    return lines


def corpus_card(
    traces: list[Trace],
    splits: dict[str, list[Trace]],
    *,
    seed: int = DEFAULT_CORPUS_SEED,
    walks: int = DEFAULT_WALKS_PER_LEAD,
) -> str:
    """Build the source B corpus card (Markdown) from the traces."""
    split_groups = {
        name: len({group_of(trace) for trace in splits[name]})
        for name in SPLIT_NAMES
    }
    n_lifecycle = sum(1 for t in traces if t.agent == "leadpipe-synthesiser")
    n_anchor = len(traces) - n_lifecycle
    prob_rows = " | ".join(
        f"{band} {CONTINUE_PROBABILITIES[band]:.0%}"
        for band in ("hot", "warm", "nurture", "low", "none")
    )
    general = dict(LATENCY_DISTRIBUTIONS["general"])
    latency_row = " | ".join(
        f"{general[band]:.0%}"
        for band in ("instant", "fast", "slow", "delayed", "overnight")
    )
    lines = [
        "# Corpus card: WorkLang corpus source B (LeadPipe)",
        "",
        f"This corpus is {SYNTHETIC_SENTENCE}.",
        "",
        "Source B is generated from LeadPipe's demo world: seeded "
        "lifecycle walks over the repo's own TRANSITIONS map for each "
        "of the 12 seed leads, plus the demo walkthrough executed "
        f"against the fake store as an anchor family "
        f"({SYNTHESISER_NAME} {SYNTHESISER_VERSION}).",
        "",
        *_counts_section(traces),
        "",
        "## Lifecycle walks",
        "",
        f"Each seed lead contributes {walks} walk(s): {n_lifecycle} "
        "lifecycle traces. A trace opens with the recorded beginning "
        "(capture, the seed's score when it has one, then stage "
        "events along a shortest legal TRANSITIONS path from NEW to "
        "the lead's seeded stage, preferring paths that avoid "
        "BOOKED so no unrecorded booking is implied), then a seeded "
        "random walk over TRANSITIONS from the seeded stage, at "
        f"most {MAX_FUTURE_STEPS} future transitions. The seeded "
        "next_action's follow_up fires when it comes due: at the "
        "start of the future segment when already due, after the "
        "first future transition otherwise. Entering BOOKED emits a "
        "booking event. After each future transition the lead stays "
        "engaged with the declared probability for its band "
        f"({prob_rows}); these are generator parameters, not "
        "measured behaviour. Outcomes are end-state: WON or LOST "
        "when the walk arrives there, booked when the walk stands "
        "at BOOKED at the end, stalled when contact ceases with the "
        "seeded next_action still unactioned or the walk is "
        "truncated unresolved, dropped when contact ceases with "
        "nothing pending.",
        "",
        "## Anchor family",
        "",
        f"{n_anchor} anchor traces from one execution of the demo "
        "walkthrough sequence (seed, then its nine steps, approval "
        "mode) against the fake store seeded at the fixed moment "
        "LeadPipe's own tests use, Monday 2026-11-23 09:00 UTC; the "
        "walkthrough script itself pins no seed moment. The 12 "
        "seeded leads' traces are built from their seeded audit "
        "entries only (capture, the one deduplicated capture, "
        "score); no stage transitions are recorded in the audit log "
        "and none are invented, and each closes with the outcome its "
        "post-run state names (stage WON, LOST, or BOOKED, else "
        "stalled while a next_action is pending, else dropped). The "
        "walkthrough lead's trace is recorded dispatch by dispatch, "
        "reads included, and closes booked. Note that in LeadPipe a "
        "booking does not move the stage, so booking events and "
        "stage events are separate signals in this source.",
        "",
        "## Splits",
        "",
        "Group-aware split by trace, 80/10/10 train/validation/test. "
        "A group is one seed lead across all its walks, or the whole "
        "walkthrough anchor family; no group appears in two splits. "
        "Trace counts per split follow from group sizes.",
        "",
        f"- Split seed: {seed}",
    ]
    for name in SPLIT_NAMES:
        lines.append(
            f"- {name}: {len(splits[name])} traces "
            f"({split_groups[name]} groups)"
        )
    lines += [
        "",
        "## Latency declaration",
        "",
        "Latency values in this corpus are generator-declared, not "
        "measured. Neither simulator emits inter-event timing, so "
        "the synthesiser declares the gap before each event by "
        "drawing from the declared general distribution below, with "
        "the trace's seeded generator. The first event of every "
        'trace carries latency "none" (it has no predecessor).',
        "",
        "| Vertical | instant | fast | slow | delayed | overnight |",
        "| --- | --- | --- | --- | --- | --- |",
        f"| general | {latency_row} |",
        "",
        "## Channels",
        "",
        "LeadPipe leads carry a real source, and every event of a "
        "lead's traces carries it as the channel, with one recorded "
        "exception below. (The walkthrough lead's deduplicated "
        "second capture carries instagram, the source that call "
        "actually arrived by.)",
        "",
        "## Vertical",
        "",
        'Every source B trace carries vertical "general". LeadPipe '
        "carries no vertical field, and assigning the QualBench "
        "verticals from company names would be inventing data.",
        "",
        "## Known conflicts and emitter decisions",
        "",
        "DESIGN.md enumerates the channel bucket set as the seven "
        "source values in LeadPipe's demo seed and Source enum. The "
        "Source enum in fact has nine values, and the demo seed "
        "itself uses two with no channel bucket: walk_in (Grace "
        "Liu) and other (Noah Smith). The schema cannot hold them, "
        "so both degrade to the channel stream's absence value "
        '"unknown" in this corpus; the degradation is a documented '
        "emitter decision, and whether they deserve channel tokens "
        "is a schema v2 question, recorded in DESIGN.md.",
        "",
        "## Provenance",
        "",
        f"- Generator: {SYNTHESISER_NAME} {SYNTHESISER_VERSION}, "
        f"corpus seed {seed}",
        f"- Sentence required wherever this corpus is described: "
        f"\"{SYNTHETIC_SENTENCE}\"",
        "",
    ]
    return "\n".join(lines)


def write_corpus(
    out_dir: Path,
    traces: list[Trace],
    splits: dict[str, list[Trace]],
    card: str,
) -> dict[str, Path]:
    """Write the full corpus, the split files, and the corpus card."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    def _write_jsonl(name: str, rows: list[Trace]) -> Path:
        path = out_dir / name
        path.write_text(
            "".join(trace_to_json(trace) + "\n" for trace in rows),
            encoding="utf-8",
        )
        return path

    paths["corpus"] = _write_jsonl("corpus-b.jsonl", traces)
    for name in SPLIT_NAMES:
        paths[name] = _write_jsonl(f"corpus-b-{name}.jsonl", splits[name])
    card_path = out_dir / "CORPUS_CARD.md"
    card_path.write_text(card, encoding="utf-8")
    paths["card"] = card_path
    return paths
