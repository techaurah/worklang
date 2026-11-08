# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

Nothing yet.

## [0.1.0] - 2026-11-07

First public release: the complete open v0 artefact set described
in DESIGN.md and measured in TECHNICAL_REPORT.md.

### Added

- Trace and event schema (`src/worklang/schema.py`) with
  validation and JSONL record conversion.
- Deterministic, rule-defined tokenizer: one event encodes as one
  composite position of 7 tokens, decode is exact by construction,
  and the vocabulary is enumerated at 69 tokens with the id space
  reserved to 128 for growth.
- Corpus v0, frozen: 409 synthetic traces (4,637 events, 33,686
  tokens) merged from two simulator sources, QualBench agent runs
  (source A) and seeded LeadPipe lifecycle walks (source B), with
  a corpus card for each source and for the merge. The corpus is
  100 percent synthetic and contains no real customer data.
- Baselines: a majority constant and an order-2 n-gram, plus the
  evaluation harness that scores every system on the same
  held-out positions.
- worklang-mini, a 5,246,523 parameter model trained on corpus
  v0. On the held-out test split (46 traces, 544 scored
  positions) it scores family accuracy 0.8511 against the
  n-gram's 0.7004 and full-composite exact match 0.2537 against
  0.2059. On outcome_class the n-gram stays ahead, 0.9430 to
  0.8952; that exception is recorded in MODEL_CARD.md, not
  hidden.
- Futures sampling and the demo (`demo/index.html`): an authored
  after-hours med spa prefix, sampled with and without an instant
  automated reply, with the synthetic-data caveat on the page at
  the same visual weight as the numbers.
- TECHNICAL_REPORT.md (the full v0 story, including limitations
  and calibration, pooled total variation 0.2262), MODEL_CARD.md,
  DESIGN.md, and RESEARCH.md.
