# Contributing to WorkLang

Thanks for considering a contribution. WorkLang's core (schema,
tokenizer, corpus, baselines) is dependency-free Python, so a first
PR needs nothing but Python 3.11 or newer. Only the mini model and
the demo sampler need PyTorch.

## Dev setup

```bash
git clone https://github.com/techaurah/worklang.git
cd worklang
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest
ruff check .
```

If pytest is green and ruff is quiet, your environment is correct.
Nothing in the test suite touches the network.

To work on the mini model or the futures demo, add the CPU build of
PyTorch (the schema, tokenizer, corpus, and baselines stay
dependency-free):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[model]"
```

## Where things live

- `src/worklang/vocab.py` and `src/worklang/vocab.json`: the
  enumerated vocabulary (69 tokens, id space reserved to 128).
  vocab.py is the source of truth; vocab.json is generated from it.
- `src/worklang/schema.py`: trace and event dataclasses with
  validation, plus JSONL record conversion.
- `src/worklang/tokenizer.py`: encode and decode. Decode is exact
  by construction, and the round trip tests guard that.
- `src/worklang/corpus_a.py`, `corpus_b.py`, `corpus_v0.py`: the
  two corpus sources and the frozen v0 merge.
- `src/worklang/baselines.py` and `evaluation.py`: the majority and
  order-2 n-gram baselines, and the harness that scores every
  system on the same positions.
- `src/worklang/mini.py`: worklang-mini: training, scoring, and
  the futures calibration measurement.
- `src/worklang/demo.py` and `demo/`: the futures demo page, its
  precomputed data, and the trained weights it sampled from.
- `scripts/`: the corpus builders, the vocabulary builder,
  baseline evaluation, training, and futures sampling.
- `tests/`: vocabulary, schema, round trip, emitter, synthesiser,
  corpus split and determinism, baseline, evaluation, mini, and
  demo tests.

## Good first contributions

- A converter that brings a new trace format into the schema, with
  a fixture and a round trip test (ROADMAP.md gives the planned
  order: CRM CSV, then chat log exports, then n8n execution
  exports).
- Tests for a documented edge case in the tokenizer, schema
  validation, or the eval harness.
- A rerun of a results script whose output differs from the
  committed artefact, reported as an issue with both numbers.
- Documentation fixes, especially where a document and an artefact
  disagree.

## Rules

- Open an issue before any scope change. Corpus v0 is frozen:
  changing it means a new corpus version, re-measured baselines,
  and new results, and that is a maintainer decision.
- Every PR links an issue (`Closes #...`) and includes test
  evidence: the command you ran and its result.
- No invented statistics. Every number in a document must be
  produced by a script from a real run and must name the artefact
  that records it (a results file, a corpus card, the model card).
- Never commit real customer data, credentials, or a real `.env`.
  Corpus material is synthetic, or real traces collected with
  consent and anonymised before tokenisation. The schema drops
  names, contact details, message text, and exact timestamps by
  design; do not add a way to carry them.
- Generated artefacts are regenerated, never hand edited:
  vocab.json comes from `scripts/build_vocab.py`, the corpora from
  the `scripts/build_corpus_*.py` scripts, and the demo data from
  `scripts/sample_futures.py`.
- Keep the tokenizer deterministic: the same event must always
  encode to the same tokens, and decode must stay exact.
- Style: `ruff check` must pass. CI runs ruff and pytest on
  Python 3.11 and 3.12.
- Do not describe WorkLang as the first foundation model for event
  logs. Process-FM is prior art for generic process logs;
  WorkLang's ground is agent executions and business outcome
  futures.

## Response rhythm

Issues are triaged weekly at minimum. PRs get a first human response
within a few days. If a claimed issue goes quiet for two weeks, we will
unassign it kindly so someone else can pick it up.
