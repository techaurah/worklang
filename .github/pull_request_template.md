## What changed and why

<!-- One short paragraph. Link the issue this addresses. -->

Closes #

## How it was tested

<!-- The exact command(s) you ran and the result. "Tested locally"
is not evidence. -->

- [ ] `python -m pytest` passes
- [ ] `ruff check` passes

## Checklist

- [ ] Every number quoted comes from a script run and names its artefact (results file, corpus card, model card)
- [ ] No real customer data, credentials, or real `.env` added; corpus material is synthetic, or consented and anonymised
- [ ] Generated artefacts (vocab.json, corpora, demo data) regenerated with their scripts, not hand edited
- [ ] Tokenizer stays deterministic and decode stays exact (round trip tests green)
- [ ] CHANGELOG.md updated under Unreleased (for user-visible changes)
