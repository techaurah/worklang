# Roadmap: WorkLang

Where WorkLang is going after v0.1.0. This is a plan, not a
promise: items land when they are built, tested, and measured, in
roughly this order. Every item below comes from the v0 technical
report's own list of what v1 needs.

## Corpus first

- **Real, consented, anonymised traces.** Corpus v0 is 100
  percent synthetic, so the mini has learned the simulators'
  habits, not real business behaviour, and no futures claim gets
  stronger until the corpus does. V1's first requirement is a
  corpus of real agent executions collected with consent,
  anonymised before tokenisation (the schema already has no field
  that can hold names, emails, or phone numbers), with real
  inter-event times so the latency stream carries measured signal
  or is dropped.
- **A larger corpus, and a re-earned bar.** A bigger corpus
  changes the baseline question too: the n-gram bar must be
  re-measured on any new corpus, and the mini's results re-earned
  against it.
- **Converters, in order:** CRM CSV first, then chat log exports,
  then n8n execution exports, each with a fixture and a round
  trip test, so real traces can enter the schema without hand
  editing.
- **A third corpus source:** the course capstone trace emitter.
  The schema's source field already has a value waiting for it.

## Schema v2 decisions

- Stage tokens for the two stage strings the QualBench agents
  report that v0 cannot represent (HANDOFF, UNQUALIFIED). V0
  records them as conflicts rather than normalising them
  silently.
- Channel buckets for the LeadPipe source values walk_in and
  other, which have no bucket in v0.
- An UNK policy for tool names beyond the reserved id space, if
  real tooling outgrows it.

## Model scaling study

- Before any larger rung is built, measure how the results move
  with corpus size and model size under the same held-out
  protocol, and publish the measurements either way. The outcome
  decides whether a bigger model is the right next step, or the
  corpus is.

## Deliberately held back

- **worklang-base** (the larger rung of the ladder), the
  **vertical fine-tunes** (med spa first), and the **hosted live
  futures endpoint** are the production engine. They are named
  here rather than pretending the ladder ends at the mini, and
  none of them is scheduled.

## How priorities are set

Priorities are set from GitHub issues: open an issue for
concrete, actionable work. Items that unblock the real-corpus
step move first, because until then every number is measured
against a synthetic corpus.
