# WorkLang: Design

Flagship 2 from the Master Playbook (breakout-research/MASTER-PLAYBOOK.md).
Pattern source: Kronos (treat a domain as a language, tokenise it, train a
small autoregressive model, sample futures). Prior art and competitor scan:
RESEARCH.md in this folder. The scan's hard rule binds this design: no
"first foundation model for event logs" claim, ever. Process-FM got there
for generic process logs. WorkLang's ground is agent executions and
business outcome futures.

Honesty standards, same as Flagship 1: no invented statistics, estimates
labelled as estimates, every published number produced by a script from a
real run, no em dashes.

## The one-line idea

WorkLang is a small foundation model for the language of agent workflows:
it tokenises agent executions into a fixed semantic alphabet, learns to
predict what happens next, and answers questions about a live pipeline by
sampling its futures (this lead books, stalls, or drops, and with what
share of sampled trajectories behind each outcome).

## The tokenizer

One event becomes one composite position of 7 tokens in a fixed order:
family first (the coarse subtoken, Kronos s1 position), then the fine
attribute subtokens. Unlike Kronos, nothing here is learned: events are
already discrete, so the tokenizer is a deterministic rule set, versioned
with the schema, and decode is exact by construction.

Event families (coarse subtoken), grounded in what our simulators emit:
1. enquiry: an inbound contact or user turn (QualBench transcript user
   turns; LeadPipe capture_lead).
2. reply: an agent outbound message (QualBench transcript agent turns).
3. tool_call: a named tool invocation and its result class (QualBench
   ToolExecutor calls; LeadPipe tool calls).
4. booking: an appointment created, confirmed, or changed (QualBench
   calendar.book outcomes; LeadPipe book_appointment).
5. follow_up: a scheduled chase fires (LeadPipe next_action coming due;
   QualBench no-show and recall scenarios).
6. handoff: control passes to a human (QualBench handoff flag).
7. stage: a pipeline stage transition is recorded (LeadPipe update_stage
   and its TRANSITIONS map; QualBench final stage reporting).
8. outcome: the trace terminates (exactly one per trace, carrying the
   outcome class below).

Fine subtokens (fixed bucket sets, every stream has an explicit none or
unknown value so absence is data, not silence):
- channel (9): website, instagram, facebook, whatsapp, email, call,
  referral (the seven source values in LeadPipe's demo seed and Source
  enum), plus chat and unknown.
- score_band (5): hot, warm, nurture, low, none. Bands use LeadPipe's own
  thresholds from scoring.py: HOT 80 to 100, WARM 65 to 79, NURTURE 50 to
  64, below 50 LOW. QualBench scores are floats on the same 0 to 100
  scale and map through the same function.
- latency_band (6): instant (under 1 minute), fast (1 to 15 minutes),
  slow (15 minutes to 2 hours), delayed (2 to 24 hours), overnight (over
  24 hours or across a business-hours boundary), none. See the conflicts
  section: in corpus v0 these values are declared by the generator, not
  measured, and the corpus card says so.
- stage (11): the ten LeadPipe stages (NEW, CONTACTED, QUALIFIED,
  NURTURE, DISCOVERY, BOOKED, PROPOSAL, NEGOTIATION, WON, LOST) plus
  none. This is the stage after the event.
- tool (14): the union of tool names the two simulators can emit.
  QualBench executor: calendar.check_availability, calendar.book,
  crm.upsert_lead, crm.get_lead, report_outcome (5). LeadPipe public
  tools: capture_lead, get_lead, list_hot_leads, score_lead,
  update_stage, check_availability, book_appointment,
  daily_pipeline_summary (8). Plus none. Names stay source-faithful even
  where two names mean similar things; merging them is a v2 schema
  decision, not a silent normalisation.
- outcome_class (6): booked, won, lost, stalled, dropped, none. Only
  meaningful on the outcome family; stalled means the trace ends with a
  pending next action unactioned, dropped means contact ceased with no
  next action. The distinction matters because the futures surface
  reports them separately.

Special and trace-level tokens: PAD, BOS, EOS, UNK (4), plus a vertical
conditioning token emitted once after BOS: real-estate, med-spa,
home-services, dental, agencies, general (6).

Vocabulary size estimate: 4 + 6 + 8 + 9 + 5 + 6 + 11 + 14 + 6 = 69
tokens. This is an estimate only in the sense that the tool stream can
grow; it is derived by enumerating the bucket sets above, and because the
vocabulary is rule defined (not learned), it becomes exact the day the
tokenizer ships. The schema reserves id space up to 128 for growth
without breaking saved corpora.

Round trip requirement: for every schema-valid trace, decode(encode(
trace)) equals the trace exactly, at the event level. Encoding is lossy
by design at the raw-record level, and the lossy fields are named, not
hidden: message text, names, contact details, exact timestamps, and any
attribute outside the bucket sets are dropped. A property test runs the
round trip over 100 percent of the generated corpus, not a sample.

## Trace schema

One JSON object per trace (JSONL corpus files), keeping the classical
event log skeleton (a case is a trace; an event has an activity and a
position in the case) and adding agent-specific fields:

- trace_id: stable id including source and scenario or lead slug.
- source: qualbench, leadpipe-demo, capstone, or converter name.
- synthetic: boolean. Always true in corpus v0, and the field is
  mandatory so no future real trace can silently mix in.
- vertical: one of the six vertical tokens.
- agent: which agent produced it (reference-agent, naive-agent, or the
  generator name), with agent_version.
- events: ordered list of {seq, family, channel, score_band,
  latency_band, stage, tool, outcome_class}. All seven attribute fields
  present on every event (none and unknown are values, not omissions).
- outcome: the terminal outcome_class, duplicated at trace level for
  cheap filtering.
- generator: {name, version, seed} for full reproducibility.

Grounding, from reading the two repos on 2026-11-04:
- QualBench runner.RunResult already holds everything an emitter needs:
  the transcript (user and agent turns), tool executions via
  ToolExecutor (the 5 tools above), calendar bookings, and final stage,
  score, fields, and handoff. The published results JSONs do NOT contain
  traces (they carry per-scenario assertion outcomes only), so corpus
  generation needs a new emitter in the runner, not a reformat of
  existing files. No agent contract changes: the emitter reads RunResult
  after a run.
- LeadPipe's demo world is a snapshot, not a sequence: 12 seeded leads
  with stage, source, score components and band, next_action and relative
  timing, plus audit entries for seeded captures (one deduplicated). Its
  executable sequence is examples/demo_walkthrough.py, which runs real
  tool calls against the fake store. Lifecycle traces are therefore
  synthesised by walking the repo's own TRANSITIONS map (every step a
  legal transition by construction), seeded deterministically, and the
  walkthrough run is included as an anchor trace family.

## Corpus plan (v0)

Generation method, exact:
1. QualBench source: run all 50 scenarios against both shipped agents
   (reference-agent and naive-agent) through the new trace emitter.
   Variants come from seeded parameter perturbation of scenario inputs
   the loader already supports (names, budgets, requested times, seed
   calendar busy blocks), never from editing expected outcomes. Each
   variant run is a real run through the unmodified runner and agents.
2. LeadPipe source: for each of the 12 seed leads, generate lifecycle
   traces by seeded random walks over TRANSITIONS from the lead's seeded
   stage, emitting capture, score, stage, follow_up, booking and outcome
   events consistent with the lead's band and next_action; plus the demo
   walkthrough executed and emitted.
3. Splits: by trace, 80/10/10 train/validation/test with a fixed seed
   published in the corpus card. Variants of one scenario never straddle
   the split (they share a group id), so the test set measures
   generalisation to unseen scenarios and leads, not memorised variants.
4. Corpus card: generated by script at build time, stating trace count,
   event count, token count, per-source and per-vertical counts, outcome
   distribution, the latency declaration (below), the split seed, and
   the sentence "100 percent synthetic, generated by TechAura's own
   simulators, contains no real customer data" wherever the corpus is
   described.

No corpus size is claimed in advance. The card publishes the real counts
when generation runs; the design only fixes the method.

## The mini model

Scope, deliberately small (anchors from RESEARCH.md: Process-FM's
smallest released suite model is 6.32M params; Kronos-mini is 4.1M):
- Decoder-only transformer, PyTorch, d_model 256, 6 layers, 8 heads,
  context 256 events (1792 tokens). Parameter count estimate: 5 to 6M,
  dominated by the transformer blocks; the embedding table is negligible
  at a 69 token vocabulary. Marked as an estimate until the model code
  prints the real count, which the results file must quote.
- CPU only, by design: trains on this class of VM and samples futures on
  a laptop. If it needs a GPU to be interesting, the scope is wrong.
- Objective: causal next-event prediction. At each position the model
  predicts the family stream first, then each attribute stream
  conditioned on the family (Kronos coarse-to-fine order, applied within
  the position). Loss is the sum of per-stream cross-entropies. No
  masked modelling, no auxiliary heads in v0.
- Training data: corpus v0 prefixes at every length (the standard PPM
  prefix extraction), train split only.

Evaluation, all numbers from an eval script into a results JSON:
- Held-out next-event accuracy on the test split: family accuracy,
  per-stream attribute accuracy, and full-composite exact match (all 7
  tokens of the next event correct).
- Baseline 1, majority class: always predict the most frequent family
  (and most frequent value per attribute stream) from the train split.
- Baseline 2, n-gram: order-2 over composite event symbols with backoff
  to order-1 and unigram, trained on the same train split. This is a
  strong baseline on small synthetic corpora and it may win. If it wins,
  the results file and the technical report say so, and the mini does
  not ship as a model claim until it beats it or the corpus grows.
- Futures calibration: for test prefixes, sample futures and compare the
  aggregated outcome distribution against the empirical outcome
  distribution of held-out traces sharing the prefix group; report total
  variation distance. A futures surface that is not calibrated is a
  random story generator, so this metric gates the demo.

## The futures demo: after-hours missed lead

The flow, grounded in scenarios that already exist (med-spa pack
after-hours-booking; real-estate after-hours-inquiry): a lead messages a
med spa at 21:47 on Instagram, outside business hours. Nobody replies
until morning. What usually happens next?

- Input: an authored prefix trace (labelled as an authored example, not
  corpus data): BOS, vertical med-spa, enquiry (channel instagram,
  stage NEW, latency none), then nothing. The demo also offers the
  counterfactual prefix with an instant automated reply event inserted.
- Method: sample N trajectories from the mini at a fixed temperature
  printed on the page (N and temperature are page configuration, stated
  on the page, not hidden defaults), run each to its outcome event or
  the context limit, aggregate outcome classes.
- Output: the distribution over booked, stalled, dropped (plus other
  outcome classes if they sample above zero), shown as percentages of
  sampled trajectories, the with-reply and without-reply distributions
  side by side, three decoded sample trajectories written out as plain
  event lines, and the standing caveat in the same visual weight as the
  numbers: trained on synthetic data from our own simulators, shares are
  model behaviour, not measured business statistics.
- Form at launch: a static, zero-install page with precomputed outputs
  regenerated on every model version change. A live endpoint where
  visitors paste their own prefix is a later, separately reviewed step.

## Artefacts to build

1. worklang schema and tokenizer package (encode, decode, vocab.json,
   round trip property tests).
2. QualBench trace emitter (reads RunResult, no agent contract changes)
   plus corpus generation script and corpus card.
3. LeadPipe lifecycle synthesiser (TRANSITIONS walks) and walkthrough
   emitter.
4. Converters, in playbook order: CRM CSV first (it matches the trace
   schema most directly), then chat log export, then n8n execution
   export. Each converter ships with a fixture and a round trip test,
   and marks its output synthetic: false only when the input is a real
   export, with an anonymisation pass (names, emails, phones dropped
   before tokenisation, since the schema has no field that can hold
   them).
5. Baselines (majority, n-gram) and the eval harness, before any neural
   code, so the bar exists before the model does.
6. worklang-mini model, training script, results JSON, model card.
7. Futures sampler and the after-hours demo page (precomputed).
8. Technical report at launch (method, corpus card, eval vs baselines,
   calibration, limitations), GitHub plus Hugging Face release, MIT.

## Build order

1. Schema and tokenizer, with round trip property tests green.
2. QualBench emitter and corpus source A; corpus card prints real stats.
3. LeadPipe synthesiser and corpus source B; merged corpus v0 frozen
   with its card.
4. Baselines evaluated and recorded. This is the bar.
5. Mini trained and evaluated against step 4, calibration measured.
   Publish the results file either way.
6. Futures sampler and demo page (precomputed), technical report,
   release when Tech says go. Med-spa fine-tune is the second beat,
   after launch, per the playbook.

## Risks and the held-back tier

Risks, plainly:
- Synthetic-only corpus: the mini learns our simulators' habits, not
  real business behaviour. Mitigation is honesty plus roadmap: v0 is a
  pipeline proof (tokenizer, corpus, eval, futures surface), and real
  consented anonymised traces are the documented next corpus, never a
  silent swap.
- The n-gram baseline may beat the transformer at this corpus size.
  That is a publishable result, not a failure to hide.
- Latency is the weakest attribute: declared, not measured, in v0 (see
  conflicts). Any future real corpus must carry real inter-event times
  or drop the stream.
- The fixed small vocabulary is the core bet. If real agent tooling
  produces tool names far beyond the reserved id space, the tool stream
  needs a versioned schema v2 with an UNK policy, which the schema
  already anticipates.
- Prior art moves fast: Process-FM exists today; a platform vendor
  could ship agent-trace modelling. The moat is the corpus, the
  converters, and the futures surface, in that order.

Held back, Kronos style (their large tier was never open sourced): the
open release is the tokenizer, schema, converters, corpus, mini weights
and eval harness. Held back: worklang-base (the larger rung of the
ladder), the vertical fine-tunes (med-spa first), and the hosted live
futures endpoint. Those are the AaaS production engine, and the README
says they are held back and why, rather than pretending the ladder ends
at mini.

## Conflicts between the playbook spec and what the simulators emit

Found during grounding on 2026-11-04, recorded rather than smoothed
over:

1. Latency band has no source data. Neither simulator emits inter-event
   timing: the QualBench runner records no timestamps or durations, and
   scenario turns carry no machine-readable times (a few state a clock
   time in prose). LeadPipe generates lead timestamps relative to a
   supplied now, but no latencies between events. Resolution: corpus v0
   latency values are drawn by the generator from per-vertical
   distributions stated in the corpus card and marked declared, never
   measured. The after-hours demo only relies on declared clock context
   (business hours and message time), which the scenarios do carry.
2. The playbook lists the course capstone world as a corpus source. No
   runnable capstone trace emitter exists yet. Resolution: corpus v0 is
   QualBench plus LeadPipe only; the capstone joins as a corpus v1
   source when it exists, and the schema's source field already has a
   value waiting for it.
3. QualBench channels are undeclared. Scenarios carry pack and vertical
   context but no channel; LeadPipe leads carry a source. Resolution:
   the emitter takes channel from an optional scenario field, defaulting
   to unknown rather than inventing a channel per scenario. Demo prefix
   traces are authored examples and may set a channel explicitly.

## Progress log
- 2026-11-04: Design phase. RESEARCH.md (competitor and prior-art scan:
  Kronos as pattern source, Process-FM and Process-Graph-FM as direct
  prior art, classical PPM, LLM-on-logs work, transaction foundation
  models, process mining platforms) and this DESIGN.md written. Trace
  schema grounded in the QualBench runner (RunResult, ToolExecutor) and
  the LeadPipe demo world (12 seed leads, TRANSITIONS map, walkthrough)
  by reading both repos. Three spec conflicts found and recorded above
  (latency has no source data, capstone emitter does not exist yet,
  QualBench channels undeclared). No code written, nothing published.
  Next: step 1, schema and tokenizer with round trip tests, when Tech
  says go.
- 2026-11-04: Build step 1 done, local only, nothing published. Repo
  root created at flagships/worklang/repo/ (src layout, pyproject.toml,
  README stub that states plainly this is a work in progress build with
  no launch claims). Delivered: src/worklang/vocab.py (the enumerated
  vocabulary: 69 tokens, ids 0 to 68 contiguous, id space reserved to
  128; src/worklang/vocab.json generated from the constants by
  scripts/build_vocab.py), src/worklang/schema.py (Event, Generator,
  Trace dataclasses with validation, plus dict and JSON conversion),
  src/worklang/tokenizer.py (encode and decode for events and traces,
  plus score_to_band using the LeadPipe thresholds). Schema rules
  written as checks from the design text: all seven attribute fields
  present on every event, seq equals the 0-based position, exactly one
  outcome event and it is last, outcome_class is none everywhere except
  on the outcome event, and the trace level outcome duplicates the
  terminal outcome_class. The lossy boundary is asserted in tests, not
  hidden: from_dict drops non-schema keys, which is where message text,
  names, contact details, and exact timestamps die. Tests: 49 passed
  (12 vocabulary, 13 schema, 24 tokenizer), including an exact round
  trip over a generated matrix of 112 events covering every family and
  every bucket value in the vocabulary, and a vocabulary enumeration
  test that rebuilds the vocab from the source constants and asserts
  the count is 69. Ruff: all checks passed. One interpretation flagged
  for review, not treated as settled: the design writes generator as
  an explicit {name, version, seed} object but describes agent with a
  companion agent_version field, so the schema uses two flat trace
  level fields, agent and agent_version; if a nested agent object was
  intended, it is a small schema change. Next: step 2, QualBench trace
  emitter and corpus source A, when Tech says go.
- 2026-11-04: Build step 2 done, local only, nothing published.
  Delivered in the worklang repo: src/worklang/qualbench_emitter.py,
  src/worklang/corpus_a.py, scripts/build_corpus_a.py, generated
  corpus under corpus/source-a/ (corpus-a.jsonl plus train,
  validation, and test split files and CORPUS_CARD.md), and 14 new
  tests. Consumption route: the QualBench Python API only
  (discover_packs, run_scenario, load_agent_file); the QualBench repo
  is untouched and no agent contract changed. RunResult does not
  retain ordered tool executions, so the emitter wraps the agent
  session in a recording proxy implementing the same start/send
  protocol and replays payloads and responses in the runner's own
  order against the final RunResult state; CLI outputs were not used
  because the results JSONs carry assertion outcomes only. Corpus
  source A, real counts from the generation run (seed 20261104, 3
  variants per scenario per agent): 300 traces, 3,902 events, 28,214
  tokens; per vertical 60 traces each (agencies, dental,
  home-services, med-spa, real-estate); per agent 150 reference-agent
  and 150 naive-agent; outcomes booked 151, stalled 140, lost 9, won
  0, dropped 0 (both shipped agents always report an outcome, so
  dropped never occurs in this source; that is a property of the
  runs, stated rather than smoothed over). Group-aware split:
  240 train, 30 validation, 30 test traces (40, 5, and 5 scenario
  groups), verified disjoint by group id on the written files.
  Round trip is exact over 100 percent of the generated corpus
  (300 of 300 traces), as the design requires. Latency bands are
  generator-declared from the per-vertical distributions printed in
  the corpus card, never measured; the first event of each trace
  carries none. Channels are unknown on every event: no scenario
  declares one and none was invented. Variants perturb only the seed
  calendar busy blocks (one seeded whole-hour offset, layout
  preserved); names, budgets, and requested times live inside prose
  turns the loader exposes only as opaque strings, so perturbing
  them would edit scenario content rather than an input, and v0 does
  not do it. One new conflict found and recorded, not smoothed over:
  QualBench agents report stages HANDOFF and UNQUALIFIED, which the
  enumerated stage vocabulary (ten LeadPipe stages plus none) does
  not contain. Following the design's no-silent-normalisation rule
  for tool names, neither enters the stage stream: HANDOFF coincides
  with the handoff flag and surfaces as handoff events, and a run
  whose final stage is UNQUALIFIED closes with outcome lost (an
  active disqualification, not a stall or a drop). Whether the two
  deserve stage tokens is a schema v2 question. Tests: 63 passed
  (49 step 1 plus 14 new: emitter tests over 6 scenarios covering
  all 5 packs including 2 safety scenarios, split no-straddle,
  perturbation non-mutation, determinism by full rebuild, and card
  contents). Ruff: all checks passed. Next: step 3, LeadPipe
  synthesiser and corpus source B, when Tech says go.
- 2026-11-05: Build step 3 done, local only, nothing published.
  The LeadPipe synthesiser (src/worklang/leadpipe_synthesiser.py,
  consuming LeadPipe through its Python API only; the LeadPipe repo
  is unmodified) turns the demo snapshot into lifecycle traces, and
  corpus_b.py plus corpus_v0.py freeze source B and the merged
  corpus v0. Lifecycle traces: 8 seeded walks for each of the 12
  seed leads, 96 traces. Each opens with the recorded beginning
  (capture, the seed's score when it has one, then stage events
  along a shortest legal TRANSITIONS path from NEW to the lead's
  seeded stage, preferring paths that avoid BOOKED so no unrecorded
  booking is implied), then a seeded random walk over TRANSITIONS
  from the seeded stage, at most 8 future transitions, every step
  legal by construction. The seeded next_action's follow_up fires
  when it comes due (at the start of the future segment when
  already due, after the first future transition otherwise);
  entering BOOKED emits a booking event; after each future
  transition the lead stays engaged with declared per-band
  probabilities (hot 85, warm 75, nurture 60, low 45, none 60
  percent; generator parameters, printed in the card). Outcomes
  are end-state (WON, LOST, standing at BOOKED, stalled when
  contact ceases with the seeded action unactioned or the walk is
  truncated unresolved, dropped otherwise), deliberately different
  from source A because a LeadPipe lifecycle continues past
  BOOKED toward PROPOSAL and WON. Anchor family: the demo
  walkthrough executed once against the fake store seeded at the
  fixed moment LeadPipe's own tests use, Monday 2026-11-23 09:00
  UTC (the walkthrough script pins no seed moment; pinning it is
  an emitter decision), 13 traces: the 12 seeded leads built from
  their seeded audit entries only (capture, the one deduplicated
  capture, score; no stage transitions are recorded there and none
  are invented), plus the walkthrough lead recorded dispatch by
  dispatch, reads included, closing booked. In LeadPipe a booking
  does not move the stage, so booking and stage events are
  separate signals in this source. Source B carries vertical
  general on every trace (LeadPipe has no vertical field; inferring
  the QualBench verticals from company names would be inventing
  data) and generator-declared latency from the general row of
  source A's declared distributions. Source B totals: 109 traces,
  735 events, 5,472 tokens; outcomes booked 6, dropped 26, lost 55,
  stalled 10, won 12. Group-aware split with the corpus seed
  20261104: 85 train, 8 validation, 16 test traces (10, 1, and 2
  of the 13 groups: 12 seed leads plus the walkthrough family).
  One new conflict found and recorded, not smoothed over: this
  design enumerates the channel bucket set as the seven source
  values in LeadPipe's demo seed and Source enum, but the Source
  enum has nine values. Seven are channel buckets verbatim
  (website, instagram, facebook, whatsapp, email, referral, call);
  the seed itself uses the other two, walk_in (Grace Liu) and
  other (Noah Smith), which have no bucket and degrade to the
  channel stream's absence value unknown, following step 2's
  precedent for out-of-vocabulary values. Schema v2 question,
  alongside HANDOFF and UNQUALIFIED. Corpus v0, the frozen merge
  in corpus/v0/: 409 traces, 4,637 events, 33,686 tokens; 300
  qualbench plus 109 leadpipe-demo; combined outcomes booked 157,
  stalled 150, lost 64, dropped 26, won 12; merged splits 325
  train (50 groups), 38 validation (6 groups), 46 test (7
  groups). The merge unions the per-source split files and
  verifies, rather than assumes, that no group id appears in more
  than one merged split and that every trace of both corpora
  appears exactly once; both failure modes are tested. A full
  rebuild of source B and v0 from the seeds is byte-identical to
  the committed files. Tests: 90 passed (63 before step 3 plus 27
  new: TRANSITIONS legality over every generated walk, band and
  channel consistency per lead, follow_up firing for due-action
  leads, anchor family shape and the walkthrough lead's exact
  dispatch sequence, validate-and-tokenize over 100 percent of
  source B, split no-straddle for B and for the merged corpus
  across both sources, determinism by genuine rebuild). Ruff: all
  checks passed. Next: step 4, baselines, when Tech says go.
- 2026-11-05: Build step 4 done, local only, nothing published.
  Delivered in the worklang repo: src/worklang/baselines.py
  (MajorityBaseline and NGramBaseline), src/worklang/evaluation.py
  (the harness: prefix extraction at every length including the
  empty prefix, per-stream and composite scoring, corpus evaluation
  and the results writer), scripts/evaluate_baselines.py, and the
  results document it produced, results/baselines-v0.json. Both
  baselines are fitted on the train split only and scored on the
  test split only; the harness reads exactly those two files and
  never touches the validation split, which a test proves with a
  poisoned validation file whose contents would change every
  headline number if they leaked into fitting. Protocol on corpus
  v0: 325 train traces, 46 test traces, 544 scored positions (every
  event position of every test trace). Real numbers from the run.
  Majority class: family 0.2794, channel 0.7537, score_band 0.3732,
  latency_band 0.2868, stage 0.4320, tool 0.6710, outcome_class
  0.9154, full-composite exact match 0.0386 (21 of 544). Order-2
  n-gram with backoff: family 0.7004, channel 0.8051, score_band
  0.7757, latency_band 0.2684, stage 0.7463, tool 0.7518,
  outcome_class 0.9430, full-composite exact match 0.2059 (112 of
  544). Stated plainly, as the design requires: the n-gram is a
  strong bar, not a straw man. It beats majority on every stream
  except one and multiplies its composite score by more than five,
  so the step 5 mini has to clear family 0.7004 and composite
  0.2059 on this split, and if it does not, the results file says
  so and the mini does not ship as a model claim. The one exception
  is latency_band, where the n-gram (0.2684) is worse than the
  majority constant (0.2868) and both are poor in absolute terms;
  that is the design's known weak stream behaving as warned, since
  v0 latency is generator-declared per vertical and carries almost
  no signal from the event history for a sequence model to exploit.
  Tests: 103 passed (90 before step 4 plus 13 new: hand-counted
  majority and n-gram fixtures including both backoff paths and a
  memorised sequence, hand-scored harness arithmetic, the
  split-discipline file test, and byte-identical determinism of
  the results file across re-runs, the script included). Ruff:
  all checks passed. Next: step 5, the mini model, when Tech
  says go.
- 2026-11-05: Build step 5 done, local only, nothing published.
  Delivered in the worklang repo: src/worklang/mini.py (the
  worklang-mini model, training, scoring, and futures calibration),
  scripts/train_mini.py, the results document it produced,
  results/mini-v0.json, MODEL_CARD.md at the repo root, and
  tests/test_mini.py. The model is the design's decoder-only
  transformer (d_model 256, 6 layers, 8 heads, context 256 events,
  embedding over the reserved 128 row space, family head first with
  attribute heads conditioned on the family, loss the sum of
  per-stream cross-entropies). The real parameter count, counted
  from the built module, is 5,246,523, inside the design's 5 to 6M
  estimate. Training ran at the script's defaults, detached,
  because a full run is far longer than any foreground exec window
  on this VM: CPU only (PyTorch 2.15.0+cpu, 4 threads), 48 epochs,
  learning rate 0.003, batch size 32, weight decay 0.01, seed
  20261104, train split only (325 traces). Loss fell from 11.7235
  in the first epoch to 1.8095 in the final epoch. The script ran
  the whole train, score, and calibrate sequence twice with the
  same seed (wall times 2758.6 and 2415.8 seconds, about 86 minutes
  in total) and the two runs agree exactly: the largest absolute
  delta across all twelve compared metrics is 0.0. Real numbers
  from the run, scored through the step 4 harness on the same 544
  test positions. Mini: family 0.8511, channel 0.9154, score_band
  0.9522, latency_band 0.3438, stage 0.8070, tool 0.8713,
  outcome_class 0.8952, full-composite exact match 0.2537 (138 of
  544). The bar is cleared on both gated metrics: family 0.8511
  against the n-gram's 0.7004, composite 0.2537 against 0.2059, so
  the mini ships as a model claim for corpus v0 and the model card
  says so in its first paragraph. Stated plainly, as the design
  requires: the mini beats the n-gram on six of seven streams, and
  the exception is outcome_class, where the n-gram (0.9430) and
  even the majority constant (0.9154) beat the mini (0.8952),
  because outcome tokens are mostly "none" away from terminal
  positions and the constant predictors exploit that base rate
  harder. latency_band stays the weakest stream for every system,
  the mini's 0.3438 included, which is the declared-not-measured
  latency behaving as warned. Futures calibration, 32 samples per
  prefix at temperature 1.0 over the 7 test groups: mean group
  total variation 0.2331, pooled total variation 0.2262, 1,472
  futures sampled and all 1,472 completed, none truncated. The
  pooled sampled distribution (booked 48.0 percent, lost 29.0,
  stalled 19.2, won 2.0, dropped 1.9) sits visibly off the pooled
  empirical distribution of the 46 test traces (booked 37.0,
  stalled 30.4, lost 17.4, dropped 8.7, won 6.5): the mini
  over-samples booked and lost and under-samples stalled, dropped,
  and won, and 0.2262 is recorded as a measured gap, not a
  calibrated surface. Tests: 110 passed (103 before step 5 plus 7
  new mini tests); the full suite needs the sibling checkouts'
  dependencies (pydantic, PyYAML) importable alongside torch in
  the venv, as in step 4. Ruff: all checks passed. Next: step 6,
  futures sampler and demo page, technical report, release when
  Tech says go.
- 2026-11-05: Build step 6 done, local only, nothing published.
  This completes the build order: all six steps are built. The
  release (GitHub plus Hugging Face, MIT) has NOT happened; it
  waits for Tech's go, and nothing in this step pre-empts it.
  Delivered in the worklang repo: src/worklang/demo.py (authored
  prefixes, sampling configuration, weights handling, sampler,
  page renderer), scripts/sample_futures.py, demo/demo-data.json
  and demo/index.html (rendered from demo/index.template.html,
  a single self-contained page: no external requests, no
  tracking), demo/worklang-mini-v0.pt (the trained weights),
  TECHNICAL_REPORT.md at the repo root, a refreshed README.md,
  and tests/test_demo.py. One gap found and closed on the record:
  step 5 proved training bit-for-bit reproducible but never saved
  a weights file, so the sampler trains the mini with exactly
  the step 5 configuration (TrainConfig defaults, corpus v0
  train split) and saves the weights before sampling; the
  counted parameter total of the saved model is 5,246,523,
  matching results/mini-v0.json. The demo samples the design's
  scenario (a lead messages a med spa at 21:47 on Instagram;
  authored prefixes, labelled as authored on the page): 200
  futures per prefix at temperature 1.0, generation cap 64 new
  events, seed 20261105, configuration printed on the page and
  recorded in the data file with the full per-sample outcome
  list and three decoded example trajectories per prefix. Real
  numbers from the run. Without reply: 200 completed, 0
  truncated; booked 39.5 percent (79 of 200), stalled 57.0
  (114), lost 3.5 (7), dropped 0, won 0. With an instant
  automated reply: 200 completed, 0 truncated; booked 32.0
  (64), stalled 60.0 (120), lost 7.0 (14), dropped 0.5 (1),
  won 0.5 (1). Stated plainly in the technical report, as the
  honesty rules require: in the model's sampled futures the
  instant reply does not raise the booked share; stalled is the
  modal outcome under both prefixes. That is model behaviour on
  a synthetic corpus with a measured calibration gap, and the
  page carries the standing caveat at the same visual weight as
  the numbers, in the design's words: trained on synthetic data
  from our own simulators, shares are model behaviour, not
  measured business statistics. The technical report tells the
  full v0 story and quotes the results files exactly, including
  the outcome_class exception (n-gram 0.9430 and majority
  0.9154 above the mini's 0.8952) and the calibration gap
  (mean group total variation 0.2331, pooled 0.2262, booked
  and lost over-sampled). Tests: 117 passed (110 before step 6
  plus 7 new demo tests: prefix shape, validate-and-decode over
  sampled futures, data consistency against its recorded
  configuration, page embedding and self-containment, weights
  round trip). Ruff: all checks passed. One environment note:
  the VM restarted mid-step and wiped /tmp, killing the first
  sampling run and the scratch venv; the run was rebuilt with
  the venv inside the repo and the torch target install under
  the home directory cache, and completed on the second run.
  Build order status: steps 1 to 6 complete. Remaining, all
  gated on Tech: the release itself, the med-spa fine-tune
  (second beat, after launch), and the v1 work the technical
  report names (real consented traces, schema v2 questions,
  converters).
