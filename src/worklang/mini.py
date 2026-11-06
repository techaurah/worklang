"""worklang-mini: the step 5 neural model (DESIGN.md, The mini model).

Architecture, fixed by the design: a decoder-only transformer in
PyTorch, d_model 256, 6 layers, 8 heads, context 256 events (1792
tokens), CPU only. The input is the tokenizer's flat stream: BOS, the
vertical conditioning token, then the 7 tokens of each event, so one
event is one composite position of 7 tokens in the flat sequence.

Prediction follows the design's Kronos coarse-to-fine order applied
within a position. The hidden state at a prediction point (the
vertical token for the first event, the outcome_class token of event
t for event t + 1) feeds a family head first; each attribute head
then reads that same hidden state plus the embedding of the family
token, so attribute predictions are conditioned on the family. In
training the conditioning family is the target event's true family
(teacher forcing within the position, exactly as the flat sequence
teacher-forces every other token); at inference it is the predicted
family for argmax prediction and the sampled family for sampling,
because a predictor may not peek at the target. The loss is the sum
of the seven per-stream cross-entropies, averaged over prediction
points, as the design specifies ("loss is the sum of per-stream
cross-entropies"; the average over points is the batching reduction,
recorded here so the number is interpretable).

The embedding table is sized to the reserved id space (128), not the
assigned vocabulary (69): DESIGN.md reserves ids up to 128 so the
vocabulary can grow without breaking saved corpora, and a table sized
to the reservation keeps that promise for the model as well. The
output heads predict only over assigned values: one family head over
the 8 families and one attribute head per remaining stream over that
stream's bucket set. Reserved ids are therefore never predicted.

Scoring reuses the step 4 harness (evaluation.score_predictor) one
trace at a time so the mini sees exactly the same 544 test positions
and the same counting; the per-trace counts are summed and the
accuracies re-derived with the harness's own rounding. One trace at a
time because the harness predictor signature receives only the event
prefix, while the mini also conditions on the trace's vertical token.

Futures calibration follows the design's measurement with one
interpretation recorded, not hidden: the prefix for a test trace is
its full event history except the terminal outcome event (the longest
prefix that does not leak the answer, and a fixed rule with no tuned
length). Futures are sampled from that prefix per trace, pooled per
prefix group (the split group of corpus_v0.group_of), and compared
against the empirical outcome distribution of the held-out test
traces in the same group. Total variation distance is over the five
real outcome classes. A sampled future that reaches the generation
cap without an outcome event is truncated: it contributes no outcome
to the sampled distribution and is counted and reported, never
silently dropped.
"""

from __future__ import annotations

import dataclasses
import random
from collections import Counter
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .corpus_v0 import group_of
from .errors import MiniError
from .evaluation import score_predictor
from .schema import Event, Trace
from .tokenizer import encode_event, encode_trace
from .vocab import (
    EVENT_STREAMS,
    FAMILIES,
    OUTCOME_CLASSES,
    RESERVED_VOCAB_SIZE,
    STREAMS,
    token_id,
)

ATTRIBUTE_STREAMS = EVENT_STREAMS[1:]
TOKENS_PER_EVENT = len(EVENT_STREAMS)
OUTCOME_VALUES = tuple(c for c in OUTCOME_CLASSES if c != "none")

_BOS = token_id("special", "BOS")
_PAD = token_id("special", "PAD")

# Each stream's values occupy a contiguous id range in STREAMS order;
# a token's index within its stream is its id minus the stream offset.
STREAM_OFFSETS: dict[str, int] = {}
_offset = 0
for _name, _values in STREAMS.items():
    STREAM_OFFSETS[_name] = _offset
    _offset += len(_values)


@dataclasses.dataclass(frozen=True)
class MiniConfig:
    """Model hyperparameters; the defaults are the design's scope."""

    d_model: int = 256
    n_layer: int = 6
    n_head: int = 8
    max_events: int = 256
    dropout: float = 0.0

    @property
    def max_len(self) -> int:
        """Flat sequence length of a maximum-length trace (no EOS)."""
        return 2 + TOKENS_PER_EVENT * self.max_events


class _Block(nn.Module):
    def __init__(self, config: MiniConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = nn.MultiheadAttention(
            config.d_model,
            config.n_head,
            dropout=config.dropout,
            batch_first=True,
        )
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model),
            nn.GELU(),
            nn.Linear(4 * config.d_model, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        attended, _ = self.attn(
            h, h, h, attn_mask=attn_mask, need_weights=False
        )
        x = x + attended
        return x + self.mlp(self.ln2(x))


class WorkLangMini(nn.Module):
    """The worklang-mini decoder-only transformer."""

    def __init__(self, config: MiniConfig | None = None) -> None:
        super().__init__()
        self.config = config or MiniConfig()
        d = self.config.d_model
        # Sized to the reserved id space (128), not the assigned 69:
        # see the module docstring.
        self.token_embedding = nn.Embedding(RESERVED_VOCAB_SIZE, d)
        # +1 for the EOS that closes a training sequence.
        self.position_embedding = nn.Embedding(self.config.max_len + 1, d)
        self.blocks = nn.ModuleList(
            _Block(self.config) for _ in range(self.config.n_layer)
        )
        self.ln_f = nn.LayerNorm(d)
        self.family_head = nn.Linear(d, len(FAMILIES))
        self.attribute_heads = nn.ModuleDict(
            {
                stream: nn.Linear(d, len(STREAMS[stream]))
                for stream in ATTRIBUTE_STREAMS
            }
        )
        self._masks: dict[int, torch.Tensor] = {}
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _causal_mask(self, length: int, device: torch.device) -> torch.Tensor:
        mask = self._masks.get(length)
        if mask is None or mask.device != device:
            mask = torch.triu(
                torch.full((length, length), float("-inf"), device=device),
                diagonal=1,
            )
            self._masks[length] = mask
        return mask

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Hidden states for a batch of flat token sequences."""
        length = ids.shape[1]
        positions = torch.arange(length, device=ids.device)
        x = self.token_embedding(ids) + self.position_embedding(positions)
        mask = self._causal_mask(length, ids.device)
        for block in self.blocks:
            x = block(x, mask)
        return self.ln_f(x)

    def family_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Family logits from hidden states of any leading shape."""
        return self.family_head(hidden)

    def attribute_logits(
        self, hidden: torch.Tensor, family_token_ids: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Per-stream attribute logits, conditioned on the family.

        The conditioning is the family token's embedding added to the
        hidden state, with one head per attribute stream.
        """
        conditioned = hidden + self.token_embedding(family_token_ids)
        return {
            stream: self.attribute_heads[stream](conditioned)
            for stream in ATTRIBUTE_STREAMS
        }


def count_parameters(model: nn.Module) -> int:
    """The real parameter count, counted from the built module."""
    return sum(parameter.numel() for parameter in model.parameters())


def sequence_ids(vertical: str, events: Sequence[Event]) -> list[int]:
    """Flat ids for a prefix: BOS, vertical token, event tokens."""
    ids = [_BOS, token_id("vertical", vertical)]
    for event in events:
        ids.extend(encode_event(event))
    return ids


def _last_hidden(
    model: WorkLangMini, vertical: str, prefix: Sequence[Event]
) -> torch.Tensor:
    ids = torch.tensor([sequence_ids(vertical, prefix)], dtype=torch.long)
    return model(ids)[0, -1]


def predict_next_event(
    model: WorkLangMini, vertical: str, prefix: Sequence[Event]
) -> dict[str, str]:
    """Argmax next-event prediction for one prefix.

    The family is the argmax of the family head; the attributes are
    argmaxes of heads conditioned on that predicted family (never on
    the target's true family, which a predictor cannot know).
    """
    model.eval()
    with torch.no_grad():
        hidden = _last_hidden(model, vertical, prefix)
        family_index = int(torch.argmax(model.family_logits(hidden)))
        family_token = STREAM_OFFSETS["family"] + family_index
        attributes = model.attribute_logits(
            hidden, torch.tensor(family_token)
        )
    predicted = {"family": FAMILIES[family_index]}
    for stream in ATTRIBUTE_STREAMS:
        index = int(torch.argmax(attributes[stream]))
        predicted[stream] = STREAMS[stream][index]
    return predicted


def _predictor_for(model: WorkLangMini, vertical: str):
    def predict(prefix: Sequence[Event]) -> dict[str, str]:
        return predict_next_event(model, vertical, prefix)

    return predict


def score_mini(model: WorkLangMini, test_traces: Sequence[Trace]) -> dict:
    """Score the mini on the step 4 protocol, via the step 4 harness.

    Runs evaluation.score_predictor once per test trace (the mini
    needs the trace's vertical, which the harness signature does not
    carry) and sums the counted results, so the positions and counts
    are the harness's own over the identical extraction.
    """
    per_stream_correct = {stream: 0 for stream in EVENT_STREAMS}
    composite_correct = 0
    positions = 0
    for trace in test_traces:
        block = score_predictor(_predictor_for(model, trace.vertical), [trace])
        positions += block["positions"]
        composite_correct += block["composite_correct"]
        for stream in EVENT_STREAMS:
            per_stream_correct[stream] += block["per_stream_correct"][stream]
    if positions == 0:
        raise MiniError("no test positions to score (empty test split)")

    def accuracy(correct: int) -> float:
        return round(correct / positions, 6)

    return {
        "positions": positions,
        "per_stream_correct": per_stream_correct,
        "composite_correct": composite_correct,
        "family_accuracy": accuracy(per_stream_correct["family"]),
        "stream_accuracies": {
            stream: accuracy(per_stream_correct[stream])
            for stream in EVENT_STREAMS
        },
        "composite_exact_match": accuracy(composite_correct),
    }


def _batch_point_loss(
    model: WorkLangMini, batch_ids: list[list[int]]
) -> tuple[torch.Tensor, int]:
    """Mean per-point loss (sum of per-stream CEs) over a batch.

    Prediction points for a trace with n events sit at flat positions
    7e + 1 for e in 0..n-1 (the vertical token predicts event 0; the
    outcome_class token of event e - 1 predicts event e). The target
    event's tokens are read straight from the teacher-forced sequence.
    Trailing PAD positions cannot influence real positions under the
    causal mask, so right padding needs no padding mask.
    """
    max_len = max(len(ids) for ids in batch_ids)
    padded = torch.tensor(
        [ids + [_PAD] * (max_len - len(ids)) for ids in batch_ids],
        dtype=torch.long,
    )
    hidden = model(padded)
    point_losses: list[torch.Tensor] = []
    for row, ids in enumerate(batch_ids):
        n_events = (len(ids) - 3) // TOKENS_PER_EVENT
        points = torch.tensor(
            [7 * e + 1 for e in range(n_events)], dtype=torch.long
        )
        h = hidden[row, points]
        targets = torch.tensor(
            [
                [ids[2 + 7 * e + j] for j in range(TOKENS_PER_EVENT)]
                for e in range(n_events)
            ],
            dtype=torch.long,
        )
        family_targets = targets[:, 0] - STREAM_OFFSETS["family"]
        loss = F.cross_entropy(
            model.family_logits(h), family_targets, reduction="none"
        )
        attributes = model.attribute_logits(h, targets[:, 0])
        for j, stream in enumerate(ATTRIBUTE_STREAMS, start=1):
            stream_targets = targets[:, j] - STREAM_OFFSETS[stream]
            loss = loss + F.cross_entropy(
                attributes[stream], stream_targets, reduction="none"
            )
        point_losses.append(loss)
    all_points = torch.cat(point_losses)
    return all_points.mean(), int(all_points.shape[0])


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    """Training hyperparameters actually used, recorded in results."""

    epochs: int = 48
    lr: float = 3e-3
    batch_size: int = 32
    weight_decay: float = 0.01
    seed: int = 20261104
    threads: int = 4


def train_model(
    train_traces: Sequence[Trace],
    *,
    config: MiniConfig | None = None,
    train_config: TrainConfig | None = None,
) -> tuple[WorkLangMini, dict]:
    """Train the mini on the train split only, CPU only, fixed seed.

    Determinism: seeds are set for torch and random, the thread count
    is fixed, and deterministic algorithms are requested. Whether a
    re-run reproduces the metrics exactly is an empirical question the
    training script answers by running this function twice with the
    same seed and recording both runs; this function claims nothing.
    """
    tcfg = train_config or TrainConfig()
    torch.manual_seed(tcfg.seed)
    random.seed(tcfg.seed)
    torch.set_num_threads(tcfg.threads)
    torch.use_deterministic_algorithms(True)
    model = WorkLangMini(config)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay
    )
    sequences = [encode_trace(trace) for trace in train_traces]
    batches_per_epoch = -(-len(sequences) // tcfg.batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=tcfg.epochs * batches_per_epoch
    )
    shuffle = random.Random(tcfg.seed)
    order = list(range(len(sequences)))
    epoch_losses: list[float] = []
    model.train()
    for _epoch in range(tcfg.epochs):
        shuffle.shuffle(order)
        total_loss = 0.0
        total_points = 0
        for start in range(0, len(order), tcfg.batch_size):
            batch = [sequences[i] for i in order[start : start + tcfg.batch_size]]
            loss, points = _batch_point_loss(model, batch)
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            scheduler.step()
            total_loss += float(loss.detach()) * points
            total_points += points
        epoch_losses.append(total_loss / total_points)
    info = {
        "epochs": tcfg.epochs,
        "lr": tcfg.lr,
        "batch_size": tcfg.batch_size,
        "weight_decay": tcfg.weight_decay,
        "seed": tcfg.seed,
        "threads": tcfg.threads,
        "train_traces": len(sequences),
        "first_epoch_loss": round(epoch_losses[0], 6),
        "final_epoch_loss": round(epoch_losses[-1], 6),
    }
    return model, info


def _sample_index(
    logits: torch.Tensor, temperature: float, generator: torch.Generator
) -> int:
    probs = torch.softmax(logits / temperature, dim=-1)
    return int(torch.multinomial(probs, 1, generator=generator))


def sample_future(
    model: WorkLangMini,
    vertical: str,
    prefix: Sequence[Event],
    *,
    temperature: float = 1.0,
    seed: int = 0,
    max_new_events: int = 64,
) -> tuple[tuple[Event, ...], bool]:
    """Sample one future continuation of a prefix.

    Generates events one at a time (family sampled first, attributes
    sampled from heads conditioned on the sampled family) until an
    outcome event is generated or the cap is reached. Schema validity
    is enforced structurally, as the module docstring states: a
    non-outcome event carries outcome_class none, and an outcome
    event's class is sampled with none masked out. Returns the
    generated events and whether an outcome event closed the future.
    """
    if temperature <= 0:
        raise MiniError(f"temperature must be positive, got {temperature}")
    generator = torch.Generator().manual_seed(seed)
    events = list(prefix)
    generated: list[Event] = []
    model.eval()
    with torch.no_grad():
        for _step in range(max_new_events):
            hidden = _last_hidden(model, vertical, events)
            family_index = _sample_index(
                model.family_logits(hidden), temperature, generator
            )
            family = FAMILIES[family_index]
            family_token = STREAM_OFFSETS["family"] + family_index
            attributes = model.attribute_logits(
                hidden, torch.tensor(family_token)
            )
            values: dict[str, str] = {"family": family}
            for stream in ATTRIBUTE_STREAMS:
                logits = attributes[stream]
                if stream == "outcome_class":
                    if family != "outcome":
                        values[stream] = "none"
                        continue
                    logits = logits.clone()
                    none_index = STREAMS[stream].index("none")
                    logits[none_index] = float("-inf")
                index = _sample_index(logits, temperature, generator)
                values[stream] = STREAMS[stream][index]
            event = Event(seq=len(events), **values)
            generated.append(event)
            events.append(event)
            if family == "outcome":
                return tuple(generated), True
    return tuple(generated), False


def _total_variation(
    sampled: Counter, empirical: Counter, classes: Sequence[str]
) -> float:
    sampled_total = sum(sampled.values())
    empirical_total = sum(empirical.values())
    return 0.5 * sum(
        abs(
            sampled.get(c, 0) / sampled_total
            - empirical.get(c, 0) / empirical_total
        )
        for c in classes
    )


def futures_calibration(
    model: WorkLangMini,
    test_traces: Sequence[Trace],
    *,
    samples_per_prefix: int = 32,
    temperature: float = 1.0,
    seed: int = 20261104,
    max_new_events: int = 64,
) -> dict:
    """The design's futures calibration measurement.

    One prefix per test trace (its events except the terminal outcome
    event); futures sampled per prefix and pooled per prefix group;
    total variation distance against the group's empirical test
    outcomes, per group and pooled. See the module docstring for the
    prefix interpretation and the truncation rule.
    """
    groups: dict[str, list[Trace]] = {}
    for trace in test_traces:
        groups.setdefault(group_of(trace), []).append(trace)
    per_group: dict[str, dict] = {}
    pooled_sampled: Counter = Counter()
    pooled_empirical: Counter = Counter()
    pooled_truncated = 0
    for group, traces in sorted(groups.items()):
        sampled: Counter = Counter()
        truncated = 0
        for trace_index, trace in enumerate(traces):
            prefix = trace.events[:-1]
            for sample_index in range(samples_per_prefix):
                sample_seed = (
                    seed * 1_000_003 + trace_index * 10_007 + sample_index
                )
                future, completed = sample_future(
                    model,
                    trace.vertical,
                    prefix,
                    temperature=temperature,
                    seed=sample_seed,
                    max_new_events=max_new_events,
                )
                if completed:
                    sampled[future[-1].outcome_class] += 1
                else:
                    truncated += 1
        empirical: Counter = Counter(t.outcome for t in traces)
        completed_total = sum(sampled.values())
        per_group[group] = {
            "traces": len(traces),
            "prefixes": len(traces),
            "samples": len(traces) * samples_per_prefix,
            "completed_samples": completed_total,
            "truncated_samples": truncated,
            "empirical_outcomes": dict(sorted(empirical.items())),
            "sampled_outcomes": dict(sorted(sampled.items())),
            "total_variation": round(
                _total_variation(sampled, empirical, OUTCOME_VALUES), 6
            ),
        }
        pooled_sampled.update(sampled)
        pooled_empirical.update(empirical)
        pooled_truncated += truncated
    group_tvs = [block["total_variation"] for block in per_group.values()]
    return {
        "protocol": (
            "One prefix per test trace: all events except the terminal "
            "outcome event. Futures sampled per prefix, pooled per "
            "prefix group (corpus_v0.group_of); total variation "
            "distance over the five real outcome classes between the "
            "sampled outcome distribution and the empirical outcome "
            "distribution of the group's test traces. Futures reaching "
            "the generation cap without an outcome event are truncated: "
            "counted, and excluded from the sampled distribution."
        ),
        "samples_per_prefix": samples_per_prefix,
        "temperature": temperature,
        "seed": seed,
        "max_new_events": max_new_events,
        "groups": per_group,
        "mean_group_total_variation": round(
            sum(group_tvs) / len(group_tvs), 6
        ),
        "pooled_total_variation": round(
            _total_variation(pooled_sampled, pooled_empirical, OUTCOME_VALUES),
            6,
        ),
        "pooled": {
            "completed_samples": sum(pooled_sampled.values()),
            "truncated_samples": pooled_truncated,
            "empirical_outcomes": dict(sorted(pooled_empirical.items())),
            "sampled_outcomes": dict(sorted(pooled_sampled.items())),
        },
    }


__all__ = [
    "ATTRIBUTE_STREAMS",
    "MiniConfig",
    "TrainConfig",
    "WorkLangMini",
    "count_parameters",
    "futures_calibration",
    "predict_next_event",
    "sample_future",
    "score_mini",
    "sequence_ids",
    "train_model",
]
