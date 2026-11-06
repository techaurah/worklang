"""Tests for worklang-mini (DESIGN.md build step 5).

Fast by construction: everything except the parameter-count test
uses a tiny configuration (d_model 32, 2 layers), and the training
tests run on four hand-built traces for a handful of epochs. The
parameter-count test builds the real design configuration because
the count is a deliverable: DESIGN.md estimated 5 to 6M parameters
before the code existed, and the test pins the counted value inside
that stated band.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from worklang.errors import MiniError  # noqa: E402
from worklang.mini import (  # noqa: E402
    ATTRIBUTE_STREAMS,
    MiniConfig,
    TrainConfig,
    WorkLangMini,
    count_parameters,
    predict_next_event,
    sample_future,
    score_mini,
    sequence_ids,
    train_model,
)
from worklang.schema import Event, Generator, Trace  # noqa: E402
from worklang.tokenizer import decode_trace, encode_trace  # noqa: E402
from worklang.vocab import EVENT_STREAMS, STREAMS  # noqa: E402

TINY = MiniConfig(d_model=32, n_layer=2, n_head=4, max_events=16)


def _event(seq: int, family: str, outcome_class: str = "none") -> Event:
    return Event(
        seq=seq,
        family=family,
        channel="website",
        score_band="none",
        latency_band="none",
        stage="none",
        tool="none",
        outcome_class=outcome_class,
    )


def _trace(trace_id: str, families: list[str], outcome: str) -> Trace:
    events = tuple(
        _event(i, family) for i, family in enumerate(families)
    ) + (_event(len(families), "outcome", outcome),)
    return Trace(
        trace_id=trace_id,
        source="qualbench",
        synthetic=True,
        vertical="general",
        agent="reference-agent",
        agent_version="1.0.0",
        events=events,
        outcome=outcome,
        generator=Generator(name="test", version="0", seed=1),
    )


def _toy_traces() -> list[Trace]:
    """Four identical traces: a pattern a model can actually learn."""
    return [
        _trace(f"toy:{i}", ["enquiry", "reply"], "booked") for i in range(4)
    ]


def test_head_shapes_and_stream_sizes() -> None:
    model = WorkLangMini(TINY)
    trace = _toy_traces()[0]
    ids = torch.tensor([sequence_ids(trace.vertical, trace.events[:1])])
    hidden = model(ids)
    assert hidden.shape == (1, ids.shape[1], TINY.d_model)
    family = model.family_logits(hidden[0, -1])
    assert family.shape == (len(STREAMS["family"]),)
    family_token = torch.tensor(10)  # first family token id
    attributes = model.attribute_logits(hidden[0, -1], family_token)
    assert tuple(attributes) == ATTRIBUTE_STREAMS
    for stream in ATTRIBUTE_STREAMS:
        assert attributes[stream].shape == (len(STREAMS[stream]),)


def test_parameter_count_in_design_band() -> None:
    model = WorkLangMini()
    counted = count_parameters(model)
    assert counted == sum(p.numel() for p in model.parameters())
    # DESIGN.md's pre-build estimate, quoted in the design log.
    assert 5_000_000 <= counted <= 6_000_000


def test_predict_next_event_returns_bucket_values() -> None:
    model = WorkLangMini(TINY)
    trace = _toy_traces()[0]
    predicted = predict_next_event(model, trace.vertical, trace.events[:1])
    assert tuple(predicted) == EVENT_STREAMS
    for stream in EVENT_STREAMS:
        assert predicted[stream] in STREAMS[stream]


def test_score_mini_counts_every_position() -> None:
    model = WorkLangMini(TINY)
    traces = _toy_traces()[:2]
    scored = score_mini(model, traces)
    assert scored["positions"] == sum(len(t.events) for t in traces)
    for stream in EVENT_STREAMS:
        assert 0 <= scored["per_stream_correct"][stream] <= scored["positions"]
        assert 0.0 <= scored["stream_accuracies"][stream] <= 1.0
    assert 0 <= scored["composite_correct"] <= scored["positions"]


def test_tiny_training_reduces_loss_and_is_reproducible() -> None:
    traces = _toy_traces()
    config = TrainConfig(epochs=30, lr=5e-3, batch_size=4, threads=2)
    _model_a, info_a = train_model(traces, config=TINY, train_config=config)
    _model_b, info_b = train_model(traces, config=TINY, train_config=config)
    assert info_a["final_epoch_loss"] < 0.5 * info_a["first_epoch_loss"]
    # Same seed, same data, CPU: the rerun reproduces the run exactly.
    assert info_a["final_epoch_loss"] == info_b["final_epoch_loss"]
    assert info_a["first_epoch_loss"] == info_b["first_epoch_loss"]


def test_sample_future_produces_schema_valid_trace() -> None:
    model = WorkLangMini(TINY)
    trace = _toy_traces()[0]
    prefix = trace.events[:1]
    future, completed = sample_future(
        model, trace.vertical, prefix, temperature=1.0, seed=7,
        max_new_events=32,
    )
    if completed:
        events = tuple(prefix) + future
        rebuilt = Trace(
            trace_id="sampled:1",
            source="qualbench",
            synthetic=True,
            vertical=trace.vertical,
            agent="worklang-mini",
            agent_version="0",
            events=events,
            outcome=future[-1].outcome_class,
            generator=Generator(name="test", version="0", seed=7),
        )
        decoded = decode_trace(encode_trace(rebuilt))
        assert decoded.vertical == rebuilt.vertical
        assert decoded.events == rebuilt.events
    else:
        # A truncated future never invents an outcome: every generated
        # event is a non-outcome event carrying outcome_class none.
        assert future
        for event in future:
            assert event.family != "outcome"
            assert event.outcome_class == "none"


def test_sample_future_rejects_nonpositive_temperature() -> None:
    model = WorkLangMini(TINY)
    trace = _toy_traces()[0]
    with pytest.raises(MiniError):
        sample_future(model, trace.vertical, trace.events[:1], temperature=0.0)
