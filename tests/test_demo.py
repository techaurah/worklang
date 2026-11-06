"""Tests for the futures demo (DESIGN.md build step 6).

Fast by construction, like test_mini.py: a tiny untrained model,
because sampling validity is structural (sample_future enforces the
schema), a handful of samples per prefix, and a weights round trip
on the tiny config. What is pinned here is the demo contract: the
authored prefixes are exactly the design's, the data document is
internally consistent and records the configuration it was sampled
with, every sampled future that completes rebuilds into a schema
valid trace that tokenises and decodes exactly, and the rendered
page embeds the same data with the caveat present and no external
references.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from worklang.demo import (  # noqa: E402
    CAVEAT,
    DATA_PLACEHOLDER,
    PREFIXES,
    build_demo_data,
    ensure_trained_model,
    load_weights,
    prefix_by_id,
    render_page,
    sample_prefix_futures,
    save_weights,
)
from worklang.errors import DemoError  # noqa: E402
from worklang.mini import MiniConfig, WorkLangMini, count_parameters  # noqa: E402
from worklang.schema import Event, Generator, Trace  # noqa: E402
from worklang.tokenizer import decode_trace, encode_trace  # noqa: E402

TINY = MiniConfig(d_model=32, n_layer=2, n_head=4, max_events=16)
REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = (REPO_ROOT / "demo" / "index.template.html").read_text(
    encoding="utf-8"
)


def _tiny_model() -> WorkLangMini:
    torch.manual_seed(11)
    return WorkLangMini(TINY)


def _metadata(model: WorkLangMini) -> dict:
    return {
        "model_name": "worklang-mini",
        "config": {
            "d_model": TINY.d_model,
            "n_layer": TINY.n_layer,
            "n_head": TINY.n_head,
            "max_events": TINY.max_events,
            "dropout": TINY.dropout,
        },
        "parameter_count": count_parameters(model),
        "provenance": "test fixture",
        "weights_source": "test fixture",
    }


def _rebuild_trace(example: dict, vertical: str) -> Trace:
    events = tuple(Event.from_dict(item) for item in example["events"])
    return Trace(
        trace_id="demo:test",
        source="demo-sample",
        synthetic=True,
        vertical=vertical,
        agent="worklang-mini",
        agent_version="0",
        events=events,
        outcome=example["outcome"],
        generator=Generator(name="worklang-demo", version="0", seed=1),
    )


def test_prefixes_are_the_designs_authored_prefixes() -> None:
    without = prefix_by_id("without-reply")
    assert len(without.events) == 1
    enquiry = without.events[0]
    assert enquiry.family == "enquiry"
    assert enquiry.channel == "instagram"
    assert enquiry.stage == "NEW"
    assert enquiry.latency_band == "none"
    assert enquiry.outcome_class == "none"

    with_reply = prefix_by_id("with-reply")
    assert len(with_reply.events) == 2
    assert with_reply.events[0] == enquiry
    reply = with_reply.events[1]
    assert reply.family == "reply"
    assert reply.channel == "instagram"
    assert reply.latency_band == "instant"
    assert reply.seq == 1
    assert tuple(p.prefix_id for p in PREFIXES) == (
        "without-reply",
        "with-reply",
    )


def test_sampled_futures_validate_and_decode() -> None:
    model = _tiny_model()
    for index, prefix in enumerate(PREFIXES):
        futures = sample_prefix_futures(
            model, index, prefix, samples=8, seed=99, max_new_events=8
        )
        assert len(futures) == 8
        for future, completed in futures:
            if completed:
                events = tuple(prefix.events) + future
                trace = Trace(
                    trace_id="demo:test",
                    source="demo-sample",
                    synthetic=True,
                    vertical="med-spa",
                    agent="worklang-mini",
                    agent_version="0",
                    events=events,
                    outcome=future[-1].outcome_class,
                    generator=Generator(
                        name="worklang-demo", version="0", seed=99
                    ),
                )
                decoded = decode_trace(encode_trace(trace))
                assert decoded.vertical == trace.vertical
                assert decoded.events == trace.events
            else:
                assert future
                for event in future:
                    assert event.family != "outcome"
                    assert event.outcome_class == "none"


def test_demo_data_is_consistent_and_records_its_config() -> None:
    model = _tiny_model()
    data = build_demo_data(
        model,
        _metadata(model),
        samples=12,
        temperature=0.9,
        seed=123,
        max_new_events=8,
    )
    assert data["sampling"] == {
        "samples_per_prefix": 12,
        "temperature": 0.9,
        "seed": 123,
        "max_new_events": 8,
        "seed_scheme": data["sampling"]["seed_scheme"],
    }
    assert data["model"]["parameter_count"] == count_parameters(model)
    assert data["caveat"] == CAVEAT
    assert data["scenario"]["vertical"] == "med-spa"
    assert [block["id"] for block in data["prefixes"]] == [
        "without-reply",
        "with-reply",
    ]
    for block in data["prefixes"]:
        outcomes = block["sampled_outcomes"]
        assert len(outcomes) == 12
        assert block["samples"] == 12
        assert block["truncated"] == outcomes.count("truncated")
        distribution = block["distribution"]
        counts = Counter(o for o in outcomes if o != "truncated")
        assert distribution["completed"] == sum(counts.values())
        share_total = 0.0
        for name, entry in distribution["classes"].items():
            assert entry["count"] == counts.get(name, 0)
            share_total += entry["share_percent"]
        if distribution["completed"]:
            assert share_total == pytest.approx(100.0, abs=0.5)
        # Examples are real sampled futures: each rebuilds into a
        # schema valid trace and round trips exactly.
        assert len(block["examples"]) == min(
            3, distribution["completed"]
        )
        for example in block["examples"]:
            trace = _rebuild_trace(example, data["scenario"]["vertical"])
            decoded = decode_trace(encode_trace(trace))
            assert decoded.events == trace.events
            assert example["outcome"] in outcomes


def test_rendered_page_embeds_the_same_data_with_caveat() -> None:
    model = _tiny_model()
    data = build_demo_data(
        model, _metadata(model), samples=6, seed=5, max_new_events=8
    )
    page = render_page(data, TEMPLATE)
    assert DATA_PLACEHOLDER not in page
    marker = '<script type="application/json" id="demo-data">'
    embedded = page.split(marker, 1)[1].split("</script>", 1)[0]
    parsed = json.loads(embedded)
    assert parsed["prefixes"][0]["distribution"] == (
        data["prefixes"][0]["distribution"]
    )
    assert parsed["sampling"] == data["sampling"]
    assert CAVEAT in page
    # Self-contained: no external loads of any kind.
    assert "http://" not in page
    assert "https://" not in page
    assert "src=" not in page
    assert "href=" not in page
    assert page.count("<script") == 2


def test_no_em_dash_in_page_or_data() -> None:
    model = _tiny_model()
    data = build_demo_data(
        model, _metadata(model), samples=4, seed=5, max_new_events=8
    )
    em_dash = chr(0x2014)
    assert em_dash not in json.dumps(data)
    assert em_dash not in render_page(data, TEMPLATE)


def test_weights_round_trip_and_ensure_loads(tmp_path: Path) -> None:
    model = _tiny_model()
    path = tmp_path / "mini.pt"
    save_weights(model, path, provenance="test fixture")
    loaded, metadata = load_weights(path)
    assert metadata["parameter_count"] == count_parameters(model)
    from worklang.mini import predict_next_event

    prefix = PREFIXES[0].events
    assert predict_next_event(loaded, "med-spa", prefix) == (
        predict_next_event(model, "med-spa", prefix)
    )
    # With the file present, ensure loads it and needs no traces.
    ensured, ensured_meta = ensure_trained_model(path)
    assert ensured_meta["parameter_count"] == count_parameters(model)
    assert "loaded" in ensured_meta["weights_source"]


def test_ensure_without_weights_or_traces_raises(
    tmp_path: Path,
) -> None:
    with pytest.raises(DemoError):
        ensure_trained_model(tmp_path / "missing.pt")
