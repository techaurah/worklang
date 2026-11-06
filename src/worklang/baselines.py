"""The two v0 baselines (DESIGN.md, The mini model, Evaluation).

Build step 4 puts the bar in place before any neural code exists:

- Baseline 1, majority class: always predict the most frequent family,
  and the most frequent value per attribute stream, counted over the
  train split.
- Baseline 2, n-gram: order-2 over composite event symbols (one full
  event as one symbol), with backoff to order-1 and then to the
  unigram distribution, trained on the same train split.

Both classes only ever see the traces handed to fit; the evaluation
harness (evaluation.py) is what enforces the split discipline of
fitting on the train split and scoring on the test split.

A composite symbol is the event's 7 stream values in EVENT_STREAMS
order (seq is positional metadata, not part of the symbol). A
prediction is a full composite event in that same form, returned as a
{stream: value} dict, so per-stream accuracies for the n-gram come
from its predicted composite event exactly as the design specifies.

Ties are broken deterministically, never by data order: highest count
wins, and equal counts resolve to the earliest value in vocabulary
order (the stream's bucket order for single streams; the token id
tuple order for composite symbols). Same corpus in, same predictions
out, on every run.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from .errors import BaselineError
from .schema import Event, Trace
from .vocab import EVENT_STREAMS, STREAMS

# One composite event symbol: the 7 stream values in EVENT_STREAMS order.
Symbol = tuple[str, ...]


def event_symbol(event: Event) -> Symbol:
    """The composite symbol of one event: its 7 stream values in order."""
    return tuple(getattr(event, stream) for stream in EVENT_STREAMS)


def symbol_values(symbol: Symbol) -> dict[str, str]:
    """A composite symbol as a {stream: value} prediction dict."""
    return dict(zip(EVENT_STREAMS, symbol, strict=True))


def _mode_stream(stream: str, counts: Counter) -> str:
    """Most frequent value of one stream; ties to earliest bucket order."""
    order = STREAMS[stream]
    return min(counts, key=lambda value: (-counts[value], order.index(value)))


def _mode_symbol(counts: Counter) -> Symbol:
    """Most frequent composite symbol; ties to lowest token id tuple."""

    def key(symbol: Symbol) -> tuple[int, tuple[int, ...]]:
        ids = tuple(
            STREAMS[stream].index(value)
            for stream, value in zip(EVENT_STREAMS, symbol, strict=True)
        )
        return (-counts[symbol], ids)

    return min(counts, key=key)


class MajorityBaseline:
    """Baseline 1: per-stream majority class of the train split.

    The prediction ignores the prefix entirely: every position gets
    the train split's most frequent family and, independently, the
    most frequent value of each attribute stream.
    """

    name = "majority"

    def __init__(self) -> None:
        self._modes: dict[str, str] | None = None

    def fit(self, traces: Sequence[Trace]) -> MajorityBaseline:
        counts: dict[str, Counter] = {stream: Counter() for stream in EVENT_STREAMS}
        events = 0
        for trace in traces:
            for event in trace.events:
                events += 1
                for stream in EVENT_STREAMS:
                    counts[stream][getattr(event, stream)] += 1
        if events == 0:
            raise BaselineError("majority baseline fitted on zero events")
        self._modes = {
            stream: _mode_stream(stream, counts[stream]) for stream in EVENT_STREAMS
        }
        return self

    def predict(self, prefix: Sequence[Event]) -> dict[str, str]:
        if self._modes is None:
            raise BaselineError("majority baseline predicted before fit")
        return dict(self._modes)


class NGramBaseline:
    """Baseline 2: order-2 n-gram over composite symbols, with backoff.

    Trained on the train split's per-trace event sequences. Given a
    prefix, predict from the longest context that has any continuation
    in training: the last two symbols (order-2), else the last one
    (order-1), else the unigram distribution. The vertical
    conditioning token is not part of the symbol stream; the design
    defines this baseline over composite event symbols only.
    """

    name = "ngram_order2"
    order = 2

    def __init__(self) -> None:
        self._unigram: Counter | None = None
        self._order1: dict[Symbol, Counter] | None = None
        self._order2: dict[tuple[Symbol, Symbol], Counter] | None = None

    def fit(self, traces: Sequence[Trace]) -> NGramBaseline:
        unigram: Counter = Counter()
        order1: dict[Symbol, Counter] = {}
        order2: dict[tuple[Symbol, Symbol], Counter] = {}
        for trace in traces:
            symbols = [event_symbol(event) for event in trace.events]
            unigram.update(symbols)
            for index, symbol in enumerate(symbols):
                if index >= 1:
                    order1.setdefault(symbols[index - 1], Counter())[symbol] += 1
                if index >= 2:
                    context = (symbols[index - 2], symbols[index - 1])
                    order2.setdefault(context, Counter())[symbol] += 1
        if not unigram:
            raise BaselineError("n-gram baseline fitted on zero events")
        self._unigram = unigram
        self._order1 = order1
        self._order2 = order2
        return self

    def predict(self, prefix: Sequence[Event]) -> dict[str, str]:
        if self._unigram is None or self._order1 is None or self._order2 is None:
            raise BaselineError("n-gram baseline predicted before fit")
        symbols = [event_symbol(event) for event in prefix]
        if len(symbols) >= 2:
            continuations = self._order2.get((symbols[-2], symbols[-1]))
            if continuations:
                return symbol_values(_mode_symbol(continuations))
        if len(symbols) >= 1:
            continuations = self._order1.get(symbols[-1])
            if continuations:
                return symbol_values(_mode_symbol(continuations))
        return symbol_values(_mode_symbol(self._unigram))


__all__ = [
    "MajorityBaseline",
    "NGramBaseline",
    "Symbol",
    "event_symbol",
    "symbol_values",
]
