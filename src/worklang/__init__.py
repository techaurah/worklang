"""WorkLang: schema, tokenizer, corpus, baselines, and eval harness.

WorkLang is a small foundation model for the language of agent
workflows, per DESIGN.md in the flagship workspace. This package
ships the trace schema and the deterministic event tokenizer
(build step 1), the QualBench trace emitter and corpus source A
generation (build step 2, see worklang.qualbench_emitter and
worklang.corpus_a; those modules import the QualBench API lazily,
so importing worklang itself never requires a QualBench checkout),
the LeadPipe corpus source B and the frozen merged corpus v0
(build step 3), and the majority and n-gram baselines with the
evaluation harness (build step 4, see worklang.baselines and
worklang.evaluation). The mini model and the futures demo are
later build steps and do not exist yet.
"""

from .errors import SchemaError, TokenError, WorkLangError
from .schema import Event, Generator, Trace, trace_from_json, trace_to_json
from .tokenizer import (
    DecodedTrace,
    decode_event,
    decode_trace,
    encode_event,
    encode_trace,
    score_to_band,
)
from .vocab import RESERVED_VOCAB_SIZE, SCHEMA_VERSION, VOCAB_SIZE

__version__ = "0.0.1"

__all__ = [
    "RESERVED_VOCAB_SIZE",
    "SCHEMA_VERSION",
    "VOCAB_SIZE",
    "DecodedTrace",
    "Event",
    "Generator",
    "SchemaError",
    "TokenError",
    "Trace",
    "WorkLangError",
    "decode_event",
    "decode_trace",
    "encode_event",
    "encode_trace",
    "score_to_band",
    "trace_from_json",
    "trace_to_json",
]
