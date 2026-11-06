"""The WorkLang vocabulary: a fixed, rule defined token set.

DESIGN.md fixes the vocabulary by enumerating bucket sets per stream.
Nothing is learned, so the enumeration below is exact: 69 tokens, with
ids assigned contiguously from 0 in the order the design enumerates the
streams (specials, verticals, families, then the fine attribute streams
in event field order). Id space is reserved up to 128 so the vocabulary
can grow without breaking saved corpora.

Stream notes, straight from the design:

- channel: the seven LeadPipe source values (website, instagram,
  facebook, whatsapp, email, call, referral) plus chat and unknown.
  The absence value on this stream is "unknown", not "none".
- score_band: bands use LeadPipe thresholds (HOT 80 to 100, WARM 65 to
  79, NURTURE 50 to 64, below 50 LOW), plus none. See
  tokenizer.score_to_band for the mapping function.
- stage: the ten LeadPipe stages, spelled as LeadPipe spells them,
  plus none. The value is the stage after the event.
- tool: the union of tool names the two simulators can emit, kept
  source-faithful. calendar.book and book_appointment (and the other
  near-duplicate pairs) are separate tokens on purpose; merging them is
  a v2 schema decision, not a silent normalisation.
- outcome_class: only meaningful on the outcome family; every other
  event carries none.
"""

from __future__ import annotations

import json
from pathlib import Path

from .errors import TokenError

SCHEMA_VERSION = "0.1"

RESERVED_VOCAB_SIZE = 128

SPECIALS = ("PAD", "BOS", "EOS", "UNK")

VERTICALS = (
    "real-estate",
    "med-spa",
    "home-services",
    "dental",
    "agencies",
    "general",
)

FAMILIES = (
    "enquiry",
    "reply",
    "tool_call",
    "booking",
    "follow_up",
    "handoff",
    "stage",
    "outcome",
)

CHANNELS = (
    "website",
    "instagram",
    "facebook",
    "whatsapp",
    "email",
    "call",
    "referral",
    "chat",
    "unknown",
)

SCORE_BANDS = ("hot", "warm", "nurture", "low", "none")

LATENCY_BANDS = ("instant", "fast", "slow", "delayed", "overnight", "none")

STAGES = (
    "NEW",
    "CONTACTED",
    "QUALIFIED",
    "NURTURE",
    "DISCOVERY",
    "BOOKED",
    "PROPOSAL",
    "NEGOTIATION",
    "WON",
    "LOST",
    "none",
)

TOOLS = (
    # QualBench ToolExecutor tools.
    "calendar.check_availability",
    "calendar.book",
    "crm.upsert_lead",
    "crm.get_lead",
    "report_outcome",
    # LeadPipe public tools.
    "capture_lead",
    "get_lead",
    "list_hot_leads",
    "score_lead",
    "update_stage",
    "check_availability",
    "book_appointment",
    "daily_pipeline_summary",
    # Absence value.
    "none",
)

OUTCOME_CLASSES = ("booked", "won", "lost", "stalled", "dropped", "none")

# Stream name to bucket values, in id assignment order.
STREAMS: dict[str, tuple[str, ...]] = {
    "special": SPECIALS,
    "vertical": VERTICALS,
    "family": FAMILIES,
    "channel": CHANNELS,
    "score_band": SCORE_BANDS,
    "latency_band": LATENCY_BANDS,
    "stage": STAGES,
    "tool": TOOLS,
    "outcome_class": OUTCOME_CLASSES,
}

# The streams that make up one event, in token order: family first
# (the coarse subtoken), then the fine attribute subtokens.
EVENT_STREAMS = (
    "family",
    "channel",
    "score_band",
    "latency_band",
    "stage",
    "tool",
    "outcome_class",
)


def _build_tables() -> tuple[dict[str, int], dict[int, tuple[str, str]]]:
    token_to_id: dict[str, int] = {}
    id_to_token: dict[int, tuple[str, str]] = {}
    next_id = 0
    for stream, values in STREAMS.items():
        for value in values:
            token_to_id[f"{stream}:{value}"] = next_id
            id_to_token[next_id] = (stream, value)
            next_id += 1
    return token_to_id, id_to_token


TOKEN_TO_ID, ID_TO_TOKEN = _build_tables()

VOCAB_SIZE = len(TOKEN_TO_ID)


def token_id(stream: str, value: str) -> int:
    """Return the id of one stream value, or raise TokenError."""
    key = f"{stream}:{value}"
    try:
        return TOKEN_TO_ID[key]
    except KeyError:
        valid = STREAMS.get(stream)
        if valid is None:
            raise TokenError(
                f"unknown stream {stream!r}; expected one of: "
                + ", ".join(STREAMS)
            ) from None
        raise TokenError(
            f"value {value!r} is not in the {stream} bucket set; "
            f"expected one of: {', '.join(valid)}"
        ) from None


def token_at(token_id_: int) -> tuple[str, str]:
    """Return the (stream, value) pair for an id, or raise TokenError.

    Ids in the reserved range (VOCAB_SIZE to RESERVED_VOCAB_SIZE - 1)
    are not assigned yet and raise, as do negative and larger ids.
    """
    if isinstance(token_id_, bool) or not isinstance(token_id_, int):
        raise TokenError(f"token id must be an int, got {token_id_!r}")
    try:
        return ID_TO_TOKEN[token_id_]
    except KeyError:
        raise TokenError(
            f"token id {token_id_} is not assigned; the vocabulary has "
            f"{VOCAB_SIZE} tokens (ids 0 to {VOCAB_SIZE - 1}) and id "
            f"space is reserved up to {RESERVED_VOCAB_SIZE}"
        ) from None


def vocab_as_dict() -> dict:
    """Return the vocabulary as a JSON serialisable dict (vocab.json)."""
    return {
        "schema_version": SCHEMA_VERSION,
        "vocab_size": VOCAB_SIZE,
        "reserved_vocab_size": RESERVED_VOCAB_SIZE,
        "streams": {name: list(values) for name, values in STREAMS.items()},
        "token_to_id": dict(TOKEN_TO_ID),
    }


def write_vocab_json(path: Path | None = None) -> Path:
    """Write vocab.json (default: alongside this module) and return it."""
    target = path if path is not None else Path(__file__).with_name("vocab.json")
    target.write_text(json.dumps(vocab_as_dict(), indent=2) + "\n", encoding="utf-8")
    return target
