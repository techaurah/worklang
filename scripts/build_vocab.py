#!/usr/bin/env python3
"""Regenerate src/worklang/vocab.json from the vocabulary constants.

The constants in worklang.vocab are the source of truth; vocab.json is
a generated artefact for non-Python consumers. Run after any schema
change: python scripts/build_vocab.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from worklang.vocab import write_vocab_json  # noqa: E402

if __name__ == "__main__":
    print(f"wrote {write_vocab_json()}")
