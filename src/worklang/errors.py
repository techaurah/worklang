"""Error types for WorkLang.

Every validation failure raises one of these with a message that names
the field, the offending value, and what was expected. They subclass
ValueError so callers can catch them generically.
"""


class WorkLangError(ValueError):
    """Base class for all WorkLang schema and tokenizer errors."""


class SchemaError(WorkLangError):
    """A trace or event failed schema validation (DESIGN.md trace schema)."""


class TokenError(WorkLangError):
    """Tokenisation or detokenisation failed (DESIGN.md tokenizer spec)."""


class BaselineError(WorkLangError):
    """A baseline was used before fitting or fitted on no data."""


class MiniError(WorkLangError):
    """The mini model was used before training or misconfigured."""


class DemoError(WorkLangError):
    """The futures demo could not be built or its data is inconsistent."""
