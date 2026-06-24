"""Token parsing helpers for bracketed Fragment-SELFIES strings."""

from __future__ import annotations

import re
from dataclasses import dataclass

TOKEN_RE = re.compile(r"\[[^\]]+\]")

POP_TOKEN = "[pop]"
DOT_TOKEN = "[.]"
SELFIES_START = "[SELFIES]"
SELFIES_END = "[ENDSELFIES]"
FRAGMENT_START = "[Frag]"


@dataclass(frozen=True)
class FragmentToken:
    attachment_index: int | None = None


def split_tokens(value: str) -> list[str]:
    """Split a bracketed token string.

    The function is intentionally strict: non-whitespace text outside bracketed
    tokens is rejected so malformed generated strings do not silently change
    meaning.
    """

    tokens = TOKEN_RE.findall(value)
    remainder = TOKEN_RE.sub("", value).strip()
    if remainder:
        raise ValueError(f"unparsed text outside tokens: {remainder!r}")
    return tokens


def make_fragment_token(attachment_index: int | None = None) -> str:
    if attachment_index is None:
        return FRAGMENT_START
    return f"[Frag@{attachment_index}]"


def parse_fragment_token(symbol: str) -> FragmentToken | None:
    if symbol == FRAGMENT_START:
        return FragmentToken(None)
    if not (symbol.startswith("[Frag@") and symbol.endswith("]")):
        return None
    try:
        return FragmentToken(int(symbol[6:-1]))
    except ValueError:
        return None


def make_attachment_token(index: int) -> str:
    return f"[Attach:{index}]"


def parse_attachment_token(symbol: str) -> int | None:
    if not (symbol.startswith("[Attach:") and symbol.endswith("]")):
        return None
    try:
        return int(symbol[8:-1])
    except ValueError:
        return None
