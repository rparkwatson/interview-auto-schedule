"""Stable identifiers for the v2 scheduling domain."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Any


_SPACE_RE = re.compile(r"\s+")
_NUMERIC_MODULUS = 10**12


def normalize_name(name: str) -> str:
    """Return the canonical comparison form of a display name."""

    value = unicodedata.normalize("NFKC", str(name or ""))
    return _SPACE_RE.sub(" ", value).strip().casefold()


def _group_prefix(group: Any) -> str:
    value = getattr(group, "value", group)
    normalized = str(value or "").strip().casefold().replace("_", "-")
    if normalized not in {"student", "adcom"}:
        raise ValueError(f"Unsupported interviewer group: {value!r}")
    return {"student": "STU", "adcom": "ADC"}[normalized]


def generated_interviewer_id(group: Any, name: str) -> str:
    """Generate a deterministic, readable ``STU/ADC-<digits>`` identifier.

    The digest is deliberately independent of Python's randomized ``hash()``.
    Collision detection remains a validation responsibility because all finite
    digests can collide and duplicate normalized names are usually data errors.
    """

    normalized = normalize_name(name)
    if not normalized:
        raise ValueError("An interviewer name is required to generate an ID")
    prefix = _group_prefix(group)
    digest = hashlib.sha256(normalized.encode("utf-8")).digest()
    number = int.from_bytes(digest[:8], "big") % _NUMERIC_MODULUS
    return f"{prefix}-{number:012d}"


def interviewer_id(group: Any, name: str, explicit_id: str | None = None) -> str:
    """Use a non-blank explicit ID or generate the canonical deterministic ID."""

    if explicit_id is not None:
        value = str(explicit_id).strip()
        if not value:
            raise ValueError("Explicit interviewer IDs cannot be blank")
        return value
    return generated_interviewer_id(group, name)
