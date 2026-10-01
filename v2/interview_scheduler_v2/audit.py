"""Canonical fingerprints and run provenance without a Git/runtime dependency."""

from dataclasses import fields, is_dataclass
from datetime import date, datetime, timezone
from enum import Enum
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
from collections.abc import Mapping
from uuid import uuid4
from zoneinfo import ZoneInfo

from .version import APPLICATION_VERSION


def _canonical(value):
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return {"iso8601": value.isoformat(), "timezone": str(value.tzinfo)}
    if isinstance(value, date):
        return value.isoformat()
    if is_dataclass(value):
        return {field.name: _canonical(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(_canonical(key)): _canonical(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted(_canonical(item) for item in value)
    if isinstance(value, (tuple, list)):
        # Slot/person ordering affects adjacency and seeded tie breaking.
        return [_canonical(item) for item in value]
    return value


def canonical_json(value) -> str:
    return json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def fingerprint(value) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _package_source_fingerprint() -> str:
    root = Path(__file__).resolve().parent
    digest = sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode("utf-8") + b"\0")
        digest.update(path.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return digest.hexdigest()


def run_provenance(problem, config, relaxation_mode) -> dict[str, str | int]:
    generated_at = datetime.now(timezone.utc)
    settings = {
        "run_id": str(uuid4()),
        "generated_at_utc": generated_at.isoformat(),
        "generated_at_et": generated_at.astimezone(ZoneInfo("America/New_York")).isoformat(),
        "application_version": APPLICATION_VERSION,
        "package_source_sha256": _package_source_fingerprint(),
        "schema_version": problem.schema_version,
        "fingerprint_format": "canonical-json-v1",
        "input_sha256": fingerprint(problem),
        "configuration_sha256": fingerprint({"config": config, "relaxation_mode": relaxation_mode}),
        "python_version": platform.python_version(),
        "solver_stages_json": "[]",
    }
    for package in ("ortools", "streamlit", "pandas", "openpyxl", "xlsxwriter", "tzdata"):
        try:
            settings[f"version:{package}"] = version(package)
        except PackageNotFoundError:
            settings[f"version:{package}"] = "not installed"
    for group, policy in config.group_policies.items():
        settings[f"group_policy:{group.value}"] = canonical_json(policy)
    for person_id, policy in config.person_policies.items():
        settings[f"person_policy:{person_id}"] = canonical_json(policy)
    return settings
