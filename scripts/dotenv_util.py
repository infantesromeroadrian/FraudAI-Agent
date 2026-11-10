"""Minimal dotenv reader shared by the smoke test and the shell helpers (stdlib only).

The file is PARSED, never executed or expanded: no shell, no ``$(...)``, no variable
substitution. Only the requested keys are returned and values are never logged.
"""

from __future__ import annotations

from pathlib import Path


def read_dotenv(path: Path, wanted: set[str]) -> dict[str, str]:
    """Values of the ``wanted`` keys in a dotenv file (the last definition wins)."""
    values: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return values
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.removeprefix("export ").strip()
        if key not in wanted:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    return values
