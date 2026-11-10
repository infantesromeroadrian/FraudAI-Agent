#!/usr/bin/env python3
"""Fail when an alert rule or dashboard references a metric the application cannot emit.

The known metrics are read from ``src/fraudai/core/metrics.py`` (the names given to
prometheus_client Counter/Histogram/Gauge/Info), expanded with the series suffixes
prometheus_client really exposes, plus the synthetic ``up`` series of a scrape job.
Usage: check_rule_metrics.py <rules.yml|dashboard.json>...
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
METRICS_SOURCE = ROOT / "src" / "fraudai" / "core" / "metrics.py"
SCRAPE_SERIES = {"up"}
_DECLARATION = re.compile(r"=\s*(Counter|Histogram|Gauge|Info|Summary)\(\s*\"([a-z_0-9]+)\"")
_IDENTIFIER = re.compile(r"[a-zA-Z_:][a-zA-Z0-9_:]*")
_PROMQL_WORDS = {
    "sum", "rate", "increase", "histogram_quantile", "by", "without", "and", "or", "unless",
    "on", "ignoring", "group_left", "group_right", "avg", "min", "max", "count", "time",
    "hour", "bool", "offset", "irate", "avg_over_time", "max_over_time", "abs", "clamp_min",
    "clamp_max", "absent", "vector", "scalar", "topk", "bottomk", "le",
}  # fmt: skip


def exposed_series() -> set[str]:
    """Series names the application exposes, from the declarations in metrics.py."""
    names: set[str] = set(SCRAPE_SERIES)
    for kind, name in _DECLARATION.findall(METRICS_SOURCE.read_text()):
        if kind == "Counter":
            base = name.removesuffix("_total")
            names |= {f"{base}_total", f"{base}_created"}
        elif kind in {"Histogram", "Summary"}:
            names |= {f"{name}_bucket", f"{name}_sum", f"{name}_count", f"{name}_created"}
        elif kind == "Info":
            names.add(f"{name}_info")
        else:
            names.add(name)
    return names


def rule_expressions(text: str) -> list[str]:
    """The ``expr:`` values of a Prometheus rules file (folded or single-line)."""
    expressions: list[str] = []
    lines = text.splitlines()
    for index, line in enumerate(lines):
        match = re.match(r"(\s*)expr:\s*(.*)$", line)
        if not match:
            continue
        indent, rest = len(match.group(1)), match.group(2).strip()
        if rest and rest not in {">", "|", ">-", "|-"}:
            expressions.append(rest)
            continue
        block: list[str] = []
        for follower in lines[index + 1 :]:
            if follower.strip() and len(follower) - len(follower.lstrip()) <= indent:
                break
            block.append(follower.strip())
        expressions.append(" ".join(block))
    return expressions


def referenced_metrics(expression: str) -> set[str]:
    stripped = re.sub(r"\"[^\"]*\"", '""', expression)  # label values
    stripped = re.sub(r"\{[^}]*\}", "", stripped)  # label matchers
    stripped = re.sub(r"\[[^\]]*\]", "", stripped)  # range selectors
    stripped = re.sub(r"\b(by|without|on|ignoring)\s*\([^)]*\)", " ", stripped)
    found = {token for token in _IDENTIFIER.findall(stripped) if token.lower() not in _PROMQL_WORDS}
    return {token for token in found if not re.fullmatch(r"\d+(\.\d+)?", token)}


def dashboard_expressions(text: str) -> list[str]:
    expressions: list[str] = []

    def walk(node: object) -> None:
        if isinstance(node, dict):
            if isinstance(node.get("expr"), str):
                expressions.append(node["expr"])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(json.loads(text))
    return expressions


def main(paths: list[str]) -> int:
    known = exposed_series()
    missing: dict[str, set[str]] = {}
    for raw in paths:
        path = Path(raw)
        text = path.read_text()
        expressions = (
            dashboard_expressions(text) if path.suffix == ".json" else rule_expressions(text)
        )
        for expression in expressions:
            for metric in referenced_metrics(expression):
                if metric not in known:
                    missing.setdefault(metric, set()).add(path.name)
    if missing:
        for metric, files in sorted(missing.items()):
            print(f"UNKNOWN METRIC {metric} (used in {', '.join(sorted(files))})")
        return 1
    print(f"OK: every metric in {len(paths)} file(s) is exposed by the application or the scrape")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
