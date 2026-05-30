"""Minimal YAML-like aircraft configuration loader.

The project intentionally avoids a hard PyYAML dependency. The parser supports
simple ``key: value`` files with comments and numeric/string scalars, which is
sufficient for the checked-in aircraft configs.
"""
from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any

from .parameters import UAVParameters


CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs" / "aircraft"


def _parse_scalar(raw: str) -> Any:
    value = raw.strip().strip('"').strip("'")
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        if any(ch in value for ch in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def load_aircraft_config(name: str = "small_mav") -> dict[str, Any]:
    path = CONFIG_ROOT / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Aircraft config not found: {path}")
    data: dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        body = line.split("#", 1)[0].strip()
        if not body or ":" not in body:
            continue
        key, value = body.split(":", 1)
        key = key.strip()
        if not key or not value.strip():
            continue
        data[key] = _parse_scalar(value)
    return data


def parameters_from_config(name: str = "small_mav") -> UAVParameters:
    config = load_aircraft_config(name)
    allowed = {field.name for field in fields(UAVParameters)}
    kwargs = {key: value for key, value in config.items() if key in allowed}
    return UAVParameters(**kwargs)
