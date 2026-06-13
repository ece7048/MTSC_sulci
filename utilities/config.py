"""Configuration helpers for command-line and YAML-driven runs."""

from __future__ import annotations

import ast
import json
from pathlib import Path


def parse_scalar(value):
    """Parse a command-line or YAML scalar into a Python value."""
    if not isinstance(value, str):
        return value

    text = value.strip()
    if text == "":
        return ""

    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None

    if "," in text and not text.startswith(("[", "{", "'", '"')):
        return [parse_scalar(part) for part in text.split(",")]

    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text.strip("\"'")


def _parse_simple_yaml(text):
    """Parse the small YAML subset used by the example config file."""
    root = {}
    stack = [(-1, root)]

    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if ":" not in line:
            raise ValueError(f"Invalid config line: {raw_line}")

        indent = len(line) - len(line.lstrip(" "))
        key, raw_value = line.strip().split(":", 1)
        value = raw_value.strip()

        while stack and indent <= stack[-1][0]:
            stack.pop()

        parent = stack[-1][1]
        if value == "":
            child = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = parse_scalar(value)

    return root


def load_config(path):
    """Load a JSON/YAML config file, returning an empty dict when omitted."""
    if not path:
        return {}

    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")

    if config_path.suffix.lower() == ".json":
        return json.loads(text)

    try:
        import yaml
    except ModuleNotFoundError:
        return _parse_simple_yaml(text)

    return yaml.safe_load(text) or {}


def section(config, name):
    """Return a named config section or an empty mapping."""
    value = config.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Config section '{name}' must be a mapping.")
    return value


def merge_parameters(defaults, config_values, cli_values):
    """Merge default, config-file, and explicit CLI values."""
    params = dict(defaults)
    params.update({k: v for k, v in config_values.items() if v is not None})
    params.update({k: v for k, v in cli_values.items() if v is not None})
    return params


def cli_overrides(namespace, skip=("config",)):
    """Collect explicitly provided argparse values and parse scalar strings."""
    values = vars(namespace)
    return {
        key: parse_scalar(value)
        for key, value in values.items()
        if key not in skip and value is not None
    }
