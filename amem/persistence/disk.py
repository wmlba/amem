"""Disk-based persistence utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import msgpack


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Any, path: Path):
    ensure_dir(path.parent)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: Path) -> Any:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def save_msgpack(data: Any, path: Path):
    ensure_dir(path.parent)
    with open(path, "wb") as f:
        msgpack.pack(data, f, use_bin_type=True)


def load_msgpack(path: Path) -> Any:
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return msgpack.unpack(f, raw=False)
