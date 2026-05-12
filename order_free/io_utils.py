from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def list_images(image_dir: str) -> List[str]:
    image_dir_path = Path(image_dir)
    image_paths = []
    for path in image_dir_path.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            image_paths.append(str(path.resolve()))
    return sorted(image_paths)


def stable_hash(payload: Any) -> str:
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=_json_default)
    return hashlib.sha1(serialized.encode("utf-8")).hexdigest()


def descriptor_cache_key(image_path: str, extractor_name: str, config_hash: str) -> str:
    stat = os.stat(image_path)
    payload = {
        "path": os.path.abspath(image_path),
        "mtime_ns": stat.st_mtime_ns,
        "size": stat.st_size,
        "extractor": extractor_name,
        "config_hash": config_hash,
    }
    return stable_hash(payload)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: str, payload: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=_json_default)


def write_csv(path: str, fieldnames: List[str], rows: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_placeholder(path: str, message: str) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(message.rstrip() + "\n")
