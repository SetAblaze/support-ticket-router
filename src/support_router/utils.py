"""Служебные функции: хеши файлов, JSON без NaN, версии пакетов."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
from pathlib import Path

import numpy as np


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value

def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(data), ensure_ascii=False, indent=2,
                               allow_nan=False), encoding="utf-8")

def runtime_versions():
    names = ["numpy", "pandas", "scipy", "scikit-learn", "joblib", "matplotlib",
             "seaborn", "wordcloud", "gradio", "nbformat", "nbclient", "pytest", "onnxruntime", "tokenizers"]
    out = {}
    for name in names:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out
