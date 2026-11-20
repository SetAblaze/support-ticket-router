"""Нормализация текста и разбор лингвистических флагов Bitext."""
from __future__ import annotations

import hashlib
import re
from typing import Any

import pandas as pd

from .config import LINGUISTIC_FLAGS


def parse_linguistic_flags(value) -> tuple[str, ...]:
    """Parse single-character Bitext codes, ignoring separators and nulls.

    Unknown letter codes are kept so EDA can flag them for investigation.
    Multiple flags per utterance are possible. Return unique sorted codes.
    """
    if value is None:
        return ()
    try:
        if bool(pd.isna(value)):
            return ()
    except (TypeError, ValueError):
        pass
    raw = str(value).strip().upper()
    if raw in {"", "NAN", "NONE", "NULL", "N/A", "<NA>"}:
        return ()
    return tuple(sorted(set(re.findall(r"[A-Z]", raw))))

def flag_membership(series: pd.Series, flag: str) -> pd.Series:
    if flag not in LINGUISTIC_FLAGS:
        raise ValueError(f"Unknown documented Bitext flag: {flag!r}")
    return series.map(lambda raw: flag in parse_linguistic_flags(raw)).astype(bool)

_PLACEHOLDER = re.compile(r"\{\{[^{}]*\}\}|\{[^{}]*\}|<[a-z][a-z ]*>", re.I)

_WS = re.compile(r"\s+")

def preprocess(text: Any) -> str:
    """Deterministic, idempotent; retains negation before vectorizer tokenization."""
    if not isinstance(text, str):
        return ""
    t = text.replace("’", "'").replace("‘", "'").replace("`", "'")
    t = _PLACEHOLDER.sub(" ", t).lower()
    for old, new in [("can't", "can not"), ("cannot", "can not"),
                     ("won't", "will not"), ("shan't", "shall not")]:
        t = re.sub(r"\b" + re.escape(old) + r"\b", new, t)
    t = re.sub(r"\b([a-z]+)n't\b", r"\1 not", t)
    return _WS.sub(" ", t).strip()

def preprocess_batch(texts):
    return [preprocess(t) for t in texts]

def text_id(text: str) -> str:
    return hashlib.sha256(preprocess(text).encode("utf-8")).hexdigest()
