"""Operator feedback loop: proposal -> confirmation or correction -> verified label.

The service never treats its own proposal as ground truth. Only an operator
decision is stored. The loop makes post-deployment accuracy measurable and
produces a queue of corrected examples for the next retraining round.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .config import CATEGORIES, CATEGORY_TO_QUEUE, MANUAL_QUEUE
from .text import preprocess

VERIFIED = "VERIFIED"
CORRECTED = "CORRECTED"


def _final_category(operator_category):
    if operator_category is None:
        return None
    if operator_category not in CATEGORIES and operator_category != "OTHER":
        raise ValueError(f"Unknown operator category: {operator_category}")
    return operator_category


def make_feedback_event(result, operator_category=None, *, confirmed=None, reviewer_id=None,
                        timestamp=None, note=None, redacted_text=None):
    """Build one immutable feedback record from a routing result."""
    proposed = result.get("suggested_category")
    if proposed is None:
        raise ValueError("Feedback requires a proposed category")
    final = _final_category(operator_category)
    if confirmed is None:
        confirmed = final is None or final == proposed
    if confirmed and final is not None and final != proposed:
        raise ValueError("confirmed=True contradicts a different operator category")
    if not confirmed and final is None:
        raise ValueError("A correction requires operator_category")
    outcome = VERIFIED if confirmed else CORRECTED
    final = proposed if confirmed else final
    now = timestamp or datetime.now(timezone.utc).isoformat()
    text = result.get("_feedback_text")
    return {
        "event_id": uuid.uuid4().hex,
        "created_at_utc": now,
        "model_id": result.get("model_id"),
        "policy_version": result.get("policy_version"),
        "proposed_category": proposed,
        "proposed_queue": CATEGORY_TO_QUEUE.get(proposed, MANUAL_QUEUE),
        "confidence": result.get("confidence"),
        "similarity": result.get("similarity"),
        "final_category": final,
        "final_queue": CATEGORY_TO_QUEUE.get(final, MANUAL_QUEUE) if final else MANUAL_QUEUE,
        "outcome": outcome,
        "reviewer_id": reviewer_id,
        "note": note,
        "text_sha256": hashlib.sha256(preprocess(text).encode("utf-8")).hexdigest() if text else None,
        "redacted_text": redacted_text,
    }


def append_feedback(path, event):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, ensure_ascii=False) + "\n")


def load_feedback(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def feedback_metrics(events, min_support=30):
    """Measured accuracy from operator decisions, with rare-slice warnings."""
    if not events:
        return {"n": 0, "note": "No verified feedback yet"}
    verified = sum(event["outcome"] == VERIFIED for event in events)
    corrected = sum(event["outcome"] == CORRECTED for event in events)
    per_category = {}
    confusion = {}
    for event in events:
        proposed = event.get("proposed_category")
        final = event.get("final_category")
        row = per_category.setdefault(proposed, {"proposed_n": 0, "verified_n": 0, "corrected_n": 0})
        row["proposed_n"] += 1
        row["verified_n"] += event["outcome"] == VERIFIED
        row["corrected_n"] += event["outcome"] == CORRECTED
        confusion.setdefault(proposed, {})
        confusion[proposed][final] = confusion[proposed].get(final, 0) + 1
    for row in per_category.values():
        row["measured_accuracy"] = row["verified_n"] / row["proposed_n"] if row["proposed_n"] else None
        row["support_status"] = ("INSUFFICIENT_SUPPORT" if row["proposed_n"] < min_support
                                 else "ADEQUATE")
    return {
        "n": len(events), "verified": verified, "corrected": corrected,
        "measured_accuracy": verified / len(events),
        "correction_rate": corrected / len(events),
        "per_category": per_category,
        "confusion_proposed_to_final": confusion,
        "caveat": ("Operator confirmations are not an external audit: a rubber-stamped wrong "
                   "proposal would look correct"),
    }


def retraining_candidates(events):
    """Corrected events are the only rows that carry new information."""
    return [event for event in events if event["outcome"] == CORRECTED]


def simulate_stream(bundle, frame, *, seed=42, operator_error=0.0, reviewer_id="simulated"):
    """Simulate operator decisions on a labelled stream for the feedback demo.

    ``operator_error`` is the chance that the operator confirms a wrong proposal
    or rejects a correct one. It makes the loop's own error visible instead of
    pretending that feedback is perfect.
    """
    import numpy as np

    from .routing import route_ticket

    rng = np.random.default_rng(seed)
    events = []
    for row in frame.itertuples(index=False):
        result = route_ticket(row.text, bundle)
        result["_feedback_text"] = row.text
        truth = row.category
        proposal = result.get("suggested_category")
        noisy = rng.random() < operator_error
        if noisy and proposal == truth:
            alternatives = [c for c in CATEGORIES if c != truth]
            final = alternatives[int(rng.integers(len(alternatives)))]
            confirmed = False
        elif noisy:
            final, confirmed = None, True
        else:
            final, confirmed = (None, True) if proposal == truth else (truth, False)
        event = make_feedback_event(
            result, final, confirmed=confirmed, reviewer_id=reviewer_id, redacted_text=None)
        events.append(event)
    return events
