"""Operator feedback loop: proposal -> explicit decision -> verified label.

The service never treats its own proposal as ground truth.  A feedback event
can only be created with an explicit operator decision:

* ``confirmed=True``  -> ``VERIFIED`` (the proposed category is accepted);
* ``confirmed=False`` -> ``CORRECTED`` (a different category is required);
* no decision         -> ``ValueError``, never an implicit ``VERIFIED``.

The module also detects duplicate deliveries and conflicting decisions by two
reviewers.  Conflicts go to a manual-resolution queue and are excluded from the
measured accuracy until a human resolves them.
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
PENDING = "PENDING"
OUTCOMES = (VERIFIED, CORRECTED, PENDING)


def _final_category(operator_category):
    if operator_category is None:
        return None
    if operator_category not in CATEGORIES and operator_category != "OTHER":
        raise ValueError(f"Unknown operator category: {operator_category}")
    return operator_category


def _fingerprint(model_id, proposed, final, reviewer_id, text_sha256, outcome):
    payload = "|".join(str(value) for value in
                       (model_id, proposed, final, reviewer_id, text_sha256, outcome))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def make_feedback_event(result, operator_category=None, *, confirmed=None, reviewer_id=None,
                        timestamp=None, note=None, redacted_text=None, event_id=None):
    """Build one immutable feedback record from an explicit operator decision."""
    proposed = result.get("suggested_category")
    if proposed is None:
        raise ValueError("Feedback requires a proposed category")
    if proposed not in CATEGORIES and proposed != "OTHER":
        raise ValueError(f"Unknown proposed category: {proposed}")
    final = _final_category(operator_category)
    if confirmed is None:
        raise ValueError("Explicit operator confirmation is required")
    if not isinstance(confirmed, bool):
        raise ValueError("confirmed must be True or False")
    if confirmed:
        if final is not None and final != proposed:
            raise ValueError("confirmed=True contradicts a different operator category")
        final, outcome = proposed, VERIFIED
    else:
        if final is None:
            raise ValueError("A correction requires operator_category")
        if final == proposed:
            raise ValueError("A correction must name a category different from the proposal")
        outcome = CORRECTED
    now = timestamp or datetime.now(timezone.utc).isoformat()
    text = result.get("_feedback_text")
    text_sha = hashlib.sha256(preprocess(text).encode("utf-8")).hexdigest() if text else None
    model_id = result.get("model_id")
    return {
        "event_id": event_id or uuid.uuid4().hex,
        "event_fingerprint": _fingerprint(model_id, proposed, final, reviewer_id, text_sha, outcome),
        "created_at_utc": now,
        "model_id": model_id,
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
        "text_sha256": text_sha,
        "redacted_text": redacted_text,
        "privacy_note": "text_sha256 is an identifier, not an anonymity guarantee",
    }


def append_feedback(path, event):
    """Append once; return False for an idempotent duplicate delivery."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_feedback(path)
    seen_ids = {item.get("event_id") for item in existing}
    seen_fingerprints = {item.get("event_fingerprint") for item in existing}
    if event.get("event_id") in seen_ids or event.get("event_fingerprint") in seen_fingerprints:
        return False
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, ensure_ascii=False) + "\n")
    return True


def load_feedback(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _conflict_groups(events):
    groups = {}
    for event in events:
        if event.get("outcome") not in (VERIFIED, CORRECTED):
            continue
        key = event.get("text_sha256")
        if not key:
            continue
        groups.setdefault(key, []).append(event)
    conflicts = {}
    for key, rows in groups.items():
        finals = {row.get("final_category") for row in rows}
        if len(finals) > 1:
            conflicts[key] = rows
    return conflicts


def feedback_metrics(events, min_support=30):
    """Measured accuracy from explicit operator decisions, excluding conflicts."""
    if not events:
        return {"n": 0, "note": "No verified feedback yet", "conflict_n": 0,
                "conflict_queue": [], "outcome_counts": {}}
    conflicts = _conflict_groups(events)
    conflict_ids = {row["event_id"] for rows in conflicts.values() for row in rows}
    clean = [event for event in events
             if event.get("outcome") in (VERIFIED, CORRECTED)
             and event.get("event_id") not in conflict_ids]
    verified = sum(event["outcome"] == VERIFIED for event in clean)
    corrected = sum(event["outcome"] == CORRECTED for event in clean)
    per_category = {}
    confusion = {}
    for event in clean:
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
    conflict_queue = [
        {"text_sha256": key,
         "decisions": [{"event_id": row["event_id"], "reviewer_id": row.get("reviewer_id"),
                        "proposed_category": row.get("proposed_category"),
                        "final_category": row.get("final_category"),
                        "outcome": row.get("outcome")}
                       for row in rows]}
        for key, rows in conflicts.items()
    ]
    return {
        "n": len(clean), "verified": verified, "corrected": corrected,
        "measured_accuracy": verified / len(clean) if clean else None,
        "correction_rate": corrected / len(clean) if clean else None,
        "per_category": per_category,
        "confusion_proposed_to_final": confusion,
        "conflict_n": len(conflict_queue),
        "conflict_queue": conflict_queue,
        "outcome_counts": {outcome: sum(e.get("outcome") == outcome for e in events)
                           for outcome in OUTCOMES},
        "caveat": ("Operator confirmations are not an external audit: a rubber-stamped wrong "
                   "proposal would look correct. Conflicting decisions are excluded until resolved."),
    }


def retraining_candidates(events):
    """Corrected, non-conflicting events are the only rows with new information."""
    conflicts = _conflict_groups(events)
    conflict_ids = {row["event_id"] for rows in conflicts.values() for row in rows}
    return [event for event in events
            if event.get("outcome") == CORRECTED and event.get("event_id") not in conflict_ids]


def simulate_stream(bundle, frame, *, seed=42, operator_error=0.0, reviewer_id="simulated"):
    """Simulate operator decisions on a labelled stream for the feedback demo.

    ``operator_error`` is the chance that the operator confirms a wrong proposal
    or rejects a correct one.  It makes the loop's own error visible instead of
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
