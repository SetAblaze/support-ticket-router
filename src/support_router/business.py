"""Retail support triage and the relative cost of a wrong queue."""
import math
import re

from .config import CATEGORY_TO_QUEUE
from .text import preprocess

# Unit: one manual review. These are editable educational assumptions.
MISROUTE_COST = {"ACCOUNT": 20, "PAYMENT": 20, "REFUNDS": 20, "CANCELLATION_FEE": 20,
                "ORDER": 10, "SHIPPING": 10}

_IMPACTS = {
    "DUPLICATE_DEBIT": r"\b(?:i (?:was|got)|(?:my (?:card|account)|you) (?:was |has been )?)\s*charged (?:me )?(?:twice|two times)\b",
    "ACCESS_BLOCKED": r"\b(?:i(?: am|'m) locked out of my account|my account (?:is|was|has been) (?:locked|blocked))\b",
    "CHECKOUT_BLOCKED": r"\b(?:i (?:can not|am unable to) (?:check out|complete (?:my|the) (?:purchase|payment))|my payment keeps (?:failing|getting declined))\b",
}
_PATTERNS = {key: re.compile(pattern) for key, pattern in _IMPACTS.items()}


def misroute_costs(overrides=None):
    costs = MISROUTE_COST.copy()
    for category, value in (overrides or {}).items():
        if category not in CATEGORY_TO_QUEUE:
            raise ValueError(f"Unknown support category: {category}")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 1:
            raise ValueError(f"Cost for {category} must be finite and at least one manual review")
        costs[category] = float(value)
    return costs


def triage(text, flags, context_reasons, context_priority):
    reasons = [flag for flag in flags if flag in ("POSSIBLE_FRAUD", "POSSIBLE_THREAT", "POSSIBLE_LEGAL_CLAIM")]
    reasons.extend(r for r in context_reasons if r in {"VERIFIED_VIP_DEMO_POLICY", "REPEAT_OPEN_TICKET_DEMO_POLICY",
                                                      "VERIFIED_DEADLINE_OVERDUE"})
    message = re.sub(r'"[^"\n]*"|“[^”\n]*”', " ", preprocess(text))
    # An example or hypothetical inside one clause is weak evidence of active impact.
    for clause in re.split(r"[.;?!\n]|\bbut\b", message):
        if re.search(r"\b(?:if|suppose|hypothetically)\b", clause):
            continue
        reasons.extend(code for code, pattern in _PATTERNS.items() if pattern.search(clause))
    if "POSSIBLE_THREAT" in flags or "POSSIBLE_FRAUD" in flags:
        level = 1
    elif reasons:
        level = min(context_priority, 2)
    else:
        level = context_priority
    return f"P{level}", list(dict.fromkeys(reasons))


def confidence_floor(policy, category):
    base = policy.get("confidence_threshold", 1.)
    costs = policy.get("misroute_cost")
    if not costs:
        return float(base)
    return max(float(base), 1 - policy["review_cost"] / costs.get(category, policy["default_misroute_cost"]))
