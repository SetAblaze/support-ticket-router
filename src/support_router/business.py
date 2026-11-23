"""Retail support triage and the relative cost of a wrong queue."""
import math
import re

import numpy as np

from .config import CATEGORIES, CATEGORY_TO_QUEUE
from .text import preprocess

# Unit: one manual review. These are editable educational assumptions.
MISROUTE_COST = {"ACCOUNT": 20, "PAYMENT": 20, "REFUNDS": 20, "CANCELLATION_FEE": 20,
                "ORDER": 10, "SHIPPING": 10}
DEFAULT_MISROUTE_COST = 5.
REVIEW_ACTION = "REVIEW"

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


def _finite_cost(value, name, *, allow_zero=True):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"{name} must be finite and {'non-negative' if allow_zero else 'positive'}")
    return value


def confidence_floor(policy, category):
    base = _finite_cost(policy.get("confidence_threshold", 1.), "confidence_threshold")
    if base > 1:
        raise ValueError("confidence_threshold must be at most one")
    costs = policy.get("misroute_cost")
    if not costs:
        return float(base)
    review_cost = _finite_cost(policy.get("review_cost", 1.), "review_cost", allow_zero=False)
    default_cost = _finite_cost(policy.get("default_misroute_cost", DEFAULT_MISROUTE_COST),
                                "default_misroute_cost", allow_zero=False)
    category_cost = _finite_cost(costs.get(category, default_cost), f"misroute_cost[{category}]",
                                 allow_zero=False)
    return max(float(base), 1 - review_cost / category_cost)


def loss_matrix(classes, costs=None, review_cost=1., default_misroute_cost=DEFAULT_MISROUTE_COST):
    """Return ``L(y, a)`` for every true category ``y`` and action ``a``.

    Routing to the true category costs nothing. Routing to any other category
    costs that category's misroute cost. Handing the ticket to an operator costs
    one review, independently of the true category.
    """
    classes = list(classes)
    if not classes or len(classes) != len(set(classes)):
        raise ValueError("classes must be non-empty and unique")
    if REVIEW_ACTION in classes:
        raise ValueError("REVIEW is an action, not a trained class")
    if "OTHER" in classes:
        raise ValueError("OTHER is a rejection label, not a trained business class")
    costs = dict(costs or MISROUTE_COST)
    unknown = set(costs) - set(classes)
    if unknown:
        raise ValueError(f"Unknown cost categories: {sorted(unknown)}")
    if REVIEW_ACTION in costs:
        raise ValueError("REVIEW is an action, not a cost category")
    review_cost = _finite_cost(review_cost, "review_cost", allow_zero=False)
    default_misroute_cost = _finite_cost(default_misroute_cost, "default_misroute_cost",
                                         allow_zero=False)
    for category, value in costs.items():
        costs[category] = _finite_cost(value, f"misroute_cost[{category}]", allow_zero=False)
    actions = classes + [REVIEW_ACTION]
    matrix = np.zeros((len(classes), len(actions)), dtype=float)
    for i, truth in enumerate(classes):
        wrong = float(costs.get(truth, default_misroute_cost))
        for j, action in enumerate(actions):
            if action == REVIEW_ACTION:
                matrix[i, j] = float(review_cost)
            elif action != truth:
                matrix[i, j] = wrong
    return classes, actions, matrix


def expected_losses(probabilities, classes, costs=None, review_cost=1.,
                    default_misroute_cost=DEFAULT_MISROUTE_COST):
    """``R(a|x) = sum_y P(y|x) L(y,a)`` for each routing action and review."""
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim == 1:
        probabilities = probabilities[None, :]
    classes = list(classes)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(classes):
        raise ValueError("Probability columns do not match classes")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("Invalid class probabilities")
    if not np.allclose(probabilities.sum(axis=1), 1., atol=1e-6):
        raise ValueError("Class probabilities must sum to one")
    _, actions, matrix = loss_matrix(classes, costs, review_cost, default_misroute_cost)
    return actions, probabilities @ matrix


def cost_sensitive_action(probabilities, classes, costs=None, review_cost=1.,
                          default_misroute_cost=DEFAULT_MISROUTE_COST):
    """Choose the action with the lowest expected loss for each row.

    Ties between a category and ``REVIEW`` deliberately prefer ``REVIEW``:
    the cheaper-looking automation must be strictly better to be selected.
    """
    actions, losses = expected_losses(probabilities, classes, costs, review_cost, default_misroute_cost)
    review_index = len(classes)
    chosen = []
    for row in losses:
        best_class_loss = float(np.min(row[:review_index]))
        if row[review_index] <= best_class_loss + 1e-12:
            chosen.append(REVIEW_ACTION)
        else:
            chosen.append(actions[int(np.argmin(row[:review_index]))])
    return chosen, losses, actions


def cost_matrix(classes=None, costs=None, review_cost=1.,
                default_misroute_cost=DEFAULT_MISROUTE_COST):
    """Serialisable loss matrix for reports and business review."""
    classes = list(classes or CATEGORIES)
    _, actions, matrix = loss_matrix(classes, costs, review_cost, default_misroute_cost)
    return {"classes": classes, "actions": actions,
            "matrix": matrix.tolist(),
            "misroute_cost": dict(costs or MISROUTE_COST),
            "review_cost": float(review_cost),
            "default_misroute_cost": float(default_misroute_cost)}
