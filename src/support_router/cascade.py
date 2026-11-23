"""A local MiniLM second opinion for unfamiliar, single-topic English tickets."""
from dataclasses import dataclass

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

from . import manual_cases, semantic_encoder
from .business import confidence_floor, cost_matrix, misroute_costs
from .config import CATEGORY_TO_QUEUE, SEED
from .multiple_requests import request_parts
from .paths import encoder_dir
from .text import preprocess, preprocess_batch


@dataclass
class SemanticHead:
    classifier: CalibratedClassifierCV
    centers: np.ndarray
    center_categories: np.ndarray

    def scores(self, vectors):
        probabilities = self.classifier.predict_proba(vectors)
        winners = self.classifier.classes_[probabilities.argmax(axis=1)]
        similarities = vectors @ self.centers.T
        support = np.where(winners[:, None] == self.center_categories, similarities, -np.inf).max(axis=1)
        return probabilities, winners, support


def candidate(result, text, policy):
    return (result["priority"] == "P3" and not result["reason_codes"]
            and result["category"] in CATEGORY_TO_QUEUE
            and (result["confidence"] < confidence_floor(policy, result["category"])
                 or result["similarity"] < policy["similarity_threshold"])
            and not request_parts(text))


def second_opinion(result, text, bundle):
    head = getattr(bundle, "semantic_head", None)
    if head is None or not candidate(result, text, bundle.policy):
        return
    thresholds = bundle.policy["semantic_gate"]
    if not thresholds["enabled"]:
        result["semantic_status"] = "NO_VALIDATED_GATE"
        return
    try:
        vector = semantic_encoder.serving_vector(preprocess(text), getattr(bundle, "encoder_directory", None) or encoder_dir())
        if vector is None:
            result["semantic_status"] = "INPUT_TOO_LONG"
            return
        probabilities, categories, support = head.scores(vector[None, :])
        if not np.isfinite(probabilities).all() or not np.isfinite(support).all():
            raise ValueError("Non-finite semantic score")
    except Exception:
        result["semantic_status"] = "UNAVAILABLE"
        return
    p = probabilities[0]
    category = str(categories[0])
    result.update(semantic_confidence=float(p.max()), semantic_similarity=float(support[0]))
    if not (p.max() >= thresholds["confidence"] and support[0] >= thresholds["semantic_similarity"]):
        result["semantic_status"] = "ABSTAINED"
        return
    result.update(semantic_status="SUGGESTED", classification_source="minilm",
                  category=category, suggested_category=category, topic_queue=CATEGORY_TO_QUEUE[category],
                  confidence=float(p.max()), top3={str(head.classifier.classes_[i]): float(p[i])
                                                  for i in np.argsort(-p)[:3]})
    result["reason_codes"].append("SEMANTIC_SUGGESTION_NEEDS_REVIEW")


def fit_business_cascade(bundle, train, calibration, val, ood_dev, *, cost_overrides=None):
    """Fit on train, calibrate on calibration, select an operator gate on primary rejects only."""
    from .routing import batch_route

    if any(set(a.group) & set(b.group) for a, b in ((train, calibration), (calibration, val), (train, val))):
        raise ValueError("Cascade splits must have disjoint template groups")
    costs = misroute_costs(cost_overrides)
    bundle.policy.update(version="retail-cascade-v1", misroute_cost=costs,
                         review_cost=1, default_misroute_cost=5,
                         cost_matrix=cost_matrix(costs=costs, review_cost=1, default_misroute_cost=5))
    if bundle.model_name.startswith("semantic_"):
        return  # An already-semantic primary does not need the same encoder again.
    vectors = semantic_encoder.encode(preprocess_batch(train.text))
    with threadpool_limits(limits=1):
        base = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED).fit(vectors, train.category)
        classifier = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid")
        classifier.fit(semantic_encoder.encode(preprocess_batch(calibration.text)), calibration.category)
    groups = train.reset_index(drop=True).groupby(["category", "intent"]).indices
    head = SemanticHead(classifier, normalize(np.stack([vectors[idx].mean(axis=0) for idx in groups.values()])),
                        np.asarray([cat for cat, _ in groups]))
    dev = manual_cases.frames()[0]
    texts = [*val.text, *dev.text, *ood_dev.text]
    labels = np.array([*val.category, *dev.category, *(["OTHER"] * len(ood_dev))])
    routes = batch_route(bundle, texts)
    eligible = []
    for i, (r, text) in enumerate(zip(routes, texts, strict=True)):
        if set(r["reason_codes"]) <= {"LOW_CONFIDENCE", "LOW_TRAINING_SIMILARITY", "AUTO_POLICY_DISABLED",
                                      "EXPECTED_LOSS_PREFERS_REVIEW", "EXPECTED_LOSS_DIFFERS_FROM_PREDICTION",
                                      "EXPECTED_LOSS_ROUTE_NOT_BETTER_THAN_REVIEW"}:
            if candidate({**r, "reason_codes": []}, text, bundle.policy) and not semantic_encoder.was_truncated(preprocess(text)):
                eligible.append(i)
    gate = {"enabled": False, "eligible_id": int((labels[eligible] != "OTHER").sum()),
            "eligible_ood": int((labels[eligible] == "OTHER").sum()), "wrong_hint_cost": 4, "review_cost": 1}
    choices = []
    if eligible:
        probabilities, pred, support = head.scores(semantic_encoder.encode(preprocess_batch([texts[i] for i in eligible])))
        truth = labels[eligible]
        for threshold in (.75, .80, .85, .90, .95):
            for similarity in (.25, .35, .45, .55, .65, .75):
                use = (probabilities.max(axis=1) >= threshold) & (support >= similarity)
                errors = int((use & (pred != truth)).sum())
                n = int(use.sum())
                gain = n - gate["wrong_hint_cost"] * errors
                if n >= 10 and gain > 0 and not (use & (truth == "OTHER")).any():
                    choices.append((gain, -errors, threshold, similarity, n))
    if choices:
        _, neg_errors, confidence, similarity, n = max(choices)
        gate.update(enabled=True, confidence=confidence, semantic_similarity=similarity,
                    development_suggestions=n, development_errors=-neg_errors)
    bundle.policy["semantic_gate"] = gate
    bundle.semantic_head = head
