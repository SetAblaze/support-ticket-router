"""Regression tests for the post-audit methodology fixes."""
import numpy as np
import pandas as pd
import pytest

import support_router as sc
from support_router.business import cost_sensitive_action, expected_losses, loss_matrix
from support_router.evaluation import (
    cascade_metrics,
    cascade_ood_metrics,
    ood_acceptance_curve,
    per_category_calibration,
    risk_coverage_curve,
)
from support_router.feedback import feedback_metrics, make_feedback_event, retraining_candidates
from support_router.multiple_requests import request_parts


class StubModel:
    model_name = "stub"
    model_id = "stub-model"

    def __init__(self, probabilities, classes, policy):
        self._probabilities = np.asarray(probabilities, dtype=float)
        self.classes_ = np.asarray(classes, dtype=object)
        self.policy = policy

    def predict_proba(self, texts):
        return np.tile(self._probabilities, (len(texts), 1))

    def similarity(self, texts):
        return np.full(len(texts), .9)


def _policy(**overrides):
    policy = {"auto_enabled": True, "confidence_threshold": .85, "similarity_threshold": .45,
              "version": "test", "review_cost": 1., "default_misroute_cost": 5.,
              "misroute_cost": {"CONTACT": 2., "PAYMENT": 20.}}
    policy.update(overrides)
    return policy


def test_expected_loss_matches_audit_example():
    classes = ["CONTACT", "PAYMENT"]
    actions, losses = expected_losses([.86, .14], classes,
                                      costs={"CONTACT": 20., "PAYMENT": 20.},
                                      review_cost=1.)
    row = dict(zip(actions, losses[0], strict=True))
    assert row["CONTACT"] == pytest.approx(2.8)
    assert row["PAYMENT"] == pytest.approx(17.2)
    assert row["REVIEW"] == pytest.approx(1.)
    chosen, _, _ = cost_sensitive_action([.86, .14], classes,
                                         costs={"CONTACT": 20., "PAYMENT": 20.}, review_cost=1.)
    assert chosen == ["REVIEW"]


def test_loss_matrix_is_square_with_review():
    classes, actions, matrix = loss_matrix(["A", "B"], costs={"A": 10.}, review_cost=1.)
    assert actions == ["A", "B", "REVIEW"]
    assert matrix.shape == (2, 3)
    assert matrix[0, 0] == 0 and matrix[0, 1] == 10 and matrix[0, 2] == 1


def test_route_blocks_when_expected_loss_prefers_review():
    model = StubModel([.86, .14], ["CONTACT", "PAYMENT"], _policy())
    result = sc.route_ticket("Please tell me how to reach your staff.", model)
    assert result["status"] == "MANUAL_REVIEW"
    assert result["decision_category"] == "OTHER"
    assert result["expected_loss_action"] == "REVIEW"
    assert "EXPECTED_LOSS_PREFERS_REVIEW" in result["reason_codes"]
    assert result["category_probabilities"]["PAYMENT"] == pytest.approx(.14)


def test_route_accepts_when_expected_loss_prefers_prediction():
    model = StubModel([.97, .03], ["CONTACT", "PAYMENT"], _policy())
    result = sc.route_ticket("Please tell me how to reach your staff.", model)
    assert result["status"] == "AUTO_RECOMMENDED"
    assert result["expected_loss_action"] == "CONTACT"
    assert result["accepted_category"] == "CONTACT"


def test_risk_coverage_curve_is_monotone_in_coverage():
    probabilities = np.array([[.9, .1], [.6, .4], [.55, .45]])
    labels = ["A", "A", "B"]
    curve = risk_coverage_curve(probabilities, labels, ["A", "B"], thresholds=(.5, .8, .95))
    coverages = [row["coverage"] for row in curve["curve"]]
    assert coverages == sorted(coverages, reverse=True)
    assert curve["curve"][-1]["accepted_n"] == 0
    assert ood_acceptance_curve(probabilities, ["A", "B"], thresholds=(.8,))["curve"][0]["accepted_n"] == 1


def test_per_category_calibration_flags_rare_support():
    frame = pd.DataFrame({"text": [f"text {i}" for i in range(6)],
                          "category": ["ACCOUNT"] * 5 + ["NEWSLETTER"]})
    model = StubModel([.9, .1], ["ACCOUNT", "NEWSLETTER"], _policy())
    report = per_category_calibration(model, frame, min_support=3)
    assert report["per_category"]["NEWSLETTER"]["support"] == 1
    assert report["per_category"]["NEWSLETTER"]["support_status"] == "INSUFFICIENT_SUPPORT"
    assert "NEWSLETTER" in report["insufficient_support"]
    assert report["per_category"]["ACCOUNT"]["brier"] >= 0


def test_cascade_metrics_separate_fixes_from_regressions():
    routes = [
        {"primary_category": "PAYMENT", "category": "REFUNDS", "classification_source": "minilm"},
        {"primary_category": "ORDER", "category": "PAYMENT", "classification_source": "minilm"},
        {"primary_category": "ACCOUNT", "category": "ACCOUNT", "classification_source": "primary"},
    ]
    labels = ["REFUNDS", "ORDER", "ACCOUNT"]
    metrics = cascade_metrics(routes, labels)
    assert metrics["corrected"] == 1
    assert metrics["regressed"] == 1
    assert metrics["suggestions_n"] == 2
    assert metrics["suggestion_precision"] == pytest.approx(.5)
    ood = cascade_ood_metrics([{"classification_source": "minilm"}, {"classification_source": "primary"}])
    assert ood["false_hints"] == 1 and ood["false_hint_rate"] == pytest.approx(.5)


def test_independent_benchmark_is_disjoint_and_labelled():
    id_frame, ood_frame, risk_frame, multi_frame = sc.independent_cases.frames()
    assert len(id_frame) == 110 and len(ood_frame) == 20
    assert len(risk_frame) == 16 and len(multi_frame) == 12
    assert id_frame.groupby("difficulty").size().to_dict() == {"natural": 55, "no_keywords": 55}
    assert set(id_frame.category) == set(sc.CATEGORIES)
    root = sc.project_root()
    development = set()
    for path in ("data/eval/hard_test.csv", "data/eval/ood_eval.csv", "data/eval/risk_regression.csv"):
        development |= set(pd.read_csv(root / path).text)
    for frame in (sc.manual_cases.frames()[0], sc.manual_cases.frames()[1],
                  sc.scenario_data.frames()[0], sc.scenario_data.frames()[1]):
        development |= set(frame.text)
    assert not (set(id_frame.text) | set(ood_frame.text) | set(risk_frame.text) | set(multi_frame.text)) & development
    assert sc.independent_cases.sha256() == "97d8404692fc8c8210ea86c8d2fbc7cb68cffbdec60c15e14a3cf070767eac9c"


def test_independent_risk_and_multi_sets_run_without_crashing():
    _, _, risk_frame, multi_frame = sc.independent_cases.frames()
    for text in risk_frame.text:
        assert isinstance(sc.detect_signals(text)["risk_flags"], list)
    for text in multi_frame.text:
        assert isinstance(request_parts(text), list)
    assert not any(request_parts(text) for text in
                   sc.independent_cases.frames()[0].query("difficulty == 'natural'").text)


def test_feedback_loop_metrics_and_retraining_queue():
    result = {"suggested_category": "PAYMENT", "confidence": .9, "similarity": .8,
              "model_id": "m", "policy_version": "p", "_feedback_text": "my card failed"}
    verified = make_feedback_event(result, confirmed=True, reviewer_id="op")
    corrected = make_feedback_event(result, "REFUNDS", confirmed=False, reviewer_id="op")
    assert verified["outcome"] == "VERIFIED"
    assert corrected["outcome"] == "CORRECTED"
    metrics = feedback_metrics([verified, corrected], min_support=1)
    assert metrics["n"] == 2 and metrics["measured_accuracy"] == pytest.approx(.5)
    assert metrics["per_category"]["PAYMENT"]["corrected_n"] == 1
    assert retraining_candidates([verified, corrected]) == [corrected]
    with pytest.raises(ValueError):
        make_feedback_event(result, "UNKNOWN", confirmed=False)
