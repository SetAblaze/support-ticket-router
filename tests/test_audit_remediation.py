"""Regression tests for the second audit-remediation round."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

import support_router as sc
from support_router.business import cost_sensitive_action, expected_losses, loss_matrix
from support_router.evaluation import (
    calibration_by_source,
    cascade_detailed_metrics,
    classification_metrics,
    end_to_end_policy_curve,
    multi_request_metrics,
)
from support_router.feedback import (
    append_feedback,
    feedback_metrics,
    load_feedback,
    make_feedback_event,
)
from support_router.identity import ExperimentIdentityError, validate_report_identity
from support_router.multiple_requests import INTENT_SCHEMA, validate_request_parts


class StubBundle:
    model_name = "stub"
    model_id = "stub-model"
    semantic_head = None

    def __init__(self, probabilities, classes, policy):
        self._probabilities = np.asarray(probabilities, dtype=float)
        self.classes_ = np.asarray(classes, dtype=object)
        self.policy = policy

    def predict_proba(self, texts):
        return np.tile(self._probabilities, (len(texts), 1))

    def similarity(self, texts):
        return np.full(len(texts), .9)


def _policy(**overrides):
    policy = {"auto_enabled": False, "confidence_threshold": .85, "similarity_threshold": .45,
              "version": "test", "review_cost": 1., "default_misroute_cost": 5.,
              "misroute_cost": {"ACCOUNT": 20., "PAYMENT": 20.}}
    policy.update(overrides)
    return policy


def test_identity_binds_current_reports_and_rejects_tampering():
    root = sc.project_root()
    meta = json.loads((root / "models/router_meta.json").read_text(encoding="utf-8"))
    report = json.loads((root / "reports/independent_benchmark.json").read_text(encoding="utf-8"))
    assert validate_report_identity(meta, report)["status"] == "CURRENT"
    tampered = json.loads(json.dumps(report))
    tampered["provenance"]["artifact_sha256"] = "0" * 64
    with pytest.raises(ExperimentIdentityError):
        validate_report_identity(meta, tampered)
    unbound = {"provenance": {"model_id": meta["model_id"]}}
    with pytest.raises(ExperimentIdentityError):
        validate_report_identity(meta, unbound)


def test_prepare_data_verify_is_reproducible():
    root = sc.project_root()
    result = subprocess.run([sys.executable, "scripts/prepare_data.py", "--verify"],
                            cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "SPLIT_MATCH" in result.stdout


def test_feedback_requires_explicit_decision_and_detects_conflicts(tmp_path):
    result = {"suggested_category": "PAYMENT", "model_id": "m", "policy_version": "p",
              "_feedback_text": "my card failed"}
    with pytest.raises(ValueError):
        make_feedback_event(result, confirmed=None)
    verified = make_feedback_event(result, confirmed=True, reviewer_id="a")
    duplicate = make_feedback_event(result, confirmed=True, reviewer_id="a")
    duplicate["event_id"] = verified["event_id"]
    path = tmp_path / "events.jsonl"
    assert append_feedback(path, verified) is True
    assert append_feedback(path, duplicate) is False
    assert len(load_feedback(path)) == 1
    corrected = make_feedback_event({**result, "_feedback_text": "my card failed"},
                                    "REFUNDS", confirmed=False, reviewer_id="b")
    metrics = feedback_metrics([verified, corrected])
    assert metrics["n"] == 0 and metrics["conflict_n"] == 1


def test_classification_metrics_rejects_repeated_labels_and_unknown_classes():
    with pytest.raises(ValueError):
        classification_metrics(["A", "B"], ["A", "B"], labels=["A", "B", "A"])
    with pytest.raises(ValueError):
        classification_metrics(["A"], ["C"], labels=["A", "B"])
    metrics = classification_metrics(["A", "B"], ["A", "A"], labels=["A", "B"])
    assert metrics["errors"] == 1 and metrics["labels"] == ["A", "B"]


def test_few_shot_modes_are_exact_and_group_cv_is_disjoint():
    spec = importlib.util.spec_from_file_location("compare_representations",
                                                  sc.project_root() / "scripts/compare_representations.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    train_path = sc.project_root() / "data/processed/train.csv"
    if not train_path.exists():
        pytest.skip("processed train split is not built; run scripts/prepare_data.py --rebuild")
    train = pd.read_csv(train_path, keep_default_na=False)
    for mode in ("balanced", "proportional"):
        sample, meta = module._sample_100(train, 0, mode)
        assert len(sample) == 100 and not sample.text.duplicated().any()
        assert sample.category.nunique() == len(sc.CATEGORIES)
        cv, folds, method, _ = module._cv_for(sample.text.tolist(), sample.category.tolist(),
                                              sample.group.tolist(), 0)
        for train_idx, test_idx in cv.split(sample.text, sample.category, sample.group):
            assert not set(sample.group.iloc[train_idx]) & set(sample.group.iloc[test_idx])
        assert method in {"StratifiedGroupKFold", "GroupKFold", "StratifiedKFold"}


def test_end_to_end_policy_curve_uses_the_real_decision_function():
    frame = pd.DataFrame({"text": ["I need help with my payment", "Please change my address"],
                          "category": ["PAYMENT", "SHIPPING"]})
    ood = pd.DataFrame({"text": ["Who wrote Hamlet?"]})
    bundle = StubBundle([.9, .05, .05], ["PAYMENT", "SHIPPING", "ACCOUNT"], _policy())
    curve = end_to_end_policy_curve(bundle, frame, thresholds=(.5, .95), ood_frame=ood)
    assert curve["mode"] == "SHADOW_POLICY_END_TO_END"
    assert curve["curve"][0]["automatic_n"] >= curve["curve"][-1]["automatic_n"]
    operational = StubBundle([.9, .05, .05], ["PAYMENT", "SHIPPING", "ACCOUNT"], _policy())
    assert sc.routing_metrics(sc.batch_route(operational, frame.text.tolist()),
                              frame.category.tolist())["automatic_n"] == 0


def test_cost_sensitive_tie_prefers_review_and_rejects_invalid_inputs():
    chosen, _, _ = cost_sensitive_action([.5, .5], ["A", "B"], costs={"A": 2., "B": 2.},
                                         review_cost=1.)
    assert chosen == ["REVIEW"]
    with pytest.raises(ValueError):
        expected_losses([.9, .2], ["A", "B"], costs={"A": 2., "B": 2.})
    with pytest.raises(ValueError):
        loss_matrix(["A", "OTHER"], costs={"A": 2.})


def test_risk_dev_and_frozen_sets_are_separate_and_clean():
    root = sc.project_root()
    for name in ("risk_dev", "independent_risk"):
        frame = pd.read_csv(root / "data/eval" / f"{name}.csv", keep_default_na=False)
        for row in frame.itertuples():
            got = set(sc.detect_signals(row.text)["risk_flags"])
            expected = set(str(row.expected_flags).split()) if str(row.expected_flags).strip() else set()
            assert got == expected, (name, row.text, expected, got)
    assert not (set(pd.read_csv(root / "data/eval/risk_dev.csv").text)
                & set(pd.read_csv(root / "data/eval/independent_risk.csv").text))


def test_multi_request_schema_and_metrics():
    assert "get_refund" in INTENT_SCHEMA
    validate_request_parts([{"intent": "get_refund", "category": "REFUNDS",
                             "queue": sc.CATEGORY_TO_QUEUE["REFUNDS"]}])
    with pytest.raises(ValueError):
        validate_request_parts([{"intent": "get_refund", "category": "PAYMENT",
                                 "queue": sc.CATEGORY_TO_QUEUE["PAYMENT"]}])
    frame = pd.DataFrame({"text": ["I need a refund and an invoice."],
                          "expected_intents": ["get_refund get_invoice"]})
    report = multi_request_metrics(frame, controls={"single": ["I need a refund."]})
    assert report["exact_matches"] == 1
    assert report["controls"]["single"]["false_splits"] == 0


def test_calibration_reliability_and_cascade_detail():
    frame = pd.DataFrame({"text": ["a", "b", "c", "d"],
                          "category": ["A", "A", "B", "B"]})
    model = StubBundle([.9, .1], ["A", "B"], _policy())
    report = calibration_by_source(model, {"synthetic": frame})
    assert report["synthetic"]["overall"]["reliability"]
    assert report["synthetic"]["per_category"]["A"]["positive_n"] == 2
    routes = [
        {"primary_category": "A", "category": "A", "classification_source": "primary",
         "semantic_status": "NOT_REQUESTED"},
        {"primary_category": "B", "category": "A", "classification_source": "minilm",
         "semantic_status": "SUGGESTED"},
    ]
    detail = cascade_detailed_metrics(routes, ["A", "A"], latency_ms=[1., 2.])
    assert detail["corrected"] == 1 and detail["suggestions_n"] == 1
    assert detail["latency"]["mean_ms"] == 1.5


def test_current_release_report_is_generated_from_json_and_manifest_verifies():
    from support_router.reporting import build_current_release

    root = sc.project_root()
    current = build_current_release(root)
    assert current["validation"]["independent_benchmark"]["status"] == "CURRENT"
    assert current["validation"]["historical_results"]["status"] == "HISTORICAL"
    results_md = (root / "reports/results.md").read_text(encoding="utf-8")
    assert current["experiment_id"] in results_md
    assert current["identity"]["model_id"] in results_md
    assert "Исторический прогон" in results_md
    result = subprocess.run([sys.executable, "scripts/release_manifest.py", "--verify"],
                            cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "MANIFEST_OK" in result.stdout


def test_ci_workflow_and_notebook_stage13_are_present():
    root = sc.project_root()
    workflow = yaml.safe_load((root / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
    assert "fast" in workflow["jobs"] and "data-and-release" in workflow["jobs"]
    notebook = json.loads((root / "notebooks/Этапы_1-12_Colab.ipynb").read_text(encoding="utf-8"))
    text = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "RUN_MODE" in text
    assert "validate_report_identity" in text
    assert "write_current_release" in text
    assert "265 passed" not in text
    assert all(cell.get("execution_count") is None for cell in notebook["cells"]
               if cell["cell_type"] == "code")
