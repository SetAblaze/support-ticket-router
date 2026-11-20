"""Метрики Prometheus и правила алертов: метрики считаются, правила ссылаются только на них."""
import re
import shutil
import subprocess

import pytest
import yaml
from fastapi.testclient import TestClient
from prometheus_client import CollectorRegistry
from prometheus_client.parser import text_string_to_metric_families

import support_router as s
from support_router import api

ROOT = s.project_root()
RULES = ROOT / "monitoring" / "alert_rules.yml"
RULE_TESTS = ROOT / "monitoring" / "alert_rules_test.yml"
needs_model = pytest.mark.skipif(not (s.models_dir() / "router.joblib").exists(), reason="нужна обученная модель")


def samples(client):
    out = {}
    for family in text_string_to_metric_families(client.get("/metrics").text):
        for sample in family.samples:
            out[(sample.name, tuple(sorted(sample.labels.items())))] = sample.value
    return out


def value(snapshot, name, **labels):
    return snapshot.get((name, tuple(sorted(labels.items()))), 0.0)


@pytest.fixture()
def client():
    app = api.create_app(s.models_dir(), registry=CollectorRegistry())
    with TestClient(app) as c:
        yield c


@needs_model
def test_decision_metrics_are_counted(client):
    client.post("/v1/route", json={"text": "I want to cancel my order"})
    client.post("/v1/route", json={"text": ""})
    client.post("/v1/route", json={"text": "I'm going to kill you"})
    snap = samples(client)
    assert value(snap, "router_decisions_total", status="MANUAL_REVIEW", queue="Queue_Manual_Review",
                 priority="P3") == 1
    assert value(snap, "router_decisions_total", status="NEEDS_CLARIFICATION", queue="none", priority="P3") == 1
    assert value(snap, "router_decisions_total", status="MANUAL_REVIEW", queue="Queue_Manual_Review",
                 priority="P1") == 1
    assert value(snap, "router_risk_flags_total", flag="POSSIBLE_THREAT") == 1
    assert value(snap, "router_reason_codes_total", reason="AUTO_POLICY_DISABLED") == 2
    assert value(snap, "router_shadow_decisions_total", status="AUTO_RECOMMENDED") == 1
    assert value(snap, "router_suggested_category_total", category="ORDER") == 1
    assert value(snap, "router_confidence_count") == 2 and value(snap, "router_training_similarity_count") == 2
    assert value(snap, "router_text_length_chars_count") == 3
    assert value(snap, "router_model_loaded") == 1


@needs_model
def test_model_info_and_request_metrics(client):
    client.post("/v1/route", json={"text": 1})
    client.get("/no/such/path")
    snap = samples(client)
    model_id = client.get("/ready").json()["model_id"]
    info = [k for k in snap if k[0] == "router_model_info"]
    assert len(info) == 1 and ("model_id", model_id) in info[0][1]
    assert value(snap, "router_http_requests_total", endpoint="/v1/route", method="POST", status_code="422") == 1
    assert value(snap, "router_http_requests_total", endpoint="other", method="GET", status_code="404") == 1
    assert value(snap, "router_http_request_duration_seconds_count", endpoint="/v1/route") == 1


@needs_model
def test_model_error_is_counted(client):
    class Broken:
        classes_ = ["ORDER"]
        policy = {"auto_enabled": False, "version": "broken"}
        model_id = "broken-model"

        def predict_proba(self, texts):
            raise RuntimeError("boom")

    client.app.state.router.update(model=Broken(), shadow=Broken())
    body = client.post("/v1/route", json={"text": "Where is my refund?"}).json()
    assert body["status"] == "MODEL_ERROR" and body["queue"] == s.MANUAL_QUEUE
    batch = client.post("/v1/route/batch", json={"tickets": [{"text": "a refund"}, {"text": "b invoice"}]}).json()
    assert [r["status"] for r in batch["results"]] == ["MODEL_ERROR", "MODEL_ERROR"]
    assert value(samples(client), "router_model_errors_total") == 3


def rule_expressions():
    rules = yaml.safe_load(RULES.read_text(encoding="utf-8"))
    return [rule for group in rules["groups"] for rule in group["rules"]]


@needs_model
def test_alert_rules_use_only_existing_metrics(client):
    client.post("/v1/route", json={"text": "I want to cancel my order"})
    client.post("/v1/route", json={"text": "I'm going to kill you"})
    exposed = {name for name, _ in samples(client)} | {"up"}
    for name in list(exposed):
        exposed.add(re.sub(r"_(bucket|count|sum)$", "", name))
    for rule in rule_expressions():
        used = set(re.findall(r"\b(router_[a-z_]+|up)\b", rule["expr"]))
        missing = used - exposed
        assert not missing, (rule["alert"], missing)


def test_every_alert_has_runbook_and_severity():
    runbook = (ROOT / "docs" / "runbook.md").read_text(encoding="utf-8")
    for rule in rule_expressions():
        assert rule["labels"]["severity"] in {"critical", "warning", "info"}, rule["alert"]
        assert rule["annotations"].get("runbook"), rule["alert"]
        assert f"### {rule['alert']}" in runbook, f"нет раздела в runbook: {rule['alert']}"


@pytest.mark.skipif(shutil.which("promtool") is None, reason="promtool не установлен")
def test_promtool_checks_and_unit_tests_pass():
    check = subprocess.run(["promtool", "check", "rules", str(RULES)], capture_output=True, text=True)
    assert check.returncode == 0, check.stdout + check.stderr
    unit = subprocess.run(["promtool", "test", "rules", str(RULE_TESTS)], capture_output=True, text=True,
                          cwd=RULE_TESTS.parent)
    assert unit.returncode == 0, unit.stdout + unit.stderr
