"""HTTP-сервис FastAPI: некорректные запросы, пустые обращения, маршрутизация, готовность."""
import json
import logging

import pytest
from fastapi.testclient import TestClient
from prometheus_client import CollectorRegistry

import support_router as s
from support_router import api

MODELS = s.models_dir()
needs_model = pytest.mark.skipif(not (MODELS / "router.joblib").exists(), reason="нужна обученная модель в models/")


@pytest.fixture(scope="module")
def client():
    app = api.create_app(MODELS, registry=CollectorRegistry())
    with TestClient(app) as c:
        yield c


def post(client, text, **extra):
    return client.post("/v1/route", json={"text": text, **extra})


# --- Живость и готовность ---------------------------------------------------------------

def test_health_is_always_ok(client):
    assert client.get("/health").json() == {"status": "ok"}


@needs_model
def test_ready_reports_model(client):
    r = client.get("/ready")
    assert r.status_code == 200 and r.json()["model_id"].startswith("router-")


def test_not_ready_without_model(tmp_path):
    app = api.create_app(tmp_path, registry=CollectorRegistry())
    with TestClient(app) as c:
        assert c.get("/health").status_code == 200
        ready = c.get("/ready")
        assert ready.status_code == 503 and ready.json()["status"] == "not_ready"
        assert c.post("/v1/route", json={"text": "Where is my refund?"}).status_code == 503
        assert "router_model_loaded 0.0" in c.get("/metrics").text


def test_tampered_model_is_not_loaded(tmp_path):
    if not (MODELS / "router.joblib").exists():
        pytest.skip("нужна обученная модель")
    (tmp_path / "router.joblib").write_bytes((MODELS / "router.joblib").read_bytes() + b"x")
    (tmp_path / "router_meta.json").write_text((MODELS / "router_meta.json").read_text(encoding="utf-8"), encoding="utf-8")
    app = api.create_app(tmp_path, registry=CollectorRegistry())
    with TestClient(app) as c:
        r = c.get("/ready")
        assert r.status_code == 503 and "SHA256" in r.json()["reason"]


# --- Некорректные запросы ---------------------------------------------------------------

@pytest.mark.parametrize("body", [
    {}, {"text": None}, {"text": 123}, {"text": ["a"]}, {"text": {"a": 1}},
    {"txt": "I want a refund"}, {"text": "ok", "unexpected": 1},
    {"text": "ok", "ticket_id": "id with spaces"}, {"text": "ok", "ticket_id": "x" * 129},
    {"text": "a" * (api.MAX_TEXT_CHARS + 1)},
])
def test_invalid_body_is_422(client, body):
    r = client.post("/v1/route", json=body)
    assert r.status_code == 422
    assert "detail" in r.json()


def test_not_json_is_422(client):
    r = client.post("/v1/route", content=b"I want a refund", headers={"content-type": "application/json"})
    assert r.status_code == 422


def test_error_does_not_echo_the_text(client):
    secret = "my card 4111 and email bob@example.com " * 200
    r = client.post("/v1/route", json={"text": secret})
    assert r.status_code == 422 and "bob@example.com" not in r.text


def test_wrong_method_and_path(client):
    assert client.get("/v1/route").status_code == 405
    assert client.post("/v1/unknown", json={"text": "x"}).status_code == 404


@pytest.mark.parametrize("body", [{"tickets": []}, {"tickets": [{"text": "x"}] * (api.MAX_BATCH + 1)},
                                  {"tickets": [{"text": 1}]}, {"items": [{"text": "x"}]}])
def test_invalid_batch_is_422(client, body):
    assert client.post("/v1/route/batch", json=body).status_code == 422


# --- Пустые и странные обращения ----------------------------------------------------------

@needs_model
@pytest.mark.parametrize("text", ["", "   ", "\n\t", "{{Order Number}}"])
def test_empty_ticket_asks_for_text(client, text):
    r = post(client, text)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "NEEDS_CLARIFICATION" and body["queue"] is None
    assert body["reason_codes"] == ["EMPTY_OR_INVALID_INPUT"]


@needs_model
@pytest.mark.parametrize("text", ["?!?!", "😀😀😀", "Где мой заказ?", "我的订单在哪里", "a" * api.MAX_TEXT_CHARS,
                                  "where is my order​", "'; DROP TABLE tickets; --"])
def test_strange_ticket_goes_to_operator(client, text):
    body = post(client, text).json()
    assert body["status"] in {"MANUAL_REVIEW", "NEEDS_CLARIFICATION"}
    assert body["delivery_status"] == "NOT_SENT"


# --- Маршрутизация ------------------------------------------------------------------------

@needs_model
@pytest.mark.parametrize("text", ["I want to cancel my order", "Where is my refund?", "I can't log in",
                                  "Reset my password and check my refund", "I'm going to sue you"])
def test_api_returns_the_same_decision_as_route_ticket(client, text):
    model, _ = s.load_bundle(MODELS, trust_local_artifact=True)
    body = post(client, text, ticket_id="T-42").json()
    direct = s.route_ticket(text, model)
    for key in ["status", "category", "suggested_category", "queue", "topic_queue", "priority",
                "risk_flags", "reason_codes", "model_id", "policy_version"]:
        assert body[key] == direct[key], key
    assert body["confidence"] == pytest.approx(direct["confidence"], abs=1e-12)
    assert body["ticket_id"] == "T-42"
    assert body["shadow"]["status"] == s.route_ticket(text, s.shadow_bundle(model))["status"]


@needs_model
def test_working_mode_never_sends_by_itself(client):
    texts = ["I want to cancel my order", "Where is my refund?", "Can I pay with PayPal?",
             "How do I subscribe to your newsletter?", "Please send me the invoice for my last purchase"]
    results = client.post("/v1/route/batch", json={"tickets": [{"text": t} for t in texts]}).json()["results"]
    assert all(r["status"] != "AUTO_RECOMMENDED" for r in results)
    assert any(r["shadow"]["status"] == "AUTO_RECOMMENDED" for r in results)


@needs_model
@pytest.mark.parametrize("text,flag,priority", [
    ("I'm going to kill you", "POSSIBLE_THREAT", "P1"),
    ("Someone hacked my account", "POSSIBLE_FRAUD", "P1"),
    ("I will sue you", "POSSIBLE_LEGAL_CLAIM", "P2"),
])
def test_risk_routing(client, text, flag, priority):
    body = post(client, text).json()
    assert flag in body["risk_flags"] and body["priority"] == priority
    assert body["status"] == "MANUAL_REVIEW"


@needs_model
@pytest.mark.parametrize("text,reason", [
    ("Где мой заказ?", "POSSIBLY_UNSUPPORTED_LANGUAGE"),
    ("Reset my password and send my refund", "POSSIBLE_MULTIPLE_REQUESTS"),
    ("The app crashes on launch", "TECHNICAL_ISSUE_NEEDS_REVIEW"),
])
def test_review_reasons(client, text, reason):
    assert reason in post(client, text).json()["reason_codes"]


@needs_model
def test_batch_keeps_order_and_ids(client):
    tickets = [{"text": "I want to cancel my order", "ticket_id": "a"}, {"text": "", "ticket_id": "b"},
               {"text": "I'm going to kill you", "ticket_id": "c"}]
    results = client.post("/v1/route/batch", json={"tickets": tickets}).json()["results"]
    assert [r["ticket_id"] for r in results] == ["a", "b", "c"]
    assert results[1]["status"] == "NEEDS_CLARIFICATION" and results[2]["priority"] == "P1"
    single = [post(client, t["text"]).json() for t in tickets]
    for one, many in zip(single, results):
        assert one["status"] == many["status"] and one["queue"] == many["queue"]


@needs_model
def test_model_card(client):
    card = client.get("/v1/model").json()
    assert card["policy"]["auto_enabled"] is False
    assert card["scikit_learn"] == "1.8.0" and len(card["classes"]) == 11


# --- Журнал и заголовки -------------------------------------------------------------------

@needs_model
def test_log_has_no_ticket_text(client, caplog):
    logger = logging.getLogger("support_router.api")
    logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.INFO, logger="support_router.api"):
            post(client, "Refund to bob@example.com, phone +44 20 7946 0958", ticket_id="T-7")
    finally:
        logger.removeHandler(caplog.handler)
    events = [json.loads(r.getMessage()) for r in caplog.records]
    decision = next(e for e in events if e["event"] == "decision")
    assert decision["ticket_id"] == "T-7" and decision["text_length"] > 0
    assert "bob@example.com" not in caplog.text and "7946" not in caplog.text


def test_request_id_is_returned(client):
    r = client.get("/health", headers={"x-request-id": "abc-123"})
    assert r.headers["x-request-id"] == "abc-123"
    assert len(client.get("/health").headers["x-request-id"]) == 32


@needs_model
def test_policy_pair_scores_each_request_once(client, monkeypatch):
    model = client.app.state.router["model"]
    calls = []
    predict = model.predict_proba
    similarity = model.similarity

    def record_predict(texts):
        calls.append(("predict", len(texts)))
        return predict(texts)

    def record_similarity(texts):
        calls.append(("similarity", len(texts)))
        return similarity(texts)

    monkeypatch.setattr(model, "predict_proba", record_predict)
    monkeypatch.setattr(model, "similarity", record_similarity)
    assert post(client, "Send me an invoice").status_code == 200
    assert calls == [("predict", 1), ("similarity", 1)]
    calls.clear()
    response = client.post("/v1/route/batch", json={"tickets": [{"text": "Send me an invoice"},
                                                                {"text": "Where is my order?"}]})
    assert response.status_code == 200
    assert calls == [("predict", 2), ("similarity", 2)]


@needs_model
def test_batch_scoring_failure_isolated_by_ticket(client, monkeypatch):
    model = client.app.state.router["model"]
    predict = model.predict_proba

    def selective_failure(texts):
        if "broken scoring" in texts:
            raise ValueError("scoring error")
        return predict(texts)

    monkeypatch.setattr(model, "predict_proba", selective_failure)
    texts = ["broken scoring", "Where is my order?", ""]
    response = client.post("/v1/route/batch", json={"tickets": [{"text": t} for t in texts]})
    bad, valid, empty = response.json()["results"]
    assert bad["status"] == bad["shadow"]["status"] == "MODEL_ERROR"
    assert valid["category"] == "ORDER"
    assert empty["status"] == empty["shadow"]["status"] == "NEEDS_CLARIFICATION"


# --- Operator feedback ---------------------------------------------------------------

@needs_model
def test_feedback_requires_explicit_decision_and_auth(tmp_path, monkeypatch):
    monkeypatch.setenv("ROUTER_FEEDBACK_API_KEY", "test-key")
    monkeypatch.setenv("ROUTER_FEEDBACK_PATH", str(tmp_path / "events.jsonl"))
    app = api.create_app(MODELS, registry=CollectorRegistry())
    with TestClient(app) as c:
        payload = {"model_id": c.get("/ready").json()["model_id"],
                   "policy_version": c.get("/v1/model").json()["policy"]["version"],
                   "proposed_category": "PAYMENT", "confirmed": True, "reviewer_id": "op-1"}
        assert c.post("/v1/feedback", json=payload).status_code == 401
        missing = dict(payload); missing.pop("confirmed")
        assert c.post("/v1/feedback", json=missing,
                      headers={"x-feedback-key": "test-key"}).status_code == 422
        wrong = {**payload, "confirmed": False, "operator_category": "PAYMENT"}
        assert c.post("/v1/feedback", json=wrong,
                      headers={"x-feedback-key": "test-key"}).status_code == 422


@needs_model
def test_feedback_is_idempotent_and_model_bound(tmp_path, monkeypatch):
    monkeypatch.setenv("ROUTER_FEEDBACK_API_KEY", "test-key")
    monkeypatch.setenv("ROUTER_FEEDBACK_PATH", str(tmp_path / "events.jsonl"))
    app = api.create_app(MODELS, registry=CollectorRegistry())
    with TestClient(app) as c:
        model = c.get("/v1/model").json()
        payload = {"model_id": model["model_id"], "policy_version": model["policy"]["version"],
                   "proposed_category": "PAYMENT", "confirmed": True, "reviewer_id": "op-1",
                   "event_id": "evt-00000001", "redacted_text": "my card failed"}
        first = c.post("/v1/feedback", json=payload, headers={"x-feedback-key": "test-key"})
        second = c.post("/v1/feedback", json=payload, headers={"x-feedback-key": "test-key"})
        assert first.status_code == 200 and first.json()["status"] == "stored"
        assert second.status_code == 200 and second.json()["status"] == "duplicate"
        mismatch = {**payload, "event_id": "evt-00000002", "model_id": "router-other"}
        assert c.post("/v1/feedback", json=mismatch,
                      headers={"x-feedback-key": "test-key"}).status_code == 409
        summary = c.get("/v1/feedback/summary", headers={"x-feedback-key": "test-key"})
        assert summary.status_code == 200 and summary.json()["n"] == 1
