import copy
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

import support_router as sc
from support_router.api import create_app
from support_router.business import confidence_floor, misroute_costs
from support_router.routing import route_policy_pair

UNFAMILIAR = "I can't get in anymore, it says my login details are wrong."


@pytest.fixture(scope="module")
def model():
    return sc.load_bundle(sc.models_dir(), trust_local_artifact=True)[0]


@pytest.mark.parametrize("text,priority,reason", [
    ("I was charged twice.", "P2", "DUPLICATE_DEBIT"),
    ("My account is locked.", "P2", "ACCESS_BLOCKED"),
    ("I cannot complete my purchase.", "P2", "CHECKOUT_BLOCKED"),
    ("I was charged twice and someone stole my card. My card was stolen.", "P1", "POSSIBLE_FRAUD"),
    ("This is urgent!!! Please email a receipt.", "P3", None),
    ("I was not charged twice.", "P3", None),
    ('The help page says "my account is locked".', "P3", None),
    ("What should I do if I am locked out of my account?", "P3", None),
])
def test_impact_precedence(model, text, priority, reason):
    live, shadow = route_policy_pair(text, model, sc.shadow_bundle(model))
    assert live["priority"] == priority
    if reason:
        assert reason in live["priority_reasons"]
        assert shadow["status"] == "MANUAL_REVIEW"
        assert live["semantic_status"] == "NOT_REQUESTED"
    else:
        assert not live["priority_reasons"]


def test_crm_context(model):
    now = datetime(2026, 1, 2, tzinfo=timezone.utc)
    ctx = {"source": "trusted_crm", "vip_verified": "false", "open_related_ticket": False,
           "deadline_utc": "2026-01-01T00:00:00Z"}
    r = sc.route_ticket("Track my order", sc.shadow_bundle(model), context=ctx, now=now)
    assert (r["priority"], r["priority_reasons"], r["status"]) == ("P2", ["VERIFIED_DEADLINE_OVERDUE"], "MANUAL_REVIEW")


def test_cost_thresholds_at_boundary(model):
    policy = model.policy
    assert confidence_floor(policy, "PAYMENT") == pytest.approx(.95)
    assert confidence_floor(policy, "SHIPPING") == pytest.approx(.90)
    assert confidence_floor(policy, "NEWSLETTER") == policy["confidence_threshold"]
    stub = copy.copy(sc.shadow_bundle(model))
    stub.semantic_head = None
    k = list(stub.classes_).index("REFUNDS")
    for probability, status in ((.9499, "MANUAL_REVIEW"), (.95, "AUTO_RECOMMENDED")):
        probs = [(1 - probability) / (len(stub.classes_) - 1)] * len(stub.classes_)
        probs[k] = probability
        r = sc.route_ticket("Please check my refund", stub, scored=(probs, .80))
        assert (r["primary_confidence_threshold"], r["status"]) == (.95, status)


def test_fallback_single_batch_and_blocks():
    with TestClient(create_app()) as api:
        tickets = [{"text": t} for t in (UNFAMILIAR, "What is the orbital velocity of Saturn?",
                                         "Reset my password and check my refund.", "Где мой заказ?")]
        results = api.post("/v1/route/batch?detect_multiple=true", json={"tickets": tickets}).json()["results"]
        for ticket, result in zip(tickets, results, strict=True):
            single = api.post("/v1/route?detect_multiple=true", json=ticket).json()
            for key in ("category", "classification_source", "semantic_status", "status", "priority", "shadow"):
                assert result[key] == single[key]
        hint = results[0]
        assert hint["classification_source"] == "minilm" and hint["primary_category"] != hint["category"] == "ACCOUNT"
        assert hint["confidence"] == hint["semantic_confidence"] == hint["top3"]["ACCOUNT"]
        assert hint["shadow"]["status"] == hint["status"] == "MANUAL_REVIEW"
        assert [r["semantic_status"] for r in results[1:]] == ["ABSTAINED", "NOT_REQUESTED", "NOT_REQUESTED"]


def test_missing_optional_encoder_leaves_primary_available(model, tmp_path):
    local = copy.copy(sc.shadow_bundle(model))
    local.encoder_directory = tmp_path
    r = sc.route_ticket(UNFAMILIAR, local)
    assert (r["semantic_status"], r["classification_source"], r["status"]) == ("UNAVAILABLE", "primary", "MANUAL_REVIEW")
    assert r["category"] == r["primary_category"] and not list(tmp_path.iterdir())
    assert sc.route_ticket("I want to cancel my order", local)["semantic_status"] == "NOT_REQUESTED"

def test_custom_business_costs(model):
    policy = {**model.policy, "misroute_cost": misroute_costs({"PAYMENT": 40})}
    assert confidence_floor(policy, "PAYMENT") == .975
    assert confidence_floor(policy, "REFUNDS") == .95
    for invalid in ({"PAYMENT": float("nan")}, {"PAYMENT": -1}, {"PAYMENT": True}, {"UNKNOWN": 2}):
        with pytest.raises(ValueError):
            misroute_costs(invalid)
