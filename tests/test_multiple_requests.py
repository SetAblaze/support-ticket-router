import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import app as demo
import support_router as sc
from support_router.api import create_app
from support_router.multiple_requests import request_parts
from support_router.routing import route_policy_pair

DUAL = "My payment failed and I need to change my shipping address"
EXPECTED = [
    {"intent": "payment_issue", "category": "PAYMENT", "queue": "Queue_Payments"},
    {"intent": "change_shipping_address", "category": "SHIPPING", "queue": "Queue_Delivery"},
]


@pytest.fixture(scope="module")
def router():
    return demo.load_router()


@pytest.mark.parametrize("text,intents", [
    (DUAL, ["payment_issue", "change_shipping_address"]),
    ("My account is locked and my payment didn't go through.", ["recover_account", "payment_issue"]),
    ("My payment failed and I need to update my card.", ["payment_issue", "change_payment_method"]),
    ("Reset my password. Also send me a receipt.", ["recover_password", "get_invoice"]),
    ("Track my order; cancel my order from yesterday.", ["track_order", "cancel_order"]),
    ("I need an invoice, a refund and a new password.", ["get_invoice", "get_refund", "recover_password"]),
    ("Update my credit card\nTrack my parcel", ["change_payment_method", "track_order"]),
    ("I do not want a refund, send an invoice and cancel my order.", ["get_invoice", "cancel_order"]),
    ("I already paid the cancellation fee and I need a refund.", []),
    ("Don't track my order. Send me an invoice.", []),
    ("Send my refund to the card linked to my account.", []),
    ("I need an invoice and a receipt", []),
    ("Track my order, please. Where is my parcel?", []),
    ("Do not cancel my order. Send me an invoice.", []),
    ("Reset my password. My invoice number is 421.", []),
    ("I need a refund or an invoice", []),
])
def test_distinct_requests(text, intents):
    assert [part["intent"] for part in request_parts(text)] == intents


def test_context_risk_and_shadow(router):
    model, shadow, _ = router
    live, candidate = route_policy_pair(DUAL + ". My card was stolen.", model, shadow, detect_multiple=True)
    for r in (live, candidate):
        assert r["request_parts"] == EXPECTED
        assert r["status"] == "MANUAL_REVIEW" and r["queue"] == sc.MANUAL_QUEUE
        assert r["priority"] == "P1" and "POSSIBLE_FRAUD" in r["risk_flags"]
        assert r["reason_codes"].count("POSSIBLE_MULTIPLE_REQUESTS") == 1
        assert r["delivery_status"] == "NOT_SENT"
    assert sc.route_ticket(DUAL, model)["request_parts"] == []


def test_single_and_batch_http_agree():
    with TestClient(create_app()) as client:
        tickets = [{"text": DUAL, "ticket_id": "a"}, {"text": "Where is my invoice?", "ticket_id": "b"},
                   {"text": "", "ticket_id": "c"}]
        response = client.post("/v1/route/batch?detect_multiple=true", json={"tickets": tickets})
        assert response.status_code == 200
        results = response.json()["results"]
        assert [r["request_parts"] for r in results] == [EXPECTED, [], []]
        for ticket, result in zip(tickets, results, strict=True):
            single = client.post("/v1/route?detect_multiple=true", json=ticket).json()
            for field in ("ticket_id", "status", "request_parts", "reason_codes", "shadow"):
                assert single[field] == result[field]
        assert results[0]["shadow"]["status"] == "MANUAL_REVIEW"
        assert client.post("/v1/route", json=tickets[0]).json()["request_parts"] == []


def test_operator_and_csv_get_both_queues(router, tmp_path):
    model, shadow, _ = router
    message = demo.make_handler(model, shadow)(DUAL, [], True)[8]
    assert all(part["queue"] in message for part in EXPECTED)
    source, dest = tmp_path / "in.csv", tmp_path / "out.csv"
    pd.DataFrame({"text": [DUAL, "Send me a bill"]}).to_csv(source, index=False)
    demo.batch_file(source, dest, model, shadow, detect_multiple=True)
    assert pd.read_csv(dest).request_parts.map(json.loads).tolist() == [EXPECTED, []]


def test_synthetic_multi_topic_regression_set():
    data = Path(sc.project_root(), "data/eval")
    ood = pd.read_csv(data / "ood_eval.csv")
    multi = ood[ood.topic == "multiple_topics"].set_index("id")
    expected = {
        86: ["recover_password", "get_refund"],
        87: ["change_shipping_address", "get_invoice"],
        88: ["payment_issue", "subscribe_newsletter"],
        89: ["cancel_order", "delete_account"],
        90: ["track_order", "check_refund_policy"],
        91: ["get_invoice", "get_refund", "recover_password"],
        92: ["change_shipping_address", "check_payment_methods"],
        93: ["review", "delivery_options"],
        94: ["payment_issue", "contact_human_agent"],
        95: ["track_order", "check_cancellation_fee"],
        96: ["subscribe_newsletter", "delete_account"],
        97: ["complaint", "get_refund"],
        98: ["invoice_issue", "track_order"],
        99: ["delivery_options", "check_payment_methods"],
        100: ["change_order", "change_shipping_address", "get_invoice"],
    }
    assert multi.text.map(lambda t: [p["intent"] for p in request_parts(t)]).to_dict() == expected
    assert not any(request_parts(t) for t in ood.loc[ood.topic != "multiple_topics", "text"])
