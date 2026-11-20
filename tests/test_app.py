"""Этап 11: приложение app.py (этап 10) поверх модели, сохранённой ноутбуком."""
import subprocess
import sys

import pandas as pd
import pytest

import support_router as s

ROOT = s.project_root()
pytestmark = pytest.mark.skipif(
    not (ROOT / "models/router.joblib").exists() or not (ROOT / "app.py").exists(),
    reason="нужны models/ и app.py, их создаёт ноутбук")


@pytest.fixture(scope="module")
def router():
    import app
    return app.load_router(ROOT / "models")


def test_loaded_model_has_operational_auto_disabled(router):
    model, shadow, meta = router
    assert meta["policy"]["auto_enabled"] is False
    assert model.policy == meta["policy"]
    assert shadow.policy["auto_enabled"] == meta["policy"]["candidate_auto_enabled"]
    assert shadow.model_id == model.model_id and shadow.policy["version"].endswith("-SHADOW")


@pytest.mark.parametrize("text", ["I want to cancel my order", "Где мой заказ?", "", "I'm going to sue you"])
def test_json_answer_is_route_ticket_plus_shadow(router, text):
    import app
    model, shadow, _ = router
    out = app.route_with_shadow(text, model, shadow)
    direct = s.route_ticket(text, model)
    for key in ["category", "suggested_category", "queue", "status", "priority",
                "reason_codes", "risk_flags", "model_id", "policy_version"]:
        assert out[key] == direct[key], key
    assert out["shadow"]["status"] == s.route_ticket(text, shadow)["status"]


def test_ui_handler_shows_the_same_decision(router):
    import app
    model, shadow, _ = router
    handle = app.make_handler(model, shadow)
    for text in ["I want to cancel my order", "I can't log in", "Reset my password and check my refund"]:
        shown = handle(text, [])
        direct = s.route_ticket(text, model)
        assert shown[0] == (direct["category"] or "Не определена")
        assert shown[2] == direct["status"] and shown[3] == direct["priority"]
        assert shown[4] == "; ".join(direct["reason_codes"])
        assert len(shown[7]) == 1  # история этой сессии
    empty = handle("", [])
    assert empty[7] == [] and empty[8] == "Нужен текст обращения."


def test_ui_history_is_masked_and_per_session(router):
    import app
    model, shadow, _ = router
    handle = app.make_handler(model, shadow)
    first = handle("Invoice please, write to bob@example.com or +44 20 7946 0958", [])
    other = handle("Where is my refund?", [])
    assert len(first[7]) == len(other[7]) == 1
    assert "bob@example.com" not in str(first[7]) and "7946" not in str(first[7])
    assert "[EMAIL]" in str(first[7])


@pytest.mark.parametrize("status,expected", [
    ("NEEDS_CLARIFICATION", "Нужен текст"), ("MANUAL_REVIEW", "оператору"),
    ("AUTO_RECOMMENDED", "не отправлено"), ("MODEL_ERROR", "Ошибка модели")])
def test_message_for_every_status(status, expected):
    import app
    assert expected in app.deployed_message({"category": "ORDER", "status": status})
    assert "не определена" in app.deployed_message({"category": None, "status": "MANUAL_REVIEW"})


def test_batch_file_keeps_order_masks_and_matches(router, tmp_path):
    import app
    model, shadow, _ = router
    texts = ["I want to cancel my order", "", "mail me at bob@example.com, +44 20 7946 0958",
             "Где мой заказ?", "I'm going to kill you"]
    src, dst = tmp_path / "in.csv", tmp_path / "out.csv"
    pd.DataFrame({"id": list("abcde"), "text": texts}).to_csv(src, index=False)
    app.batch_file(src, dst, model, shadow)
    raw = dst.read_text(encoding="utf-8")
    saved = pd.read_csv(dst, keep_default_na=False, dtype=str)
    assert list(saved.columns) == ["id"] + app.BATCH_COLUMNS
    assert list(saved["id"]) == list("abcde")
    assert "bob@example.com" not in raw and "7946" not in raw
    for text, row in zip(texts, saved.to_dict("records")):
        direct = s.route_ticket(text, model)
        assert row["status"] == direct["status"] and row["priority"] == direct["priority"]
        assert row["reason_codes"] == ";".join(direct["reason_codes"])
    assert saved.loc[1, "status"] == "NEEDS_CLARIFICATION"
    assert saved.loc[4, "priority"] == "P1" and "POSSIBLE_THREAT" in saved.loc[4, "risk_flags"]


def test_batch_needs_text_column(router, tmp_path):
    import app
    model, shadow, _ = router
    src = tmp_path / "bad.csv"
    pd.DataFrame({"message": ["hi"]}).to_csv(src, index=False)
    with pytest.raises(SystemExit):
        app.batch_file(src, tmp_path / "out.csv", model, shadow)


def test_explanation_is_short_and_safe(router):
    import app
    model, _, _ = router
    rows = app.explanation_rows(model, "I want to cancel my order")
    assert 0 < len(rows) <= 6 and all(len(r) == 2 for r in rows)
    assert app.explanation_rows(model, "") == [] and app.explanation_rows(model, None) == []


def test_page_and_api_are_served(router):
    gr = pytest.importorskip("gradio")
    from fastapi import FastAPI
    from starlette.testclient import TestClient

    import app
    model, shadow, meta = router
    demo = app.build_ui(model, shadow, meta)
    with TestClient(gr.mount_gradio_app(FastAPI(), demo, path="/")) as client:
        page, config = client.get("/"), client.get("/config")
    assert page.status_code == 200 and "text/html" in page.headers["content-type"]
    assert config.status_code == 200
    names = {d.get("api_name") for d in config.json().get("dependencies", [])}
    assert {"analyze", "route_json"} <= names


def test_cli_check_in_fresh_process():
    run = subprocess.run([sys.executable, "app.py", "--check"], cwd=ROOT,
                         capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
    assert run.returncode == 0, run.stderr[-2000:]
    assert "CHECK OK" in run.stdout
