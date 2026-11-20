"""Этап 11: нестандартные формулировки и регрессия найденного дефекта правил риска.

Правила проверяются на заглушке (поведение не зависит от модели), а общие
гарантии системы проверяются на настоящей модели, сохранённой ноутбуком.
"""

import numpy as np
import pytest

import support_router as s

ROOT = s.project_root()
needs_model = pytest.mark.skipif(not (ROOT / "models/router.joblib").exists(),
                                 reason="нужна модель, её сохраняет ноутбук")


class Confident:
    """Заглушка: уверенная модель, автоотправка включена, чтобы блокировки было видно."""
    classes_ = np.array(["ORDER", "PAYMENT"])
    policy = {"auto_enabled": True, "confidence_threshold": .8, "similarity_threshold": .4,
              "version": "test"}
    model_id = "stage11-stub"

    def predict_proba(self, texts):
        return np.tile([.95, .05], (len(texts), 1))

    def similarity(self, texts):
        return np.full(len(texts), .9)


@pytest.fixture(scope="module")
def model():
    bundle, _ = s.load_bundle(ROOT / "models", trust_local_artifact=True)
    return bundle


# Дефекты найдены на этапе 11 и исправлены в версии 3.1.1 (тогда ещё support_core.py). Правила рисков уже
# ловили эти фразы в одном написании и пропускали в другом: сокращения «I'm» и «gonna»,
# другой порядок слов, прошедшее время, синоним в том же месте шаблона.
@pytest.mark.parametrize("text,flag,priority", [
    ("I'm going to kill you", "POSSIBLE_THREAT", "P1"),
    ("I’m going to hurt the driver", "POSSIBLE_THREAT", "P1"),
    ("Im going to hurt you", "POSSIBLE_THREAT", "P1"),
    ("I am gonna attack your staff", "POSSIBLE_THREAT", "P1"),
    ("I'm going to make you pay for this", "POSSIBLE_THREAT", "P1"),
    ("The courier threatened to hurt me", "POSSIBLE_THREAT", "P1"),
    ("Someone hacked my account and ordered things I never asked for", "POSSIBLE_FRAUD", "P1"),
    ("My card got hacked", "POSSIBLE_FRAUD", "P1"),
    ("I never placed this order", "POSSIBLE_FRAUD", "P1"),
    ("This transaction isn't mine", "POSSIBLE_FRAUD", "P1"),
    ("There is a charge I don't recognise on my card", "POSSIBLE_FRAUD", "P1"),
    ("I'm going to sue you", "POSSIBLE_LEGAL_CLAIM", "P2"),
    ("I'm gonna take you to court", "POSSIBLE_LEGAL_CLAIM", "P2"),
])
def test_risk_phrase_variants_are_flagged(text, flag, priority):
    r = s.route_ticket(text, Confident())
    assert flag in r["risk_flags"] and r["priority"] == priority
    assert r["status"] == "MANUAL_REVIEW" and r["queue"] == s.MANUAL_QUEUE


@pytest.mark.parametrize("text", [
    "I'm not going to hurt anyone, I just want my refund",
    "I'm going to make an order tomorrow",
    "I'm gonna kill some time while I wait for the parcel",
    "Im having issues signing up",
    "I'm going to the post office to send it back",
    'The film line "I\'m going to kill you" scared me',
    "How do you protect my account from being hacked?",
    "I never received my order",
    "I never ordered this item, you sent the wrong one",
    "The order I placed yesterday has not arrived",
    "The threat of rain delayed my parcel",
])
def test_fixed_rules_keep_hard_negatives(text):
    flags = s.detect_signals(text)["risk_flags"]
    assert not {"POSSIBLE_THREAT", "POSSIBLE_LEGAL_CLAIM", "POSSIBLE_FRAUD"} & set(flags)


# Известные пропуски сознательно не исправлены: это новые виды фраз, а не варианты
# существующих правил. Без экспертной разметки рисков не измерить, сколько ложных
# тревог дадут новые правила. strict=True: если пропуск исправят, тест напомнит
# перенести фразу в список выше.
@pytest.mark.xfail(reason="не исправлено на этапе 11: нужна экспертная разметка рисков", strict=True)
@pytest.mark.parametrize("text", ["I see a fraudulent charge on my card", "I have been scammed by your seller",
                                  "i m going to kill you", "I'm gonna kill ya"])
def test_known_gaps_in_risk_rules(text):
    assert s.detect_signals(text)["risk_flags"]


@pytest.mark.parametrize("text", [None, 123, 4.5, "", "   ", "\n\t", "{{Order Number}}"])
def test_no_text_asks_for_clarification(text):
    r = s.route_ticket(text, Confident())
    assert r["status"] == "NEEDS_CLARIFICATION" and r["queue"] is None


@pytest.mark.parametrize("text", ["?!?!", "😀😀😀", "1234567890", ";;;;;; ...... !!!!!!"])
def test_no_letters_goes_to_operator(text):
    r = s.route_ticket(text, Confident())
    assert r["status"] == "MANUAL_REVIEW" and r["category"] is None
    assert "INSUFFICIENT_SUPPORTED_TEXT" in r["reason_codes"]


@pytest.mark.parametrize("text", ["Где мой заказ?", "Привет, I need a refund", "我的订单在哪里",
                                  "أين طلبي", "Quiero cancelar mi pedido"])
def test_foreign_language_goes_to_operator_without_category(text):
    r = s.route_ticket(text, Confident())
    assert "POSSIBLY_UNSUPPORTED_LANGUAGE" in r["reason_codes"]
    assert r["status"] == "MANUAL_REVIEW" and r["category"] is None


STRANGE = [None, 123, "", "   ", "?!?!", "😀😀😀", "a" * 20000, "where is my order​​",
           "<b>cancel</b> my order please", "'; DROP TABLE tickets; --", "https://example.com/help",
           "Где мой заказ?", "我的订单在哪里", "I WANT A REFUND NOW!!!", "pls cancel my order asap thx",
           "  I   want\n to cancel \t my order  ", "I'm going to kill you", "{{Order Number}}"]


@needs_model
@pytest.mark.parametrize("text", STRANGE)
def test_real_model_never_crashes_and_never_sends_by_itself(model, text):
    r = s.route_ticket(text, model)
    assert r["status"] in {"MANUAL_REVIEW", "NEEDS_CLARIFICATION"}
    assert r["delivery_status"] == "NOT_SENT" and not r["business_policy_approved"]
    if r["confidence"] is not None:
        assert 0 <= r["confidence"] <= 1 and 0 <= r["similarity"] <= 1 + 1e-6
    shadow = s.route_ticket(text, s.shadow_bundle(model))
    assert shadow["status"] != "MODEL_ERROR"


@needs_model
def test_batch_equals_single_on_strange_inputs(model):
    texts = [t for t in STRANGE if isinstance(t, str)]
    for text, b in zip(texts, s.batch_route(model, texts)):
        live = s.route_ticket(text, model)
        for key in ["category", "queue", "status", "priority", "reason_codes"]:
            assert b[key] == live[key], (text, key)
        if live["confidence"] is not None:
            assert abs(b["confidence"] - live["confidence"]) < 1e-9


@needs_model
def test_whitespace_and_case_do_not_change_the_decision(model):
    base = s.route_ticket("I want to cancel my order", model)
    for variant in ["  I   want\n to cancel \t my order  ", "I WANT TO CANCEL MY ORDER"]:
        r = s.route_ticket(variant, model)
        assert r["category"] == base["category"]
        assert abs(r["confidence"] - base["confidence"]) < 1e-9
