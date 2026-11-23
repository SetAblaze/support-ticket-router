"""Маршрутизация обращений в поддержку: единое приложение (этап 10).

Модель, правила маршрутизации и интерфейс собраны в одну программу.
Решения принимает только support_router.route_ticket: тот же код работает
в ноутбуке, в тестах и здесь. Модель читается из models/ (router.joblib
и router_meta.json), её сохраняет ноутбук. load_bundle сверяет хеш файла
модели, версию scikit-learn и хеш модулей решения, поэтому устаревшая
или чужая модель не загрузится молча.

Запуск:
  python app.py                               интерфейс на http://127.0.0.1:7860
  python app.py --share                       плюс временная ссылка gradio.live
  python app.py --batch in.csv --out out.csv  разметить CSV (колонка text или utterance)
  python app.py --check                       загрузить модель и проверить контрольные фразы

Ограничения те же, что в ноутбуке: только английский, автоотправка выключена,
обращения никуда не передаются. joblib загружает pickle, поэтому открывайте
только модель, которую обучили сами.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for folder in (ROOT / "src", ROOT):  # пакет support_router лежит в src/
    if folder.exists() and str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import pandas as pd  # noqa: E402

import support_router as sc  # noqa: E402

MODELS_DIR = sc.models_dir()
HISTORY_COLUMNS = ["Время UTC", "Текст (маскир.)", "Категория", "Очередь", "Приоритет",
                   "Статус", "Причины", "model_id", "policy_version"]
CATEGORY_HELP = (
    "ACCOUNT: доступ и профиль; CANCELLATION_FEE: стоимость отмены; CONTACT: связь с поддержкой; "
    "DELIVERY: способы и общие сроки доставки; FEEDBACK: отзыв или жалоба; INVOICES: счета и "
    "платёжные документы; NEWSLETTER: подписка и отписка от рассылки; ORDER: оформление, изменение, "
    "отмена и отслеживание конкретного заказа; PAYMENT: способы и ошибки оплаты; REFUNDS: возврат "
    "денег; SHIPPING: адрес доставки.")
EXAMPLES = ["I want to cancel my order", "I can't log in", "Where is my refund?",
            "My card was charged twice for the same purchase",
            "I moved last week, so my shipping address needs changing",
            "I'm going to sue you", "What causes the northern lights",
            "Где мой заказ?", "Reset my password and check my refund"]
CHECK_TEXTS = ["I want to cancel my order", "I can't log in", "Where is my refund?",
               "I'm going to kill you", "I will not hurt you", "", "qzxv qqqq",
               "Где мой заказ?", "Reset my password and check my refund"]
BATCH_COLUMNS = ["row", "text_masked", "suggested_category", "category", "queue", "topic_queue",
                 "status", "priority", "confidence", "similarity", "risk_flags", "reason_codes",
                 "shadow_status", "shadow_queue", "model_id", "policy_version", "priority_reasons",
                 "classification_source", "semantic_status", "primary_category", "primary_confidence", "primary_confidence_threshold"]


def load_router(models_dir=MODELS_DIR):
    """Рабочая модель, её теневая копия (только для анализа) и метаданные."""
    model, meta = sc.load_bundle(models_dir, trust_local_artifact=True)
    return model, sc.shadow_bundle(model), meta


def deployed_message(result):
    """Тот же текст, что в интерфейсе этапа 9."""
    if result["status"] == "NEEDS_CLARIFICATION":
        return "Нужен текст обращения."
    if result["status"] == "MODEL_ERROR":
        return "Ошибка модели. Обращение остаётся оператору."
    if result.get("request_parts"):
        parts = [f"{part['intent'].replace('_', ' ')} ({part['category']} → {part['queue']})"
                 for part in result["request_parts"]]
        return "Оператору нужно проверить каждый запрос. " + " · ".join(parts)
    if result.get("semantic_status") in ("ABSTAINED", "UNAVAILABLE", "INPUT_TOO_LONG", "NO_VALIDATED_GATE"):
        return "Первичный прогноз требует проверки. У MiniLM нет подтверждённой подсказки."
    if result.get("classification_source") == "minilm":
        return "MiniLM предложила тему для незнакомой формулировки. Очередь должен подтвердить оператор."
    if result["category"] is None:
        return "Категория не определена для этого ввода. Нужна ручная проверка."
    if result["status"] == "MANUAL_REVIEW":
        return ("Показана предполагаемая тема. Автоматическое направление не подтверждено: "
                "оператору нужно проверить решение.")
    return "Сформирована рекомендация категории и очереди. Обращение не отправлено."


def shadow_message(shadow_result, shadow_policy):
    """Что сделала бы политика-кандидат, если бы автоотправку включили."""
    if shadow_result["status"] == "NEEDS_CLARIFICATION":
        return "Теневая политика: нужен текст обращения."
    if shadow_result["status"] == "AUTO_RECOMMENDED":
        return (f"Теневая политика (только для анализа) отправила бы сама: {shadow_result['queue']}. "
                f"Пороги: уверенность ≥ {shadow_result.get('primary_confidence_threshold') or shadow_policy.get('confidence_threshold')}, "
                f"сходство с обучением ≥ {shadow_policy.get('similarity_threshold')}.")
    return "Теневая политика (только для анализа) тоже оставила бы обращение оператору."


def explanation_rows(model, text):
    """Слова и сочетания букв, сильнее всего повлиявшие на линейную часть модели."""
    if not isinstance(text, str) or not sc.preprocess(text):
        return []
    try:
        info = sc.local_explanation(model, text)
    except Exception:  # объяснение вспомогательное и не должно ломать ответ
        return []
    rows = []
    for item in info.get("terms", []):
        term = item["term"]
        kind, _, value = term.partition("__")
        kind = {"w": "слова", "c": "буквы"}.get(kind, kind)
        rows.append([f"{kind}: «{value}»", round(item["contribution"], 3)])
    return rows


def route_with_shadow(text, model, shadow, detect_multiple=False):
    """Ответ для внешних систем: рабочее решение плюс теневое для сравнения."""
    from support_router.routing import route_policy_pair

    result, shadow_result = route_policy_pair(text, model, shadow, detect_multiple=detect_multiple)
    result["shadow"] = {"status": shadow_result["status"], "queue": shadow_result["queue"],
                        "policy_version": shadow_result["policy_version"]}
    return result


def model_card(meta):
    test = (meta.get("test") or {}).get("classification") or {}
    shadow = (meta.get("test") or {}).get("shadow_routing") or {}
    policy = meta.get("policy") or {}
    lines = [f"**Модель:** {meta.get('model_name')} (`{meta.get('model_id')}`)",
             f"**Политика:** {policy.get('version')}, автоотправка "
             f"{'включена' if policy.get('auto_enabled') else 'выключена'}; теневые пороги: уверенность ≥ "
             f"{policy.get('confidence_threshold')}, сходство ≥ {policy.get('similarity_threshold')}"]
    if policy.get("misroute_cost"):
        lines.append("Порог зависит от цены ошибочной очереди. Для незнакомых формулировок доступна подсказка MiniLM.")
    if test:
        lines.append(f"**Test Bitext:** macro-F1 {test.get('macro_f1', 0):.4f}, accuracy "
                     f"{test.get('accuracy', 0):.4f}, ошибок {test.get('errors')} из {test.get('n')}")
    if shadow:
        lines.append(f"**Теневой режим на test:** сама решила бы {shadow.get('automatic_n')} из "
                     f"{shadow.get('n')}, ошибок среди них {shadow.get('automatic_category_errors')}")
    lines.append("Test Bitext состоит из перефразировок тех же шаблонов, что и обучение, поэтому "
                 "высокие цифры на нём не обещают такой же точности на живых обращениях.")
    return "\n\n".join(lines)


def make_handler(model, shadow):
    """Обработчик кнопки «Анализировать»: рабочее решение, теневое и объяснение."""
    def process(user_text, history, detect_multiple=False):
        result, updated = sc.web_request(user_text, history, model, detect_multiple=detect_multiple)
        shadow_result = sc.route_ticket(user_text, shadow, detect_multiple=detect_multiple)
        return (result["category"] or "Не определена", result["queue"] or "", result["status"],
                result["priority"], "; ".join(result["reason_codes"]), result["top3"],
                pd.DataFrame(updated[::-1], columns=HISTORY_COLUMNS), updated,
                deployed_message(result) + (" Причина приоритета: " + ", ".join(result["priority_reasons"])
                                            if result["priority_reasons"] else ""), shadow_message(shadow_result, shadow.policy),
                pd.DataFrame([] if result["classification_source"] == "minilm" else explanation_rows(model, user_text), columns=["Признак", "Вклад"]))
    return process


def build_ui(model, shadow, meta):
    import gradio as gr

    process = make_handler(model, shadow)

    with gr.Blocks(title="Support Ticket Router", analytics_enabled=False) as demo:
        gr.Markdown("# Маршрутизация обращений в поддержку")
        gr.Markdown("**Только исследовательский режим:** автоотправки нет, предварительная категория "
                    "и очередь требуют проверки оператором. Демонстрация работает с обращениями на "
                    "английском; вводите обезличенные тексты.")
        with gr.Accordion("Что означают категории", open=False):
            gr.Markdown(CATEGORY_HELP)
        with gr.Accordion("О модели", open=False):
            gr.Markdown(model_card(meta))
        session_history = gr.State([])
        with gr.Row():
            with gr.Column():
                user_text = gr.Textbox(lines=4, label="Обращение клиента")
                multiple = gr.Checkbox(label="Распознавать несколько запросов в одном обращении", value=False)
                gr.Examples(examples=[[t] for t in EXAMPLES], inputs=user_text)
                submit = gr.Button("Анализировать", variant="primary")
                message = gr.Markdown()
                shadow_note = gr.Markdown()
            with gr.Column():
                category = gr.Textbox(label="Предполагаемая категория обращения", interactive=False)
                queue = gr.Textbox(label="Рекомендованная очередь", interactive=False)
                status = gr.Textbox(label="Статус", interactive=False)
                priority = gr.Textbox(label="Предварительный приоритет", interactive=False)
                reasons = gr.Textbox(label="Причины решения", interactive=False)
                top3 = gr.Label(label="Top-3 категорий (оценки модели)", num_top_classes=3)
        with gr.Accordion("Что повлияло на категорию (линейная часть модели, не причина)", open=False):
            explanation = gr.Dataframe(headers=["Признак", "Вклад"], interactive=False)
        history_table = gr.Dataframe(headers=HISTORY_COLUMNS, interactive=False,
                                     label="История этой сессии (до 100 записей)")
        outputs = [category, queue, status, priority, reasons, top3, history_table,
                   session_history, message, shadow_note, explanation]
        submit.click(process, inputs=[user_text, session_history, multiple], outputs=outputs, api_name="analyze")
        user_text.submit(process, inputs=[user_text, session_history, multiple], outputs=outputs,
                         api_visibility="private")

        def route_json(text: str, detect_multiple: bool = False) -> dict:
            """Полный ответ route_ticket в JSON плюс теневое решение для сравнения."""
            return route_with_shadow(text, model, shadow, detect_multiple=detect_multiple)

        gr.api(route_json, api_name="route_json")
    return demo


def batch_file(src, dst, model, shadow, *, detect_multiple=False):
    frame = pd.read_csv(src, dtype=str, keep_default_na=False)
    column = next((c for c in ("text", "utterance") if c in frame.columns), None)
    if column is None:
        raise SystemExit("В CSV нужна колонка text или utterance")
    texts = frame[column].tolist()
    deployed = sc.batch_route(model, texts, detect_multiple=detect_multiple) if texts else []
    shadowed = sc.batch_route(shadow, texts, detect_multiple=detect_multiple) if texts else []
    rows = []
    for i, (text, r, s) in enumerate(zip(texts, deployed, shadowed)):
        row = {"row": i, "text_masked": sc.redact_text(text),
               "suggested_category": r["suggested_category"], "category": r["category"],
               "queue": r["queue"], "topic_queue": r["topic_queue"], "status": r["status"],
               "priority": r["priority"], "confidence": r["confidence"], "similarity": r["similarity"],
               "risk_flags": ";".join(r["risk_flags"]), "reason_codes": ";".join(r["reason_codes"]),
               "shadow_status": s["status"], "shadow_queue": s["queue"],
               "model_id": r["model_id"], "policy_version": r["policy_version"]}
        row.update({key: r[key] for key in ("classification_source", "semantic_status", "primary_category",
                                            "primary_confidence", "primary_confidence_threshold")})
        row["priority_reasons"] = ";".join(r["priority_reasons"])
        if detect_multiple:
            row["request_parts"] = json.dumps(r["request_parts"], ensure_ascii=False)
        if "id" in frame.columns:
            row = {"id": frame.at[i, "id"], **row}
        rows.append(row)
    columns = (["id"] if "id" in frame.columns else []) + BATCH_COLUMNS
    if detect_multiple:
        columns = columns + ["request_parts"]
    out = pd.DataFrame(rows, columns=columns)
    out.to_csv(dst, index=False)
    return out


def self_check(model, shadow, meta):
    """Контрольные фразы: модель загружается и рабочий режим никогда не отправляет сам."""
    rows = []
    for text in CHECK_TEXTS:
        r = sc.route_ticket(text, model)
        s = sc.route_ticket(text, shadow)
        assert r["status"] != "AUTO_RECOMMENDED", "рабочий режим не должен отправлять сам"
        rows.append({"text": text, "category": r["category"], "status": r["status"],
                     "priority": r["priority"], "risk_flags": ";".join(r["risk_flags"]),
                     "shadow_status": s["status"]})
    print(f"Модель {meta['model_name']} ({meta['model_id']}) загружена, политика "
          f"{meta['policy']['version']}, автоотправка: {meta['policy']['auto_enabled']}")
    print(pd.DataFrame(rows).to_string(index=False))
    print("CHECK OK")


def main(argv=None):
    # Windows: в консоль и в канал печатаем в UTF-8, иначе русские строки роняют процесс.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure") and (stream.encoding or "").lower().replace("-", "") != "utf8":
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Маршрутизация обращений в поддержку (демонстрационная версия)")
    parser.add_argument("--models", default=str(MODELS_DIR), help="папка с router.joblib и router_meta.json")
    parser.add_argument("--detect-multiple", action="store_true", help="выделить отдельные запросы в пакетном режиме")
    parser.add_argument("--batch", help="CSV с колонкой text или utterance")
    parser.add_argument("--out", help="куда записать результат пакетной разметки")
    parser.add_argument("--check", action="store_true", help="проверить модель на контрольных фразах")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="временная публичная ссылка gradio.live")
    args = parser.parse_args(argv)
    model, shadow, meta = load_router(args.models)
    if args.batch:
        out = args.out or str(Path(args.batch).with_name(Path(args.batch).stem + "_routed.csv"))
        result = batch_file(args.batch, out, model, shadow, detect_multiple=args.detect_multiple)
        print(f"Размечено {len(result)} строк: {out}")
        print(result.status.value_counts().to_string())
        return
    if args.check:
        self_check(model, shadow, meta)
        return
    import gradio as gr
    demo = build_ui(model, shadow, meta)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share,
                theme=gr.themes.Soft())


if __name__ == "__main__":
    main()
