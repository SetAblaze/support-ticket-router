# HTTP API

Сервис запускается командой `uvicorn support_router.api:app --port 8000` или в Docker (`docker compose up --build`). Интерактивная документация со всеми схемами: http://localhost:8000/docs.

## Адреса

| Метод и адрес | Что делает | Ответы |
|---|---|---|
| `POST /v1/route` | решение по одному обращению | 200, 422 неверный запрос, 503 модель не загружена |
| `POST /v1/route/batch` | решение по пачке до 100 обращений, порядок сохраняется | 200, 422, 503 |
| `GET /v1/model` | паспорт модели: номер, политика, версия scikit-learn, метрики test | 200, 503 |
| `GET /health` | процесс жив | всегда 200 |
| `GET /ready` | модель загружена и прошла проверки | 200 или 503 с причиной |
| `GET /metrics` | метрики Prometheus | 200 |
| `POST /v1/feedback` | явное подтверждение или исправление оператора | 200, 401, 409, 422, 503 |
| `GET /v1/feedback/summary` | агрегаты по сохранённым событиям | 200, 401, 503 |

Каждый ответ несёт заголовок `x-request-id`: его можно передать в запросе, иначе сервис создаст свой. По нему запрос находится в журнале.

## Запрос

```json
{"text": "I want to cancel my order", "ticket_id": "CRM-1042"}
```

- `text`: строка до 5000 символов. Пустая строка допустима: ответ попросит прислать текст.
- `ticket_id`: необязательный номер обращения в CRM (буквы, цифры, `_ . : -`, до 128 символов), возвращается как есть.
- Лишние поля, число вместо строки, отсутствие `text`, текст длиннее 5000 символов и тело не в JSON дают 422. В ответе об ошибке присланный текст не повторяется.

## Ответ

```json
{
  "request_parts": [],
  "ticket_id": "CRM-1042",
  "status": "MANUAL_REVIEW",
  "category": "ORDER",
  "suggested_category": "ORDER",
  "queue": "Queue_Manual_Review",
  "topic_queue": "Queue_Orders",
  "priority": "P3",
  "priority_reasons": [],
  "classification_source": "primary",
  "primary_category": "ORDER",
  "primary_confidence": 0.9884,
  "primary_confidence_threshold": 0.9,
  "semantic_status": "NOT_REQUESTED",
  "semantic_confidence": null,
  "semantic_similarity": null,
  "confidence": 0.9884,
  "similarity": 0.9727,
  "top3": {
    "ORDER": 0.9884,
    "ACCOUNT": 0.0088,
    "CANCELLATION_FEE": 0.0008
  },
  "category_probabilities": {
    "ORDER": 0.9884,
    "ACCOUNT": 0.0088,
    "CANCELLATION_FEE": 0.0008
  },
  "expected_loss": {
    "ORDER": 0.0149,
    "REVIEW": 1.0
  },
  "expected_loss_action": "ORDER",
  "expected_loss_review": 1.0,
  "risk_flags": [],
  "reason_codes": [
    "AUTO_POLICY_DISABLED"
  ],
  "model_id": "router-17e31a0d6ea04c7b",
  "policy_version": "retail-cascade-v1",
  "delivery_status": "NOT_SENT",
  "created_at_utc": "2026-11-23T18:55:05.356932+00:00",
  "shadow": {
    "status": "AUTO_RECOMMENDED",
    "queue": "Queue_Orders",
    "policy_version": "retail-cascade-v1-SHADOW"
  }
}
```

| Поле | Значение |
|---|---|
| `status` | `NEEDS_CLARIFICATION` (нет текста), `MANUAL_REVIEW` (к оператору), `AUTO_RECOMMENDED` (только в теневом режиме), `MODEL_ERROR` (модель не смогла оценить, к оператору) |
| `queue` | куда положить обращение сейчас; пока автоотправка выключена, это всегда ручная очередь |
| `topic_queue` | очередь темы: подсказка оператору |
| `priority` | срочность P1–P3 по правилам магазина. Основания возвращает `priority_reasons` |
| `reason_codes` | почему обращение ушло оператору: другой язык, несколько тем, сбой сайта, низкая уверенность, малое сходство с обучением и т.д. |
| `category_probabilities` | полный вектор вероятностей 11 тем; используется для ожидаемых потерь |
| `expected_loss` | `R(a\|x)` для каждой очереди и ручной проверки `REVIEW` |
| `expected_loss_action` | очередь или `REVIEW` с минимальными ожидаемыми потерями |
| `expected_loss_review` | стоимость ручной проверки в тех же единицах |
| `shadow` | что сделала бы политика-кандидат; ничего не отправляет |

## Источник предложенной темы

`classification_source` принимает `primary`, `minilm` или null при отсутствии прогноза. `primary_confidence_threshold` — применённый к основной теме порог. `similarity` всегда относится к лексическому сходству. Для MiniLM отдельно доступны `semantic_confidence` и `semantic_similarity`. Значения `semantic_status` — `NOT_REQUESTED`, `SUGGESTED`, `ABSTAINED`, `INPUT_TOO_LONG`, `UNAVAILABLE` или `NO_VALIDATED_GATE`. При `SUGGESTED` очередь для проверки находится в `topic_queue`, фактическая очередь — ручная. Действующие пороги и условные цены доступны через `GET /v1/model`. [Правила приоритета и маршрутизации](business_policy.md)

## Ожидаемые потери

Поле `expected_loss` показывает `R(a|x) = Σ P(y|x) L(y,a)`: ожидаемую цену
маршрута в каждую очередь и цену ручной проверки. Политика принимает тему
только если маршрут в неё дешевле проверки и совпадает с лучшим действием.
Иначе появляется причина `EXPECTED_LOSS_PREFERS_REVIEW`,
`EXPECTED_LOSS_DIFFERS_FROM_PREDICTION` или
`EXPECTED_LOSS_ROUTE_NOT_BETTER_THAN_REVIEW`, и обращение уходит оператору.
Матрица потерь и пороги доступны в `GET /v1/model`.
[Подробнее](business_policy.md)

## Обратная связь оператора

`POST /v1/feedback` требует заголовок `x-feedback-key`, совпадающий с
`ROUTER_FEEDBACK_API_KEY`. Без настроенного ключа endpoint отвечает 503, с
неверным ключом — 401. Поле `confirmed` обязательно: если его не передать,
событие не создаётся. `operator_category` можно передать только при
`confirmed=false`; он должен отличаться от предложенной категории. Событие
привязывается к `model_id` и `policy_version` загруженной модели; при
несовпадении возвращается 409.

```json
{
  "ticket_id": "CRM-1042",
  "proposed_category": "PAYMENT",
  "operator_category": "REFUNDS",
  "confirmed": false,
  "reviewer_id": "operator-7",
  "redacted_text": "my card payment was declined",
  "event_id": "crm-1042-1"
}
```

Текст должен быть уже обезличен. Лимиты задаются `ROUTER_MAX_TEXT_CHARS` и
`ROUTER_MAX_BATCH`; путь хранения — `ROUTER_FEEDBACK_PATH` (по умолчанию
`var/feedback/events.jsonl`, вне git). `GET /v1/feedback/summary` возвращает
число событий, подтверждений, исправлений, конфликтов и метрики точности по
подтверждённым событиям. [Правила feedback](runbook.md)

## Режим нескольких запросов

Добавьте параметр `?detect_multiple=true` к `/v1/route` или `/v1/route/batch`. В пакетном запросе режим действует для всех строк. Поле `request_parts` содержит распознанные действия в порядке упоминания. Для сообщения `My payment failed and I need to change my shipping address` ответ включает:

```json
{
  "request_parts": [
    {"intent": "payment_issue", "category": "PAYMENT", "queue": "Queue_Payments"},
    {"intent": "change_shipping_address", "category": "SHIPPING", "queue": "Queue_Delivery"}
  ],
  "status": "MANUAL_REVIEW",
  "queue": "Queue_Manual_Review"
}
```

Поле `category` по-прежнему показывает прогноз модели для всего сообщения. При обнаружении нескольких запросов обе политики оставляют исходный тикет оператору с причиной `POSSIBLE_MULTIPLE_REQUESTS`. Очереди в списке служат подсказками. При выключенном режиме, пустом вводе или недостатке признаков двух разных действий `request_parts` равен `[]`. Штатное правило может отдельно пометить сообщение как неоднозначное.

## Примеры

```bash
curl -s -X POST http://localhost:8000/v1/route \
  -H 'content-type: application/json' \
  -d '{"text": "Someone hacked my account", "ticket_id": "CRM-7"}'

curl -s -X POST http://localhost:8000/v1/route/batch \
  -H 'content-type: application/json' \
  -d '{"tickets": [{"text": "Where is my refund?"}, {"text": ""}, {"text": "I will sue you"}]}'
```
