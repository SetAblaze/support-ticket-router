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
  "ticket_id": "CRM-1042",
  "status": "MANUAL_REVIEW",
  "category": "ORDER",
  "suggested_category": "ORDER",
  "queue": "Queue_Manual_Review",
  "topic_queue": "Queue_Orders",
  "priority": "P3",
  "confidence": 0.988,
  "similarity": 0.973,
  "top3": {"ORDER": 0.988, "ACCOUNT": 0.009, "CANCELLATION_FEE": 0.001},
  "risk_flags": [],
  "reason_codes": ["AUTO_POLICY_DISABLED"],
  "model_id": "router-7f7d5b68a20f49b5",
  "policy_version": "selective-v3-shadow",
  "delivery_status": "NOT_SENT",
  "created_at_utc": "2026-11-18T11:28:24.436197+00:00",
  "shadow": {"status": "AUTO_RECOMMENDED", "queue": "Queue_Orders", "policy_version": "selective-v3-shadow-SHADOW"}
}
```

| Поле | Значение |
|---|---|
| `status` | `NEEDS_CLARIFICATION` (нет текста), `MANUAL_REVIEW` (к оператору), `AUTO_RECOMMENDED` (только в теневом режиме), `MODEL_ERROR` (модель не смогла оценить, к оператору) |
| `queue` | куда положить обращение сейчас; пока автоотправка выключена, это всегда ручная очередь |
| `topic_queue` | очередь темы: подсказка оператору |
| `priority` | P1 угроза или мошенничество, P2 судебная претензия, P3 остальное |
| `reason_codes` | почему обращение ушло оператору: другой язык, несколько тем, сбой сайта, низкая уверенность, малое сходство с обучением и т.д. |
| `shadow` | что сделала бы политика-кандидат; ничего не отправляет |

## Примеры

```bash
curl -s -X POST http://localhost:8000/v1/route \
  -H 'content-type: application/json' \
  -d '{"text": "Someone hacked my account", "ticket_id": "CRM-7"}'

curl -s -X POST http://localhost:8000/v1/route/batch \
  -H 'content-type: application/json' \
  -d '{"tickets": [{"text": "Where is my refund?"}, {"text": ""}, {"text": "I will sue you"}]}'
```
