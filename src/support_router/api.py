"""HTTP-сервис маршрутизации на FastAPI.

    uvicorn support_router.api:app --host 0.0.0.0 --port 8000

Решение принимает та же функция route_ticket, что в ноутбуке, тестах и интерфейсе
Gradio. Сервис ничего не отправляет в CRM: он возвращает тему, очередь, приоритет
и причины, а куда направить обращение, решает вызывающая система и оператор.
Автоотправка выключена политикой модели; поле shadow показывает, что сделала бы
политика-кандидат, и служит только для наблюдения.

Текст обращения не пишется ни в журнал, ни в метрики. Модель загружается при старте
из SUPPORT_ROUTER_MODELS_DIR (по умолчанию models/ в корне проекта) с проверкой хешей
и версии scikit-learn; если проверка не прошла, сервис жив (/health), но не готов
(/ready и /v1/route отвечают 503).
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field

from .artifacts import decision_code_sha256, load_bundle
from .config import VERSION
from .monitoring import RouterMetrics
from .paths import models_dir
from .routing import batch_policy_pairs, route_policy_pair, shadow_bundle

MAX_TEXT_CHARS = int(os.environ.get("ROUTER_MAX_TEXT_CHARS", "5000"))
MAX_BATCH = int(os.environ.get("ROUTER_MAX_BATCH", "100"))
KNOWN_PATHS = {"/", "/health", "/ready", "/metrics", "/v1/model", "/v1/route", "/v1/route/batch",
               "/docs", "/openapi.json"}

log = logging.getLogger("support_router.api")
if not log.handlers:  # журнал сервиса: одна JSON-строка на событие, без текста обращения
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(_handler)
    log.setLevel(os.environ.get("ROUTER_LOG_LEVEL", "INFO"))
    log.propagate = False


class TicketIn(BaseModel):
    """Одно обращение. Пустой текст допустим: ответ попросит прислать текст."""
    model_config = ConfigDict(extra="forbid")
    text: str = Field(max_length=MAX_TEXT_CHARS, description="Текст обращения на английском",
                      examples=["I want to cancel my order"])
    ticket_id: str | None = Field(default=None, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$",
                                  description="Номер обращения во внешней системе, возвращается как есть",
                                  examples=["CRM-1"])


class BatchIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tickets: list[TicketIn] = Field(min_length=1, max_length=MAX_BATCH)


class ShadowOut(BaseModel):
    status: str
    queue: str | None
    policy_version: str


class RequestPart(BaseModel):
    intent: str
    category: str
    queue: str


class DecisionOut(BaseModel):
    request_parts: list[RequestPart]
    ticket_id: str | None
    status: Literal["NEEDS_CLARIFICATION", "MANUAL_REVIEW", "AUTO_RECOMMENDED", "MODEL_ERROR"]
    category: str | None
    suggested_category: str | None
    queue: str | None
    topic_queue: str | None
    priority: Literal["P1", "P2", "P3"]
    priority_reasons: list[str]
    classification_source: Literal["primary", "minilm"] | None
    primary_category: str | None
    primary_confidence: float | None
    primary_confidence_threshold: float | None
    semantic_status: str
    semantic_confidence: float | None
    semantic_similarity: float | None
    confidence: float | None
    similarity: float | None
    top3: dict[str, float]
    category_probabilities: dict[str, float] = Field(default_factory=dict)
    expected_loss: dict[str, float] | None = None
    expected_loss_action: str | None = None
    expected_loss_review: float | None = None
    risk_flags: list[str]
    reason_codes: list[str]
    model_id: str
    policy_version: str
    delivery_status: str
    created_at_utc: str
    shadow: ShadowOut


class BatchOut(BaseModel):
    results: list[DecisionOut]


def _json_log(event, **fields):
    log.info(json.dumps({"event": event, **fields}, ensure_ascii=False, default=str))


def _decision(result, shadow_result, ticket_id):
    out = {key: result.get(key) for key in DecisionOut.model_fields if key not in ("ticket_id", "shadow")}
    out["ticket_id"] = ticket_id
    out["shadow"] = {"status": shadow_result["status"], "queue": shadow_result["queue"],
                     "policy_version": shadow_result["policy_version"]}
    return out


def create_app(models_directory=None, registry=None) -> FastAPI:
    """Отдельный экземпляр приложения со своей моделью и своим реестром метрик."""
    state = {"model": None, "shadow": None, "meta": None, "error": "model is not loaded yet"}
    metrics = RouterMetrics(registry)

    def load_model():
        directory = models_directory or models_dir()
        try:
            # Модель собрана этой же командой; joblib загружает pickle, поэтому чужие файлы не открываем.
            model, meta = load_bundle(directory, trust_local_artifact=True)
        except Exception as exc:  # сервис остаётся живым, но не готовым
            state.update(model=None, shadow=None, meta=None, error=f"{type(exc).__name__}: {exc}")
            metrics.model_loaded.set(0)
            _json_log("model_load_failed", directory=str(directory), error=state["error"])
            return
        state.update(model=model, shadow=shadow_bundle(model), meta=meta, error=None)
        metrics.set_model(meta, VERSION)
        _json_log("model_loaded", directory=str(directory), model_id=meta["model_id"],
                  policy_version=meta["policy"].get("version"), auto_enabled=meta["policy"].get("auto_enabled"))

    @asynccontextmanager
    async def lifespan(_app):
        load_model()
        yield

    app = FastAPI(title="Support Ticket Router", version=VERSION, lifespan=lifespan,
                  description="Тема, очередь, приоритет и риски обращения в поддержку. "
                              "Сервис ничего не отправляет сам: автоотправка выключена.")
    app.state.router = state
    app.state.metrics = metrics
    app.state.reload_model = load_model

    @app.middleware("http")
    async def observe(request: Request, call_next):
        started = time.perf_counter()
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        request.state.request_id = request_id
        endpoint = request.url.path if request.url.path in KNOWN_PATHS else "other"
        try:
            response = await call_next(request)
        except Exception:
            seconds = time.perf_counter() - started
            metrics.observe_request(endpoint, request.method, 500, seconds)
            log.exception("unhandled error, request_id=%s", request_id)
            return JSONResponse({"detail": "internal error", "request_id": request_id}, status_code=500,
                                headers={"x-request-id": request_id})
        seconds = time.perf_counter() - started
        metrics.observe_request(endpoint, request.method, response.status_code, seconds)
        response.headers["x-request-id"] = request_id
        if endpoint != "/metrics":
            _json_log("request", request_id=request_id, endpoint=endpoint, method=request.method,
                      status_code=response.status_code, duration_ms=round(seconds * 1000, 2))
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        # Без поля input: в ответе не повторяем присланный текст обращения.
        errors = [{"loc": list(e.get("loc", [])), "msg": e.get("msg"), "type": e.get("type")}
                  for e in exc.errors()]
        return JSONResponse({"detail": errors}, status_code=422)

    def require_model():
        if state["model"] is None:
            raise HTTPException(status_code=503, detail={"status": "not_ready", "reason": state["error"]})
        return state["model"], state["shadow"]

    def record_decision(request, ticket, result, shadow_result):
        metrics.observe_decision(result, shadow_result, len(ticket.text))
        _json_log("decision", request_id=request.state.request_id, ticket_id=ticket.ticket_id,
                  status=result["status"], priority=result["priority"], category=result["suggested_category"],
                  queue=result["queue"], risk_flags=result["risk_flags"], reason_codes=result["reason_codes"],
                  confidence=result["confidence"], similarity=result["similarity"],
                  classification_source=result["classification_source"], semantic_status=result["semantic_status"],
                  shadow_status=shadow_result["status"], text_length=len(ticket.text),
                  model_id=result["model_id"])

    @app.get("/")
    def index():
        return {"service": "support-ticket-router", "version": VERSION, "docs": "/docs",
                "route": "POST /v1/route", "batch": "POST /v1/route/batch"}

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready")
    def ready():
        if state["model"] is None:
            return JSONResponse({"status": "not_ready", "reason": state["error"]}, status_code=503)
        return {"status": "ready", "model_id": state["meta"]["model_id"]}

    @app.get("/v1/model")
    def model_card():
        require_model()
        meta = state["meta"]
        test = (meta.get("test") or {})
        classification = test.get("classification") or {}
        shadow = test.get("shadow_routing") or {}
        return {"model_id": meta["model_id"], "model_name": meta["model_name"], "classes": meta["classes"],
                "policy": {k: meta["policy"].get(k) for k in ["version", "auto_enabled", "candidate_auto_enabled",
                                                               "confidence_threshold", "similarity_threshold",
                                                               "status", "misroute_cost", "default_misroute_cost",
                                                               "review_cost", "semantic_gate"]},
                "scikit_learn": meta["versions"].get("scikit-learn"), "code_version": VERSION,
                "decision_code_sha256": decision_code_sha256(),
                "test": {"n": classification.get("n"), "macro_f1": classification.get("macro_f1"),
                         "errors": classification.get("errors"), "shadow_coverage": shadow.get("coverage"),
                         "shadow_errors": shadow.get("automatic_queue_errors")}}

    @app.post("/v1/route", response_model=DecisionOut)
    def route(ticket: TicketIn, request: Request, detect_multiple: bool = False):
        model, shadow = require_model()
        result, shadow_result = route_policy_pair(ticket.text, model, shadow, detect_multiple=detect_multiple)
        record_decision(request, ticket, result, shadow_result)
        return _decision(result, shadow_result, ticket.ticket_id)

    @app.post("/v1/route/batch", response_model=BatchOut)
    def route_batch(batch: BatchIn, request: Request, detect_multiple: bool = False):
        model, shadow = require_model()
        texts = [t.text for t in batch.tickets]
        pairs = batch_policy_pairs(texts, model, shadow, detect_multiple=detect_multiple)
        out = []
        for ticket, (result, shadow_result) in zip(batch.tickets, pairs, strict=True):
            record_decision(request, ticket, result, shadow_result)
            out.append(_decision(result, shadow_result, ticket.ticket_id))
        return {"results": out}

    @app.get("/metrics")
    def prometheus_metrics():
        return Response(metrics.render(), media_type=metrics.content_type)

    return app


app = create_app()


def main():
    import argparse

    import uvicorn
    parser = argparse.ArgumentParser(description="HTTP-сервис маршрутизации обращений")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run("support_router.api:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
