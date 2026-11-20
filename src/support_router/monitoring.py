"""Метрики Prometheus для сервиса маршрутизации.

Текст обращения в метрики не попадает: только статус, тема, очередь, приоритет,
флаги, причины, длина текста, уверенность, сходство и задержка. У всех меток
ограниченный набор значений (11 тем, 9 очередей, 4 статуса, 3 приоритета, коды
причин из rules.py и routing.py), поэтому число временных рядов не растёт с трафиком.
"""
from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest

LATENCY_BUCKETS = (.005, .01, .025, .05, .1, .25, .5, 1., 2.5, 5.)
SCORE_BUCKETS = (.1, .2, .3, .4, .45, .5, .6, .7, .8, .85, .9, .95, .99, 1.)
LENGTH_BUCKETS = (0, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000)


class RouterMetrics:
    """Все метрики сервиса в отдельном реестре: у каждого экземпляра приложения свой."""

    content_type = CONTENT_TYPE_LATEST

    def __init__(self, registry: CollectorRegistry | None = None):
        self.registry = registry or CollectorRegistry()
        r = self.registry
        self.requests = Counter("router_http_requests", "HTTP-запросы к сервису",
                                ["endpoint", "method", "status_code"], registry=r)
        self.latency = Histogram("router_http_request_duration_seconds", "Время ответа сервиса",
                                 ["endpoint"], buckets=LATENCY_BUCKETS, registry=r)
        self.decisions = Counter("router_decisions", "Решения рабочего режима",
                                 ["status", "queue", "priority"], registry=r)
        self.shadow = Counter("router_shadow_decisions", "Решения теневой политики (ничего не отправляют)",
                              ["status"], registry=r)
        self.categories = Counter("router_suggested_category", "Тема, предложенная моделью",
                                  ["category"], registry=r)
        self.risk_flags = Counter("router_risk_flags", "Флаги риска", ["flag"], registry=r)
        self.reasons = Counter("router_reason_codes", "Причины решения", ["reason"], registry=r)
        self.confidence = Histogram("router_confidence", "Уверенность модели в предложенной теме",
                                    buckets=SCORE_BUCKETS, registry=r)
        self.similarity = Histogram("router_training_similarity", "Сходство текста с обучающими фразами",
                                    buckets=SCORE_BUCKETS, registry=r)
        self.text_length = Histogram("router_text_length_chars", "Длина обращения в символах",
                                     buckets=LENGTH_BUCKETS, registry=r)
        self.model_errors = Counter("router_model_errors", "Ошибки модели при оценке обращения", registry=r)
        self.model_loaded = Gauge("router_model_loaded", "1, если модель загружена и прошла проверки",
                                  registry=r)
        self.model_info = Gauge("router_model_info", "Паспорт загруженной модели",
                                ["model_id", "model_name", "policy_version", "code_version"], registry=r)

    def observe_request(self, endpoint, method, status_code, seconds):
        self.requests.labels(endpoint, method, str(status_code)).inc()
        self.latency.labels(endpoint).observe(seconds)

    def observe_decision(self, result, shadow_result, text_length):
        self.decisions.labels(result["status"], result["queue"] or "none", result["priority"]).inc()
        self.shadow.labels(shadow_result["status"]).inc()
        self.text_length.observe(text_length)
        if result.get("suggested_category"):
            self.categories.labels(result["suggested_category"]).inc()
        for flag in result["risk_flags"]:
            self.risk_flags.labels(flag).inc()
        for reason in result["reason_codes"]:
            self.reasons.labels(reason).inc()
        if result.get("confidence") is not None:
            self.confidence.observe(result["confidence"])
        if result.get("similarity") is not None:
            self.similarity.observe(result["similarity"])
        if result["status"] == "MODEL_ERROR":
            self.model_errors.inc()

    def set_model(self, meta, code_version):
        self.model_loaded.set(1)
        self.model_info.labels(meta["model_id"], meta["model_name"], meta["policy"].get("version", "missing"),
                               code_version).set(1)

    def render(self) -> bytes:
        return generate_latest(self.registry)
