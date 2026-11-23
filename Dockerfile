# Образ HTTP-сервиса маршрутизации. Модель внутри образа: код и модель выкладываются
# и откатываются вместе, а load_bundle при старте проверяет, что они совпадают.
# Перед сборкой обучите модель: ноутбук или `python -m support_router.train`.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    PYTHONPATH=/app/src \
    SUPPORT_ROUTER_HOME=/app \
    SUPPORT_ROUTER_MODELS_DIR=/app/models

WORKDIR /app

# Сначала зависимости: этот слой пересобирается, только когда меняется список пакетов.
COPY requirements-api.txt .
RUN --mount=type=secret,id=pip_ca \
    if [ -f /run/secrets/pip_ca ]; then export PIP_CERT=/run/secrets/pip_ca; fi && \
    pip install -r requirements-api.txt

COPY src ./src
COPY models/router.joblib models/router_meta.json ./models/
COPY models/encoder ./models/encoder

RUN chmod -R a+rX /app/src /app/models && useradd --create-home --uid 10001 router
USER router

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=30s --retries=3 \
  CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status == 200 else 1)"

CMD ["uvicorn", "support_router.api:app", "--host", "0.0.0.0", "--port", "8000"]
