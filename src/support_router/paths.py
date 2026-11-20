"""Где лежат модели проекта: корень репозитория, переменная окружения или текущая папка."""
from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    """Корень проекта: SUPPORT_ROUTER_HOME, иначе репозиторий (src/support_router), иначе текущая папка."""
    env = os.environ.get("SUPPORT_ROUTER_HOME")
    if env:
        return Path(env).resolve()
    package_dir = Path(__file__).resolve().parent
    if package_dir.parent.name == "src":
        return package_dir.parent.parent
    return Path.cwd().resolve()


def models_dir() -> Path:
    """Папка с router.joblib и router_meta.json: SUPPORT_ROUTER_MODELS_DIR или <корень>/models."""
    env = os.environ.get("SUPPORT_ROUTER_MODELS_DIR")
    return Path(env).resolve() if env else project_root() / "models"


def encoder_dir() -> Path:
    """Папка семантического энкодера: SUPPORT_ENCODER_DIR или <модели>/encoder."""
    env = os.environ.get("SUPPORT_ENCODER_DIR")
    return Path(env) if env else models_dir() / "encoder"
