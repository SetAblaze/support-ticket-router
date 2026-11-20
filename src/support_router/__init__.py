"""Маршрутизация обращений в поддержку интернет-магазина.

Пакет собран из модуля support_core.py версии 3.1.1 без изменения логики:
функции перенесены по смыслу в отдельные модули, а здесь собраны в один
набор имён, чтобы ноутбук, тесты и приложения писали `import support_router as sc`.
"""
from __future__ import annotations

from . import category_examples, manual_cases, scenario_data, semantic_encoder  # noqa: F401
from .artifacts import (  # noqa: F401
    DECISION_MODULES,
    decision_code_sha256,
    load_bundle,
    save_bundle,
)
from .config import (  # noqa: F401
    CATEGORIES,
    CATEGORY_TO_QUEUE,
    LINGUISTIC_FLAGS,
    MANUAL_QUEUE,
    PREPROCESS_VERSION,
    SCHEMA_VERSION,
    SEED,
    VERSION,
)
from .data import (  # noqa: F401
    assert_split_integrity,
    balanced_group_holdout,
    clean_frame,
    cross_split_similarity,
    lexical_groups,
    make_splits,
)
from .evaluation import (  # noqa: F401
    calibration_diagnostics,
    classification_metrics,
    intent_protocol,
    loio_evaluate,
    paired_error_test,
    paired_group_bootstrap,
    paired_intent_bootstrap,
    risk_recall,
    routing_metrics,
    wilson_lower,
    wilson_upper,
)
from .explain import (  # noqa: F401
    local_explanation,
)
from .models import (  # noqa: F401
    CONFIG_NAMES,
    RouterBundle,
    checked_proba,
    fit_calibrated_bundle,
    fit_model,
    make_model,
)
from .paths import (  # noqa: F401
    encoder_dir,
    models_dir,
    project_root,
)
from .routing import (  # noqa: F401
    batch_route,
    redact_text,
    route_ticket,
    schedule_demo,
    select_policy,
    shadow_bundle,
    web_request,
)
from .rules import (  # noqa: F401
    _context_effects,
    _input_reasons,
    _risk_signals,
    detect_signals,
)
from .text import (  # noqa: F401
    _PLACEHOLDER,
    _WS,
    flag_membership,
    parse_linguistic_flags,
    preprocess,
    preprocess_batch,
    text_id,
)
from .utils import (  # noqa: F401
    json_ready,
    runtime_versions,
    sha256_file,
    write_json,
)

__version__ = VERSION

__all__ = [
    'VERSION', 'PREPROCESS_VERSION', 'SCHEMA_VERSION', 'SEED',
    'CATEGORY_TO_QUEUE', 'CATEGORIES', 'MANUAL_QUEUE', 'LINGUISTIC_FLAGS',
    'parse_linguistic_flags', 'flag_membership', '_PLACEHOLDER', '_WS',
    'preprocess', 'preprocess_batch', 'text_id', 'sha256_file',
    'json_ready', 'write_json', 'runtime_versions', 'clean_frame',
    'lexical_groups', 'balanced_group_holdout', 'make_splits', 'assert_split_integrity',
    'cross_split_similarity', 'make_model', 'CONFIG_NAMES', 'fit_model',
    'checked_proba', 'RouterBundle', 'fit_calibrated_bundle', 'intent_protocol',
    'loio_evaluate', 'classification_metrics', 'wilson_upper', 'calibration_diagnostics',
    'routing_metrics', 'paired_error_test', 'paired_group_bootstrap', 'wilson_lower',
    'paired_intent_bootstrap', 'risk_recall', '_risk_signals', '_input_reasons',
    'detect_signals', '_context_effects', 'route_ticket', 'batch_route',
    'select_policy', 'shadow_bundle', 'redact_text', 'web_request',
    'schedule_demo', 'DECISION_MODULES', 'decision_code_sha256', 'save_bundle',
    'load_bundle', 'local_explanation', 'project_root', 'models_dir',
    'encoder_dir', 'category_examples', 'manual_cases', 'scenario_data',
    'semantic_encoder', '__version__',
]
