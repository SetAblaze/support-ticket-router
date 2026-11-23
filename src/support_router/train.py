"""Обучение итоговой модели вне ноутбука: те же шаги и те же функции, что в этапах 4-7.

    python -m support_router.train --out models

Шаги: очистка (clean_frame) → группы похожих фраз (lexical_groups) → разбиение по группам
(make_splits) → обучение выбранной конфигурации с калибровкой на отдельной части
(fit_calibrated_bundle) → выбор теневой политики на val и OOD-dev (select_policy) →
единственная оценка на test → сохранение с паспортом (save_bundle).

Конфигурацию модели выбирает ноутбук на этапе 6 по LOIO; здесь она передаётся параметром,
по умолчанию linsvc_wc_C1. Автоотправка в политике всегда выключена.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

from . import scenario_data
from .artifacts import save_bundle
from .business import misroute_costs
from .cascade import fit_business_cascade
from .config import SEED
from .data import assert_split_integrity, clean_frame, lexical_groups, make_splits
from .evaluation import classification_metrics, routing_metrics
from .models import CONFIG_NAMES, fit_calibrated_bundle
from .paths import models_dir, project_root
from .routing import batch_route, select_policy, shadow_bundle
from .utils import sha256_file, write_json

DEFAULT_RAW = "data/raw/20000-Utterances-Training-dataset-for-chatbots-virtual-assistant-Bitext-sample.csv"
SPLIT_ROLES = {"train": "model selection and fitting", "calibration": "sigmoid calibration only",
               "val": "routing policy selection", "test": "final in-corpus evaluation"}


def build_splits(raw_path, processed_dir):
    """Этапы 4: очистка, группы и разбиение; части сохраняются и читаются обратно, как в ноутбуке."""
    raw = pd.read_csv(raw_path)
    frame, short, cleaning = clean_frame(raw)
    frame["group"], grouping = lexical_groups(frame.text.tolist(), threshold=.85)
    splits = make_splits(frame)
    processed_dir.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        part.to_csv(processed_dir / f"{name}.csv", index=False, lineterminator="\n")
    short.to_csv(processed_dir / "short_examples_retained.csv", index=False)
    # Ноутбук обучает модель на частях, прочитанных из CSV (ячейка 24): повторяем это дословно.
    splits = {k: pd.read_csv(processed_dir / f"{k}.csv", keep_default_na=False) for k in SPLIT_ROLES}
    overlap = assert_split_integrity(splits)
    manifest = {"seed": SEED, "raw_sha256": sha256_file(raw_path), "cleaning": cleaning,
                "grouping": grouping, "overlap_checks": overlap, "roles": SPLIT_ROLES,
                "splits": {k: {"rows": len(v), "sha256": sha256_file(processed_dir / f"{k}.csv"),
                               "class_counts": v.category.value_counts().to_dict()}
                           for k, v in splits.items()},
                "built_by": "support_router.train"}
    write_json(processed_dir / "split_manifest.json", manifest)
    return splits, manifest


def train(raw_path, out_dir, model_name="linsvc_wc_C1", processed_dir=None, log=print, *, cost_overrides=None):
    started = time.time()
    raw_path, out_dir = Path(raw_path), Path(out_dir)
    processed_dir = Path(processed_dir) if processed_dir else out_dir.parent / "data" / "processed"
    if model_name not in CONFIG_NAMES:
        raise ValueError(f"Неизвестная конфигурация {model_name!r}; доступны: {CONFIG_NAMES}")
    costs = misroute_costs(cost_overrides)
    splits, manifest = build_splits(raw_path, processed_dir)
    train_part, calibration, val, test = (splits[k] for k in SPLIT_ROLES)
    log("Части: " + ", ".join(f"{k} {len(v)}" for k, v in splits.items()))

    bundle = fit_calibrated_bundle(model_name, train_part, calibration)
    ood_dev, _ = scenario_data.frames()
    policy, _ = select_policy(bundle, val, ood_dev, target_error_upper=.01, min_auto=30,
                              false_auto_cost=25., review_cost=1.)
    fit_business_cascade(bundle, train_part, calibration, val, ood_dev, cost_overrides=costs)
    assert policy["auto_enabled"] is False, "автоотправка в рабочей политике должна быть выключена"
    bundle.metadata = {"split_manifest": manifest,
                       "model_selection": {"selected": model_name,
                                           "how": "LOIO over 25 intents in notebook stage 6"},
                       "fit_rows": len(train_part), "calibration_rows": len(calibration),
                       "policy_rows": len(val), "synthetic_ood_dev_rows": len(ood_dev),
                       "training_extension": [], "operational_auto_enabled": False,
                       "real_external_validation": False, "feature_columns": ["text"],
                       "excluded_feature_columns": ["intent", "flags", "group", "row_id", "category"],
                       "trained_by": "support_router.train"}

    texts, y_test = test.text.tolist(), test.category.to_numpy()
    classification = classification_metrics(y_test, bundle.predict(texts))
    deployed = routing_metrics(batch_route(bundle, texts), y_test)
    shadow = routing_metrics(batch_route(shadow_bundle(bundle), texts), y_test)
    path, meta = save_bundle(bundle, out_dir, test_metrics={"classification": classification,
                                                           "shadow_routing": shadow,
                                                           "operational_routing": deployed})
    summary = {"model_id": bundle.model_id, "model_name": model_name, "artifact": str(path),
               "test_macro_f1": classification["macro_f1"], "test_errors": classification["errors"],
               "shadow_coverage_test": shadow["coverage"], "policy": {k: policy[k] for k in
               ["version", "auto_enabled", "candidate_auto_enabled", "confidence_threshold",
                "similarity_threshold"]}, "seconds": round(time.time() - started, 1)}
    log(json.dumps(summary, ensure_ascii=False, indent=2))
    return bundle, meta, summary


def main(argv=None):
    # Windows: в консоль и в канал печатаем в UTF-8, иначе русские строки роняют процесс.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure") and (stream.encoding or "").lower().replace("-", "") != "utf8":
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Обучить итоговую модель маршрутизации без ноутбука")
    parser.add_argument("--raw", default=str(project_root() / DEFAULT_RAW), help="исходный CSV Bitext")
    parser.add_argument("--out", default=str(models_dir()), help="куда сохранить router.joblib и паспорт")
    parser.add_argument("--processed", default=None, help="куда сохранить части выборки (data/processed)")
    parser.add_argument("--model", default="linsvc_wc_C1", choices=CONFIG_NAMES,
                        help="конфигурация, выбранная на этапе 6")
    parser.add_argument("--misroute-cost", action="append", default=[], metavar="CATEGORY=UNITS",
                        help="цена ошибки в единицах ручного разбора, например PAYMENT=40 (можно повторять)")
    args = parser.parse_args(argv)
    try:
        entries = (item.split("=", 1) for item in args.misroute_cost)
        costs = misroute_costs({category: float(cost) for category, cost in entries})
    except ValueError as exc:
        parser.error(str(exc))
    if not Path(args.raw).exists():
        sys.exit(f"Нет файла {args.raw}: восстановите CSV из репозитория или укажите путь через --raw")
    processed = args.processed or str(Path(args.out).resolve().parent / "data" / "processed")
    train(args.raw, args.out, model_name=args.model, processed_dir=processed, cost_overrides=costs)


if __name__ == "__main__":
    main()
