"""Single source of truth for the current release report.

Numbers shown in ``reports/results.md`` and in the notebook are read from JSON
reports and the loaded model passport.  The historical notebook run stays in
``reports/results.json`` and is never presented as the current release.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from . import independent_cases
from .config import VERSION
from .identity import ExperimentIdentityError, identity_from_meta, validate_report_identity
from .paths import project_root
from .utils import sha256_file, write_json

REPORT_FILES = {
    "independent_benchmark": "reports/independent_benchmark.json",
    "business_cascade": "reports/business_cascade.json",
    "representation_comparison": "reports/representation_comparison.json",
    "feedback_simulation": "reports/feedback_simulation.json",
    "release_manifest": "reports/release_manifest.json",
    "historical_results": "reports/results.json",
}


def _read(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _validation(meta, report, name, root):
    if report is None:
        return {"status": "MISSING"}
    if name == "representation_comparison":
        return {"status": "MODEL_INDEPENDENT",
                "reason": "compares several representations; not a release artifact"}
    if name == "historical_results":
        return {"status": "HISTORICAL",
                "reason": "notebook output for an earlier artifact; never current"}
    try:
        result = validate_report_identity(meta, report, root=root)
        return {"status": "CURRENT", "experiment_id": result["experiment_id"]}
    except ExperimentIdentityError as exc:
        return {"status": "STALE_OR_UNBOUND", "reason": str(exc)}


def _metric(report, *keys, default=None):
    value = report
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def build_current_release(root: Path | None = None) -> dict:
    root = Path(root) if root is not None else project_root()
    meta = _read(root / "models/router_meta.json")
    if meta is None:
        raise FileNotFoundError("models/router_meta.json is required")
    reports = {name: _read(root / path) for name, path in REPORT_FILES.items()}
    identity = identity_from_meta(meta, root)
    validation = {name: _validation(meta, report, name, root)
                  for name, report in reports.items()}
    independent = reports["independent_benchmark"] or {}
    cascade = reports["business_cascade"] or {}
    representation = reports["representation_comparison"] or {}
    feedback = reports["feedback_simulation"] or {}
    return {
        "schema": "current-release-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_version": VERSION,
        "identity": identity.as_dict(),
        "experiment_id": identity.experiment_id,
        "benchmark_protocol": {"version": independent_cases.VERSION,
                               "sha256": independent_cases.sha256()},
        "validation": validation,
        "sources": {name: {"path": path,
                           "sha256": sha256_file(root / path) if (root / path).exists() else None}
                    for name, path in REPORT_FILES.items() if name != "release_manifest"},
        "metrics": {
            "test_bitext": {
                "classification": _metric(meta, "test", "classification"),
                "shadow_routing": _metric(meta, "test", "shadow_routing"),
                "operational_routing": _metric(meta, "test", "operational_routing"),
            },
            "independent": {
                "natural": _metric(independent, "deployed", "natural"),
                "no_keywords": _metric(independent, "deployed", "no_keywords"),
                "ood": _metric(independent, "ood"),
                "cascade": _metric(independent, "cascade"),
                "risk": _metric(independent, "risk"),
                "multi_request": _metric(independent, "multi_request"),
                "calibration": _metric(independent, "calibration"),
                "policy_evaluation": _metric(independent, "policy_evaluation"),
                "criteria": _metric(independent, "criteria"),
                "disjointness": _metric(independent, "disjointness"),
            },
            "business_cascade": {
                "bitext_test": _metric(cascade, "bitext_test", "detailed"),
                "hard": _metric(cascade, "hard"),
                "ood": _metric(cascade, "ood", "detailed"),
            },
            "representation": {
                "full_train": _metric(representation, "full_train"),
                "few_shot": _metric(representation, "few_shot"),
                "llm_zero_few_shot": _metric(representation, "llm_zero_few_shot"),
            },
            "feedback": _metric(feedback, "runs"),
        },
    }


def write_current_release(root: Path | None = None) -> Path:
    root = Path(root) if root is not None else project_root()
    payload = build_current_release(root)
    path = root / "reports/current_release.json"
    write_json(path, payload)
    return path


def _pct(value):
    return "—" if value is None else f"{value:.1%}"


def _num(value):
    return "—" if value is None else f"{value:,}".replace(",", " ")


def _table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    lines += ["| " + " | ".join(str(value) for value in row) + " |" for row in rows]
    return "\n".join(lines)


def build_results_markdown(current: dict, historical: dict | None = None) -> str:
    metrics = current["metrics"]
    test = metrics["test_bitext"]
    independent = metrics["independent"]
    natural, no_keywords = independent.get("natural") or {}, independent.get("no_keywords") or {}
    shadow = test.get("shadow_routing") or {}
    classification = test.get("classification") or {}
    criteria = independent.get("criteria") or {}
    checks = criteria.get("checks") or {}
    cascade = independent.get("cascade") or {}
    risk = independent.get("risk") or {}
    multi = independent.get("multi_request") or {}
    representation = metrics.get("representation") or {}
    feedback = metrics.get("feedback") or {}
    validation = current.get("validation") or {}

    lines = [
        "# Результаты проекта",
        "",
        "Этот файл собран из `reports/current_release.json`. Числа не вводятся вручную.",
        "",
        "## Текущий релиз",
        "",
        _table(["Параметр", "Значение"], [
            ["Модель", f"`{current['identity']['model_id']}` / `{current['identity']['model_name']}`"],
            ["Версия кода", current["identity"]["code_version"]],
            ["Политика", f"`{current['identity']['policy_version']}`"],
            ["experiment_id", f"`{current['experiment_id']}`"],
            ["Test Bitext, macro-F1", f"{classification.get('macro_f1', 0):.4f}, ошибок {classification.get('errors', '—')}"],
            ["Test Bitext, теневой режим", f"{shadow.get('automatic_n', '—')} сам, ошибок очереди {shadow.get('automatic_queue_errors', '—')}, к оператору {shadow.get('manual_n', '—')}"],
            ["Независимый `natural`", f"{natural.get('top1_correct', '—')}/{natural.get('n', '—')} = {_pct(natural.get('top1_accuracy'))}, macro-F1 {natural.get('macro_f1', 0):.3f}"],
            ["Независимый `no_keywords`", f"{no_keywords.get('top1_correct', '—')}/{no_keywords.get('n', '—')} = {_pct(no_keywords.get('top1_accuracy'))}, macro-F1 {no_keywords.get('macro_f1', 0):.3f}"],
            ["OOD, ложных авторешений", f"{independent.get('ood', {}).get('shadow_false_auto_n', '—')}/{independent.get('ood', {}).get('n', '—')} = {_pct(independent.get('ood', {}).get('shadow_false_auto_rate'))}"],
            ["Критерии", f"{sum(bool(v) for v in checks.values())} из {len(checks)} пройдено; all_passed = {criteria.get('all_passed', '—')}"],
            ["Статус отчётов", ", ".join(f"{k}: {v.get('status')}" for k, v in validation.items())],
        ]),
        "",
        "Вывод: на Bitext качество почти идеальное, но на новых формулировках модель не готова к "
        "самостоятельной маршрутизации. Автоотправка выключена.",
        "",
        "## Сравнение признаков",
        "",
    ]
    full = representation.get("full_train") or {}
    few = representation.get("few_shot") or {}
    rows = []
    for name, value in full.items():
        balanced = (few.get("balanced") or {}).get(name, {}).get("aggregate", {})
        proportional = (few.get("proportional") or {}).get(name, {}).get("aggregate", {})
        rows.append([name,
                     _pct((value.get("natural") or {}).get("top1_accuracy")),
                     _pct((value.get("no_keywords") or {}).get("top1_accuracy")),
                     _pct((balanced.get("natural") or {}).get("mean_top1")),
                     _pct((balanced.get("no_keywords") or {}).get("mean_top1")),
                     _pct((proportional.get("natural") or {}).get("mean_top1")),
                     _pct((proportional.get("no_keywords") or {}).get("mean_top1"))])
    lines += [_table(["Признаки", "Полный train natural", "Полный train no_keywords",
                      "100 balanced natural", "100 balanced no_keywords",
                      "100 proportional natural", "100 proportional no_keywords"], rows), ""]
    lines += [f"LLM zero/few-shot: `{(representation.get('llm_zero_few_shot') or {}).get('status', 'NOT_RUN')}`.", ""]
    lines += ["## Каскад", "",
              _table(["Набор", "Подсказок", "Точность", "Исправил", "Испортил"], [
                  [mode, cascade.get(mode, {}).get("suggestions_n", "—"),
                   _pct(cascade.get(mode, {}).get("suggestion_precision")),
                   cascade.get(mode, {}).get("corrected", "—"),
                   cascade.get(mode, {}).get("regressed", "—")]
                  for mode in ("natural", "no_keywords")
              ]), ""]
    lines += ["## Риски и несколько просьб", "",
              f"- Риски: precision {risk.get('any_risk_flag', {}).get('precision')}, "
              f"recall {risk.get('any_risk_flag', {}).get('recall')}, "
              f"нижняя граница Wilson {risk.get('any_risk_flag', {}).get('recall_wilson_lower_95')}.",
              f"- Несколько просьб: точных совпадений {multi.get('exact_matches', '—')} из {multi.get('n', '—')}, "
              f"precision {multi.get('intent_precision')}, recall {multi.get('intent_recall')}.",
              ""]
    calibration = (independent.get("calibration") or {}).get("calibration_split") or {}
    rare = calibration.get("insufficient_support") or []
    lines += ["## Калибровка: классы с недостаточной поддержкой", ""]
    if rare:
        rows = []
        for category in rare:
            row = (calibration.get("per_category") or {}).get(category, {})
            rows.append([category, row.get("support", "—"), row.get("support_status", "—"),
                         f"{row.get('brier', 0):.3f}", f"{row.get('log_loss', 0):.3f}"])
        lines += [_table(["Категория", "Support", "Статус", "Brier", "Log loss"], rows), ""]
    else:
        lines += ["Нет классов с недостаточной поддержкой в текущем отчёте.", ""]
    lines += ["## Обратная связь оператора", "",
              _table(["Набор", "Идеальный оператор", "Ошибка оператора 20%"], [
                  [name, _pct((runs.get("operator_error_0.0") or {}).get("metrics", {}).get("measured_accuracy")),
                   _pct((runs.get("operator_error_0.2") or {}).get("metrics", {}).get("measured_accuracy"))]
                  for name, runs in feedback.items()
              ]), ""]
    lines += ["## Исторический прогон ноутбука 3.1.2", "",
              "Эти числа относятся к более раннему артефакту и политике. Их нельзя сравнивать с текущим "
              "релизом как один эксперимент. Полные данные остаются в `reports/results.json`.", ""]
    if historical:
        stage = historical.get("stage11") or {}
        loio_rows = historical.get("loio_summary") or []
        selected = (historical.get("selection") or {}).get("selected")
        loio = next((row for row in loio_rows if row.get("model") == selected),
                    loio_rows[0] if loio_rows else {})
        lines += [_table(["Проверка", "Исторический прогон"], [
            ["Модель", f"`{historical.get('model_id', '—')}`"],
            ["Test Bitext, macro-F1", f"{(historical.get('test') or {}).get('macro_f1', '—')}"],
            ["LOIO, 25 интентов", f"{loio.get('all25_mean_recall', '—')}"],
            ["Теневой режим на test", f"{(historical.get('test_shadow_routing') or {}).get('automatic_n', '—')} сам, ошибок {(historical.get('test_shadow_routing') or {}).get('automatic_queue_errors', '—')}"],
            ["OOD, ложных авторешений", f"{(stage.get('ood_new') or {}).get('shadow_false_auto', '—')}"],
        ]), ""]
    lines += ["## Что изменено после аудита", "",
              "Полная карта: [`docs/audit_change_map.md`](../docs/audit_change_map.md).", "",
              "- замороженный независимый набор и протокол;",
              "- cost-sensitive правило `R(a|x)` и валидация матрицы потерь;",
              "- независимые risk–coverage и OOD-кривые;",
              "- калибровка по категориям с флагом `INSUFFICIENT_SUPPORT`;",
              "- отдельные метрики каскада и контур обратной связи оператора;",
              "- проверка воспроизводимости разбиения и привязка отчётов к модели.", ""]
    return "\n".join(lines)


def write_results_markdown(root: Path | None = None) -> Path:
    root = Path(root) if root is not None else project_root()
    current = build_current_release(root)
    historical = _read(root / "reports/results.json")
    path = root / "reports/results.md"
    path.write_text(build_results_markdown(current, historical), encoding="utf-8")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=project_root())
    args = parser.parse_args(argv)
    current = write_current_release(args.root)
    markdown = write_results_markdown(args.root)
    print(json.dumps({"current_release": str(current), "results_md": str(markdown)},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
