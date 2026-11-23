# Карта изменений после аудита

Дата: 23 ноября 2026 года. Ветка: `codex/router-review-fixes`.

Документ связывает каждое замечание аудита с кодом, данными, отчётами, тестами
и ячейками ноутбука. Актуальные коммиты, хеши и `experiment_id` находятся в
[`reports/release_manifest.json`](../reports/release_manifest.json); они не
дублируются здесь вручную.

## 1. Текущий релиз

| Параметр | Значение |
|---|---|
| Модель | `router-17e31a0d6ea04c7b` / `linsvc_wc_C1` |
| Политика | `retail-cascade-v1`, автоотправка выключена |
| Test Bitext | macro-F1 0.9963, 10 ошибок |
| Теневой режим на test | 3 059 из 3 122, ошибок очереди 0 |
| Независимый `natural` | 28/55 = 50.9%, macro-F1 0.439 |
| Независимый `no_keywords` | 12/55 = 21.8%, macro-F1 0.173 |
| OOD | 1 ложное теневое авторешение из 20 |
| Риски, замороженный набор | precision 1.0, recall 1.0 на 8 положительных; Wilson lower 0.676 |
| Многотемные | 9/12 exact, precision 1.0, recall 0.75 |
| Критерии | 3 из 4; `no_keywords` ниже 50% |
| Автотесты | 279 passed, 1 skipped |
| Автоотправка | выключена, только теневой режим |

Исторический прогон 3.1.2 (`router-937116a8646a48e3`) остаётся в
`reports/results.json` и в отдельном разделе `reports/results.md`.

## 2. Полная карта замечаний

| № | Приоритет | Замечание аудита | Что сделано | Где | Доказательство | Тест |
|---|---|---|---|---|---|---|
| 1 | P0 | LOIO использовался и для выбора модели, и как независимая оценка | LOIO явно назван выбором архитектуры; независимая оценка вынесена в замороженный набор | `src/support_router/evaluation.py`, `docs/independent_benchmark_protocol.md` | `selection.loio_role`; `protocol.role` | `tests/test_audit_fixes.py` |
| 2 | P0 | Нет независимого бенчмарка на новые формулировки | Создан `independent-v1-2026-11-23`: 110 ID-фраз в двух режимах, 20 OOD, 16 риск, 12 многотемных | `src/support_router/independent_cases.py`, `data/eval/independent_*.csv` | SHA-256 `97d84046…eac9c` | `test_independent_benchmark_is_disjoint_and_labelled` |
| 3 | P1 | Тест Bitext слишком простой | Bitext, `natural`, `no_keywords` публикуются раздельно; Bitext не выдаётся за robustness | `reports/results.md`, `reports/independent_benchmark.json` | 50.9% / 21.8% | `test_release_reports_are_internally_consistent` |
| 4 | P1 | Порог уверенности не учитывает полную стоимость ошибки | Матрица `L(y,a)` и правило `R(a|x)=Σ P(y|x)L(y,a)`; при равенстве выбирается `REVIEW` | `src/support_router/business.py`, `src/support_router/routing.py` | `expected_loss`, `EXPECTED_LOSS_*` | `test_expected_loss_matches_audit_example`, `test_cost_sensitive_tie_prefers_review_and_rejects_invalid_inputs` |
| 5 | P1 | Пороги выбраны и проверены на одних данных | Добавлена end-to-end risk–coverage кривая через тот же `batch_route`, что API; Wilson после выбора не называется гарантией | `src/support_router/evaluation.py`, `scripts/run_independent_benchmark.py` | `policy_evaluation` | `test_end_to_end_policy_curve_uses_the_real_decision_function` |
| 6 | P1 | Надёжность второй ступени MiniLM не доказана | Каскад оценивается отдельно: исправленные/внесённые ошибки, точность подсказок, ложные подсказки, задержка | `src/support_router/evaluation.py`, `scripts/evaluate_cascade.py` | `natural` 7/1, `no_keywords` 14/1, OOD 1/20 | `test_calibration_reliability_and_cascade_detail` |
| 7 | P1 | Калибровка редких классов не проверена | Калибровка по категориям и источникам, Brier, log loss, ECE, reliability, `INSUFFICIENT_SUPPORT` | `src/support_router/evaluation.py` | `CANCELLATION_FEE` 24, `NEWSLETTER` 6, `SHIPPING` 10 | `test_per_category_calibration_flags_rare_support` |
| 8 | P2 | Риск-правила и многотемные обращения проверены на авторских примерах | Отдельный `risk_dev.csv`, замороженный `independent_risk.csv`, control-фразы для ложных разделений, схема интентов | `src/support_router/rules.py`, `src/support_router/multiple_requests.py`, `data/eval/risk_dev.csv` | dev 15/15, frozen 8/8; multi 9/12 | `test_risk_dev_and_frozen_sets_are_separate_and_clean`, `test_multi_request_schema_and_metrics` |
| 9 | P2 | Нет версионирования результатов и обратной связи оператора | Релизный манифест v2 с source/build/release commit, `--verify`, единый `current_release.json`; feedback API требует явное решение | `scripts/release_manifest.py`, `src/support_router/reporting.py`, `src/support_router/api.py`, `src/support_router/feedback.py` | `MANIFEST_OK`, `reports/current_release.json` | `test_current_release_report_is_generated_from_json_and_manifest_verifies` |
| 10 | P0 | Нужно отделить выбор модели от независимой оценки | Добавлены `identity.py`, `experiment_id`, проверка отчётов при загрузке; старые отчёты не выдают себя за текущие | `src/support_router/identity.py`, `reports/current_release.json` | `validation` по каждому отчёту | `test_identity_binds_current_reports_and_rejects_tampering` |
| 11 | P0 | Нужен новый контрольный набор | Замороженный набор не менялся; semantic similarity к train sample 2000: max 0.811, 0 фраз ≥0.85; статус post-selection | `scripts/run_independent_benchmark.py`, `docs/independent_benchmark_protocol.md` | `disjointness.semantic_similarity_to_train_sample` | `test_independent_benchmark_is_disjoint_and_labelled` |
| 12 | P1 | Cost-sensitive политика должна проверяться | Валидация вероятностей, классов, стоимостей; tie → `REVIEW`; матрица печатается в отчёте | `src/support_router/business.py` | `cost_matrix` | `test_cost_sensitive_tie_prefers_review_and_rejects_invalid_inputs` |
| 13 | P1 | Пороги на отдельном ID/OOD holdout | `end_to_end_policy_curve` на независимом наборе; OOD false auto; операционный режим отдельно | `src/support_router/evaluation.py` | `policy_evaluation.natural/no_keywords` | `test_end_to_end_policy_curve_uses_the_real_decision_function` |
| 14 | P1 | Риск-правила: независимая оценка | Development и frozen наборы разделены, тексты не пересекаются | `data/eval/risk_dev.csv`, `data/eval/independent_risk.csv` | dev 15/15, frozen 8/8 | `test_risk_dev_and_frozen_sets_are_separate_and_clean` |
| 15 | P2 | Редкие категории и калибровка | `calibration_by_source`, reliability-бины, Brier/log loss/ECE, статус поддержки | `src/support_router/evaluation.py` | `calibration.by_source` | `test_calibration_reliability_and_cascade_detail` |
| 16 | P2 | Связь результатов с версией | `reports/current_release.json` — единственный источник для `results.md`; исторический `results.json` не перезаписывается | `src/support_router/reporting.py` | `reports/results.md` | `test_current_release_report_is_generated_from_json_and_manifest_verifies` |
| 17 | P2 | Feedback-контур | Явное `confirmed`, конфликты, `retraining_candidates`, API с ключом и лимитами, метрики | `src/support_router/feedback.py`, `src/support_router/api.py` | `reports/feedback_simulation.json` | `test_feedback_requires_explicit_decision_and_detects_conflicts`, `tests/test_api.py` |
| 18 | P2 | CI и чистая проверка | `ruff`, pytest, `prepare_data --verify`, `release_manifest --verify`, отдельный full-ML job; `verify_clean_checkout.py` | `.github/workflows/ci.yml`, `scripts/verify_clean_checkout.py` | `CI` | `test_ci_workflow_and_notebook_stage13_are_present` |
| 19 | P2 | Ноутбук с устаревшими числами | Этап 13 читает отчёты, проверяет identity, имеет режимы `verified_reports` / `full_pipeline`, собирает отчёт через `reporting.py` | `notebooks/Этапы_1-12_Colab.ipynb` | ячейки 58a–65 | `test_ci_workflow_and_notebook_stage13_are_present` |
| 20 | P2 | API/Gradio/monitoring | `/v1/feedback` с auth, лимитами, 422; `/ready`, `/v1/model`; счётчик feedback в Prometheus | `src/support_router/api.py`, `src/support_router/monitoring.py` | `docs/api.md` | `tests/test_api.py` |

## 3. Изменённые файлы

### Код

| Файл | Что изменено |
|---|---|
| `src/support_router/business.py` | Полная матрица потерь, `expected_losses`, `cost_sensitive_action`, валидация входов, tie → `REVIEW` |
| `src/support_router/routing.py` | Интеграция ожидаемых потерь, reason codes `EXPECTED_LOSS_*` |
| `src/support_router/rules.py` | Расширенные правила рисков: first-person intent, reported threat, negations, quotes, fraud/legal/abuse context |
| `src/support_router/multiple_requests.py` | `INTENT_SCHEMA`, `validate_request_parts` |
| `src/support_router/evaluation.py` | `end_to_end_policy_curve`, `cascade_detailed_metrics`, `calibration_by_source`, `multi_request_metrics`, reliability |
| `src/support_router/identity.py` | `ExperimentIdentity`, `experiment_id`, `validate_report_identity` |
| `src/support_router/reporting.py` | Единый `current_release.json` и генерация `results.md` |
| `src/support_router/feedback.py` | Явное решение, конфликты, очередь переобучения, симуляция |
| `src/support_router/api.py` | `/v1/feedback`, auth, лимиты, 422, `/ready` |
| `src/support_router/data.py` | `group_diagnostics`, `leakage_diagnostics` |
| `src/support_router/train.py` | Диагностика групп в манифесте |
| `src/support_router/monitoring.py` | Счётчик feedback |
| `src/support_router/__init__.py` | Экспорт новых функций |

### Данные

| Файл | Что добавлено |
|---|---|
| `data/eval/independent_id.csv` | 110 фраз: 55 `natural`, 55 `no_keywords` |
| `data/eval/independent_ood.csv` | 20 OOD-обращений |
| `data/eval/independent_risk.csv` | 16 замороженных риск-примеров |
| `data/eval/independent_multi.csv` | 12 многотемных обращений |
| `data/eval/risk_dev.csv` | 27 отдельных development-примеров, не пересекаются с frozen |
| `data/processed/split_manifest.json` | Воспроизводимое разбиение и диагностика групп/утечек |

### Скрипты

| Файл | Назначение |
|---|---|
| `scripts/prepare_data.py` | `--verify` / `--rebuild`; сверяет SHA-256 и размеры |
| `scripts/run_independent_benchmark.py` | Один прогон, `policy_evaluation`, риск-dev/frozen, semantic disjointness, provenance |
| `scripts/compare_representations.py` | GroupKFold, balanced/proportional 100 примеров, LLM-статус |
| `scripts/evaluate_cascade.py` | Метрики каскада и задержка, provenance |
| `scripts/simulate_feedback.py` | Симуляция операторской обратной связи |
| `scripts/release_manifest.py` | `--build` / `--verify`, source/build/release commit, хеши |
| `scripts/verify_clean_checkout.py` | Клон в пустую папку и проверки |

### Отчёты

| Файл | Что содержит |
|---|---|
| `reports/current_release.json` | Единственный источник текущих чисел |
| `reports/results.md` | Сгенерирован из `current_release.json`; исторический раздел отдельно |
| `reports/independent_benchmark.json` | Независимая оценка, OOD, риск, многотемные, каскад, калибровка, политика |
| `reports/representation_comparison.json` | TF-IDF, MiniLM, 100 balanced/proportional, LLM-статус |
| `reports/business_cascade.json` | Метрики каскада и provenance |
| `reports/feedback_simulation.json` | Симуляция feedback |
| `reports/release_manifest.json` | Коммиты, хеши, `experiment_id`, правило сравнения |
| `reports/results.json` | Исторический прогон 3.1.2; не перезаписывается текущими метриками |

### Документация и CI

| Файл | Что изменено |
|---|---|
| `README.md` | Актуальные числа, ограничения, ссылка на карту |
| `docs/evaluation_protocol.md` | Роли наборов, end-to-end кривая, калибровка, каскад, cost matrix, multi-request, CI |
| `docs/independent_benchmark_protocol.md` | Статус post-selection, semantic similarity, результаты |
| `docs/business_policy.md` | Tie → `REVIEW`, валидация матрицы |
| `docs/api.md` | `/v1/feedback`, модель, лимиты |
| `docs/runbook.md` | Feedback, identity, действия при деградации |
| `reports/README.md` | Какие отчёты текущие, какие исторические |
| `.github/workflows/ci.yml` | Lint, fast tests, data/release verify, full-ML job |
| `docs/audit_change_map.md` | Этот документ |

### Тесты

| Файл | Что проверяет |
|---|---|
| `tests/test_audit_fixes.py` | Матрица потерь, risk–coverage, калибровка, каскад, независимый набор, feedback, согласованность отчётов |
| `tests/test_audit_remediation.py` | Identity, `prepare_data --verify`, feedback decision, метрики, few-shot CV, end-to-end curve, cost tie, risk dev/frozen, multi-request, calibration, release manifest, CI/notebook |
| `tests/test_api.py` | Feedback auth, явное решение, лимиты, ошибки |

### Модель

| Файл | Что изменено |
|---|---|
| `models/router.joblib` | Переобучен после изменения decision-кода |
| `models/router_meta.json` | Новый `model_id`, политика, метрики test, cost matrix, версии |

### Ноутбук

| Файл | Что изменено |
|---|---|
| `notebooks/Этапы_1-12_Colab.ipynb` | Этап 13: режим `RUN_MODE`, проверка identity, `policy_evaluation`, reliability, cascade detail, risk dev/frozen, multi per-intent, единый отчёт, динамические выводы |

## 4. Как проверить

```bash
python -m pytest -q
python scripts/prepare_data.py --verify
python scripts/run_independent_benchmark.py --no-update-results
python scripts/compare_representations.py
python scripts/evaluate_cascade.py
python scripts/simulate_feedback.py
python -m support_router.reporting
python scripts/release_manifest.py --build
python scripts/release_manifest.py --verify
python scripts/verify_clean_checkout.py
```

В ноутбуке этап 13 — ячейки 58a–65. Режим `verified_reports` только читает и
проверяет отчёты; `full_pipeline` пересчитывает benchmark. Ячейка 55 собирает
`current_release.json` и `results.md` через `reporting.py`.

## 5. Что не изменилось

- Архитектура классификатора осталась `linsvc_wc_C1`. Аудит требует не
  заменить модель, а честно измерить её обобщение.
- Автоотправка остаётся выключенной.
- LLM zero/few-shot не запускался: в окружении нет LLM. В отчёте `NOT_RUN`.
- Исторический `reports/results.json` не перезаписывается.

## 6. Исторический прогон и текущий релиз

| Источник | Модель | Политика | Test: авто / ошибок |
|---|---|---|---|
| `reports/results.json`, версия 3.1.2 | `router-937116a8646a48e3` | `selective-v3-shadow` | 3068 / 6 |
| `models/router_meta.json`, текущий релиз | `router-17e31a0d6ea04c7b` | `retail-cascade-v1` | 3059 / 0 |

Это не противоречие: после аудита изменилась политика принятия решений.
Сравнивать строки как один эксперимент нельзя.

## 7. Ограничения, которые остаются

- Независимый набор авторский, без реальных клиентских тикетов и независимого
  эксперта; после первого отчёта он не blind.
- Semantic similarity к train sample: max 0.811, 0 фраз ≥0.85; это выборка
  2 000 фраз train, а не доказательство независимости.
- Самая большая лексическая группа — 1 891 фраза (9.6%); её семантическая
  эквивалентность не проверена.
- Риск-правила: 8 положительных примеров, Wilson lower 0.676. Это не
  защитный контур.
- `no_keywords` top-1 21.8% ниже заранее зафиксированного порога 50%.
- Синтетическая validation не является бизнес-SLA.
- Для выбора новой основной модели нужен новый замороженный набор.
