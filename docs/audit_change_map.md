# Карта изменений после аудита

Дата: 23 ноября 2026 года. Ветка: `codex/router-review-fixes`.

Этот документ отвечает на вопрос «что и где изменилось» целиком, а не по частям. Он связывает каждое
замечание аудита с кодом, данными, отчётами, тестами и ячейками ноутбука.

## 1. Версии и идентификаторы

| Параметр | Значение |
|---|---|
| Базовая точка | `2fe07ed` |
| Коммиты с исправлениями аудита | `a70966b`, `7cd5d33`, `a6c7b9f`, `4aeb1ad` |
| Модель | `router-c551bd5d209a4d1a` / `linsvc_wc_C1` |
| Политика | `retail-cascade-v1` |
| experiment_id | `f64f5ee528b6be95ee049b48196862a5ead490240445eea1013d28ccec4ef9a1` |
| Build commit в манифесте | `a6c7b9f525205870ef9cfe33db73f78702fcb03b` |
| Версия кода | `3.2.0` |
| Автотесты | 265 passed, 1 skipped |
| Автоотправка | выключена, только теневой режим |

## 2. Полная карта замечаний

| № | Приоритет | Замечание аудита | Что сделано | Где | Доказательство | Тест |
|---|---|---|---|---|---|---|
| 1 | P0 | LOIO использовался и для выбора модели, и как независимая оценка | LOIO явно назван выбором архитектуры; независимая оценка вынесена в замороженный набор | `src/support_router/evaluation.py`, `reports/results.json`, `docs/independent_benchmark_protocol.md` | `selection.loio_role`, `independent_benchmark.json` | `tests/test_audit_fixes.py` |
| 2 | P0 | Нет независимого бенчмарка на новые формулировки | Создан `independent-v1-2026-11-23`: 110 ID-фраз в двух режимах, 20 OOD, 16 риск, 12 многотемных | `src/support_router/independent_cases.py`, `data/eval/independent_*.csv`, `docs/independent_benchmark_protocol.md` | SHA-256 `97d84046…eac9c`, проверка в ячейке 59 | `test_independent_benchmark_is_disjoint_and_labelled` |
| 3 | P1 | Тест Bitext слишком простой | Основное сравнение разделено на «известные шаблоны» и «новые формулировки»; Bitext не выдаётся за доказательство robustness | `reports/results.md`, `reports/independent_benchmark.json`, `README.md` | `natural` 50.9%, `no_keywords` 21.8% | `test_release_reports_are_internally_consistent` |
| 4 | P1 | Порог уверенности не учитывает полную стоимость ошибки | Добавлена матрица `L(y,a)` и правило `R(a|x)=Σ P(y|x)L(y,a)`; решение блокируется, если минимум потерь — ручная проверка | `src/support_router/business.py`, `src/support_router/routing.py` | 6 старых ошибок `PAYMENT → REFUNDS` заблокированы; 3059 авто, 0 ошибок | `test_expected_loss_matches_audit_example`, `test_route_blocks_when_expected_loss_prefers_review` |
| 5 | P1 | Пороги выбраны и проверены на одних данных | Пороги оцениваются на независимом наборе; Wilson после выбора не называется гарантией | `scripts/run_independent_benchmark.py`, `reports/independent_benchmark.json` | risk–coverage и OOD-кривые | `test_risk_coverage_curve_is_monotone_in_coverage` |
| 6 | P1 | Надёжность второй ступени MiniLM не доказана | Каскад оценивается отдельно: исправленные ошибки, внесённые ошибки, точность подсказок, ложные подсказки на OOD | `src/support_router/cascade.py`, `scripts/evaluate_cascade.py`, `reports/business_cascade.json` | `natural` 7/1, `no_keywords` 14/1, OOD 1/20 | `test_cascade_metrics_separate_fixes_from_regressions` |
| 7 | P1 | Калибровка редких классов не проверена | Добавлена калибровка по категориям с флагом `INSUFFICIENT_SUPPORT`, Brier, log loss, ECE | `src/support_router/evaluation.py`, `reports/independent_benchmark.json` | `CANCELLATION_FEE` 24, `NEWSLETTER` 6, `SHIPPING` 10 — недостаточно | `test_per_category_calibration_flags_rare_support` |
| 8 | P2 | Риск-правила и многотемные обращения проверены на авторских примерах | Добавлены отдельные независимые наборы и честные метрики полноты | `src/support_router/independent_cases.py`, `data/eval/independent_risk.csv`, `data/eval/independent_multi.csv` | recall рисков 50%, многотемные 9/12 | `test_independent_risk_and_multi_sets_run_without_crashing` |
| 9 | P2 | Нет версионирования результатов и обратной связи оператора | Добавлены релизный манифест и контур feedback: подтверждение/исправление → метрики → очередь на переобучение | `scripts/release_manifest.py`, `src/support_router/feedback.py`, `scripts/simulate_feedback.py` | `reports/release_manifest.json`, `reports/feedback_simulation.json` | `test_feedback_loop_metrics_and_retraining_queue` |

## 3. Все изменённые файлы

### Код

| Файл | Что изменено |
|---|---|
| `src/support_router/business.py` | Полная матрица потерь, `MISROUTE_COST`, `REVIEW_ACTION`, `expected_losses`, `cost_sensitive_action`, `confidence_floor` |
| `src/support_router/routing.py` | Интеграция ожидаемых потерь, reason codes `EXPECTED_LOSS_*` |
| `src/support_router/evaluation.py` | Risk–coverage, OOD acceptance, калибровка по категориям, метрики каскада; оговорка о LOIO |
| `src/support_router/independent_cases.py` | Замороженные фразы и SHA-256 независимого набора |
| `src/support_router/feedback.py` | События оператора, метрики, очередь на переобучение |
| `src/support_router/cascade.py` | Метаданные и пороги каскада |
| `src/support_router/monitoring.py` | Наблюдение за feedback |
| `src/support_router/train.py` | Проверка целостности разбиения |
| `src/support_router/api.py` | Версия и метаданные модели |
| `src/support_router/__init__.py` | Экспорт новых модулей |

### Данные

| Файл | Что добавлено |
|---|---|
| `data/eval/independent_id.csv` | 110 фраз: 55 `natural`, 55 `no_keywords` |
| `data/eval/independent_ood.csv` | 20 OOD-обращений |
| `data/eval/independent_risk.csv` | 16 риск-примеров |
| `data/eval/independent_multi.csv` | 12 многотемных обращений |
| `data/processed/split_manifest.json` | Диагностика сходства train/test |

### Скрипты

| Файл | Назначение |
|---|---|
| `scripts/run_independent_benchmark.py` | Один прогон независимого набора, запись `independent_benchmark.json` |
| `scripts/compare_representations.py` | Сравнение TF-IDF, MiniLM, 100 примеров, LLM-статус |
| `scripts/evaluate_cascade.py` | Отдельные метрики каскада |
| `scripts/simulate_feedback.py` | Симуляция обратной связи оператора |
| `scripts/release_manifest.py` | Релизный манифест с хешами и `git_dirty` |

### Отчёты

| Файл | Что содержит |
|---|---|
| `reports/independent_benchmark.json` | Независимая оценка, OOD, риск, многотемные, каскад, калибровка |
| `reports/representation_comparison.json` | Сравнение признаков на полном train и 100 примерах |
| `reports/business_cascade.json` | Метрики каскада |
| `reports/feedback_simulation.json` | Симуляция операторской обратной связи |
| `reports/release_manifest.json` | Версии, хеши, наборы оценки |
| `reports/results.json` | Исторический прогон 3.1.2 + ссылка на текущий релиз |
| `reports/results.md` | Текущий релиз + отдельный исторический раздел |

### Документация

| Файл | Что изменено |
|---|---|
| `README.md` | Актуальные результаты, ограничения, ссылка на эту карту |
| `docs/independent_benchmark_protocol.md` | Протокол до первого прогона |
| `docs/business_policy.md` | Матрица потерь и правило `R(a|x)` |
| `docs/architecture.md` | Новые модули и поток решения |
| `docs/api.md` | Версия модели и поля ответа |
| `docs/runbook.md` | Мониторинг, feedback, действия при деградации |
| `docs/audit_change_map.md` | Этот документ |

### Тесты

| Файл | Что проверяет |
|---|---|
| `tests/test_audit_fixes.py` | Матрица потерь, блокировка по ожидаемым потерям, risk–coverage, калибровка, каскад, независимый набор, feedback, согласованность отчётов и ноутбука |

### Модель

| Файл | Что изменено |
|---|---|
| `models/router.joblib` | Переобучен на текущем разбиении |
| `models/router_meta.json` | `model_id`, политика, метрики test, cost matrix, версии |

### Ноутбук

| Файл | Что изменено |
|---|---|
| `notebooks/Этапы_1-12_Colab.ipynb` | Версия 3.2.0; добавлен этап 13 (ячейки 59–64); ячейка 55 собирает отчёт из текущего релиза и отдельно помечает исторический 3.1.2; ячейка 58 обновлена |

## 4. Как проверить

```bash
python -m pytest -q
python scripts/run_independent_benchmark.py
python scripts/compare_representations.py
python scripts/evaluate_cascade.py
python scripts/simulate_feedback.py
python scripts/release_manifest.py
```

В ноутбуке этап 13 — ячейки 59–64. Они читают готовые отчёты, проверяют SHA-256 независимого набора и
показывают:

- независимый набор и критерии;
- сравнение признаков;
- матрицу потерь и пример аудита;
- калибровку по категориям;
- каскад, риски, многотемные обращения и feedback;
- релизный манифест.

## 5. Что не изменилось

- Архитектура классификатора осталась `linsvc_wc_C1`. Это осознанно: аудит требует не заменить модель,
  а честно измерить её обобщение.
- Автоотправка остаётся выключенной. Теневой режим показывает, что было бы, но не отправляет заявки.
- LLM zero/few-shot не запускался: в окружении нет локальной или внешней LLM. В отчёте стоит `NOT_RUN`.
- Ноутбук не входил в коммиты исправлений аудита. Его обновление — отдельный полный проход
  согласованности, а не скрытая правка модели.

## 6. Исторический прогон и текущий релиз

| Источник | Модель | Политика | Test: авто / ошибок |
|---|---|---|---|
| `reports/results.json`, версия 3.1.2 | `router-937116a8646a48e3` | `selective-v3-shadow` | 3068 / 6 |
| `models/router_meta.json`, текущий релиз | `router-c551bd5d209a4d1a` | `retail-cascade-v1` | 3059 / 0 |

Это не противоречие: после аудита изменилась политика принятия решений. Сравнивать эти строки как один
эксперимент нельзя, поэтому в `reports/results.md` они находятся в разных разделах.

## 7. Ограничения, которые остаются

- Независимый набор авторский, без реальных клиентских тикетов и независимого эксперта.
- 448 из 3 122 тестовых фраз имеют соседа в train со сходством ≥ 0.85; это описательная диагностика.
- Самая большая лексическая группа — 1 891 фраза (9.6%); её семантическая эквивалентность не проверена.
- Recall риск-правил 50% на авторских положительных примерах; это не защитный контур.
- Синтетическая validation не является бизнес-SLA.
