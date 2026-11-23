# Результаты проекта

Этот файл собран из `reports/current_release.json`. Числа не вводятся вручную.

## Текущий релиз

| Параметр | Значение |
|---|---|
| Модель | `router-17e31a0d6ea04c7b` / `linsvc_wc_C1` |
| Версия кода | 3.2.0 |
| Политика | `retail-cascade-v1` |
| experiment_id | `dfa1f909a140aaf0e61c70c21fa59d3438441633622851fa35d3d43cd3a658c5` |
| Test Bitext, macro-F1 | 0.9963, ошибок 10 |
| Test Bitext, теневой режим | 3059 сам, ошибок очереди 0, к оператору 63 |
| Независимый `natural` | 28/55 = 50.9%, macro-F1 0.439 |
| Независимый `no_keywords` | 12/55 = 21.8%, macro-F1 0.173 |
| OOD, ложных авторешений | 1/20 = 5.0% |
| Критерии | 3 из 4 пройдено; all_passed = False |
| Статус отчётов | independent_benchmark: CURRENT, business_cascade: CURRENT, representation_comparison: MODEL_INDEPENDENT, feedback_simulation: CURRENT, release_manifest: CURRENT, historical_results: HISTORICAL |

Вывод: на Bitext качество почти идеальное, но на новых формулировках модель не готова к самостоятельной маршрутизации. Автоотправка выключена.

## Сравнение признаков

| Признаки | Полный train natural | Полный train no_keywords | 100 balanced natural | 100 balanced no_keywords | 100 proportional natural | 100 proportional no_keywords |
|---|---|---|---|---|---|---|
| lr_word (baseline) | 49.1% | 27.3% | 36.7% | 19.6% | 37.5% | 16.0% |
| linsvc_wc_C1 | 54.5% | 27.3% | 43.3% | 20.7% | 45.5% | 17.1% |
| semantic_lr_C1 | 76.4% | 58.2% | 81.1% | 63.3% | 70.2% | 52.4% |
| semantic_svc_C1 | 72.7% | 60.0% | — | — | — | — |

LLM zero/few-shot: `NOT_RUN`.

## Каскад

| Набор | Подсказок | Точность | Исправил | Испортил |
|---|---|---|---|---|
| natural | 26 | 84.6% | 7 | 1 |
| no_keywords | 29 | 65.5% | 14 | 1 |

## Риски и несколько просьб

- Риски: precision 1.0, recall 1.0, нижняя граница Wilson 0.6755843804891231.
- Несколько просьб: точных совпадений 9 из 12, precision 1.0, recall 0.75.

## Калибровка: классы с недостаточной поддержкой

| Категория | Support | Статус | Brier | Log loss |
|---|---|---|---|---|
| CANCELLATION_FEE | 24 | INSUFFICIENT_SUPPORT | 0.000 | 0.001 |
| NEWSLETTER | 6 | INSUFFICIENT_SUPPORT | 0.000 | 0.001 |
| SHIPPING | 10 | INSUFFICIENT_SUPPORT | 0.000 | 0.001 |

## Обратная связь оператора

| Набор | Идеальный оператор | Ошибка оператора 20% |
|---|---|---|
| natural | 61.8% | 60.0% |
| no_keywords | 45.5% | 45.5% |

## Исторический прогон ноутбука 3.1.2

Эти числа относятся к более раннему артефакту и политике. Их нельзя сравнивать с текущим релизом как один эксперимент. Полные данные остаются в `reports/results.json`.

| Проверка | Исторический прогон |
|---|---|
| Модель | `router-937116a8646a48e3` |
| Test Bitext, macro-F1 | 0.996344904462853 |
| LOIO, 25 интентов | 0.8182036379085134 |
| Теневой режим на test | 3068 сам, ошибок 6 |
| OOD, ложных авторешений | 5 |

## Что изменено после аудита

Полная карта: [`docs/audit_change_map.md`](../docs/audit_change_map.md).

- замороженный независимый набор и протокол;
- cost-sensitive правило `R(a|x)` и валидация матрицы потерь;
- независимые risk–coverage и OOD-кривые;
- калибровка по категориям с флагом `INSUFFICIENT_SUPPORT`;
- отдельные метрики каскада и контур обратной связи оператора;
- проверка воспроизводимости разбиения и привязка отчётов к модели.
