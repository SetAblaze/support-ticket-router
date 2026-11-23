# Что лежит в reports

В папке есть отчёты двух разных этапов. Их нельзя смешивать в одну таблицу.

## Текущий релиз

Модель `router-c551bd5d209a4d1a`, политика `retail-cascade-v1`, версия кода `3.2.0`.

| Файл | Что содержит |
|---|---|
| `release_manifest.json` | experiment_id, model_id, policy_version, хеши данных, наборов и отчётов |
| `results.md` | Сводный отчёт: текущий релиз + отдельный исторический раздел 3.1.2 |
| `independent_benchmark.json` | Независимый набор, OOD, риск, многотемные, каскад, калибровка |
| `representation_comparison.json` | TF-IDF, MiniLM, 100 размеченных примеров, статус LLM |
| `business_cascade.json` | Метрики второй ступени |
| `feedback_simulation.json` | Симуляция обратной связи оператора |
| `results.json` | **Исторический** прогон ноутбука 3.1.2; в поле `release` есть ссылка на текущий релиз |

## Исторический прогон 3.1.2

Модель `router-937116a8646a48e3`, политика `selective-v3-shadow`.

Эти файлы сохранены для воспроизводимости и сравнения «до/после». Они не являются текущими
метриками релиза:

- `stage11_*`, `test_predictions.csv`, `manual_*`, `ood_diagnostic.csv`;
- `loio_*`, `policy_validation_grid.csv`, `similar_train_*.csv`;
- `review_benchmark.json`, `historical_demo.csv`, `app_batch_*`, `fresh_process_predictions.json`;
- `figures/`.

Если число из исторического файла расходится с `results.md`, это ожидаемо: после аудита изменилась
не только модель, но и политика принятия решений. Сравнивать такие строки как один эксперимент нельзя.

## Как пересобрать текущие отчёты

```bash
python scripts/run_independent_benchmark.py
python scripts/compare_representations.py
python scripts/evaluate_cascade.py
python scripts/simulate_feedback.py
python scripts/release_manifest.py
```

Ноутбук, этап 13 (ячейки 59–64), читает эти отчёты и проверяет SHA-256 замороженного набора.
