"""Compare text representations on the frozen independent benchmark.

Two regimes are reported separately:

* ``full_train`` - all 11 970 training rows;
* ``few_shot_100`` - exactly 100 labelled rows, with regularisation chosen by
  cross-validation *inside those 100 rows only*.

The comparison answers the audit question: do pretrained embeddings generalise
better than TF-IDF on unfamiliar wording, and does the advantage survive a
small labelled budget? A zero/few-shot LLM can be scored through
``--llm-predictions``; when no model is configured the report says ``NOT_RUN``
instead of inventing numbers.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score

import support_router as sc
from support_router import independent_cases
from support_router.evaluation import classification_metrics
from support_router.utils import sha256_file, write_json

C_GRID = (0.1, 1.0, 10.0)
SEEDS = (0, 1, 2, 3, 4)
FROZEN_SHA256 = "97d8404692fc8c8210ea86c8d2fbc7cb68cffbdec60c15e14a3cf070767eac9c"


def _top1(predicted, labels):
    predicted, labels = np.asarray(predicted, dtype=object), np.asarray(labels, dtype=object)
    metrics = classification_metrics(labels.tolist(), predicted.tolist())
    return {"n": len(labels), "top1_accuracy": float((predicted == labels).mean()),
            "top1_correct": int((predicted == labels).sum()),
            "macro_f1": metrics["macro_f1"], "errors": metrics["errors"]}


def _sample_100(train, seed):
    """At least two rows per category, then proportional allocation to exactly 100."""
    rng = np.random.default_rng(seed)
    groups = {category: group for category, group in train.groupby("category")}
    counts = {category: 2 for category in groups}
    remaining = 100 - sum(counts.values())
    weights = {category: len(group) / len(train) for category, group in groups.items()}
    for _ in range(remaining):
        category = max(groups, key=lambda c: (weights[c] * (counts[c] + 1) - counts[c], -counts[c]))
        counts[category] += 1
    indices = []
    for category, group in groups.items():
        indices.extend(rng.choice(group.index.to_numpy(), size=min(counts[category], len(group)),
                                  replace=False))
    return train.loc[sorted(indices)].reset_index(drop=True)


def _fit_with_cv(name, texts, labels, seed):
    min_count = pd.Series(labels).value_counts().min()
    folds = int(max(2, min(5, min_count)))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    best = None
    for c in C_GRID:
        model = sc.make_model(name)
        model.set_params(clf__C=c)
        score = cross_val_score(model, texts, labels, cv=cv, scoring="accuracy").mean()
        if best is None or score > best[0]:
            best = score, c
    _, chosen_c = best
    model = sc.make_model(name)
    model.set_params(clf__C=chosen_c)
    model.fit(texts, labels)
    return model, chosen_c, float(best[0])


def _evaluate_model(model, natural, no_keywords):
    return {"natural": _top1(model.predict(natural.text.tolist()), natural.category.tolist()),
            "no_keywords": _top1(model.predict(no_keywords.text.tolist()), no_keywords.category.tolist())}


def _aggregate(rows):
    out = {}
    for slice_name in ("natural", "no_keywords"):
        values = np.asarray([row[slice_name]["top1_accuracy"] for row in rows])
        out[slice_name] = {"mean_top1": float(values.mean()), "std_top1": float(values.std(ddof=0)),
                           "min_top1": float(values.min()), "max_top1": float(values.max()),
                           "seeds": len(values)}
    return out


def _llm_block(path, natural, no_keywords):
    if not path:
        return {"status": "NOT_RUN",
                "reason": "No local or external LLM configured in this environment",
                "protocol": "docs/independent_benchmark_protocol.md"}
    frame = pd.read_csv(path, keep_default_na=False)
    predictions = dict(zip(frame.text, frame.predicted_category, strict=True))
    missing = [t for t in [*natural.text, *no_keywords.text] if t not in predictions]
    if missing:
        raise ValueError(f"LLM predictions missing {len(missing)} benchmark texts")
    return {"status": "SCORED", "source": str(path),
            "natural": _top1([predictions[t] for t in natural.text], natural.category.tolist()),
            "no_keywords": _top1([predictions[t] for t in no_keywords.text], no_keywords.category.tolist())}


def main():
    root = Path(sc.project_root())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", type=Path, default=root / "data/processed")
    parser.add_argument("--out", type=Path, default=root / "reports/representation_comparison.json")
    parser.add_argument("--llm-predictions", type=Path, default=None,
                        help="CSV with columns text,predicted_category for a fixed LLM prompt run")
    args = parser.parse_args()

    if independent_cases.sha256() != FROZEN_SHA256:
        raise ValueError("Independent benchmark differs from the frozen protocol hash")
    train_path = args.processed / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"{train_path} missing: run python -m support_router.train first")
    train = pd.read_csv(train_path, keep_default_na=False)
    id_frame, _, _, _ = independent_cases.frames()
    natural = id_frame[id_frame.difficulty == "natural"].reset_index(drop=True)
    no_keywords = id_frame[id_frame.difficulty == "no_keywords"].reset_index(drop=True)

    full = {}
    for name in ("lr_word (baseline)", "linsvc_wc_C1", "semantic_lr_C1", "semantic_svc_C1"):
        model = sc.fit_model(name, train)
        full[name] = {"chosen_C": "fixed by configuration",
                      **_evaluate_model(model, natural, no_keywords)}

    few_shot = {}
    for name in ("lr_word (baseline)", "linsvc_wc_C1", "semantic_lr_C1"):
        rows = []
        for seed in SEEDS:
            sample = _sample_100(train, seed)
            model, chosen_c, cv_accuracy = _fit_with_cv(name, sample.text.tolist(),
                                                        sample.category.tolist(), seed)
            rows.append({"seed": seed, "chosen_C": chosen_c, "cv_accuracy_on_100": cv_accuracy,
                         **_evaluate_model(model, natural, no_keywords)})
        few_shot[name] = {"runs": rows, "aggregate": _aggregate(rows)}

    report = {
        "benchmark": {"version": independent_cases.VERSION, "sha256": independent_cases.sha256(),
                      "report": "reports/independent_benchmark.json"},
        "provenance": {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
                       "train_sha256": sha256_file(train_path),
                       "train_rows": len(train)},
        "regimes": {"full_train": "all training rows", "few_shot_100": "100 rows, C chosen inside them"},
        "full_train": full,
        "few_shot_100": few_shot,
        "llm_zero_few_shot": _llm_block(args.llm_predictions, natural, no_keywords),
        "scope": "Author-written frozen benchmark. Natural and no-keyword modes are not pooled.",
    }
    write_json(args.out, report)
    print(json.dumps({"report": str(args.out),
                      "full_train": {k: {s: round(v[s]["top1_accuracy"], 3) for s in ("natural", "no_keywords")}
                                     for k, v in full.items()},
                      "few_shot_100": {k: {s: round(v["aggregate"][s]["mean_top1"], 3)
                                           for s in ("natural", "no_keywords")}
                                       for k, v in few_shot.items()},
                      "llm": report["llm_zero_few_shot"]["status"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
