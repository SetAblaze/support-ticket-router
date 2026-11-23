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
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, StratifiedKFold, cross_val_score

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


def _allocate(total, weights, capacity, minimum):
    """Deterministic largest-deficit allocation with a minimum per category."""
    allocation = {category: min(minimum, capacity[category]) for category in weights}
    if sum(allocation.values()) > total:
        raise ValueError("Minimum per class exceeds the sample budget")
    total_weight = sum(weights.values())
    while sum(allocation.values()) < total:
        candidates = [category for category in sorted(weights)
                      if allocation[category] < capacity[category]]
        if not candidates:
            raise ValueError("Not enough rows for the requested sample budget")
        category = max(candidates, key=lambda c: (
            weights[c] / total_weight * total - allocation[c], -allocation[c], c))
        allocation[category] += 1
    return allocation


def _sample_100(train, seed, mode="balanced"):
    """Return exactly 100 rows and the allocation protocol.

    ``balanced`` gives every category a nearly equal quota.  ``proportional``
    follows the train distribution, with an explicit minimum of two rows per
    category.  If that minimum distorts strict proportionality, the returned
    metadata says so.
    """
    if mode not in ("balanced", "proportional"):
        raise ValueError(f"Unknown few-shot mode: {mode}")
    rng = np.random.default_rng(seed)
    groups = {category: group for category, group in train.groupby("category")}
    counts = {category: len(group) for category, group in groups.items()}
    weights = ({category: 1.0 for category in counts} if mode == "balanced"
               else {category: float(counts[category]) for category in counts})
    allocation = _allocate(100, weights, counts, minimum=2)
    indices = []
    for category, group in groups.items():
        indices.extend(rng.choice(group.index.to_numpy(), size=allocation[category],
                                  replace=False))
    sample = train.loc[sorted(indices)].reset_index(drop=True)
    if len(sample) != 100 or sample.text.duplicated().any():
        raise AssertionError("Few-shot sample must contain exactly 100 unique rows")
    ideal = {category: 100 * weights[category] / sum(weights.values()) for category in weights}
    deviation = {category: allocation[category] - ideal[category] for category in weights}
    return sample, {
        "mode": mode, "n": len(sample), "allocation": allocation,
        "class_distribution": sample.category.value_counts().to_dict(),
        "n_groups": int(sample.group.nunique()), "min_per_class": 2,
        "max_abs_deviation_from_ideal": max(abs(value) for value in deviation.values()),
        "proportionality_note": ("minimum two rows per category can distort strict "
                                 "proportionality" if mode == "proportional" else
                                 "balanced quota, not the train prevalence"),
    }


def _cv_for(texts, labels, groups, seed):
    min_count = pd.Series(labels).value_counts().min()
    folds = int(max(2, min(5, min_count, len(set(groups)))))
    last_error = None
    while folds >= 2:
        try:
            cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
            list(cv.split(texts, labels, groups))
            return cv, folds, "StratifiedGroupKFold", None
        except ValueError as exc:
            last_error = exc
        try:
            cv = GroupKFold(n_splits=folds)
            list(cv.split(texts, labels, groups))
            return cv, folds, "GroupKFold", f"StratifiedGroupKFold unavailable: {last_error}"
        except ValueError as exc:
            last_error = exc
            folds -= 1
    # Last-resort stratified fallback is explicitly marked as leakage-prone.
    if min_count >= 2:
        cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
        return cv, 2, "StratifiedKFold", f"group-aware CV unavailable: {last_error}"
    raise ValueError("Cannot build any cross-validation split")


def _fit_with_cv(name, texts, labels, groups, seed):
    cv, folds, method, fallback = _cv_for(texts, labels, groups, seed)
    best = None
    for c in C_GRID:
        model = sc.make_model(name)
        model.set_params(clf__C=c)
        score = cross_val_score(model, texts, labels, cv=cv, groups=groups,
                                scoring="accuracy").mean()
        if best is None or score > best[0]:
            best = score, c
    _, chosen_c = best
    model = sc.make_model(name)
    model.set_params(clf__C=chosen_c)
    model.fit(texts, labels)
    return model, chosen_c, float(best[0]), {
        "method": method, "folds": folds, "fallback_reason": fallback,
        "leakage_risk": method == "StratifiedKFold",
    }


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
    cv_values = np.asarray([row["cv_accuracy"] for row in rows])
    out["cv_accuracy"] = {"mean": float(cv_values.mean()), "std": float(cv_values.std(ddof=0)),
                          "seeds": len(cv_values)}
    out["chosen_C"] = sorted({row["chosen_C"] for row in rows})
    out["cv_methods"] = sorted({row["cv"]["method"] for row in rows})
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
    for mode in ("balanced", "proportional"):
        few_shot[mode] = {}
        for name in ("lr_word (baseline)", "linsvc_wc_C1", "semantic_lr_C1"):
            rows = []
            for seed in SEEDS:
                sample, sample_meta = _sample_100(train, seed, mode)
                model, chosen_c, cv_accuracy, cv_meta = _fit_with_cv(
                    name, sample.text.tolist(), sample.category.tolist(),
                    sample.group.tolist(), seed)
                rows.append({"seed": seed, "chosen_C": chosen_c, "cv_accuracy": cv_accuracy,
                             "sample": sample_meta, "cv": cv_meta,
                             **_evaluate_model(model, natural, no_keywords)})
            few_shot[mode][name] = {"runs": rows, "aggregate": _aggregate(rows)}

    report = {
        "benchmark": {"version": independent_cases.VERSION, "sha256": independent_cases.sha256(),
                      "report": "reports/independent_benchmark.json"},
        "provenance": {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
                       "train_sha256": sha256_file(train_path),
                       "train_rows": len(train)},
        "regimes": {
            "full_train": "all training rows",
            "few_shot_balanced": "exactly 100 rows, nearly equal category quota, C chosen inside them",
            "few_shot_proportional": ("exactly 100 rows, train-proportional quota with a documented "
                                      "minimum of two rows per category, C chosen inside them"),
        },
        "full_train": full,
        "few_shot": few_shot,
        "few_shot_100": few_shot["balanced"],
        "few_shot_100_note": ("alias for backward compatibility; the balanced and proportional "
                              "regimes are separate experiments"),
        "cv_protocol": {
            "unit": "lexical group",
            "primary": "StratifiedGroupKFold when the sample supports it",
            "fallback": "GroupKFold, then a leakage-marked StratifiedKFold last resort",
            "selection_inside_sample_only": True,
        },
        "llm_zero_few_shot": _llm_block(args.llm_predictions, natural, no_keywords),
        "scope": ("Author-written frozen benchmark. Natural and no-keyword modes are not pooled. "
                  "The benchmark has already been inspected, so it is not a blind final test."),
    }
    write_json(args.out, report)
    print(json.dumps({"report": str(args.out),
                      "full_train": {k: {s: round(v[s]["top1_accuracy"], 3) for s in ("natural", "no_keywords")}
                                     for k, v in full.items()},
                      "few_shot": {mode: {k: {s: round(v["aggregate"][s]["mean_top1"], 3)
                                              for s in ("natural", "no_keywords")}
                                          for k, v in models.items()}
                                  for mode, models in few_shot.items()},
                      "llm": report["llm_zero_few_shot"]["status"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
