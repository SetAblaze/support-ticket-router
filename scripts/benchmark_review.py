"""Compare operator suggestions, shadow routing and risk rules on the frozen synthetic cases."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from support_router import (
    CATEGORY_TO_QUEUE,
    batch_route,
    detect_signals,
    fit_calibrated_bundle,
    load_bundle,
    project_root,
    scenario_data,
    select_policy,
    sha256_file,
    shadow_bundle,
    write_json,
)


def evaluate_model(bundle, hard, ood):
    texts = hard.text.tolist()
    probabilities = bundle.predict_proba(texts)
    top = np.asarray(bundle.classes_)[np.argsort(-probabilities, axis=1)[:, :3]]
    shadow = shadow_bundle(bundle)
    routes = batch_route(shadow, texts)
    slices = {}
    for name, part in hard.groupby("difficulty"):
        labels = part.category.to_numpy()
        predictions = top[part.index]
        accepted = [i for i in part.index if routes[i]["status"] == "AUTO_RECOMMENDED"]
        slices[name] = {
            "n": len(part),
            "top1_correct": int((predictions[:, 0] == labels).sum()),
            "top3_correct": int((predictions == labels[:, None]).any(axis=1).sum()),
            "shadow_auto": len(accepted),
            "shadow_wrong_queue": sum(routes[i]["queue"] != CATEGORY_TO_QUEUE[hard.at[i, "category"]]
                                      for i in accepted),
        }
    return {
        "confidence_threshold": bundle.policy["confidence_threshold"],
        "similarity_threshold": bundle.policy["similarity_threshold"],
        "by_difficulty": slices,
        "ood_n": len(ood),
        "ood_false_auto": sum(r["status"] == "AUTO_RECOMMENDED" for r in batch_route(shadow, ood.text)),
    }


def evaluate_risks(cases):
    flags = [set(detect_signals(t)["risk_flags"]) for t in cases.text]
    expected = [set(value.split()) for value in cases.expected_flags]
    report = {}
    for flag in sorted(set.union(*expected, *flags)):
        tp = sum(flag in e and flag in p for e, p in zip(expected, flags))
        fp = sum(flag not in e and flag in p for e, p in zip(expected, flags))
        fn = sum(flag in e and flag not in p for e, p in zip(expected, flags))
        report[flag] = {"tp": tp, "fp": fp, "fn": fn,
                        "precision": tp / (tp + fp) if tp + fp else None,
                        "recall": tp / (tp + fn) if tp + fn else None}
    return report


def main():
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", type=Path, default=root / "data/processed")
    parser.add_argument("--out", type=Path, default=root / "reports/review_benchmark.json")
    args = parser.parse_args()
    splits = {name: pd.read_csv(args.processed / f"{name}.csv", keep_default_na=False)
              for name in ("train", "calibration", "val")}
    deployed, meta = load_bundle(root / "models", trust_local_artifact=True)
    for name in splits:
        if sha256_file(args.processed / f"{name}.csv") != meta["provenance"]["split_manifest"]["splits"][name]["sha256"]:
            raise ValueError(f"Split differs from the deployed model: {name}")
    candidate = fit_calibrated_bundle("semantic_lr_C1", splits["train"], splits["calibration"])
    select_policy(candidate, splits["val"], scenario_data.frames()[0])
    datasets = root / "data/eval"
    hard = pd.read_csv(datasets / "hard_test.csv")
    ood = pd.read_csv(datasets / "ood_eval.csv")
    risks = pd.read_csv(datasets / "risk_regression.csv", keep_default_na=False)
    write_json(args.out, {
        "scope": "Author-written diagnostic cases; risk examples include regression fixtures.",
        "models": {b.model_name: evaluate_model(b, hard, ood) for b in (deployed, candidate)},
        "risk_n": len(risks), "risk_by_flag": evaluate_risks(risks),
    })


if __name__ == "__main__":
    main()
