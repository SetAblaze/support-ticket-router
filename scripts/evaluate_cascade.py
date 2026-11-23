"""Evaluate the deployed hybrid on fixed diagnostics, without fitting any parameters."""
import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

from support_router import batch_route, load_bundle, models_dir, project_root, sha256_file, shadow_bundle, write_json
from support_router.evaluation import cascade_metrics, cascade_ood_metrics


def measure(bundle, texts, labels=None):
    routes = batch_route(bundle, texts)
    fallback = [r for r in routes if r["classification_source"] == "minilm"]
    report = {"n": len(routes), "semantic_status": dict(Counter(r["semantic_status"] for r in routes)),
              "shadow_auto": sum(r["status"] == "AUTO_RECOMMENDED" for r in routes),
              "shadow_auto_from_fallback": sum(r["status"] == "AUTO_RECOMMENDED" for r in fallback)}
    if labels is not None:
        report.update(cascade_metrics(routes, labels))
    else:
        report.update(cascade_ood_metrics(routes))
    return report


def main():
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", type=Path, default=root / "data/processed")
    args = parser.parse_args()
    model, meta = load_bundle(models_dir(), trust_local_artifact=True)
    bundle = shadow_bundle(model)
    test_path = args.processed / "test.csv"
    if sha256_file(test_path) != meta["provenance"]["split_manifest"]["splits"]["test"]["sha256"]:
        raise ValueError("Different test split from the deployed model")
    test = pd.read_csv(test_path, keep_default_na=False)
    hard = pd.read_csv(root / "data/eval/hard_test.csv")
    ood = pd.read_csv(root / "data/eval/ood_eval.csv")
    write_json(root / "reports/business_cascade.json", {
        "model_id": model.model_id, "policy": model.policy["version"], "gate": model.policy["semantic_gate"],
        "bitext_test": measure(bundle, test.text, test.category),
        "hard": {k: measure(bundle, v.text, v.category) for k, v in hard.groupby("difficulty")},
        "ood": measure(bundle, ood.text), "ood_by_topic": {k: measure(bundle, v.text) for k, v in ood.groupby("topic")},
        "scope": "Frozen synthetic diagnostics. Some cases were previously reviewed by the authors."
    })


if __name__ == "__main__":
    main()
