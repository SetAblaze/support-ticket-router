"""Evaluate the deployed hybrid on fixed diagnostics, without fitting any parameters."""
import argparse
import time
from collections import Counter
from pathlib import Path

import pandas as pd

import support_router as sc
from support_router import (
    batch_route,
    independent_cases,
    load_bundle,
    models_dir,
    project_root,
    sha256_file,
    shadow_bundle,
    write_json,
)
from support_router.evaluation import cascade_detailed_metrics, cascade_metrics, cascade_ood_metrics


def measure(bundle, texts, labels=None):
    started = time.perf_counter()
    routes = batch_route(bundle, texts)
    elapsed_ms = (time.perf_counter() - started) * 1000
    fallback = [r for r in routes if r["classification_source"] == "minilm"]
    report = {"n": len(routes), "semantic_status": dict(Counter(r["semantic_status"] for r in routes)),
              "shadow_auto": sum(r["status"] == "AUTO_RECOMMENDED" for r in routes),
              "shadow_auto_from_fallback": sum(r["status"] == "AUTO_RECOMMENDED" for r in fallback),
              "latency": {"total_ms": elapsed_ms,
                          "mean_ms": elapsed_ms / len(routes) if routes else None,
                          "note": "batch latency; MiniLM is local ONNX"}}
    per_request_ms = elapsed_ms / len(routes) if routes else None
    latencies = [per_request_ms] * len(routes) if per_request_ms is not None else None
    if labels is not None:
        report.update(cascade_metrics(routes, labels))
        report["detailed"] = cascade_detailed_metrics(routes, labels, latency_ms=latencies)
    else:
        report.update(cascade_ood_metrics(routes))
        report["detailed"] = cascade_detailed_metrics(routes, latency_ms=latencies)
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
        "provenance": {
            "model_id": model.model_id,
            "model_name": model.model_name,
            "artifact_sha256": meta["artifact_sha256"],
            "decision_code_sha256": meta["module_sha256"],
            "policy_version": model.policy["version"],
            "policy_auto_enabled": model.policy.get("auto_enabled"),
            "code_version": sc.VERSION,
            "experiment_id": sc.identity_from_meta(meta, root).experiment_id,
            "raw_data_sha256": sha256_file(root / "data/raw/20000-Utterances-Training-dataset-for-chatbots-virtual-assistant-Bitext-sample.csv"),
            "split_manifest_sha256": sha256_file(args.processed / "split_manifest.json"),
            "benchmark_protocol_sha256": independent_cases.sha256(),
        },
        "model_id": model.model_id, "policy": model.policy["version"], "gate": model.policy["semantic_gate"],
        "bitext_test": measure(bundle, test.text, test.category),
        "hard": {k: measure(bundle, v.text, v.category) for k, v in hard.groupby("difficulty")},
        "ood": measure(bundle, ood.text), "ood_by_topic": {k: measure(bundle, v.text) for k, v in ood.groupby("topic")},
        "scope": "Frozen synthetic diagnostics. Some cases were previously reviewed by the authors."
    })


if __name__ == "__main__":
    main()
