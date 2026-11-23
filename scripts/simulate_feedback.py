"""Simulate the operator feedback loop on the frozen independent benchmark."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import support_router as sc
from support_router import independent_cases
from support_router.utils import write_json


def main():
    root = Path(sc.project_root())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=root / "models")
    parser.add_argument("--out", type=Path, default=root / "reports/feedback_simulation.json")
    args = parser.parse_args()

    bundle, meta = sc.load_bundle(args.models, trust_local_artifact=True)
    id_frame, _, _, _ = independent_cases.frames()
    natural = id_frame[id_frame.difficulty == "natural"].reset_index(drop=True)
    no_keywords = id_frame[id_frame.difficulty == "no_keywords"].reset_index(drop=True)

    runs = {}
    for label, frame in (("natural", natural), ("no_keywords", no_keywords)):
        runs[label] = {}
        for operator_error in (0.0, 0.1, 0.2):
            events = sc.simulate_stream(bundle, frame, seed=42, operator_error=operator_error)
            runs[label][f"operator_error_{operator_error:.1f}"] = {
                "metrics": sc.feedback_metrics(events),
                "retraining_candidates": len(sc.retraining_candidates(events)),
            }

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": bundle.model_id,
        "policy_version": bundle.policy.get("version"),
        "artifact_sha256": meta["artifact_sha256"],
        "benchmark_sha256": independent_cases.sha256(),
        "stream": "Frozen independent benchmark; operator behaviour is simulated",
        "runs": runs,
        "note": ("Perfect operator feedback recovers the proposal accuracy. Noisy feedback "
                 "shows how operator mistakes leak into the measured metric."),
        "scope": "Simulation only; no real operator labels",
    }
    write_json(args.out, report)
    print(json.dumps({"report": str(args.out),
                      "natural_perfect": runs["natural"]["operator_error_0.0"]["metrics"]["measured_accuracy"],
                      "no_keywords_perfect": runs["no_keywords"]["operator_error_0.0"]["metrics"]["measured_accuracy"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
