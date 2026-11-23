"""Build a release manifest that ties every result to one model and policy."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import support_router as sc
from support_router import independent_cases
from support_router.utils import sha256_file, write_json


def _git(root, *args):
    try:
        return subprocess.run(["git", "-C", str(root), *args], check=True,
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


def _hash_if_exists(path):
    return sha256_file(path) if Path(path).exists() else None


def main():
    root = Path(sc.project_root())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=root / "models")
    parser.add_argument("--out", type=Path, default=root / "reports/release_manifest.json")
    args = parser.parse_args()

    meta = json.loads((args.models / "router_meta.json").read_text(encoding="utf-8"))
    identity = {
        "code_version": sc.VERSION,
        "git_commit": _git(root, "rev-parse", "HEAD"),
        "git_branch": _git(root, "rev-parse", "--abbrev-ref", "HEAD"),
        "model_id": meta["model_id"],
        "model_name": meta["model_name"],
        "artifact_sha256": meta["artifact_sha256"],
        "decision_code_sha256": meta["module_sha256"],
        "policy_version": meta["policy"].get("version"),
        "policy_auto_enabled": meta["policy"].get("auto_enabled"),
        "raw_data_sha256": sha256_file(root / "data/raw/20000-Utterances-Training-dataset-for-chatbots-virtual-assistant-Bitext-sample.csv"),
        "split_manifest_sha256": _hash_if_exists(root / "data/processed/split_manifest.json"),
        "benchmark_protocol_sha256": independent_cases.sha256(),
    }
    experiment_id = hashlib.sha256(json.dumps(identity, sort_keys=True,
                                              separators=(",", ":")).encode("utf-8")).hexdigest()
    evaluation_sets = {name: _hash_if_exists(root / "data/eval" / f"{name}.csv")
                       for name in ("hard_test", "ood_eval", "risk_regression",
                                    "independent_id", "independent_ood",
                                    "independent_risk", "independent_multi")}
    reports = {name: _hash_if_exists(root / "reports" / f"{name}.json")
               for name in ("results", "independent_benchmark", "representation_comparison",
                            "business_cascade", "review_benchmark", "feedback_simulation")}
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": experiment_id,
        "git_dirty": bool(_git(root, "status", "--porcelain")),
        "warning": ("git_dirty=true means the manifest describes the working tree, not a "
                    "committed release; regenerate after committing"),
        "identity": identity,
        "evaluation_sets": evaluation_sets,
        "reports": reports,
        "comparison_rule": ("Compare reports only when experiment_id, evaluation-set hashes and "
                            "policy version match"),
        "note": ("A different policy changes decisions without changing the classifier; "
                 "such reports are different experiments"),
    }
    write_json(args.out, manifest)
    print(json.dumps({"manifest": str(args.out), "experiment_id": experiment_id,
                      "model_id": meta["model_id"], "policy_version": meta["policy"].get("version")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
