"""Build or verify a release manifest that ties every report to one experiment.

``--build`` records the source/build/release commits and hashes every evaluation
set and report.  ``--verify`` recomputes the identity and report hashes and fails
on any mismatch.  The manifest never hashes itself.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import support_router as sc
from support_router.identity import identity_from_meta
from support_router.utils import sha256_file, write_json

EVAL_SETS = ("hard_test", "ood_eval", "risk_regression", "risk_dev",
             "independent_id", "independent_ood", "independent_risk", "independent_multi")
REPORTS = ("results", "independent_benchmark", "representation_comparison",
           "business_cascade", "review_benchmark", "feedback_simulation")


def _git(root, *args):
    try:
        return subprocess.run(["git", "-C", str(root), *args], check=True,
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


def _hash_if_exists(path):
    return sha256_file(path) if Path(path).exists() else None


def _identity(meta, root):
    identity = identity_from_meta(meta, root)
    return identity.as_dict(), identity.experiment_id


def _commits(root, previous=None):
    head = _git(root, "rev-parse", "HEAD")
    source = (previous or {}).get("source_commit") or head
    return {"source_commit": source, "build_commit": head, "release_commit": head}


def build(root, models, out):
    meta = json.loads((models / "router_meta.json").read_text(encoding="utf-8"))
    previous = json.loads(out.read_text(encoding="utf-8")) if out.exists() else None
    identity, experiment_id = _identity(meta, root)
    manifest = {
        "schema": "release-manifest-v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": experiment_id,
        "experiment_identity": identity,
        "commits": _commits(root, previous),
        "git_branch": _git(root, "rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(_git(root, "status", "--porcelain")),
        "warning": ("git_dirty=true means the manifest describes the working tree, not a "
                    "committed release; regenerate after committing"),
        "identity": {**identity, "git_commit": _git(root, "rev-parse", "HEAD")},
        "evaluation_sets": {name: _hash_if_exists(root / "data/eval" / f"{name}.csv")
                            for name in EVAL_SETS},
        "reports": {name: _hash_if_exists(root / "reports" / f"{name}.json")
                    for name in REPORTS},
        "comparison_rule": ("Compare reports only when experiment_id, evaluation-set hashes, "
                            "report hashes and policy version match"),
        "note": ("A different policy changes decisions without changing the classifier; "
                 "such reports are different experiments"),
    }
    write_json(out, manifest)
    return manifest


def verify(root, models, out):
    problems = []
    if not out.exists():
        return [f"manifest missing: {out}"]
    manifest = json.loads(out.read_text(encoding="utf-8"))
    meta = json.loads((models / "router_meta.json").read_text(encoding="utf-8"))
    identity, experiment_id = _identity(meta, root)
    if manifest.get("experiment_id") != experiment_id:
        problems.append("experiment_id does not match the current model/data/policy identity")
    stored = manifest.get("experiment_identity") or {}
    for key, value in identity.items():
        if stored.get(key) != value:
            problems.append(f"identity.{key}: manifest={stored.get(key)!r} current={value!r}")
    if manifest.get("git_dirty"):
        problems.append("manifest was built from a dirty working tree")
    for name, expected in (manifest.get("evaluation_sets") or {}).items():
        current = _hash_if_exists(root / "data/eval" / f"{name}.csv")
        if current != expected:
            problems.append(f"evaluation set {name}: hash changed or file missing")
    for name, expected in (manifest.get("reports") or {}).items():
        current = _hash_if_exists(root / "reports" / f"{name}.json")
        if current != expected:
            problems.append(f"report {name}: hash changed or file missing")
    return problems


def main(argv=None):
    root = Path(sc.project_root())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=root / "models")
    parser.add_argument("--out", type=Path, default=root / "reports/release_manifest.json")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--build", action="store_true", help="write a new manifest (default)")
    mode.add_argument("--verify", action="store_true", help="verify the existing manifest")
    args = parser.parse_args(argv)
    if args.verify:
        problems = verify(root, args.models, args.out)
        if problems:
            print("MANIFEST_MISMATCH")
            for problem in problems:
                print(" -", problem)
            return 2
        print("MANIFEST_OK")
        return 0
    manifest = build(root, args.models, args.out)
    print(json.dumps({"manifest": str(args.out), "experiment_id": manifest["experiment_id"],
                      "model_id": manifest["identity"]["model_id"],
                      "policy_version": manifest["identity"]["policy_version"],
                      "git_dirty": manifest["git_dirty"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
