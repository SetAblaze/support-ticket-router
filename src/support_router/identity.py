"""Experiment identity: bind a report to one model, policy, data split and benchmark.

The Colab notebook and the scripts must never display a saved report as the
result of a different model.  This module is the single place that decides
whether a report belongs to the currently loaded artifact.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from . import independent_cases
from .config import VERSION
from .paths import project_root
from .utils import sha256_file

RAW_DATA = "data/raw/20000-Utterances-Training-dataset-for-chatbots-virtual-assistant-Bitext-sample.csv"


class ExperimentIdentityError(ValueError):
    """Raised when a report cannot be tied to the expected experiment."""


@dataclass(frozen=True)
class ExperimentIdentity:
    code_version: str
    model_id: str
    model_name: str
    artifact_sha256: str
    decision_code_sha256: str
    policy_version: str
    policy_auto_enabled: bool
    raw_data_sha256: str
    split_manifest_sha256: str | None
    benchmark_protocol_sha256: str

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def experiment_id(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _hash(path: Path) -> str | None:
    return sha256_file(path) if path.exists() else None


def identity_from_meta(meta: dict, root: Path | None = None) -> ExperimentIdentity:
    """Build the immutable identity of the loaded artifact and repository state."""
    root = Path(root) if root is not None else project_root()
    policy = meta.get("policy") or {}
    return ExperimentIdentity(
        code_version=str(VERSION),
        model_id=str(meta["model_id"]),
        model_name=str(meta["model_name"]),
        artifact_sha256=str(meta["artifact_sha256"]),
        decision_code_sha256=str(meta["module_sha256"]),
        policy_version=str(policy.get("version")),
        policy_auto_enabled=bool(policy.get("auto_enabled", False)),
        raw_data_sha256=str(_hash(root / RAW_DATA)),
        split_manifest_sha256=_hash(root / "data/processed/split_manifest.json"),
        benchmark_protocol_sha256=independent_cases.sha256(),
    )


def identity_from_manifest(manifest: dict) -> ExperimentIdentity:
    identity = manifest.get("identity") or {}
    return ExperimentIdentity(
        code_version=str(identity.get("code_version")),
        model_id=str(identity.get("model_id")),
        model_name=str(identity.get("model_name")),
        artifact_sha256=str(identity.get("artifact_sha256")),
        decision_code_sha256=str(identity.get("decision_code_sha256")),
        policy_version=str(identity.get("policy_version")),
        policy_auto_enabled=bool(identity.get("policy_auto_enabled", False)),
        raw_data_sha256=str(identity.get("raw_data_sha256")),
        split_manifest_sha256=identity.get("split_manifest_sha256"),
        benchmark_protocol_sha256=str(identity.get("benchmark_protocol_sha256")),
    )


def report_identity(report: dict) -> dict:
    """Extract identity fields from the report formats used in this repository."""
    manifest_identity = report.get("identity") or {}
    provenance = report.get("provenance") or manifest_identity
    protocol = report.get("protocol") or {}
    policy = report.get("policy")
    policy_version = (provenance.get("policy_version") or report.get("policy_version")
                      or (policy.get("version") if isinstance(policy, dict) else policy))
    return {
        "code_version": provenance.get("code_version") or report.get("code_version"),
        "model_id": provenance.get("model_id") or report.get("model_id"),
        "model_name": provenance.get("model_name") or report.get("model_name"),
        "artifact_sha256": provenance.get("artifact_sha256") or report.get("artifact_sha256"),
        "decision_code_sha256": (provenance.get("decision_code_sha256")
                                 or report.get("decision_code_sha256")),
        "policy_version": policy_version,
        "policy_auto_enabled": provenance.get("policy_auto_enabled",
                                              report.get("policy_auto_enabled")),
        "raw_data_sha256": provenance.get("raw_data_sha256") or report.get("raw_data_sha256"),
        "split_manifest_sha256": (provenance.get("split_manifest_sha256")
                                  or report.get("split_manifest_sha256")),
        "benchmark_protocol_sha256": (provenance.get("benchmark_protocol_sha256")
                                      or protocol.get("sha256")
                                      or report.get("benchmark_sha256")),
        "experiment_id": (provenance.get("experiment_id") or report.get("experiment_id")
                          or manifest_identity.get("experiment_id")),
    }


_REQUIRED = ("model_id", "artifact_sha256", "policy_version", "benchmark_protocol_sha256")


def validate_report_identity(meta: dict, report: dict, expected_benchmark_sha: str | None = None,
                             *, root: Path | None = None,
                             require: tuple[str, ...] = _REQUIRED) -> dict:
    """Validate that ``report`` belongs to ``meta`` and return the parsed identity.

    A report that lacks the binding fields is rejected rather than treated as
    current.  This is intentional: an old notebook output must not silently
    appear as the result of a freshly trained model.
    """
    expected = identity_from_meta(meta, root)
    actual = report_identity(report)
    missing = [field for field in require if not actual.get(field)]
    if missing:
        raise ExperimentIdentityError(
            "Report is not bound to a model; missing identity fields: " + ", ".join(missing))
    checks = {}
    for field in ("model_id", "model_name", "artifact_sha256", "decision_code_sha256",
                  "policy_version", "raw_data_sha256", "split_manifest_sha256",
                  "benchmark_protocol_sha256"):
        want = getattr(expected, field)
        got = actual.get(field)
        if want is None:
            continue
        if got is None:
            # Older reports may omit split/code hashes; only required fields are fatal.
            if field in require:
                raise ExperimentIdentityError(f"Report is missing {field}")
            continue
        if str(got) != str(want):
            raise ExperimentIdentityError(
                f"Report {field} mismatch: expected {want}, got {got}")
        checks[field] = "MATCH"
    benchmark = expected_benchmark_sha or expected.benchmark_protocol_sha256
    if str(actual["benchmark_protocol_sha256"]) != str(benchmark):
        raise ExperimentIdentityError(
            "Benchmark protocol hash mismatch: report was produced on a different frozen set")
    if actual.get("experiment_id") and actual["experiment_id"] != expected.experiment_id:
        raise ExperimentIdentityError(
            "Report experiment_id does not match the loaded model and repository state")
    return {"status": "CURRENT", "expected": expected.as_dict(), "actual": actual,
            "checks": checks, "experiment_id": expected.experiment_id}


def is_current(meta: dict, report: dict, *, root: Path | None = None) -> bool:
    try:
        validate_report_identity(meta, report, root=root)
        return True
    except ExperimentIdentityError:
        return False
