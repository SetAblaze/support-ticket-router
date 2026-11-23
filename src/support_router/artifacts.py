"""Сохранение и проверяемая загрузка модели."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib

from .config import PREPROCESS_VERSION, SCHEMA_VERSION
from .utils import runtime_versions, sha256_file, write_json

# Модули, от которых зависит решение по обращению. Если любой из них изменился,
# сохранённая модель считается несовместимой: решение могло бы тихо поменяться.
# api.py, monitoring.py, train.py, data.py и evaluation.py сюда не входят:
# их правка не меняет того, как обученная модель решает.
DECISION_MODULES = ("config.py", "text.py", "models.py", "rules.py", "routing.py", "multiple_requests.py",
                    "semantic_encoder.py", "category_examples.py")


def decision_code_sha256():
    """Один хеш по всем модулям решения; переводы строк приводятся к \\n."""
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for name in DECISION_MODULES:
        digest.update(name.encode("utf-8") + b"\0")
        digest.update((root / name).read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return digest.hexdigest()


def save_bundle(bundle, directory, test_metrics=None):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    path = directory / "router.joblib"
    joblib.dump(bundle, path, compress=3)
    auxiliary = {name:sha256_file(Path(__file__).parent/name)
                 for name in ['semantic_encoder.py','category_examples.py'] if (Path(__file__).parent/name).exists()}
    encoder_manifest = Path(directory)/'encoder/encoder_manifest.json'
    meta = {"schema_version": SCHEMA_VERSION, "model_id": bundle.model_id,
            "model_name": bundle.model_name, "model_type": type(bundle.model).__name__,
            "base_type": type(bundle.base_model.named_steps["clf"]).__name__,
            "classes": list(map(str, bundle.classes_)), "other_is_rejection_not_trained_class": True,
            "preprocess_version": PREPROCESS_VERSION, "policy": bundle.policy,
            "artifact_sha256": sha256_file(path), "module_sha256": decision_code_sha256(),
            "module_sha256_covers": list(DECISION_MODULES),
            "auxiliary_module_sha256":auxiliary,
            "encoder":json.loads(encoder_manifest.read_text()) if bundle.model_name.startswith('semantic_') else None,
            "versions": runtime_versions(), "provenance": bundle.metadata,
            "test": test_metrics, "external_validation_status": "NOT_AVAILABLE",
            "security_notice":"Adjacent SHA256 is not authenticity; joblib is unsafe for untrusted artifacts"}
    write_json(directory / "router_meta.json", meta)
    return path, meta

def load_bundle(directory, *, trust_local_artifact=False):
    # A hash in adjacent metadata detects corruption, NOT malicious replacement.
    # joblib.load is pickle deserialization and can execute arbitrary Python.
    if not trust_local_artifact:
        raise PermissionError('Only load artifacts created in this trusted runtime; explicit trust_local_artifact=True required')
    directory = Path(directory)
    meta = json.loads((directory/"router_meta.json").read_text(encoding="utf-8"))
    if sha256_file(directory/"router.joblib") != meta["artifact_sha256"]:
        raise ValueError("Artifact SHA256 mismatch")
    if meta["versions"]["scikit-learn"] != runtime_versions()["scikit-learn"]:
        raise ValueError("scikit-learn version mismatch: use recorded version or retrain")
    if meta["module_sha256"] != decision_code_sha256():
        raise ValueError("support_router decision code differs from the code that exported the artifact; retrain")
    for name,digest in meta.get('auxiliary_module_sha256',{}).items():
        if sha256_file(Path(__file__).parent/name)!=digest:raise ValueError(f'Auxiliary module mismatch: {name}')
    if meta.get('encoder'):
        from .semantic_encoder import ensure_encoder
        encoder_dir=ensure_encoder(directory/'encoder')
        if json.loads((encoder_dir/'encoder_manifest.json').read_text())!=meta['encoder']:
            raise ValueError('Encoder metadata mismatch')
    obj = joblib.load(directory/"router.joblib")
    if obj.model_id != meta["model_id"] or list(obj.classes_) != meta["classes"] or obj.policy != meta["policy"]:
        raise ValueError("Artifact metadata mismatch")
    return obj, meta
