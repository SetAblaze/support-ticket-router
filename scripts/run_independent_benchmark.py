"""One-shot evaluation on the frozen independent benchmark.

The protocol is fixed in ``docs/independent_benchmark_protocol.md``. This script
must not be used to tune thresholds, features, rules or prompts. It writes
``reports/independent_benchmark.json`` and, unless disabled, adds a compact
summary to ``reports/results.json``.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

import support_router as sc
from support_router import independent_cases
from support_router.evaluation import (
    calibration_diagnostics,
    cascade_metrics,
    cascade_ood_metrics,
    classification_metrics,
    ood_acceptance_curve,
    per_category_calibration,
    risk_coverage_curve,
    routing_metrics,
    similarity_binned_accuracy,
    wilson_lower,
    wilson_upper,
)
from support_router.multiple_requests import request_parts
from support_router.text import preprocess
from support_router.utils import sha256_file, write_json

FROZEN_SHA256 = "97d8404692fc8c8210ea86c8d2fbc7cb68cffbdec60c15e14a3cf070767eac9c"
RISK_FLAGS = {"POSSIBLE_THREAT", "POSSIBLE_FRAUD", "POSSIBLE_LEGAL_CLAIM"}


def _git_commit(root):
    try:
        return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True,
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


def _topk(probabilities, classes, k=3):
    order = np.argsort(-np.asarray(probabilities), axis=1)[:, :k]
    return np.asarray(classes, dtype=object)[order]


def _classification(probabilities, classes, labels):
    labels = list(labels)
    top = _topk(probabilities, classes)
    top1 = top[:, 0]
    correct = int((top1 == np.asarray(labels)).sum())
    metrics = classification_metrics(labels, top1, labels=labels)
    per_category = {}
    for category in labels:
        mask = np.asarray(labels) == category
        n = int(mask.sum())
        hit = int((top1[mask] == category).sum())
        per_category[category] = {"n": n, "correct": hit,
                                  "top1_accuracy": hit / n if n else None,
                                  "top1_wilson_lower_95": wilson_lower(hit, n)}
    return {"n": len(labels), "top1_correct": correct,
            "top1_accuracy": correct / len(labels) if labels else None,
            "top1_wilson_lower_95": wilson_lower(correct, len(labels)),
            "top3_correct": int((top == np.asarray(labels)[:, None]).any(axis=1).sum()),
            "macro_f1": metrics["macro_f1"], "errors": metrics["errors"],
            "per_category": per_category}


def _rules(frame):
    detected = [set(sc.detect_signals(text)["risk_flags"]) for text in frame.text]
    expected = [set(str(value).split()) if str(value).strip() else set()
                for value in frame.expected_flags]
    per_flag = {}
    for flag in sorted(set().union(*expected, *detected)):
        tp = sum(flag in e and flag in p for e, p in zip(expected, detected, strict=True))
        fp = sum(flag not in e and flag in p for e, p in zip(expected, detected, strict=True))
        fn = sum(flag in e and flag not in p for e, p in zip(expected, detected, strict=True))
        per_flag[flag] = {"tp": tp, "fp": fp, "fn": fn,
                          "precision": tp / (tp + fp) if tp + fp else None,
                          "recall": tp / (tp + fn) if tp + fn else None,
                          "recall_wilson_lower_95": wilson_lower(tp, tp + fn) if tp + fn else None}
    risk_true = [bool(e & RISK_FLAGS) for e in expected]
    risk_pred = [bool(p & RISK_FLAGS) for p in detected]
    tp = sum(t and p for t, p in zip(risk_true, risk_pred, strict=True))
    fp = sum((not t) and p for t, p in zip(risk_true, risk_pred, strict=True))
    fn = sum(t and (not p) for t, p in zip(risk_true, risk_pred, strict=True))
    return {"n": len(frame), "per_flag": per_flag,
            "any_risk_flag": {"tp": tp, "fp": fp, "fn": fn,
                              "precision": tp / (tp + fp) if tp + fp else None,
                              "recall": tp / (tp + fn) if tp + fn else None,
                              "recall_wilson_lower_95": wilson_lower(tp, tp + fn) if tp + fn else None},
            "misses": [{"text": t, "expected": sorted(e), "detected": sorted(p)}
                       for t, e, p in zip(frame.text, expected, detected, strict=True) if e != p]}


def _multi(frame, controls):
    exact, tp, expected_n, false_n = 0, 0, 0, 0
    misses = []
    for text, expected_text in zip(frame.text, frame.expected_intents, strict=True):
        got = [part["intent"] for part in request_parts(text)]
        expected = expected_text.split()
        exact += sorted(got) == sorted(expected)
        tp += len(set(got) & set(expected))
        expected_n += len(expected)
        false_n += len(set(got) - set(expected))
        if sorted(got) != sorted(expected):
            misses.append({"text": text, "expected": expected, "detected": got})
    control_rows = {}
    for name, texts in controls.items():
        split = sum(bool(request_parts(text)) for text in texts)
        control_rows[name] = {"n": len(texts), "false_splits": split,
                              "false_split_rate": split / len(texts) if len(texts) else None,
                              "false_split_upper_wilson_95": wilson_upper(split, len(texts))}
    return {"n": len(frame), "exact_matches": exact,
            "exact_match_rate": exact / len(frame) if len(frame) else None,
            "intent_precision": tp / (tp + false_n) if tp + false_n else None,
            "intent_recall": tp / expected_n if expected_n else None,
            "misses": misses, "controls": control_rows,
            "note": "Author-written regression set; not a measure of real-world recall"}


def _disjointness(new_texts, raw_texts, existing):
    exact_raw = sorted({text for text in new_texts if preprocess(text) in raw_texts})
    vectorizer = TfidfVectorizer(preprocessor=preprocess, lowercase=False, ngram_range=(1, 2),
                                 min_df=1, sublinear_tf=True)
    reference = normalize(vectorizer.fit_transform(sorted(raw_texts)))
    query = normalize(vectorizer.transform([preprocess(text) for text in new_texts]))
    similarity = np.asarray((query @ reference.T).max(axis=1).todense()).ravel()
    return {"exact_duplicates_with_bitext": exact_raw,
            "max_lexical_similarity_to_bitext": float(similarity.max()) if len(similarity) else None,
            "n_similarity_ge_085": int((similarity >= .85).sum()),
            "n_similarity_ge_080": int((similarity >= .80).sum()),
            "exact_duplicates_with_development_sets": {
                name: sorted({text for text in new_texts if text in texts}) for name, texts in existing.items()},
            "note": "Lexical checks do not prove semantic novelty"}


def _criteria(report):
    natural = report["deployed"]["natural"]["top1_accuracy"]
    no_keywords = report["deployed"]["no_keywords"]["top1_accuracy"]
    ood = report["ood"]["shadow_false_auto_rate"]
    cascade = report["cascade"]["no_keywords"]
    risk = report["risk"]["any_risk_flag"]["recall"]
    checks = {
        "no_keywords_top1_at_least_0.50": no_keywords is not None and no_keywords >= .50,
        "ood_false_auto_at_most_0.05": ood is not None and ood <= .05,
        "cascade_net_positive": cascade["corrected"] > cascade["regressed"],
        "risk_recall_at_least_0.80": risk is not None and risk >= .80,
    }
    return {"checks": checks, "natural_top1": natural, "no_keywords_top1": no_keywords,
            "ood_false_auto_rate": ood, "risk_recall": risk,
            "all_passed": all(checks.values())}


def main():
    root = Path(sc.project_root())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", type=Path, default=root / "data/processed")
    parser.add_argument("--models", type=Path, default=root / "models")
    parser.add_argument("--out", type=Path, default=root / "reports/independent_benchmark.json")
    parser.add_argument("--no-update-results", action="store_true")
    args = parser.parse_args()

    if independent_cases.sha256() != FROZEN_SHA256:
        raise ValueError("Independent benchmark differs from the frozen protocol hash")

    bundle, meta = sc.load_bundle(args.models, trust_local_artifact=True)
    id_frame, ood_frame, risk_frame, multi_frame = independent_cases.frames()
    natural = id_frame[id_frame.difficulty == "natural"].reset_index(drop=True)
    no_keywords = id_frame[id_frame.difficulty == "no_keywords"].reset_index(drop=True)
    shadow = sc.shadow_bundle(bundle)

    train_path = args.processed / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"{train_path} missing: run python -m support_router.train first")
    train = pd.read_csv(train_path, keep_default_na=False)
    baseline = sc.fit_model("lr_word (baseline)", train)

    deployed = {}
    baseline_report = {}
    for name, frame in (("natural", natural), ("no_keywords", no_keywords)):
        probabilities = bundle.predict_proba(frame.text.tolist())
        deployed[name] = _classification(probabilities, bundle.classes_, frame.category.tolist())
        deployed[name]["similarity_bins"] = similarity_binned_accuracy(bundle, frame)
        deployed[name]["policy"] = routing_metrics(sc.batch_route(shadow, frame.text.tolist()),
                                                   frame.category.tolist())
        base_probabilities = sc.checked_proba(baseline, frame.text.tolist())
        baseline_report[name] = _classification(base_probabilities, baseline.classes_,
                                                frame.category.tolist())

    ood_routes = sc.batch_route(shadow, ood_frame.text.tolist())
    ood_false = sum(route["status"] == "AUTO_RECOMMENDED" for route in ood_routes)
    ood_probabilities = bundle.predict_proba(ood_frame.text.tolist())

    natural_routes = sc.batch_route(shadow, natural.text.tolist())
    no_keyword_routes = sc.batch_route(shadow, no_keywords.text.tolist())

    calibration = pd.read_csv(args.processed / "calibration.csv", keep_default_na=False)
    test = pd.read_csv(args.processed / "test.csv", keep_default_na=False)
    hard = pd.read_csv(root / "data/eval/hard_test.csv")
    controls = {"bitext_test_single_topic": test.text.tolist(),
                "hard_test_single_topic": hard.text.tolist(),
                "independent_natural_single_topic": natural.text.tolist()}

    raw = pd.read_csv(root / "data/raw/20000-Utterances-Training-dataset-for-chatbots-virtual-assistant-Bitext-sample.csv")
    raw_texts = {preprocess(text) for text in raw.utterance.dropna().unique()}
    existing = {
        "manual_cases_dev": set(sc.manual_cases.frames()[0].text),
        "manual_cases_holdout": set(sc.manual_cases.frames()[1].text),
        "scenario_dev": set(sc.scenario_data.frames()[0].text),
        "scenario_holdout": set(sc.scenario_data.frames()[1].text),
        "hard_test": set(hard.text),
        "ood_eval": set(pd.read_csv(root / "data/eval/ood_eval.csv").text),
        "risk_regression": set(pd.read_csv(root / "data/eval/risk_regression.csv", keep_default_na=False).text),
    }
    all_new = [*id_frame.text, *ood_frame.text, *risk_frame.text, *multi_frame.text]

    report = {
        "protocol": {"version": independent_cases.VERSION, "sha256": independent_cases.sha256(),
                     "frozen_sha256": FROZEN_SHA256, "document": "docs/independent_benchmark_protocol.md"},
        "provenance": {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
                       "git_commit": _git_commit(root), "model_id": bundle.model_id,
                       "model_name": bundle.model_name, "policy_version": bundle.policy.get("version"),
                       "policy_auto_enabled": bundle.policy.get("auto_enabled"),
                       "artifact_sha256": meta["artifact_sha256"],
                       "raw_data_sha256": sha256_file(root / "data/raw/20000-Utterances-Training-dataset-for-chatbots-virtual-assistant-Bitext-sample.csv"),
                       "benchmark_files": {name: sha256_file(root / "data/eval" / f"{name}.csv")
                                           for name in ("independent_id", "independent_ood",
                                                        "independent_risk", "independent_multi")}},
        "disjointness": _disjointness(all_new, raw_texts, existing),
        "deployed": deployed,
        "baseline_tfidf_logreg": baseline_report,
        "ood": {"n": len(ood_frame), "shadow_false_auto_n": ood_false,
                "shadow_false_auto_rate": ood_false / len(ood_frame),
                "shadow_false_auto_upper_wilson_95": wilson_upper(ood_false, len(ood_frame)),
                "acceptance_curve": ood_acceptance_curve(ood_probabilities, bundle.classes_)},
        "cascade": {
            "natural": cascade_metrics(natural_routes, natural.category.tolist()),
            "no_keywords": cascade_metrics(no_keyword_routes, no_keywords.category.tolist()),
            "ood": cascade_ood_metrics(ood_routes),
        },
        "risk_coverage": {
            "natural": risk_coverage_curve(bundle.predict_proba(natural.text.tolist()),
                                           natural.category.tolist(), bundle.classes_),
            "no_keywords": risk_coverage_curve(bundle.predict_proba(no_keywords.text.tolist()),
                                               no_keywords.category.tolist(), bundle.classes_),
        },
        "calibration": {"calibration_split": per_category_calibration(bundle, calibration),
                        "overall_calibration_split": calibration_diagnostics(bundle, calibration),
                        "independent_natural": per_category_calibration(bundle, natural)},
        "risk": _rules(risk_frame),
        "multi_request": _multi(multi_frame, controls),
        "scope": "Frozen author-written benchmark, scored once. Not a customer sample.",
    }
    report["criteria"] = _criteria(report)
    write_json(args.out, report)
    print(json.dumps({"report": str(args.out), "criteria": report["criteria"],
                      "natural_top1": report["deployed"]["natural"]["top1_accuracy"],
                      "no_keywords_top1": report["deployed"]["no_keywords"]["top1_accuracy"]},
                     ensure_ascii=False, indent=2))

    if not args.no_update_results:
        results_path = root / "reports/results.json"
        if results_path.exists():
            results = json.loads(results_path.read_text(encoding="utf-8"))
            results["independent_benchmark"] = {
                "protocol_version": independent_cases.VERSION,
                "protocol_sha256": independent_cases.sha256(),
                "report": "reports/independent_benchmark.json",
                "natural_top1": report["deployed"]["natural"]["top1_accuracy"],
                "no_keywords_top1": report["deployed"]["no_keywords"]["top1_accuracy"],
                "ood_shadow_false_auto_rate": report["ood"]["shadow_false_auto_rate"],
                "criteria": report["criteria"],
            }
            selection = results.setdefault("selection", {})
            selection["loio_role"] = ("architecture selection only; the same 25 intents are not "
                                      "independent validation of the selected model")
            selection["independent_evaluation"] = "reports/independent_benchmark.json"
            results["release"] = {
                "note": ("results.json is the notebook output for an earlier artifact. Current "
                         "release metrics are in models/router_meta.json, "
                         "reports/independent_benchmark.json and reports/release_manifest.json"),
                "current_model_id": bundle.model_id,
                "current_policy_version": bundle.policy.get("version"),
                "release_manifest": "reports/release_manifest.json",
            }
            limitations = results.setdefault("limitations", [])
            for item in ("LOIO selection and reporting reuse the same 25 intents",
                         "The independent benchmark is author-written, not a customer sample"):
                if item not in limitations:
                    limitations.append(item)
            write_json(results_path, results)


if __name__ == "__main__":
    main()
