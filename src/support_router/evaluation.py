"""Метрики, LOIO, бутстреп и проверка маршрутизации."""
from __future__ import annotations

import hashlib
import math

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, log_loss, roc_auc_score

from .config import CATEGORIES, CATEGORY_TO_QUEUE, MANUAL_QUEUE, SEED
from .models import checked_proba, fit_model


def intent_protocol(frame):
    n = frame.groupby("category").intent.nunique()
    eligible = sorted(frame.loc[frame.category.isin(n[n >= 2].index), "intent"].unique())
    order = sorted(eligible, key=lambda s: hashlib.sha256((str(SEED)+s).encode()).hexdigest())
    dev = order[:min(10, max(1, len(order)//2))]
    diag = [i for i in order if i not in dev]
    excluded = frame.loc[frame.category.isin(n[n == 1].index), ["category", "intent"]].drop_duplicates()
    return dev, diag, excluded

def loio_evaluate(frame, name, intents):
    """LOIO diagnostic for architecture selection, not independent validation.

    Every intent may take part in choosing the model. The same intents cannot
    then serve as an independent estimate of the selected model; use the frozen
    benchmark in ``independent_cases`` for that.
    """
    rows, predictions = [], []
    for intent in intents:
        te = frame[frame.intent.eq(intent)]
        tr = frame[~frame.intent.eq(intent) & ~frame.group.isin(te.group)]
        missing = set(te.category) - set(tr.category)
        if missing or te.empty or tr.empty:
            rows.append({"intent": intent, "n_test": len(te), "status": "HELD_OUT_CATEGORY",
                         "recall": None, "missing_categories": sorted(missing)})
            continue
        m = fit_model(name, tr, leave_intent_out=True)
        pred = m.predict(te.text.tolist())
        # A single-intent slice is reported as recall/accuracy, not fake all-class macro F1.
        rows.append({"intent": intent, "n_train": len(tr), "n_test": len(te),
                     "remaining_class_train_n": int((tr.category == te.category.iloc[0]).sum()),
                     "class_starvation_warning": bool((tr.category == te.category.iloc[0]).sum() < 200),
                     "category": ",".join(sorted(te.category.unique())),
                     "status": "PASS", "recall": float(np.mean(pred == te.category.values)),
                     "n_groups_removed": int(frame.loc[~frame.intent.eq(intent) & frame.group.isin(te.group), "group"].nunique()),
                     "n_training_rows_removed_for_group_overlap": int(frame[~frame.intent.eq(intent)].group.isin(te.group).sum())})
        for (_, r), p in zip(te.iterrows(), pred):
            predictions.append({"row_id": r.row_id, "intent": intent,
                                "y_true": r.category, "y_pred": str(p), "model": name})
    return pd.DataFrame(rows), pd.DataFrame(predictions)

def classification_metrics(y_true, pred, labels=None):
    y_true, pred = list(y_true), list(pred)
    if len(y_true) != len(pred):
        raise ValueError("y_true and pred must have the same length")
    labels = list(labels if labels is not None else CATEGORIES)
    if len(labels) != len(set(labels)):
        raise ValueError("labels must be unique; pass CATEGORIES, not the repeated y_true list")
    unknown = (set(y_true) | set(pred)) - set(labels)
    if unknown:
        raise ValueError(f"Predictions or truth contain classes outside the declared labels: {sorted(unknown)}")
    return {"n": len(y_true), "macro_f1": float(f1_score(y_true, pred, average="macro",
             labels=labels, zero_division=0)), "accuracy": float(accuracy_score(y_true, pred)),
            "errors": int(np.sum(np.asarray(y_true) != np.asarray(pred))), "labels": labels,
            "report": classification_report(y_true, pred, labels=labels, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(y_true, pred, labels=labels).tolist()}

def wilson_upper(errors, n, z=1.96):
    if n == 0:
        return None
    p = errors/n
    return float((p + z*z/(2*n) + z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))) / (1+z*z/n))

def calibration_diagnostics(model, frame, bins=10):
    p = checked_proba(model, frame.text.tolist())
    classes = np.asarray(model.classes_)
    pred = classes[p.argmax(axis=1)]
    conf = p.max(axis=1)
    correct = (pred == frame.category.values).astype(int)
    index = {c: i for i, c in enumerate(classes)}
    onehot = np.zeros_like(p)
    onehot[np.arange(len(frame)), [index[c] for c in frame.category]] = 1
    rows = []
    bucket = np.minimum((conf*bins).astype(int), bins-1)
    for b in range(bins):
        mask = bucket == b
        if mask.any():
            confidence = float(conf[mask].mean())
            accuracy = float(correct[mask].mean())
            rows.append({"bin": b, "n": int(mask.sum()), "confidence": confidence,
                         "accuracy": accuracy, "gap": accuracy - confidence})
    auc = float(roc_auc_score(correct, conf)) if len(np.unique(correct)) == 2 else None
    return {"n": len(frame), "confidence_auroc": auc,
            "auroc_status": "AVAILABLE" if auc is not None else "UNDEFINED_SINGLE_CORRECTNESS_CLASS",
            "log_loss": float(log_loss(frame.category, p, labels=classes)),
            "multiclass_brier_sum": float(np.mean(np.sum((p-onehot)**2, axis=1))),
            "ece_top_label": float(sum(r["n"]*abs(r["confidence"]-r["accuracy"]) for r in rows)/len(frame)),
            "bins": rows, "reliability": rows,
            "note": ("Low ECE on an easy synthetic split does not prove calibration under "
                     "distribution shift; inspect bin counts and per-category support")}

def routing_metrics(results, y_true=None):
    n=len(results)
    auto=np.array([r['status']=='AUTO_RECOMMENDED' for r in results])
    out={'n':n,'automatic_n':int(auto.sum()),'manual_n':int((~auto).sum()),
         'coverage':float(auto.mean()) if n else None,
         'manual_rate':float((~auto).mean()) if n else None,
         'automatic_category_errors':None,'accuracy_automatic_category':None,
         'automatic_queue_errors':None,'accuracy_automatic_queue':None,
         'manual_is_not_a_classification_error':True}
    if y_true is not None:
        y=np.asarray(y_true,dtype=object)
        predicted=np.array([r.get('accepted_category') for r in results],dtype=object)
        queue_true=np.array([CATEGORY_TO_QUEUE.get(str(c),MANUAL_QUEUE) for c in y])
        queue_pred=np.array([r.get('queue') for r in results],dtype=object)
        err=int((predicted[auto]!=y[auto]).sum());qerr=int((queue_pred[auto]!=queue_true[auto]).sum())
        out.update(automatic_category_errors=err,
                   accuracy_automatic_category=float((predicted[auto]==y[auto]).mean()) if auto.any() else None,
                   automatic_category_error_upper_wilson_95=wilson_upper(err,int(auto.sum())),
                   automatic_queue_errors=qerr,
                   accuracy_automatic_queue=float((queue_pred[auto]==queue_true[auto]).mean()) if auto.any() else None,
                   automatic_queue_error_upper_wilson_95=wilson_upper(qerr,int(auto.sum())),
                   id_other_decision_rate=float(np.mean([r['decision_category']=='OTHER' for r in results])) if n else None)
        out['per_category']={}
        for c in sorted(set(y)):
            ix=y==c;ac=ix&auto
            out['per_category'][c]={'n':int(ix.sum()),'coverage':float(auto[ix].mean()),
                                    'auto_n':int(ac.sum()),
                                    'auto_category_accuracy':float((predicted[ac]==y[ac]).mean()) if ac.any() else None,
                                    'auto_queue_accuracy':float((queue_pred[ac]==queue_true[ac]).mean()) if ac.any() else None}
    return out

def paired_error_test(y, pred_a, pred_b):
    a, b = np.asarray(pred_a) == np.asarray(y), np.asarray(pred_b) == np.asarray(y)
    ab, ba = int((a & ~b).sum()), int((~a & b).sum())
    return {"a_correct_b_wrong": ab, "a_wrong_b_correct": ba,
            "p_value": float(binomtest(ab, ab+ba, .5).pvalue) if ab+ba else 1.,
            "interpretation": "paired binary errors, not significance of macro-F1; dependent templates limit inference"}

def paired_group_bootstrap(frame, pred_a, pred_b, iterations=2000):
    rng = np.random.default_rng(SEED)
    groups = frame.group.unique()
    group_rows = {g: np.flatnonzero(frame.group.to_numpy() == g) for g in groups}
    y, pa, pb = frame.category.to_numpy(), np.asarray(pred_a), np.asarray(pred_b)
    a, b = [], []
    for _ in range(iterations):
        ix = np.concatenate([group_rows[g] for g in rng.choice(groups, len(groups), replace=True)])
        a.append(f1_score(y[ix], pa[ix], labels=CATEGORIES, average="macro", zero_division=0))
        b.append(f1_score(y[ix], pb[ix], labels=CATEGORIES, average="macro", zero_division=0))
    return {"iterations": iterations, "unit": "lexical_group", "labels": CATEGORIES,
            "baseline_ci95": np.quantile(a, [.025, .975]).tolist(),
            "router_classifier_ci95": np.quantile(b, [.025, .975]).tolist(),
            "paired_delta_ci95": np.quantile(np.asarray(b)-a, [.025, .975]).tolist(),
            "limitation": "zero observed errors produce degenerate bootstrap; not a guarantee on new domains"}

def wilson_lower(successes,n,z=1.96):
    if n == 0:return None
    return 1-wilson_upper(n-successes,n,z=z)

def paired_intent_bootstrap(intents_and_scores, reference, candidate, iterations=2000,seed=42):
    """Paired bootstrap over *intents*; no train/test mixture and no false SE claim."""
    table=intents_and_scores.pivot(index='intent',columns='model',values='recall')
    a=table[reference].astype(float).to_numpy()
    b=table[candidate].astype(float).to_numpy()
    mask=np.isfinite(a)&np.isfinite(b)
    if not mask.sum():raise ValueError('No common evaluated intents')
    a,b=a[mask],b[mask]
    rng=np.random.default_rng(seed)
    idx=rng.integers(0,len(a),size=(iterations,len(a)))
    delta=(b[idx]-a[idx]).mean(axis=1)
    return {'reference':reference,'candidate':candidate,'n_intents':int(len(a)),
            'paired_mean_delta':float((b-a).mean()),
            'paired_delta_ci95':np.quantile(delta,[.025,.975]).tolist(),
            'intent_unit':'held_out_intent','iterations':iterations,
            'candidate_wins':int((b>a).sum()),'candidate_losses':int((b<a).sum()),
            'interpretation':'Exploratory CI with same 25 intents used for selection; not independent validation'}

def per_category_calibration(model, frame, min_support=30, bins=10):
    """Per-category calibration and support. Rare categories are marked, not celebrated."""
    p = checked_proba(model, frame.text.tolist())
    classes = np.asarray(model.classes_)
    index = {c: i for i, c in enumerate(classes)}
    y = frame.category.to_numpy()
    pred = classes[p.argmax(axis=1)]
    eps = 1e-6
    rows = {}
    for c in classes:
        i = index[c]
        prob = np.clip(p[:, i], eps, 1 - eps)
        truth = (y == c).astype(int)
        n = int(truth.sum())
        bucket = np.minimum((prob * bins).astype(int), bins - 1)
        ece = 0.
        reliability = []
        for b in range(bins):
            mask = bucket == b
            if mask.any():
                confidence = float(prob[mask].mean())
                accuracy = float(truth[mask].mean())
                ece += mask.sum() * abs(confidence - accuracy)
                reliability.append({"bin": b, "n": int(mask.sum()), "confidence": confidence,
                                    "accuracy": accuracy, "gap": accuracy - confidence})
        ece = float(ece / len(frame))
        rows[c] = {
            "support": n, "positive_n": n, "negative_n": int(len(frame) - n),
            "support_status": "INSUFFICIENT_SUPPORT" if n < min_support else "ADEQUATE",
            "prevalence": float(truth.mean()),
            "mean_predicted_probability": float(prob.mean()),
            "brier": float(np.mean((prob - truth) ** 2)),
            "log_loss": float(-np.mean(truth * np.log(prob) + (1 - truth) * np.log(1 - prob))),
            "ece": ece, "reliability": reliability,
            "precision": float(np.mean(y[pred == c] == c)) if (pred == c).any() else None,
            "recall": float(np.mean(pred[truth == 1] == c)) if n else None,
        }
    insuff = sorted(c for c, r in rows.items() if r["support_status"] == "INSUFFICIENT_SUPPORT")
    return {"n": len(frame), "min_support": min_support, "per_category": rows,
            "insufficient_support": insuff,
            "note": "Rare classes cannot support a 100% precision/recall claim"}


def calibration_by_source(model, frames):
    """Calibration split by data source; rare categories remain explicitly marked."""
    out = {}
    for name, frame in frames.items():
        per = per_category_calibration(model, frame)
        out[name] = {"overall": calibration_diagnostics(model, frame),
                     "per_category": per["per_category"],
                     "insufficient_support": per["insufficient_support"],
                     "min_support": per["min_support"]}
    return out


def similarity_binned_accuracy(bundle, frame, bins=(0., .5, .6, .7, .8, .85, .9, .95, 1.01)):
    """Accuracy by maximum lexical similarity to the training reference."""
    sim = np.asarray(bundle.similarity(frame.text.tolist()), dtype=float)
    pred = bundle.predict(frame.text.tolist())
    truth = frame.category.to_numpy()
    correct = pred == truth
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (sim >= lo) & (sim < hi)
        n = int(mask.sum())
        errs = int((~correct[mask]).sum()) if n else 0
        rows.append({"low": float(lo), "high": float(hi), "n": n,
                     "mean_similarity": float(sim[mask].mean()) if n else None,
                     "accuracy": float(correct[mask].mean()) if n else None,
                     "errors": errs,
                     "error_upper_wilson_95": wilson_upper(errs, n)})
    return {"n": len(frame), "bins": rows,
            "note": "Lexical similarity is descriptive; it does not prove semantic separation"}


def cascade_metrics(routes, y_true):
    """Separate cascade value from raw accuracy: fixed vs newly broken tickets."""
    y = np.asarray(y_true, dtype=object)
    primary = np.asarray([r.get("primary_category") for r in routes], dtype=object)
    final = np.asarray([r.get("category") for r in routes], dtype=object)
    hints = np.asarray([r.get("classification_source") == "minilm" for r in routes])
    suggest = int(hints.sum())
    hint_correct = int((hints & (final == y)).sum())
    correct = int((final == y).sum())
    primary_correct = int((primary == y).sum())
    return {"n": len(routes), "suggestions_n": suggest,
            "primary_correct": primary_correct, "cascade_correct": correct,
            "corrected": int(((final == y) & (primary != y)).sum()),
            "regressed": int(((primary == y) & (final != y)).sum()),
            "suggestion_precision": hint_correct / suggest if suggest else None,
            "hint_accuracy_on_suggested": hint_correct / suggest if suggest else None,
            "coverage_of_cascade": suggest / len(routes) if routes else None,
            "note": "A suggestion is an operator hint, not an automatic delivery"}


def cascade_ood_metrics(routes):
    """False MiniLM suggestions on tickets that do not belong to any category."""
    hints = [r for r in routes if r.get("classification_source") == "minilm"]
    return {"n": len(routes), "false_hints": len(hints),
            "false_hint_rate": len(hints) / len(routes) if routes else None,
            "false_hint_upper_wilson_95": wilson_upper(len(hints), len(routes)),
            "note": "OOD has no true category; every hint is a false suggestion"}


def risk_coverage_curve(probabilities, y_true, classes, thresholds=(.5, .6, .7, .75, .8, .85, .9, .95, .975, .99)):
    """Risk-coverage trade-off on an independent labelled set."""
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(y_true, dtype=object)
    classes = np.asarray(list(classes), dtype=object)
    pred = classes[p.argmax(axis=1)]
    conf = p.max(axis=1)
    rows = []
    for t in thresholds:
        accept = conf >= t
        n = int(accept.sum())
        errs = int((pred[accept] != y[accept]).sum()) if n else 0
        rows.append({"threshold": float(t), "coverage": float(accept.mean()),
                     "accepted_n": n, "errors": errs,
                     "accuracy_accepted": 1 - errs / n if n else None,
                     "error_upper_wilson_95": wilson_upper(errs, n)})
    return {"n": len(y), "curve": rows,
            "note": "Post-hoc curve on an independent set; not a guarantee on production"}


def ood_acceptance_curve(probabilities, classes, thresholds=(.5, .6, .7, .75, .8, .85, .9, .95, .975, .99)):
    """Acceptance rate on OOD: every accepted ticket is a false auto decision."""
    p = np.asarray(probabilities, dtype=float)
    conf = p.max(axis=1)
    return {"n": len(p), "curve": [
        {"threshold": float(t), "accepted_n": int((conf >= t).sum()),
         "accepted_rate": float((conf >= t).mean()),
         "accepted_rate_upper_wilson_95": wilson_upper(int((conf >= t).sum()), len(p))}
        for t in thresholds],
        "note": "OOD has no correct queue; accepted means a false suggestion"}


def end_to_end_policy_curve(bundle, frame, thresholds=(.5, .6, .7, .75, .8, .85, .9, .95),
                            similarity_threshold=None, ood_frame=None):
    """Evaluate the real routing policy, not only the classifier confidence.

    Each point calls the same ``route_ticket`` logic as the API with
    ``auto_enabled=True`` for the shadow experiment.  This includes confidence,
    training similarity, risk/input blockers and the cost-sensitive expected
    loss rule.  It is descriptive: do not choose a threshold on a final test.
    """
    import copy

    from .routing import batch_route

    y = frame.category.tolist()
    sim = (float(similarity_threshold) if similarity_threshold is not None
           else float(bundle.policy.get("similarity_threshold", .45)))
    rows = []
    for threshold in thresholds:
        candidate = copy.copy(bundle)
        policy = dict(bundle.policy)
        policy.update(auto_enabled=True, confidence_threshold=float(threshold),
                      similarity_threshold=sim, version=str(policy.get("version", "policy")) + "-EVAL")
        candidate.policy = policy
        routes = batch_route(candidate, frame.text.tolist())
        metrics = routing_metrics(routes, y)
        risk_positive = [bool(set(r.get("risk_flags", [])) &
                              {"POSSIBLE_THREAT", "POSSIBLE_FRAUD", "POSSIBLE_LEGAL_CLAIM"})
                         for r in routes]
        expected = [min((r.get("expected_loss") or {"REVIEW": 1.}).values()) for r in routes]
        row = {"threshold": float(threshold), "similarity_threshold": sim,
               "automatic_n": metrics["automatic_n"], "manual_n": metrics["manual_n"],
               "coverage": metrics["coverage"],
               "automatic_category_errors": metrics["automatic_category_errors"],
               "automatic_queue_errors": metrics["automatic_queue_errors"],
               "accuracy_automatic_category": metrics["accuracy_automatic_category"],
               "accuracy_automatic_queue": metrics["accuracy_automatic_queue"],
               "category_error_upper_wilson_95": metrics["automatic_category_error_upper_wilson_95"],
               "risk_positive_n": int(sum(risk_positive)),
               "risk_blocked_n": int(sum(risk_positive) - sum(
                   flag and r["status"] == "AUTO_RECOMMENDED"
                   for flag, r in zip(risk_positive, routes, strict=True))),
               "mean_min_expected_loss": float(np.mean(expected)) if expected else None}
        if ood_frame is not None:
            ood_routes = batch_route(candidate, ood_frame.text.tolist())
            row["ood_false_auto_n"] = int(sum(r["status"] == "AUTO_RECOMMENDED" for r in ood_routes))
            row["ood_false_auto_rate"] = row["ood_false_auto_n"] / len(ood_routes) if ood_routes else None
        rows.append(row)
    return {"n": len(frame), "curve": rows,
            "mode": "SHADOW_POLICY_END_TO_END",
            "note": ("Calls the same decision function as the API with auto_enabled=True for "
                     "diagnostics. The operational mode keeps auto_enabled=False.")}


def risk_recall(results,labels):
    if len(results)!=len(labels):raise ValueError('length mismatch')
    positives=np.asarray(labels,dtype=bool)
    detected=np.array([bool(set(r.get('risk_flags',[])) &
                           {'POSSIBLE_THREAT','POSSIBLE_FRAUD','POSSIBLE_LEGAL_CLAIM'}) for r in results])
    tp=int((detected&positives).sum());fn=int((~detected&positives).sum())
    fp=int((detected&~positives).sum())
    return {'n':len(labels),'positive_n':int(positives.sum()),'true_positive':tp,
            'false_negative':fn,'false_positive':fp,
            'precision':tp/(tp+fp) if tp+fp else None,
            'recall':tp/int(positives.sum()) if positives.any() else None,
            'wilson_lower_recall_95':wilson_lower(tp,int(positives.sum())),
            'dataset_note':'Only meaningful with independently human-annotated risk labels'}


def cascade_detailed_metrics(routes, y_true=None, latency_ms=None):
    """Separate first-stage, second-stage and operator-hint quality."""
    y = np.asarray(y_true, dtype=object) if y_true is not None else None
    primary = np.asarray([r.get("primary_category") for r in routes], dtype=object)
    final = np.asarray([r.get("category") for r in routes], dtype=object)
    source = np.asarray([r.get("classification_source") for r in routes], dtype=object)
    status = np.asarray([r.get("semantic_status") for r in routes], dtype=object)
    suggestions = source == "minilm"
    attempted = ~np.isin(status, ["NOT_REQUESTED"])
    out = {
        "n": len(routes),
        "primary_correct": int((primary == y).sum()) if y is not None else None,
        "primary_accuracy": float((primary == y).mean()) if y is not None and len(y) else None,
        "final_correct": int((final == y).sum()) if y is not None else None,
        "final_accuracy": float((final == y).mean()) if y is not None and len(y) else None,
        "suggestions_n": int(suggestions.sum()),
        "suggestion_rate": float(suggestions.mean()) if routes else None,
        "suggestion_precision": (float((final[suggestions] == y[suggestions]).mean())
                                 if y is not None and suggestions.any() else None),
        "corrected": (int(((final == y) & (primary != y)).sum()) if y is not None else None),
        "regressed": (int(((primary == y) & (final != y)).sum()) if y is not None else None),
        "fallback_attempted_n": int(attempted.sum()),
        "fallback_attempt_rate": float(attempted.mean()) if routes else None,
        "abstained_n": int((status == "ABSTAINED").sum()),
        "unavailable_n": int((status == "UNAVAILABLE").sum()),
        "truncated_n": int((status == "INPUT_TOO_LONG").sum()),
        "semantic_status_counts": {str(value): int((status == value).sum())
                                   for value in sorted(set(status.tolist()))},
        "latency": ("NOT_MEASURED" if latency_ms is None else {
            "n": len(latency_ms), "mean_ms": float(np.mean(latency_ms)),
            "p95_ms": float(np.quantile(latency_ms, .95))}),
        "note": "Suggestions are operator hints; only AUTO_RECOMMENDED is a routing decision",
    }
    if y is not None:
        out["per_category"] = {}
        for category in sorted(set(y)):
            mask = y == category
            out["per_category"][category] = {
                "n": int(mask.sum()),
                "primary_accuracy": float((primary[mask] == y[mask]).mean()),
                "final_accuracy": float((final[mask] == y[mask]).mean()),
                "suggestions_n": int(suggestions[mask].sum()),
            }
    return out


def multi_request_metrics(frame, controls=None, parts_fn=None):
    """Measure multi-intent extraction with per-intent support and control false splits."""
    if parts_fn is None:
        from .multiple_requests import request_parts as parts_fn
    exact, tp, expected_n, false_n = 0, 0, 0, 0
    per_intent = {}
    misses = []
    for text, expected_text in zip(frame.text, frame.expected_intents, strict=True):
        got = [part["intent"] for part in parts_fn(text)]
        expected = str(expected_text).split()
        exact += sorted(got) == sorted(expected)
        tp += len(set(got) & set(expected))
        expected_n += len(expected)
        false_n += len(set(got) - set(expected))
        for intent in set(expected) | set(got):
            row = per_intent.setdefault(intent, {"expected_n": 0, "predicted_n": 0,
                                                 "true_positive": 0})
            row["expected_n"] += intent in expected
            row["predicted_n"] += intent in got
            row["true_positive"] += intent in expected and intent in got
        if sorted(got) != sorted(expected):
            misses.append({"text": text, "expected": expected, "detected": got})
    for row in per_intent.values():
        row["recall"] = row["true_positive"] / row["expected_n"] if row["expected_n"] else None
        row["precision"] = (row["true_positive"] / row["predicted_n"]
                            if row["predicted_n"] else None)
        row["support_status"] = "INSUFFICIENT_SUPPORT" if row["expected_n"] < 3 else "ADEQUATE"
    control_rows = {}
    for name, texts in (controls or {}).items():
        split = sum(bool(parts_fn(text)) for text in texts)
        control_rows[name] = {"n": len(texts), "false_splits": split,
                              "false_split_rate": split / len(texts) if texts else None,
                              "false_split_upper_wilson_95": wilson_upper(split, len(texts))}
    precision = tp / (tp + false_n) if tp + false_n else None
    recall = tp / expected_n if expected_n else None
    return {"n": len(frame), "exact_matches": exact,
            "exact_match_rate": exact / len(frame) if len(frame) else None,
            "intent_precision": precision, "intent_recall": recall,
            "intent_f1": (2 * precision * recall / (precision + recall)
                          if precision and recall else None),
            "per_intent": per_intent, "misses": misses, "controls": control_rows,
            "note": ("Author-written regression set; this is not a measure of real-world "
                     "recall. Controls estimate false splits on single-topic text.")}
