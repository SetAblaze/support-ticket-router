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
    labels = list(labels if labels is not None else CATEGORIES)
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
            rows.append({"bin": b, "n": int(mask.sum()), "confidence": float(conf[mask].mean()),
                         "accuracy": float(correct[mask].mean())})
    auc = float(roc_auc_score(correct, conf)) if len(np.unique(correct)) == 2 else None
    return {"n": len(frame), "confidence_auroc": auc,
            "auroc_status": "AVAILABLE" if auc is not None else "UNDEFINED_SINGLE_CORRECTNESS_CLASS",
            "log_loss": float(log_loss(frame.category, p, labels=classes)),
            "multiclass_brier_sum": float(np.mean(np.sum((p-onehot)**2, axis=1))),
            "ece_top_label": float(sum(r["n"]*abs(r["confidence"]-r["accuracy"]) for r in rows)/len(frame)),
            "bins": rows}

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
