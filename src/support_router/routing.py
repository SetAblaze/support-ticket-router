"""Единственная функция решения route_ticket, политика и теневой режим."""
from __future__ import annotations

import copy
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .business import confidence_floor, triage
from .cascade import second_opinion
from .config import CATEGORY_TO_QUEUE, MANUAL_QUEUE, SCHEMA_VERSION
from .evaluation import wilson_upper
from .models import checked_proba
from .multiple_requests import request_parts
from .rules import _context_effects, detect_signals
from .text import preprocess


def _assess_ticket(text, model, context, now, scored, detect_multiple):
    """Compute input signals and model scores before applying a routing policy."""
    now = now or datetime.now(timezone.utc)
    result = {"schema_version": SCHEMA_VERSION, "category": None, "suggested_category": None,
              "decision_category": None, "accepted_category": None,
              "queue": MANUAL_QUEUE, "topic_queue": None, "confidence": None,
              "similarity": None, "top3": {}, "request_parts": [], "status": "MANUAL_REVIEW",
              "priority": "P3", "priority_reasons": [], "classification_source": None,
              "primary_category": None, "primary_confidence": None, "primary_confidence_threshold": None,
              "semantic_status": "NOT_REQUESTED", "semantic_confidence": None, "semantic_similarity": None, "risk_flags": [], "reason_codes": [],
              "model_id": getattr(model, "model_id", "unidentified"),
              "business_policy_approved": False, "delivery_status": "NOT_SENT",
              "created_at_utc": now.isoformat(), "context": {}}
    if not isinstance(text, str) or not preprocess(text):
        result.update(status="NEEDS_CLARIFICATION", reason_codes=["EMPTY_OR_INVALID_INPUT"], queue=None)
        return result, []
    signals = detect_signals(text)
    if getattr(model,'model_name','').startswith('semantic_'):
        from .semantic_encoder import was_truncated
        if was_truncated(preprocess(text)):
            signals['input_reasons'].append('INPUT_EXCEEDS_ENCODER_LIMIT')
    result["risk_flags"] = signals["risk_flags"]
    result["reason_codes"].extend(signals["input_reasons"])
    context_reasons, priority, facts = _context_effects(context, now)
    result["context"] = facts
    result["reason_codes"].extend(context_reasons)
    result["priority"], result["priority_reasons"] = triage(text, signals["risk_flags"], context_reasons, priority)
    if result["priority"] != "P3" and getattr(model, "policy", {}).get("misroute_cost"):
        result["reason_codes"].append("PRIORITY_REQUIRES_REVIEW")
    for flag in signals["risk_flags"]:
        if flag != "ABUSE_LANGUAGE":
            result["reason_codes"].append(flag + "_SPECIALIST_REVIEW")
    if any(reason in signals['input_reasons'] for reason in
           ['POSSIBLY_UNSUPPORTED_LANGUAGE','INSUFFICIENT_SUPPORTED_TEXT']):
        result['decision_category'] = 'OTHER'
        return result, context_reasons
    if detect_multiple:
        result["request_parts"] = request_parts(text)
        if result["request_parts"] and "POSSIBLE_MULTIPLE_REQUESTS" not in result["reason_codes"]:
            result["reason_codes"].append("POSSIBLE_MULTIPLE_REQUESTS")
    try:
        if scored is None:
            p = checked_proba(model, [text])[0]
            sim = float(model.similarity([text])[0]) if hasattr(model, "similarity") else 0.
        else:
            p, sim = np.asarray(scored[0], float), float(scored[1])
            # Validate cached scores as strictly as live scores.
            if p.shape != (len(model.classes_),) or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any() or not np.isclose(p.sum(), 1):
                raise ValueError("MODEL_INVALID_CACHED_PROBA")
        if not np.isfinite(sim) or sim < -1e-8 or sim > 1+1e-6:
            raise ValueError("MODEL_INVALID_SIMILARITY")
        order = np.argsort(-p)
        cat = str(model.classes_[order[0]])
        conf = float(p[order[0]])
        result.update(category=cat if cat in CATEGORY_TO_QUEUE else None,
                      suggested_category=cat, confidence=conf, similarity=sim,
                      topic_queue=CATEGORY_TO_QUEUE.get(cat),
                      top3={str(model.classes_[i]): float(p[i]) for i in order[:3]})
        result.update(classification_source="primary", primary_category=cat, primary_confidence=conf,
                      primary_confidence_threshold=confidence_floor(getattr(model, "policy", {}), cat))
        if cat == "OTHER":
            result["reason_codes"].append("MODEL_OTHER")
        elif cat not in CATEGORY_TO_QUEUE:
            result["reason_codes"].append("UNKNOWN_CATEGORY")
        second_opinion(result, text, model)
    except Exception as exc:
        result["reason_codes"].append("MODEL_CAPABILITY_OR_SCORING_ERROR")
        result["error_type"] = type(exc).__name__
        result["status"] = "MODEL_ERROR"
    return result, context_reasons


def _apply_policy(assessment, policy):
    facts, context_reasons = assessment
    result = {**facts, "reason_codes": list(facts["reason_codes"]),
              "policy_version": policy.get("version", "missing")}
    if result["confidence"] is None or result["status"] == "MODEL_ERROR":
        return result
    cat = result["suggested_category"]
    conf, sim = result["primary_confidence"], result["similarity"]
    try:
        if not policy.get("auto_enabled", False):
            result["reason_codes"].append("AUTO_POLICY_DISABLED")
        if conf < confidence_floor(policy, result["primary_category"]):
            result["reason_codes"].append("LOW_CONFIDENCE")
        if sim < float(policy.get("similarity_threshold", 1.)):
            result["reason_codes"].append("LOW_TRAINING_SIMILARITY")
        blockers = [r for r in result["reason_codes"] if r not in context_reasons]
        if blockers:
            result['decision_category'] = 'OTHER'
        else:
            result.update(queue=CATEGORY_TO_QUEUE[cat], status="AUTO_RECOMMENDED",
                          decision_category=cat,accepted_category=cat)
            result["reason_codes"].append("CATEGORY_AND_POLICY_ACCEPTED")
    except (ValueError, TypeError) as exc:
        result["reason_codes"].append("MODEL_CAPABILITY_OR_SCORING_ERROR")
        result.update(status="MODEL_ERROR", error_type=type(exc).__name__)
    result["reason_codes"] = list(dict.fromkeys(result["reason_codes"]))
    return result


def route_ticket(text, model, context=None, now=None, scored=None, *, detect_multiple=False):
    """Return a recommendation without sending the ticket to an external queue."""
    assessment = _assess_ticket(text, model, context, now, scored, detect_multiple)
    return _apply_policy(assessment, getattr(model, "policy", {}) or {})


def _batch_scores(model, texts):
    probabilities = checked_proba(model, texts)
    similarities = model.similarity(texts)
    return list(zip(probabilities, similarities, strict=True))


def batch_route(model, texts, *, detect_multiple=False):
    texts = list(texts)
    return [route_ticket(text, model, scored=score, detect_multiple=detect_multiple)
            for text, score in zip(texts, _batch_scores(model, texts), strict=True)]


def route_policy_pair(text, model, shadow, scored=None, *, detect_multiple=False):
    assessment = _assess_ticket(text, model, None, None, scored, detect_multiple)
    return _apply_policy(assessment, model.policy), _apply_policy(assessment, shadow.policy)


def batch_policy_pairs(texts, model, shadow, *, detect_multiple=False):
    try:
        scores = _batch_scores(model, texts)
    except Exception:
        # Isolate a bad input or a model that cannot score a batch.
        scores = [None] * len(texts)
    return [route_policy_pair(text, model, shadow, scored=score, detect_multiple=detect_multiple)
            for text, score in zip(texts, scores, strict=True)]


def select_policy(bundle, val, ood_dev, target_error_upper=.01, min_auto=30,
                  false_auto_cost=25., review_cost=1.):
    """Choose a CANDIDATE only. Production auto remains disabled until real
    independent ID/OOD evaluation, specialist review and formal approval.
    The author-written OOD-dev (n=24) is not independent evidence.
    """
    p=checked_proba(bundle,val.text.tolist()); pred=np.asarray(bundle.classes_)[p.argmax(1)]
    conf=p.max(1); sim=bundle.similarity(val.text.tolist())
    op=checked_proba(bundle,ood_dev.text.tolist()); oc=op.max(1)
    os=bundle.similarity(ood_dev.text.tolist())
    def allowed(t):
        signals=detect_signals(t)
        if bundle.model_name.startswith('semantic_'):
            from .semantic_encoder import was_truncated
            if was_truncated(preprocess(t)):return False
        return not signals['input_reasons'] and not any(x != 'ABUSE_LANGUAGE' for x in signals['risk_flags'])
    allowed_id=np.array([allowed(t) for t in val.text])
    allowed_ood=np.array([allowed(t) for t in ood_dev.text])
    rows=[]
    for t in [.50,.60,.70,.75,.80,.85,.90,.95,.975,.99]:
        for s in [.20,.35,.45,.50,.55,.60,.65,.70,.80,.90]:
            auto=(conf>=t)&(sim>=s)&allowed_id
            n=int(auto.sum()); errors=int((pred[auto]!=val.category.values[auto]).sum())
            out=(oc>=t)&(os>=s)&allowed_ood
            ood_false=int(out.sum())
            upper=wilson_upper(errors,n)
            # Costs are demo assumptions, not measured business costs.
            cost = (false_auto_cost*(errors+ood_false) + review_cost*(len(val)-n)) / max(1,len(val))
            rows.append({'threshold':t,'similarity_threshold':s,'auto_n':n,
                         'coverage':n/len(val),'auto_category_errors':errors,
                         'accuracy_covered_category':1-errors/n if n else None,
                         'category_error_upper_wilson_95':upper,
                         'ood_dev_false_auto':ood_false,'assumed_cost_per_val':cost,
                         'eligible':n>=min_auto and upper is not None and upper<=target_error_upper
                                    and ood_false==0})
    table=pd.DataFrame(rows)
    eligible=table[table.eligible].sort_values(
         ['assumed_cost_per_val','coverage','auto_category_errors','threshold'],
         ascending=[True,False,True,False])
    chosen=eligible.iloc[0] if len(eligible) else None
    policy={'version':'selective-v3-shadow','auto_enabled':False,
            'candidate_auto_enabled':bool(chosen is not None),
            'confidence_threshold':float(chosen.threshold) if chosen is not None else 1.,
            'similarity_threshold':float(chosen.similarity_threshold) if chosen is not None else 1.,
            'parameter_status':'SHADOW_ONLY_NO_BUSINESS_APPROVAL',
            'target_error_upper':target_error_upper,'min_auto_validation':min_auto,
            'assumed_false_auto_cost':false_auto_cost,'assumed_review_cost':review_cost,
            'selection_data':'group-disjoint val plus 24 author-written OOD-dev; no test',
            'status':'SHADOW_CANDIDATE_ONLY' if chosen is not None else 'NO_CANDIDATE',
            'blocking_requirements':['independent verified real ID>=300',
                                     'independent verified real OOD>=200',
                                     'labelled risk/P1 evaluation','business signoff'],
            'warning':'Val Wilson post-selection is NOT a risk guarantee; synthetic OOD is not external validation'}
    bundle.policy=policy
    return policy,table

def shadow_bundle(bundle):
    """Separate *non-operational* candidate used only for diagnostics."""
    import copy
    out=copy.copy(bundle)
    out.policy={**bundle.policy,'auto_enabled':bool(bundle.policy.get('candidate_auto_enabled',False)),
                'version':bundle.policy.get('version','unknown')+'-SHADOW'}
    return out

def redact_text(text):
    t = str(text)
    t = re.sub(r"\b[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}\b", "[EMAIL]", t)
    t = re.sub(r"(?<!\w)\+?\d[\d ()-]{6,}\d(?!\w)", "[NUMBER]", t)
    return t[:500]

def web_request(text, history, model, *, detect_multiple=False):
    """No global history. Returns a fresh per-session list."""
    result = route_ticket(text, model, detect_multiple=detect_multiple)
    new_history = copy.deepcopy(history or [])
    if result["status"] != "NEEDS_CLARIFICATION":
        new_history.append({"Время UTC": result["created_at_utc"], "Текст (маскир.)": redact_text(text),
                            "Категория": result["category"], "Очередь": result["queue"],
                            "Приоритет": result["priority"], "Статус": result["status"],
                            "Причины": "; ".join(result["reason_codes"]),
                            "model_id": result["model_id"], "policy_version": result["policy_version"]})
    new_history = new_history[-100:]
    return result, new_history

def schedule_demo(tickets, now, aging_minutes=60., reserve_every=3, urgent_capacity=10):
    """Toy scheduling only. Assumptions are explicit; no real SLA statistics."""
    if aging_minutes <= 0 or reserve_every < 1:
        raise ValueError("Invalid demo scheduling settings")
    rows = []
    for t in tickets:
        created = datetime.fromisoformat(t["created_at_utc"].replace("Z", "+00:00"))
        if created.tzinfo is None:
            raise ValueError("Timezone required")
        age = max(0., (now-created).total_seconds()/60)
        priority = int(str(t["priority"]).lstrip("P"))
        effective = priority if priority == 1 else max(2, priority-int(age/aging_minutes))
        rows.append({**t, "age_minutes": age, "effective_priority": effective})
    urgent = sorted([r for r in rows if r["priority"] == "P1"], key=lambda r: -r["age_minutes"])
    others = sorted([r for r in rows if r["priority"] != "P1"],
                    key=lambda r: (r["effective_priority"], -r["age_minutes"]))
    ordered = list(urgent)
    while others:
        if len(ordered) % reserve_every == reserve_every-1:
            idx = max(range(len(others)), key=lambda i: others[i]["age_minutes"])
        else:
            idx = 0
        ordered.append(others.pop(idx))
    return {"mode": "SYNTHETIC_SCHEDULING_DEMO", "ordered": ordered,
            "capacity_alert": len(urgent) > urgent_capacity,
            "parameters": {"aging_minutes": aging_minutes, "reserve_every": reserve_every,
                           "urgent_capacity": urgent_capacity}, "real_sla_metrics": None}
