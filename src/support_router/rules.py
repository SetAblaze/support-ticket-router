"""Правила рисков и причин ручной проверки (демонстрационные, не обученные)."""
from __future__ import annotations

import re
from datetime import datetime

from .text import preprocess


def _risk_signals(text):
    t = preprocess(text)
    # 3.1.1 (этап 11): preprocess разворачивает только n't, поэтому «I'm going to kill you»
    # не совпадало с шаблоном «i am going to …». Нормализуем только для правил рисков.
    t = re.sub(r"\bi(?:'| )?m\b", "i am", t)
    t = re.sub(r"\bgonna\b", "going to", t)
    # Quotes in an educational/example context do not directly establish intent.
    quoted = re.sub(r'"[^"\n]*"|“[^”\n]*”', " ", t)
    reported = bool(re.search(r"\b(threatened me|threatening me|sent me (?:a )?threat|someone threatened|"
                              r"(?:threatened|threatening) to (?:kill|hurt|attack|shoot|harm))\b", t))
    direct = bool(re.search(
        r"\bi(?: will|'ll| am going to| intend to)\s+"
        r"(?:(?:kill|hurt|attack|shoot)\s+(?:you|ya|your|the courier|the driver|staff|someone)|make you pay)\b",
        quoted))
    fraud = bool(re.search(r"\b(unauthori[sz]ed (?:payment|charge|transaction)|(?:account|card) (?:was |has been |got )?(?:hacked|stolen|compromised)|did not authori[sz]e|do not recogni[sz]e (?:this|the|a) (?:charge|payment)|"
                           r"(?:someone|somebody|hackers?) (?:hacked|broke into) my (?:account|card)|(?:charge|payment|transaction) i do not recogni[sz]e)\b", t))
    legal = bool(re.search(r"\b(i(?: will|'ll| intend to| am going to) (?:sue|take (?:you|this) to court)|formal (?:legal )?(?:claim|notice)|legal (?:notice|proceedings)|my lawyer (?:has|will|is))\b", quoted))
    abuse = bool(re.search(r"\b(?:idiots?|morons?|stupid|fuck(?:ing)?|shit|assholes?)\b", quoted))
    flags = []
    if direct or reported:
        flags.append("POSSIBLE_THREAT")
    fraud = fraud or bool(re.search(r"\b(?:i (?:did not|didn't|never) (?:make|made|place|placed|approve|approved) "
                                  r"(?:this|that|the) (?:purchase|order|payment|transaction)|"
                                  r"someone (?:used|is using) my (?:card|account)|"
                                  r"this (?:purchase|charge|payment|transaction) (?:is not|was not) mine)\b", t))
    # Require a first-person report. Advice, quotations and denials are common in support tickets.
    fraud = fraud or bool(re.search(
        r"\bi (?:see|noticed|found|have) (?:a |an )?fraudulent (?:charge|transaction)\b|"
        r"\bi(?: have|'ve) been scammed\b", quoted))
    if fraud:
        flags.append("POSSIBLE_FRAUD")
    if legal:
        flags.append("POSSIBLE_LEGAL_CLAIM")
    if abuse:
        flags.append("ABUSE_LANGUAGE")
    return flags

def _input_reasons(text):
    t = preprocess(text)
    reasons = []
    if re.search(r"[\u0400-\u04ff\u0600-\u06ff\u4e00-\u9fff¿¡]", t) or re.search(
            r"\b(bonjour|remboursement|contraseña|pedido|dirección|bonjour|pouvez|mein konto|meine bestellung)\b", t):
        reasons.append("POSSIBLY_UNSUPPORTED_LANGUAGE")
    topic_patterns = {
        "account": r"\b(password|log ?in|account|profile)\b", "refund": r"\b(refund|reimbursement|return my money)\b",
        "invoice": r"\b(invoice|bill)\b", "address": r"\b(shipping address|delivery address|address|relocate|moved house)\b",
        "payment": r"\b(card|payment method|credit card|debit card|billing card)\b",
        "order": r"\b(cancel my order|change my order|track my order)\b"}
    topics = [k for k, p in topic_patterns.items() if re.search(p, t)]
    if len(topics) >= 2 and re.search(r"\b(and|also|both|plus|as well)\b", t):
        reasons.append("POSSIBLE_MULTIPLE_REQUESTS")
    if re.search(r"\b(app|website|browser|checkout|page)\b", t) and re.search(
            r"\b(crashes?|freezes?|fatal exception|blank screen|will not load|does not load|stops responding)\b", t):
        reasons.append("TECHNICAL_ISSUE_NEEDS_REVIEW")
    if not re.search(r"[a-z]{2,}", t):
        reasons.append("INSUFFICIENT_SUPPORTED_TEXT")
    # Conservative triage for vague inputs without a recognizable shop anchor.
    anchors = r"\b(order|purchase|shop|store|package|parcel|delivery|ship|return|refund|"               r"payment|card|account|password|sign in|log in|invoice|bill|email|"               r"newsletter|discount|checkout|price|item|product|contact|customer service)\b"
    vague = r"\b(?:this|that|it|anything|everything|something|any of this|again|yesterday)\b"
    if len(t.split()) <= 10 and not re.search(anchors,t) and re.search(vague,t):
        reasons.append('INSUFFICIENT_CONTEXT')
    return reasons

def detect_signals(text):
    return {"risk_flags": _risk_signals(text), "input_reasons": _input_reasons(text),
            "validation_status": "HEURISTIC_CANDIDATES_NOT_BUSINESS_VALIDATED"}

def _context_effects(context, now):
    # This argument is for an application-owned CRM adapter, never extracted from user_text.
    if not context or context.get("source") != "trusted_crm":
        return [], 3, {"vip": "unknown", "repeat": "unknown", "sla_overdue": "unknown"}
    reasons, priority = [], 3
    facts = {"vip": bool(context.get("vip_verified", False)),
             "repeat": bool(context.get("open_related_ticket", False) and
                            isinstance(context.get("contact_count"), int) and context["contact_count"] >= 2),
             "sla_overdue": "unknown"}
    if facts["vip"]:
        reasons.append("VERIFIED_VIP_DEMO_POLICY"); priority = 2
    if facts["repeat"]:
        reasons.append("REPEAT_OPEN_TICKET_DEMO_POLICY"); priority = 2
    deadline = context.get("deadline_utc")
    if deadline:
        try:
            dt = datetime.fromisoformat(deadline.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                raise ValueError("deadline requires timezone")
            facts["sla_overdue"] = now > dt
            if facts["sla_overdue"]:
                reasons.append("VERIFIED_DEADLINE_OVERDUE"); priority = 2
        except (ValueError, TypeError, AttributeError):
            reasons.append("INVALID_CONTEXT_DEADLINE")
    return reasons, priority, facts
