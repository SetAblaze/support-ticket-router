"""Правила рисков и причин ручной проверки (демонстрационные, не обученные)."""
from __future__ import annotations

import re
from datetime import datetime

from .text import preprocess


def _risk_signals(text):
    t = preprocess(text)
    # preprocess expands n't but not the other contractions used in threats.
    t = re.sub(r"\bi(?:'| )?m\b", "i am", t)
    t = re.sub(r"\bi'll\b", "i will", t)
    t = re.sub(r"\bi've\b", "i have", t)
    t = re.sub(r"\bgonna\b", "going to", t)
    # Quotes in an educational/example context do not directly establish intent.
    quoted = re.sub(r'"[^"\n]*"|“[^”\n]*”', " ", t)
    reported = bool(re.search(
        r"\b(?:someone|somebody|he|she|they|a (?:man|woman|person|customer)|the (?:driver|courier|seller|merchant|staff|agent))\s+"
        r"(?:threatened|is threatening|has threatened)\s+"
        r"(?:(?:to\s+)?(?:kill|hurt|attack|shoot|harm)\s+)?(?:me|us|my (?:family|partner|child))\b",
        quoted))
    direct = bool(re.search(
        r"\bi (?:will|am going to|intend to|plan to)\s+(?!not\b|never\b)"
        r"(?:(?:come|go) to [a-z0-9' ]{1,60}? and\s+)?"
        r"(?:(?:kill|hurt|attack|shoot|harm|assault|beat)\s+"
        r"(?:you|ya|your|him|her|them|someone|somebody|staff|"
        r"the (?:driver|courier|seller|merchant|agent)|my (?:family|partner|child))"
        r"|make you pay)\b", quoted))
    question_like = bool(re.search(r"^\s*(?:how|what|where|can|could|should|is|are|do|does|did)\b", t))
    fraud = bool(re.search(
        r"\b(?:unauthori[sz]ed|unrecogni[sz]ed)\s+(?:payment|charge|transaction|purchase)\b|"
        r"\b(?:account|card)\s+(?:was |has been |got )?(?:hacked|stolen|compromised)\b|"
        r"\b(?:someone|somebody|hackers?)\s+(?:used|is using|hacked|broke into)\s+my\s+(?:card|account)\b|"
        r"\b(?:did not|do not|never)\s+(?:authori[sz]e|recogni[sz]e)\s+"
        r"(?:this|that|the|a)\s+(?:charge|payment|transaction|purchase)\b|"
        r"\bi (?:did not|do not|never) authori[sz]e\b|"
        r"\b(?:charge|payment|transaction)\s+i\s+(?:do not|did not)\s+recogni[sz]e\b|"
        r"\b(?:this|that)\s+(?:purchase|charge|payment|transaction)\s+(?:is not|was not)\s+mine\b|"
        r"\b(?:seller|merchant|shop|store)\b.{0,60}\b(?:took|kept|stole)\b.{0,40}"
        r"\b(?:money|payment|funds)\b.{0,40}\b(?:never|did not)\b.{0,20}"
        r"\b(?:send|sent|deliver|delivered|ship|shipped)\b", t))
    fraud = fraud or bool(re.search(
        r"\bi (?:see|noticed|found|have) (?:a |an )?fraudulent (?:charge|transaction)\b|"
        r"\bi (?:have|'ve) been scammed\b|"
        r"\bi (?:did not|never) (?:make|made|place|placed|approve|approved) (?:this|that|the) "
        r"(?:purchase|order|payment|transaction)\b", quoted))
    if question_like and not re.search(r"\b(?:my|i see|i noticed|i found|i have)\b", t):
        fraud = False
    legal = bool(re.search(
        r"\bi (?:will|intend to|am going to)\s+(?:sue|take (?:you|this) to court)\b|"
        r"\bmy (?:lawyer|solicitor)\b|"
        r"\b(?:formal|legal)\s+(?:notice|claim|proceedings)\s+(?:to|against|for)\s+me\b|"
        r"\blegal proceedings against\b", quoted))
    abuse = bool(re.search(r"\b(?:idiots?|morons?|stupid|fuck(?:ing)?|shit|assholes?)\b", quoted))
    if abuse and re.search(r"\b(?:article|film|movie|book|quote|example|word)\b", t) \
            and not re.search(r"\byou\b", quoted):
        abuse = False
    flags = []
    if direct or reported:
        flags.append("POSSIBLE_THREAT")
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
    facts = {"vip": context.get("vip_verified") is True,
             "repeat": bool(context.get("open_related_ticket") is True and
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
