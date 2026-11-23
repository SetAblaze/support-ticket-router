"""Conservative English clause rules for an optional operator-assistance mode."""
import re

from .config import CATEGORY_TO_QUEUE
from .text import preprocess

# Read a coordinated list of requested objects ("an invoice and a refund") differently
# from separate sentences: a bare noun in a new sentence may just be context.
_BOUNDARY = re.compile(r"([.!?;\n]+|,|\b(?:and|but|plus|also|as well as|by the way|additionally)\b)", re.I)
_REQUEST = re.compile(r"\b(?:i (?:also )?(?:need|want|would like)|i'd like|send me|please (?:send|get) me)\b")
_NOUN = re.compile(r"(?:an? |my |the |another |new )*(invoice|receipt|refund|reimbursement|new password|password)")
_OBJECTS = {
    "invoice": "get_invoice", "receipt": "get_invoice",
    "refund": "get_refund", "reimbursement": "get_refund",
    "password": "recover_password", "new password": "recover_password",
}
_NEGATED = re.compile(r"\b(?:do not|never|no longer|not asking (?:you )?to|instead of|without)\s+"
                      r"(?:want|need|ask|request|send|cancel|refund|reset|change|close|delete|update|subscribe|"
                      r"track|check|unsubscribe|contact|open|create|modify|recover|remove|issue|get|download)\b")

# Match a request or a reported problem. The category of a mentioned object alone
# (e.g. "card" in "send my refund to my card") is insufficient evidence.
_RULES = [
    ("recover_password", "ACCOUNT", r"(?:reset|forgot|forgotten|recover|change|lost) (?:my |the |a )?password|(?:can not|unable to) (?:log|sign) in"),
    ("delete_account", "ACCOUNT", r"(?:delete|close|remove) (?:my |the )?(?:old )?(?:account|profile)"),
    ("create_account", "ACCOUNT", r"(?:open|create|register) (?:an? |my )?(?:new )?(?:account|profile)"),
    ("edit_account", "ACCOUNT", r"(?:update|edit) (?:my |the )?(?:account|profile)"),
    ("recover_account", "ACCOUNT", r"(?:recover|unlock) (?:my |the )?account|(?:my |the )?account (?:is |was )?locked|locked out of my account"),
    ("change_shipping_address", "SHIPPING", r"(?:change|update|edit|correct|set up) (?:my |the |a )?(?:(?:shipping|delivery|postal) )?address|(?:shipping|delivery) address (?:is wrong|needs changing)"),
    ("payment_issue", "PAYMENT", r"(?:payment|transaction|card|transfer) (?:was |is |has )?(?:failed|declined|rejected)|charged (?:me )?twice|(?:payment|transaction) did not go through"),
    ("change_payment_method", "PAYMENT", r"(?:change|update) (?:my |the )?(?:(?:credit |debit )?card|payment method)"),
    ("check_payment_methods", "PAYMENT", r"(?:what|which) payment methods|(?:can i |how (?:can|do) i )pay (?:with|by)|do you accept (?:cash|paypal|cards?)"),
    ("get_invoice", "INVOICES", r"(?:send|need|want|get|download|request|resend|print|copy of) (?:me )?(?:a copy of )?(?:my |an? |the )?(?:bill|invoice|receipt)"),
    ("invoice_issue", "INVOICES", r"(?:bill|invoice) is (?:wrong|missing|incorrect)"),
    ("track_refund", "REFUNDS", r"(?:check|track) (?:my |the )?refund(?! policy)|(?:where is|where's) my refund|refund (?:still |has |is )*(?:not arrived|missing|pending)"),
    ("check_refund_policy", "REFUNDS", r"(?:what(?: is|'s)|explain|tell me|ask about|check) (?:about )?(?:the |your )?refund polic(?:y|ies)"),
    ("get_refund", "REFUNDS", r"(?:want|need|request|issue|send|get|like) (?:me )?(?:my |an? |the )?(?:refund|reimbursement)|refund (?:my|me)|(?:get|return) my money(?: back)?"),
    ("cancel_order", "ORDER", r"cancel (?:my |the |an? )?(?:order|purchase)"),
    ("change_order", "ORDER", r"(?:change|modify|update) (?:my |the |an? )?order"),
    ("track_order", "ORDER", r"track (?:my |the |an? )?(?:order|parcel|package)|where(?: is|'s) (?:my |the )?(?:order|parcel|package)|(?:package|parcel|order) is (?:late|missing)"),
    ("check_cancellation_fee", "CANCELLATION_FEE", r"(?:what(?: is|'s)|how much is|is there|check|ask about) (?:an? |the |your )?cancell?ation fees?"),
    ("delivery_options", "DELIVERY", r"(?:what(?: are)?|which|ask about|tell me about|check) (?:the |your )?delivery (?:options|methods|times?)|how long (?:does|will) delivery"),
    ("unsubscribe_newsletter", "NEWSLETTER", r"unsubscribe (?:me )?(?:from )?(?:your |the )?newsletter"),
    ("subscribe_newsletter", "NEWSLETTER", r"subscribe (?:me )?(?:to )?(?:your |the )?newsletter|sign me up for (?:your |the )?newsletter"),
    ("contact_human_agent", "CONTACT", r"(?:speak|talk) (?:with|to) (?:an? |your )?(?:human|agent|representative)|contact (?:your |the )?(?:customer service|support)"),
    ("complaint", "FEEDBACK", r"(?:file|make|lodge) (?:a )?complaint|complain about"),
    ("review", "FEEDBACK", r"(?:write|leave|submit) (?:a |my )?(?:review|feedback)"),
]
_PATTERNS = [(intent, re.compile(r"\b(?:" + pattern + r")\b")) for intent, _, pattern in _RULES]
_CATEGORIES = {intent: category for intent, category, _ in _RULES}


def request_parts(text):
    """Return ordered, distinct suggested intents only when two or more are supported."""
    pieces = _BOUNDARY.split(text)
    if len(pieces) == 1:
        return []
    intents = {}
    in_list = False
    for i in range(0, len(pieces), 2):
        clause = preprocess(pieces[i])
        if i and re.search(r"[.!?;\n]", pieces[i - 1]):
            in_list = False
        if not clause:
            continue
        if _NEGATED.search(clause):
            in_list = False
            continue
        matches = [(m.start(), intent) for intent, pattern in _PATTERNS
                   if (m := pattern.search(clause))]
        noun = _NOUN.fullmatch(clause)
        if matches:
            _, intent = min(matches, key=lambda match: match[0])
            intents[intent] = None
            in_list = bool(_REQUEST.search(clause))
        elif in_list and noun:
            intents[_OBJECTS[noun[1]]] = None
        else:
            in_list = False
    if len(intents) < 2:
        return []
    return [{"intent": intent, "category": _CATEGORIES[intent], "queue": CATEGORY_TO_QUEUE[_CATEGORIES[intent]]}
            for intent in intents]
