"""Константы проекта: версии, темы и очереди, флаги Bitext."""
from __future__ import annotations

VERSION = "3.1.2"

PREPROCESS_VERSION = "2.0-negation-placeholders"

SCHEMA_VERSION = "3.0"

SEED = 42

CATEGORY_TO_QUEUE = {
    "ACCOUNT": "Queue_Accounts", "CANCELLATION_FEE": "Queue_Billing",
    "CONTACT": "Queue_Support", "DELIVERY": "Queue_Delivery",
    "FEEDBACK": "Queue_Quality", "INVOICES": "Queue_Billing",
    "NEWSLETTER": "Queue_Marketing", "ORDER": "Queue_Orders",
    "PAYMENT": "Queue_Payments", "REFUNDS": "Queue_Refunds",
    "SHIPPING": "Queue_Delivery",
}

CATEGORIES = sorted(CATEGORY_TO_QUEUE)

MANUAL_QUEUE = "Queue_Manual_Review"

# Linguistic annotations documented for the 2020 Bitext sample (27 intents / 11 categories).
# These flags describe language-generation variants, NOT risk labels or model inputs.
LINGUISTIC_FLAGS = {
    "B": "Basic syntactic structure",
    "S": "Syntactic structure",
    "L": "Lexical variation (synonyms)",
    "M": "Morphological variation (plurals, tenses)",
    "I": "Interrogative structure",
    "C": "Complex/Coordinated syntactic structure",
    "P": "Politeness variation",
    "Q": "Colloquial variation",
    "W": "Offensive language",
    "E": "Expanded abbreviations",
    "D": "Indirect speech",
    "Z": "Noise (spelling, punctuation)",
}
