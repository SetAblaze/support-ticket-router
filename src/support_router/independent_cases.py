"""Frozen author-written benchmark created for the post-audit evaluation.

The phrases were written from the public category definitions and the audit
questions, before this benchmark was scored by any model. They are never used
for fitting, calibration, threshold selection, feature choices or prompt
engineering. The protocol is fixed in ``docs/independent_benchmark_protocol.md``.

Two labelled modes are kept separate on purpose:

* ``natural`` - ordinary customer wording;
* ``no_keywords`` - the same intents without the words that make Bitext easy.

The ``ood``, ``risk`` and ``multi`` sets are unlabelled for category routing
(except ``risk`` and ``multi`` which carry their own rule labels). They measure
abstention, rule recall and split behaviour, not classification accuracy.
"""
from __future__ import annotations

import hashlib
import json

import pandas as pd

VERSION = "independent-v1-2026-11-23"

NATURAL = {
    "ACCOUNT": [
        "I changed my email and now I cannot access my profile at all.",
        "The sign-in page keeps rejecting the password I just created.",
        "Please help me remove the personal details stored in my profile.",
        "I want to register a new profile but the form says I already have one.",
        "My security question answers are no longer accepted when I sign in.",
    ],
    "CANCELLATION_FEE": [
        "Will money be taken from my card if I call off the reservation?",
        "Is there a penalty for backing out of the booking after confirmation?",
        "How much do you keep when a customer calls off a reservation?",
        "I stopped the booking yesterday; what charge should I expect on my statement?",
        "Does ending the arrangement before the event cost anything?",
    ],
    "CONTACT": [
        "Please put me through to a member of your support team.",
        "What is the best way to reach a human advisor about my case?",
        "Could this conversation be handed over to a real adviser?",
        "Can you give me the phone line for customer care?",
        "How do I start a live conversation with your staff?",
    ],
    "DELIVERY": [
        "What delivery choices do I have if I order today?",
        "How quickly can an order reach me with the standard service?",
        "Can I arrange for the parcel to arrive tomorrow morning?",
        "Which courier services are offered at the checkout?",
        "Is there a faster option than the usual delivery time?",
    ],
    "FEEDBACK": [
        "I want to praise the adviser who solved my problem today.",
        "Where can I send a complaint about the way I was treated?",
        "The service I received was excellent and I would like to say so.",
        "Please pass on my disappointment to the team that handled my case.",
        "I want to tell you how pleased I am with the agent I spoke to.",
    ],
    "INVOICES": [
        "Could you email the billing document for last month's purchase?",
        "I need a copy of the receipt for my accounting records.",
        "Where can I find the bill for the order I received on Monday?",
        "Please resend the invoice to my work email address.",
        "The amount on my bill does not match what I was charged.",
    ],
    "NEWSLETTER": [
        "Please stop the sale announcements arriving in my inbox.",
        "I want to receive your weekly product updates by email.",
        "I no longer wish to receive advertising campaigns.",
        "How can I join the mailing list for seasonal offers?",
        "The promotional emails are too frequent; please end them.",
    ],
    "ORDER": [
        "I placed a purchase this morning and need to change the items.",
        "Where is the parcel from the purchase I made last Friday?",
        "Please stop the purchase I made a few minutes ago.",
        "I ordered the wrong size and want to amend the purchase.",
        "Can you tell me when my purchase will be dispatched?",
    ],
    "PAYMENT": [
        "The checkout rejects my debit card every time I attempt to pay.",
        "Can I settle the amount with a bank transfer?",
        "The same purchase appears twice on my bank statement.",
        "Which payment options can I use at the checkout?",
        "My bank says the transaction went through, but your store says it did not.",
    ],
    "REFUNDS": [
        "Your team accepted the return, yet the amount is still missing from my statement.",
        "What are the rules for getting my funds returned?",
        "I returned the purchase and would like to request reimbursement.",
        "How long does it usually take for a refund to appear?",
        "The item went back last week; where is my money?",
    ],
    "SHIPPING": [
        "The destination address saved for deliveries is out of date.",
        "I need parcels sent to my new flat from now on.",
        "The flat number in my destination details is wrong.",
        "Can I add a second delivery address to my profile?",
        "The last parcel went to my old street; please update the details.",
    ],
}

NO_KEYWORDS = {
    "ACCOUNT": [
        "I am shut out of my personal area and the recovery message never arrives.",
        "The site refuses the secret phrase I set up yesterday.",
        "I want my personal details erased from your system.",
        "The form says I am already registered although this is my first visit.",
        "The questions used to prove it is me are no longer accepted.",
    ],
    "CANCELLATION_FEE": [
        "If I call off the booking, will any money be taken from my card?",
        "What does backing out after confirmation cost me?",
        "How much do you keep when a reservation is called off?",
        "I stopped the booking yesterday; what will appear on my statement?",
        "Does ending the arrangement before the date cost anything?",
    ],
    "CONTACT": [
        "Please put me through to a real person.",
        "What is the best way to reach a human about my case?",
        "I prefer dealing with a human rather than this bot.",
        "Can you give me the number for customer care?",
        "How do I start a live conversation with your staff?",
    ],
    "DELIVERY": [
        "What choices do I have for getting something to my door?",
        "How quickly can an item reach me with the usual service?",
        "Can I arrange for the parcel to arrive tomorrow morning?",
        "Which services are offered at the checkout for sending goods?",
        "Is there a faster option than the usual timeframe?",
    ],
    "FEEDBACK": [
        "I want to praise the adviser who solved my problem today.",
        "Where can I send a grievance about the way I was treated?",
        "The service was excellent and I would like to say so.",
        "Please pass on my disappointment to the team that handled my case.",
        "How do I share my opinion after an experience with your staff?",
    ],
    "INVOICES": [
        "Could you email the paperwork showing what I paid last month?",
        "I need a copy of the payment document for my accountant.",
        "Where can I find the statement for the order I received on Monday?",
        "Please resend the paperwork to my work email address.",
        "The amount on the document does not match what left my bank.",
    ],
    "NEWSLETTER": [
        "Please stop the sale announcements arriving in my inbox.",
        "I want to receive your weekly product updates by email.",
        "I no longer wish to receive advertising campaigns.",
        "How can I join the group that gets seasonal offers?",
        "The promotional emails are too frequent; please end them.",
    ],
    "ORDER": [
        "I bought something this morning and need to change the items.",
        "Where is the item I paid for last Friday?",
        "I want to stop the request I sent a few minutes ago.",
        "I chose the wrong size and want to amend my basket.",
        "Can you tell me when my goods will be sent out?",
    ],
    "PAYMENT": [
        "The checkout rejects my Visa every time I try to finish.",
        "Is a bank transfer accepted by your store?",
        "My bank shows two identical amounts for one purchase.",
        "Which ways can I settle the amount at the checkout?",
        "The transaction failed although there is enough money in my bank.",
    ],
    "REFUNDS": [
        "The return was approved but the amount has not reached my bank.",
        "Please explain the conditions for getting my funds returned.",
        "I sent the item back and would like the amount credited.",
        "How long does it usually take for the amount to appear?",
        "The item went back last week; where are my funds?",
    ],
    "SHIPPING": [
        "The destination saved for parcels is out of date.",
        "I need goods sent to my new flat from now on.",
        "Please correct the house number on the destination details.",
        "Can I add a second destination to my profile?",
        "The last parcel went to my old street; please update the details.",
    ],
}

OOD = {
    "off_topic": [
        "Who wrote the play Hamlet?",
        "Explain how a combustion engine works.",
        "Recommend a book about ancient Rome.",
        "How many players are on a football team?",
        "What is the tallest mountain in Africa?",
    ],
    "out_of_scope": [
        "Is this coat available in a medium?",
        "Can you gift wrap the item in blue paper?",
        "Is the blue dress available in a larger size?",
        "Can I reserve a table for dinner on Friday?",
    ],
    "app_failure": [
        "The application shuts down the moment I launch it.",
        "Your website shows a blank page after I sign in.",
        "The checkout freezes when I press continue.",
        "The mobile app crashes every time I add an item.",
    ],
    "gibberish": [
        "zzzz pppp vvvv llll rrrr",
        "flim flam womp zizzle",
        "nnnn kkkk pppp qqqq",
    ],
    "other_language": [
        "Guten Tag, ich möchte meine Bestellung stornieren.",
        "Bonjour, je voudrais une facture s'il vous plaît.",
        "Come posso modificare il mio indirizzo di spedizione?",
        "Mi pedido todavía no ha llegado.",
    ],
}

RISK = [
    ("I am going to come to your office and hurt someone.", "POSSIBLE_THREAT"),
    ("Tell your driver I will attack him if the parcel is late.", "POSSIBLE_THREAT"),
    ("The seller threatened to harm me when I asked for a refund.", "POSSIBLE_THREAT"),
    ("There is a transaction on my statement that I did not authorise.", "POSSIBLE_FRAUD"),
    ("Somebody used my card to buy a laptop.", "POSSIBLE_FRAUD"),
    ("A marketplace seller took my money and never sent the goods.", "POSSIBLE_FRAUD"),
    ("I will take this to court unless the money is returned.", "POSSIBLE_LEGAL_CLAIM"),
    ("My solicitor will send a formal notice tomorrow.", "POSSIBLE_LEGAL_CLAIM"),
    ("You morons have lost my parcel again.", "ABUSE_LANGUAGE"),
    ("This stupid service is a waste of time.", "ABUSE_LANGUAGE"),
    ("There is no threat here; I simply want my parcel.", ""),
    ("The article said someone threatened to attack the driver.", ""),
    ('The film line "I will kill you" was scary.', ""),
    ("How do I report a transaction I did not authorise?", ""),
    ("I have never been scammed by your store.", ""),
    ("The threat of snow delayed the delivery.", ""),
]

MULTI = [
    ("I need a new password and an invoice for last month.",
     "recover_password get_invoice"),
    ("Please cancel my order and update my card.",
     "cancel_order change_payment_method"),
    ("My payment failed and I also want to unsubscribe from the newsletter.",
     "payment_issue unsubscribe_newsletter"),
    ("Where is my parcel, and can you also change my address?",
     "track_order change_shipping_address"),
    ("I want a refund and a copy of my invoice.",
     "get_refund get_invoice"),
    ("Reset my password. Also, please close my account.",
     "recover_password delete_account"),
    ("The checkout keeps failing; please send me a receipt as well.",
     "payment_issue get_invoice"),
    ("Can you track my order and tell me about delivery options?",
     "track_order delivery_options"),
    ("I need to update my address and recover my password.",
     "change_shipping_address recover_password"),
    ("Please send an invoice and explain the refund policy.",
     "get_invoice check_refund_policy"),
    ("I want to speak to a person and file a complaint.",
     "contact_human_agent complaint"),
    ("My card was declined and I need a refund for the earlier order.",
     "payment_issue get_refund"),
]


def id_frame():
    rows = []
    for difficulty, source in (("natural", NATURAL), ("no_keywords", NO_KEYWORDS)):
        for category, texts in source.items():
            rows.extend({"text": text, "category": category, "difficulty": difficulty}
                        for text in texts)
    frame = pd.DataFrame(rows)
    frame.insert(0, "id", range(1, len(frame) + 1))
    frame["role"] = "independent_post_selection_benchmark"
    frame["used_for_training"] = False
    return frame


def ood_frame():
    rows = [{"text": text, "topic": topic} for topic, texts in OOD.items() for text in texts]
    frame = pd.DataFrame(rows)
    frame.insert(0, "id", range(1, len(frame) + 1))
    frame["role"] = "independent_ood"
    frame["used_for_training"] = False
    return frame


def risk_frame():
    frame = pd.DataFrame(RISK, columns=["text", "expected_flags"])
    frame.insert(0, "id", range(1, len(frame) + 1))
    frame["role"] = "independent_risk"
    frame["used_for_training"] = False
    return frame


def multi_frame():
    frame = pd.DataFrame(MULTI, columns=["text", "expected_intents"])
    frame.insert(0, "id", range(1, len(frame) + 1))
    frame["role"] = "independent_multi_request"
    frame["used_for_training"] = False
    return frame


def frames():
    return id_frame(), ood_frame(), risk_frame(), multi_frame()


def canonical_payload():
    return {"version": VERSION, "natural": NATURAL, "no_keywords": NO_KEYWORDS,
            "ood": OOD, "risk": RISK, "multi": MULTI}


def sha256():
    payload = json.dumps(canonical_payload(), ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
