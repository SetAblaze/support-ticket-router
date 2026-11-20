"""Handwritten scenarios: NEVER training data or evidence of real-world quality."""
import pandas as pd

DEV = {
    "off_topic": [
        "Explain how volcanoes form under the ocean",
        "Give me a recipe for roasted aubergine",
        "Who composed the Moonlight Sonata",
        "How do I train for a marathon next spring",
    ],
    "gibberish": [
        "violet spoon galaxy plink plonk",
        "invoice mango moonwalk dinosaur",
        "qzxv jklp wwww uuuu",
        "account banana castle orbit nonsense",
    ],
    "unsupported_language": [
        "Помогите разобраться с доставкой моего заказа",
        "Bonjour, pouvez vous vérifier mon remboursement",
        "¿Dónde está mi pedido?",
        "أحتاج مساعدة في حسابي",
    ],
    "multiple_requests": [
        "Reset my password and check my refund",
        "I need an invoice and also a refund",
        "Change my shipping address and reset my account password",
        "Track my order and check the refund as well",
    ],
    "technical_issue": [
        "The app crashes as soon as I open it",
        "The checkout page freezes whenever I click continue",
        "Your website does not load on my computer",
        "My browser shows a fatal exception on your website",
    ],
    "insufficient_context": [
        "Something seems wrong but I do not know what",
        "This is completely ridiculous",
        "Please help with that thing from before",
        "I am upset and confused right now",
    ],
}

HOLDOUT = {
    "off_topic": [
        "Describe the rules of competitive chess",
        "What causes the northern lights",
        "Suggest exercises for learning the violin",
        "Calculate the area of a circle with radius seven",
    ],
    "gibberish": [
        "tangerine nebula floppy zigzag aardvark",
        "refund broccoli spaceship trapezoid",
        "zxzx qvqp lmnr tttx",
        "delivery pineapple wizard marzipan",
    ],
    "unsupported_language": [
        "Как восстановить доступ к личному кабинету?",
        "Bonjour, je voudrais une copie de ma facture",
        "¿Cómo recupero mi contraseña?",
        "我的订单还没有到",
    ],
    "multiple_requests": [
        "I forgot my password and my refund still has not arrived",
        "Please send an invoice plus an update on the refund",
        "Update my delivery address and also recover my account",
        "Cancel my order and check the refund status",
    ],
    "technical_issue": [
        "The website crashes every time I open the home page",
        "The app stops responding when I open the catalogue",
        "The checkout page shows a blank screen instead of the form",
        "This browser freezes on your website",
    ],
    "insufficient_context": [
        "It happened again yesterday and I need help",
        "This whole thing is a mess",
        "Can someone explain what just happened",
        "I do not understand any of this",
    ],
}

LEGACY_OOD = [
    "Tell me a short fairy tale about an astronaut cat",
    "Can you predict whether the weekend will be sunny?",
    "What is the best way to learn piano?", "Translate hello into Japanese please",
    "The app vanishes the instant I tap its icon on my phone",
    "My browser shows a fatal exception before your site loads",
    "Everything freezes when I click checkout",
    "Reset my password, return my money, and cancel my order",
    "I have a refund and an invoice issue, need help with both",
    "I want to change my address and update my card",
    "zorp widget banana nebula grumble", "invoice dragon dances inside a cabbage",
    "asdf qwer zxcv hjkl", "Можно объяснить работу магазина на русском языке?",
    "Peut-on payer avec PayPal?", "¿Cómo cambio mi dirección de envío?",
    "This is infuriating nonsense and you are all hopeless",
    "I am furious, worst service ever, no help needed just frustration",
]

DEMO = [
    ("I replaced my phone and cannot remember the security answer", "ACCOUNT"),
    ("Our finance team needs the paperwork proving we paid", "INVOICES"),
    ("My purchase still has not shown up and I want an update", "ORDER"),
    ("The cancelled concert money has not come back yet", "REFUNDS"),
    ("I moved across town; future parcels need to go to the new place", "SHIPPING"),
    ("Please stop the promotional messages that keep arriving", "NEWSLETTER"),
    ("My bank card is refused each time I try to finish", "PAYMENT"),
    ("I would like the details needed to telephone a human", "CONTACT"),
    ("Which ways are available for delivering the ticket?", "DELIVERY"),
    ("Why am I being asked to pay a charge for cancelling?", "CANCELLATION_FEE"),
    ("Can you predict whether the weekend will be sunny?", "OTHER"),
    ("The app vanishes the instant I tap its icon", "OTHER"),
    ("zorp widget banana nebula grumble", "OTHER"),
    ("Можно объяснить работу магазина на русском?", "OTHER"),
    ("Reset my password, return my money, and cancel my order", "OTHER"),
]


def frames():
    def frame(source, role):
        return pd.DataFrame([{"text": t, "topic": topic, "expected_action": "MANUAL",
                              "synthetic": True, "role": role, "training_use": False}
                             for topic, texts in source.items() for t in texts])
    return frame(DEV, "policy_development"), frame(HOLDOUT, "separate_synthetic_diagnostic")
