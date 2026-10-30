"""
Shared definitions (agreed Day 1): aspect list, aspect lexicon, urgency v0 keywords.
Imported by aspects/extract_aspects.py and labelling/draw_sample.py.
"""

# Fixed aspect list (8). Order = order shown in labelling sheets and dashboard.
ASPECTS = [
    "delivery_order",      # speed, late, not delivered, order cancelled, delivery partner
    "refund_payment",      # refund, money, payment failed, charged twice, wallet, cashback
    "product_quality",     # fresh, rotten, expired, damaged, quality, stale
    "missing_wrong_item",  # missing item, wrong item, less quantity
    "app_tech",            # app, bug, crash, login, otp, update, ui, location
    "customer_support",    # customer care, support, chat, agent, no response
    "pricing_offers",      # price, expensive, fees, handling/delivery charge, offer, coupon
    "safety",              # hygiene, insect, food poisoning, unsafe, harassment, allergic
]

# Lemma-level terms (single words) and phrases. Matched on spaCy LEMMA, lower-cased.
# Hinglish entries are matched as raw tokens (spaCy has no Hindi lemmas).
LEXICON = {
    "delivery_order": [
        "delivery", "deliver", "late", "delay", "delayed", "rider", "partner", "minute",
        "eta", "dispatch", "arrive", "cancel", "cancellation", "slot",
        "on time", "not deliver", "deliver late", "delivery boy", "delivery partner",
        "delivery time", "10 minute", "10 min", "fast delivery", "quick delivery", "order cancel",
        "jaldi", "der", "late se",
    ],
    "refund_payment": [
        "refund", "money", "payment", "pay", "paid", "charge twice", "deduct", "debit",
        "wallet", "cashback", "upi", "transaction", "amount", "return", "replacement", "credit",
        "cod", "cash on delivery", "money back", "refund not", "paisa", "paise", "wapas",
    ],
    "product_quality": [
        "quality", "fresh", "freshness", "rotten", "expired", "expiry", "expire", "stale",
        "damage", "damaged", "broken", "spoil", "spoiled", "leak", "leaked", "smell", "fungus",
        "fungal", "mould", "mold", "sour", "spoilt", "kharab", "sada",  # item names (milk, bread...) removed 30 Oct: a name alone is not a complaint
        "sadi", "ganda", "packaging", "packing",
    ],
    "missing_wrong_item": [
        "missing", "wrong item", "wrong product", "different item", "item missing",
        "not receive", "less quantity", "low quantity", "short quantity", "wrong quantity", "less weight",
        "incomplete", "wrong order",  # "quantity" alone removed 30 Oct (rule in extract_aspects.py)
        "miss",
    ],
    "app_tech": [
        # "app" alone is NOT here: "worst app" is a general opinion, not a tech issue.
        # It counts only with a technical context word (rule in extract_aspects.py).
        "bug", "crash", "glitch", "login", "log in", "otp", "ui", "interface", "loading",
        "hang", "error", "server", "notification", "checkout", "button", "screen", "website",
        "slow app", "app update", "not working", "not open", "gps", "change address",
        "add address", "location access", "payment page", "technical issue", "technical",
    ],
    "customer_support": [
        "support", "customer care", "costumer care", "costomer care", "coustomer care", "customer service",
        "costumer service", "customer support", "care team", "chat",  # "care" alone removed 30 Oct ("take care")
        "agent", "executive", "helpline", "complaint", "complain", "response", "respond",
        "reply", "call", "ticket", "escalate", "bot", "chatbot",
    ],
    "pricing_offers": [
        "price", "pricing", "expensive", "costly", "cheap", "cheaper", "fee",
        "handling fee", "delivery fee", "delivery charge", "platform fee", "surge", "charge",
        "offer", "discount", "coupon", "deal", "mrp", "membership", "minimum order",
        "mehenga", "mehnga", "sasta",
    ],
    "safety": [
        "hygiene", "unhygienic", "insect", "cockroach", "worm", "hair", "food poisoning",
        # 30 Oct: "ill" ("ill only say") and bare "sick" ("sick of this") removed
        "poisoning", "fell sick", "got sick", "made me sick", "became sick", "got ill", "fell ill",
        "became ill", "vomit",
        "allergy", "allergic", "unsafe", "safety",
        "harass", "harassment", "misbehave", "threat", "abuse", "fake product",
        "duplicate product", "contaminate", "contaminated", "keeda", "rat", "dusty",
    ],
}

# Urgency v0 keyword groups (used only to STRATIFY the labelling sample;
# Rasee owns the real urgency score on Day 6).
URGENCY_MONEY = ["refund", "money", "payment", "deduct", "debit", "charged", "paisa", "paise",
                 "wapas", "fraud", "scam", "cheat", "cheated", "chor"]
URGENCY_ORDER = ["not deliver", "not received", "never received", "cancel", "missing",
                 "wrong item", "wrong product", "late", "delay", "expired", "damaged", "rotten"]
URGENCY_SAFETY = ["insect", "cockroach", "worm", "food poisoning", "sick", "vomit", "allerg",
                  "harass", "misbehav", "threat", "abuse", "unsafe", "hospital", "fungus"]
