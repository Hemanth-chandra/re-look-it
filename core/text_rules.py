"""Text cleaning and the keyword urgency rules, copied from the project pipeline so a NEW review
is treated exactly like the 195,875 stored ones.

- clean_text: preprocessing/clean_split.py
- urgency_rules: Cell 3 of 05_relookit_final_pipeline.ipynb (keyword groups + negative review)
- script_type / clean_top: Cell 5 of the same notebook (app clean-up rules)
"""
import re

import emoji
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def clean_text(text: str) -> str:
    t = str(text)
    t = emoji.demojize(t, delimiters=(" ", " "))
    t = t.replace("_", " ")
    t = t.lower()
    t = re.sub(r"https?://\S+|www\.\S+", " url ", t)
    t = re.sub(r"\S+@\S+", " email ", t)
    t = re.sub(r"(.)\1{2,}", r"\1\1", t)
    t = re.sub(r"[^a-z0-9ऀ-ॿ\s'!?.,]", " ", t)
    t = re.sub(r"([!?.,])", r" \1 ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


GROUPS = {
    "money": ["refund", "money", "payment", "deducted", "deduct", "debited", "charged twice",
              "double charged", "wallet", "fraud", "scam", "cheated", "cheat", "paisa", "paise",
              "wapas", "refund nahi", "amount"],
    "order_failure": ["not delivered", "never delivered", "never received", "not received",
                      "didn't receive", "did not receive", "cancelled", "canceled", "missing",
                      "wrong item", "wrong product", "nahi aaya", "nahi mila", "never arrived"],
    "safety": ["expired", "rotten", "insect", "cockroach", "worm", "fungus", "food poisoning",
               "vomit", "hospital", "harass", "harassment", "unsafe", "keeda", "contaminated"],
    "support_failure": ["no response", "not responding", "no reply", "ticket closed", "chat closed",
                        "customer care is useless", "nobody answers", "no one answers",
                        "support is useless", "no help"],
    "escalation": ["consumer court", "consumer forum", "police", "legal action", "fir",
                   "complaint to", "cyber crime"],
}
PATTERNS = {g: re.compile(r"\b(?:" + "|".join(re.escape(w) for w in ws) + r")\b") for g, ws in GROUPS.items()}
REASON_NAMES = {"money": "Money", "order_failure": "Order failure", "safety": "Safety",
                "support_failure": "Support failure", "escalation": "Escalation threat"}


def urgency_rules(cleaned: str, negative: bool, word_count: int):
    """Returns (level, score, reason, matched_groups) exactly as the pipeline rules."""
    hits = {g: bool(p.search(cleaned)) for g, p in PATTERNS.items()}
    long_review = word_count >= 8
    high = negative and (hits["money"] or hits["safety"] or hits["escalation"] or
                         (hits["order_failure"] and hits["support_failure"]))
    any_hit = any(hits.values())
    medium = (not high) and (any_hit or (negative and long_review))
    level = "High" if high else ("Medium" if medium else "Low")
    score = (0.35 * hits["money"] + 0.35 * hits["safety"] + 0.30 * hits["escalation"] +
             0.25 * hits["order_failure"] + 0.20 * hits["support_failure"] +
             0.20 * negative + 0.05 * long_review)
    score = round(min(max(score, 0.0), 1.0), 3)
    parts = [REASON_NAMES[g] for g in GROUPS if hits[g]]
    if not parts:
        reason = "Negative review" if negative else "No issue found"
    else:
        reason = " + ".join(parts)
    return level, score, reason, [g for g in GROUPS if hits[g]]


# ---- clean-up helpers (notebook Cell 5) ----
STOP = {"the", "a", "an", "is", "was", "are", "were", "be", "been", "and", "or", "to", "of", "in", "on", "for",
        "it", "its", "this", "that", "i", "me", "my", "we", "you", "your", "he", "she", "they", "them", "so",
        "very", "but", "with", "at", "as", "by", "from", "have", "has", "had", "do", "did", "not", "no", "hai",
        "ka", "ki", "ke", "ko", "se", "ne", "bhi", "aur", "ho", "tha", "thi"}
BRANDS = {"blinkit", "zepto", "instamart", "bigbasket", "swiggy", "big", "basket", "app", "apps"}
STOP_ALL = set(ENGLISH_STOP_WORDS) | STOP | {"h", "ye", "yeh", "wo", "woh", "kya", "ek", "hi", "toh", "to"}


def clean_top(words):
    keep = []
    for w in words:
        w = w.strip().lower()
        if not w or w in STOP_ALL or w in BRANDS:
            continue
        if re.fullmatch(r"[\d.,:/\-]+", w):
            continue
        if not any(c.isalnum() for c in w):
            continue
        if w not in keep:
            keep.append(w)
    return keep


def _latin_or_dev(ch):
    o = ord(ch)
    return o < 0x0250 or 0x1E00 <= o <= 0x1EFF or 0x0900 <= o <= 0x097F


def _emoji(ch):
    o = ord(ch)
    return o >= 0x1F000 or 0xFE00 <= o <= 0xFE0F or o == 0x200D


def script_type(text: str) -> str:
    t = str(text)
    if t.strip() == "":
        return "empty"
    letters = [c for c in t if c.isalpha()]
    if letters:
        return "latin_or_devanagari" if any(_latin_or_dev(c) for c in letters) else "other_script"
    if any(_emoji(c) for c in t):
        return "emoji_only"
    return "symbols_or_digits_only"
