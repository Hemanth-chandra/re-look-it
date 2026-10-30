"""Keyword aspect flags for a new review: the same spaCy lexicon matcher used to build aspect_flags.csv
(aspects/extract_aspects.py). Only reviews with 3+ words are tagged, as in the pipeline."""
import spacy
import streamlit as st
from spacy.matcher import PhraseMatcher

from core.lexicon import LEXICON

APP_WORDS = {"app", "application", "update", "load", "loading", "website", "ui", "screen", "otp"}
SPEED_WORDS = {"slow", "late", "fast", "quick", "delay"}
APP_TECH_CTX = {"work", "open", "load", "slow", "hang", "stuck", "freeze", "update", "glitch",
                "error", "crash", "bug", "lag", "login", "otp", "install", "uninstall"}
QTY_PROBLEM = {"less", "short", "wrong", "missing", "low", "poor", "small", "reduce", "cheat", "lesser",
               "bad", "issue", "problem"}
REFUND_CTX = {"refund", "deduct", "twice", "double", "extra", "debit", "back"}


@st.cache_resource(show_spinner="Loading the keyword matcher…")
def _matcher():
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    lemma = PhraseMatcher(nlp.vocab, attr="LEMMA")
    raw = PhraseMatcher(nlp.vocab, attr="LOWER")
    for aspect, terms in LEXICON.items():
        lemma.add(aspect, list(nlp.pipe(terms)))
        raw.add(aspect, [nlp.make_doc(t) for t in terms])
    return nlp, lemma, raw


def keyword_aspects(cleaned_text: str) -> set:
    nlp, lemma, raw = _matcher()
    doc = nlp(cleaned_text)
    found = {nlp.vocab.strings[mid] for mid, _, _ in list(lemma(doc)) + list(raw(doc))}
    lemmas = {t.lemma_.lower() for t in doc}
    if lemmas & SPEED_WORDS:
        found.add("app_tech" if lemmas & APP_WORDS else "delivery_order")
    if lemmas & {"app", "application"} and lemmas & APP_TECH_CTX:
        found.add("app_tech")
    if "quantity" in lemmas and lemmas & QTY_PROBLEM:
        found.add("missing_wrong_item")
    if "charge" in lemmas:
        found.add("refund_payment" if lemmas & REFUND_CTX else "pricing_offers")
    return found
