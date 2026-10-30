"""The final multi-task model (v3) for live predictions on new reviews.

model.pt (~496 MB) is too big for a normal GitHub commit, so it is attached to a GitHub Release.
On first start the app downloads it once into model/ and keeps the model in memory.
Set the link in MODEL_URL below, or as a Streamlit secret called MODEL_URL.
"""
import json
import math
import os
import urllib.request
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

from core.keyword_aspects import keyword_aspects
from core.text_rules import clean_text, clean_top, script_type, urgency_rules

# >>> EDIT THIS LINE after creating the GitHub Release (see README) <<<
MODEL_URL = "https://github.com/YOUR-GITHUB-USERNAME/relookit-app/releases/download/v1.0/model.pt"

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
WEIGHTS = MODEL_DIR / "model.pt"
SETTINGS = json.loads((MODEL_DIR / "settings.json").read_text())
SENT_LABELS = SETTINGS["sentiment_labels"]
URG_LABELS = SETTINGS["urgency_labels"]
ASPECTS = SETTINGS["aspects"]
MAX_LEN = SETTINGS["max_length"]


class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states, attention_mask):
        scores = self.score(hidden_states).squeeze(-1)
        scores = scores.masked_fill(attention_mask == 0, -1e4)
        weights = torch.softmax(scores.float(), dim=-1)
        pooled = torch.bmm(weights.unsqueeze(1).to(hidden_states.dtype), hidden_states).squeeze(1)
        return pooled, weights


class MultiTaskModel(nn.Module):
    """Same layout as the notebook, so model.pt loads with all keys matching."""

    def __init__(self, n_aspects, dropout=0.3):
        super().__init__()
        self.encoder = AutoModel.from_config(AutoConfig.from_pretrained(MODEL_DIR), add_pooling_layer=False)
        hidden = self.encoder.config.hidden_size
        self.pool = AttentionPooling(hidden)
        self.dropout = nn.Dropout(dropout)
        self.sent_head = nn.Linear(hidden, 3)
        self.urg_head = nn.Linear(hidden, 3)
        self.asp_head = nn.Linear(hidden, n_aspects)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled, attn = self.pool(out.last_hidden_state, attention_mask)
        x = self.dropout(pooled)
        return self.sent_head(x), self.urg_head(x), self.asp_head(x), attn


def _model_url() -> str:
    try:
        if "MODEL_URL" in st.secrets:
            return st.secrets["MODEL_URL"]
    except Exception:
        pass
    return os.environ.get("MODEL_URL", MODEL_URL)


def _download(url: str, dest: Path):
    tmp = dest.with_suffix(".part")
    bar = st.progress(0.0, text="Downloading model weights (about 500 MB, first start only)…")
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                bar.progress(min(done / total, 1.0), text=f"Downloading model weights… {done >> 20} / {total >> 20} MB")
    tmp.replace(dest)
    bar.empty()


@st.cache_resource(show_spinner="Loading the model…")
def load_model():
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    if not WEIGHTS.exists():
        _download(_model_url(), WEIGHTS)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = MultiTaskModel(len(ASPECTS), dropout=SETTINGS.get("dropout", 0.3))
    state = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return tok, model


def _band(conf: float) -> str:
    b = SETTINGS["confidence_bands"]
    return "High" if conf >= b["high_min"] else ("Medium" if conf >= b["medium_min"] else "Low")


def _why_words(tok, model, ids, mask, sent_idx, k=3):
    """Gradient x input on the final sentiment class + predicted urgency (notebook grad_top_words)."""
    emb = model.encoder.get_input_embeddings()
    embeds = emb(ids).detach().float().requires_grad_(True)
    out = model.encoder(inputs_embeds=embeds, attention_mask=mask)
    pooled, _ = model.pool(out.last_hidden_state, mask)
    s = model.sent_head(pooled)
    u = model.urg_head(pooled)
    target = s[0, sent_idx] + u[0, u[0].argmax()]
    model.zero_grad()
    target.backward()
    scores = (embeds.grad * embeds).sum(-1).abs()[0] * mask[0]
    tokens = tok.convert_ids_to_tokens(ids[0].tolist())
    words, wscores = [], []
    for t, sc in zip(tokens, scores.tolist()):
        if t in ("<s>", "</s>", "<pad>"):
            continue
        if t.startswith("Ġ") or not words:
            words.append([t]); wscores.append(sc)
        else:
            words[-1].append(t); wscores[-1] += sc
    ranked = [tok.convert_tokens_to_string(words[i]).strip().lower() for i in np.argsort(wscores)[::-1]]
    ranked = [w.strip(".,!?;:\"'()[]…“”‘’") for w in ranked]
    return clean_top(ranked)[:k]


def analyse(text: str, stars: int | None = None, likes: int = 0) -> dict:
    """Full deployed pipeline for one review: sentiment (T + neutral bias + band), urgency
    max(model, rules), aspects model OR keyword, why-words."""
    tok, model = load_model()
    enc = tok([text], truncation=True, padding="max_length", max_length=MAX_LEN, return_tensors="pt")
    ids, mask = enc["input_ids"], enc["attention_mask"]
    with torch.no_grad():
        s, u, a, _ = model(ids, mask)
    logits = s[0].float() / SETTINGS["sentiment_temperature"]
    logits[1] += SETTINGS["neutral_bias"]
    sp = torch.softmax(logits, -1).numpy()
    up = torch.softmax(u[0].float(), -1).numpy()
    ap = torch.sigmoid(a[0].float()).numpy()

    sent_idx = int(sp.argmax())
    sentiment = SENT_LABELS[sent_idx]
    conf = float(sp.max())
    band = _band(conf)
    stype = script_type(text)
    if stype in ("other_script", "symbols_or_digits_only"):
        band = "Low"

    words = len(str(text).split())
    cleaned = clean_text(text)
    negative = (stars <= 2) if stars else (sentiment == "negative")
    r_level, r_score, r_reason, _ = urgency_rules(cleaned, negative, words)
    m_idx = int(up.argmax())
    r_idx = URG_LABELS.index(r_level)
    urgency = URG_LABELS[max(m_idx, r_idx)]
    base = max(float(up[2] + 0.5 * up[1]), r_score)
    score = round(min(1.0, base + 0.05 * math.log1p(max(likes, 0))), 3)

    thr = SETTINGS["aspect_thresholds"]
    model_asp = {asp for j, asp in enumerate(ASPECTS) if ap[j] > thr[asp]}
    kw_asp = keyword_aspects(cleaned) if words >= 3 else set()
    aspects = [asp for asp in ASPECTS if asp in (model_asp | kw_asp)] if words >= 3 else []

    if m_idx > r_idx:
        reason = "Model: urgent wording"
    elif r_reason == "Negative review":
        reason = ("Negative: " + ", ".join(aspects)) if aspects else "Negative, no specific issue"
    else:
        reason = r_reason

    why = _why_words(tok, model, ids, mask, sent_idx)
    return {
        "sentiment": sentiment, "confidence": conf, "band": band, "sent_probs": dict(zip(SENT_LABELS, sp.tolist())),
        "urgency": urgency, "urgency_model": URG_LABELS[m_idx], "urgency_rules": r_level, "urgency_score": score,
        "urgency_reason": reason, "urg_probs": dict(zip(URG_LABELS, up.tolist())),
        "aspects": aspects, "aspects_model": sorted(model_asp), "aspects_keyword": sorted(kw_asp),
        "asp_probs": dict(zip(ASPECTS, ap.tolist())), "why_words": why, "script": stype,
        "truncated": int(mask.sum()) >= MAX_LEN, "word_count": words,
    }
