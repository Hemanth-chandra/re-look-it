from pathlib import Path

import pandas as pd
import streamlit as st

from core.data import load_csv

ROOT = Path(__file__).resolve().parents[1]

st.title("About the model")
st.markdown('<p class="rl-muted">One multi-task model, trained once on 33,188 unique review texts, gives every output '
            'in this app.</p>', unsafe_allow_html=True)

st.markdown("""
#### How it works
- **Encoder:** `cardiffnlp/twitter-roberta-base-sentiment-latest` (RoBERTa-base, already tuned on tweets), reviews cut at 64 tokens.
- **Attention pooling** over the token vectors, then **three heads**: sentiment (3 classes), urgency (Low / Medium / High) and 8 issue types (multi-label).
- **Training labels:** star ratings for sentiment, keyword rules for urgency, the spaCy keyword lexicon for issues (weak supervision).
- **After training, all chosen on validation:** temperature T = 0.61 and a neutral bias of −0.9; confidence bands; one threshold per issue type.
- **Deployed rules:** urgency = the higher of the model and the keyword rules; issues = model **or** keyword; why-words = gradient × input.
- **Human check:** 400 reviews from the test set, AI pre-labelled and checked by one annotator, used only for the final evaluation and never for tuning.
""")

ba = load_csv("before_after.csv")


def val(metric, version):
    s = ba[(ba["metric"] == metric) & (ba["version"] == version)]["value"]
    return float(s.iloc[0]) if len(s) else None


st.markdown("#### Results on the 400 human-checked reviews (weighted)")
c1, c2 = st.columns(2, gap="medium")
with c1, st.container(border=True):
    st.markdown("**Sentiment**")
    t = pd.DataFrame({
        "Source": ["Star ratings", "Model v1", "Final model"],
        "Macro-F1": [val("human macro-F1 (weighted)", v) for v in ["star labels", "model v1", "model final"]],
        "Accuracy": [val("human accuracy (weighted)", v) for v in ["star labels", "model v1", "model final"]],
        "Negative → neutral": [val("human-negative predicted neutral (count)", v) for v in ["star labels", "model v1", "model final"]],
    })
    st.dataframe(t, hide_index=True, use_container_width=True, column_config={
        "Macro-F1": st.column_config.NumberColumn(format="%.3f"), "Accuracy": st.column_config.NumberColumn(format="%.3f"),
        "Negative → neutral": st.column_config.NumberColumn(format="%d")})
    st.caption("The neutral fix halved negative reviews wrongly called neutral (36 → 18).")
with c2, st.container(border=True):
    st.markdown("**Urgency**")
    t = pd.DataFrame({
        "Source": ["Model", "Keyword rules", "max(model, rules) · used"],
        "Macro-F1": [val("human macro-F1 (weighted)", v) for v in ["model", "rules", "max(model, rules)"]],
        "High F1": [val("human High F1 (weighted)", v) for v in ["model", "rules", "max(model, rules)"]],
    })
    st.dataframe(t, hide_index=True, use_container_width=True, column_config={
        "Macro-F1": st.column_config.NumberColumn(format="%.3f"), "High F1": st.column_config.NumberColumn(format="%.3f")})
    st.caption("Where model and rules disagree (41 reviews), the model is right 20 times and the rules 13.")
c3, c4 = st.columns(2, gap="medium")
with c3, st.container(border=True):
    st.markdown("**Issue types**")
    t = pd.DataFrame({
        "Source": ["Keywords", "Model", "Model or keyword · used"],
        "Mean F1": [val("human mean F1 (weighted)", v) for v in ["keywords", "model", "old OR (deployed)"]],
    })
    st.dataframe(t, hide_index=True, use_container_width=True,
                 column_config={"Mean F1": st.column_config.NumberColumn(format="%.3f")})
    st.caption("0.706 is optimistic: the OR rule was chosen after seeing the human labels.")

with c4, st.container(border=True):
    st.markdown("**Confidence bands (validation)**")
    cb = load_csv("confidence_bands.csv")
    st.dataframe(cb.rename(columns={"band": "Band", "val_accuracy": "Accuracy", "val_share": "Share of validation",
                                    "all_reviews_share": "Share of all reviews"}),
                 hide_index=True, use_container_width=True,
                 column_config={c: st.column_config.NumberColumn(format="%.3f") for c in
                                ["Accuracy", "Share of validation", "Share of all reviews"]})
    st.caption("ECE 0.156 → 0.039 after temperature scaling and the neutral bias.")

d1, d2 = st.columns(2, gap="medium")
with d1, st.container(border=True):
    st.markdown("**Ablation on validation (v1 → v3)**")
    mv = load_csv("model_versions.csv")
    st.dataframe(mv.rename(columns={"version": "Version", "val_f1_sent": "Sentiment F1", "val_f1_urg": "Urgency F1",
                                    "val_f1_asp": "Issues F1", "val_score": "Mean"}),
                 hide_index=True, use_container_width=True,
                 column_config={c: st.column_config.NumberColumn(format="%.3f") for c in
                                ["Sentiment F1", "Urgency F1", "Issues F1", "Mean"]})
    st.caption("Aspect positive weights (v3) gave the biggest jump, mostly on the rarer issue types.")
with d2, st.container(border=True):
    st.markdown("**Training curves (v3)**")
    st.image(str(ROOT / "assets/loss_curves_v3.png"), use_container_width=True)
    st.caption("Validation loss falls for all three heads from epoch 1 to 2, so the model is not overfitting.")

with st.container(border=True):
    st.markdown("#### Limitations")
    st.markdown("""
- Labels are AI pre-labelled and checked by one annotator (self-agreement κ: sentiment 0.94, urgency 0.57, issues 0.58).
- Urgency and issue heads learned from rule and keyword labels, so they partly copy them.
- 17 of the 71 human-urgent reviews (24%) are missed by both the model and the rules.
- On sentiment the model is about as accurate as star ratings; its value is that it works on text alone, explains itself and catches mild complaints hidden in 3★ reviews.
- Only 17 human-neutral reviews; safety detection is moderate; 10.5% of texts are longer than 64 tokens and get cut.
- The encoder was already tuned for sentiment, so part of the gain is transfer learning.
- Reviews in other Indian scripts (289) and heavy Hinglish are handled poorly; their confidence is set to Low.
- Spikes and app versions show *when* complaints rose and *what* they mention, not *why*.
""")
