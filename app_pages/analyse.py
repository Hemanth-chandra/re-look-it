import re

import plotly.graph_objects as go
import streamlit as st

from core.theme import (ASPECT_NAMES, BAND_COLORS, BASIL, CHILI, LINE, MUSTARD, NEUTRAL, SAFFRON, SENT_COLORS,
                        URG_COLORS, aspect_label, chips, esc, nice_reason, style_fig)

st.title("Analyse a review")
st.markdown('<p class="rl-muted">Paste any review. The model returns sentiment with a confidence band, urgency with '
            'its reason, the issues it mentions and the words that drove the result.</p>', unsafe_allow_html=True)

EXAMPLES = {
    "Expired milk, no refund": "bbdaily people deliver the expired milk packets. They also don't refund the money for the spoiled and expired milk packets.",
    "Missing item, support useless": "Zepto denied my refund despite a missing item. Extremely disappointing service and no response from customer care.",
    "Hinglish": "bilkul bekar app hai, expire saman bheja aur paisa wapas nahi mila",
    "Happy customer": "superb fast delivery, fresh vegetables, love it",
    "Fees complaint": "night fare, delivery fees and platform fees, 75rs extra on a small order. too expensive now",
}

if "an_text" not in st.session_state:
    st.session_state.an_text = EXAMPLES["Expired milk, no refund"]


def use_example():
    pick = st.session_state.get("an_example")
    if pick:
        st.session_state.an_text = EXAMPLES[pick]


st.pills("Try an example", list(EXAMPLES), key="an_example", on_change=use_example)
text = st.text_area("Review text", key="an_text", height=120, max_chars=2000)
c1, c2, c3 = st.columns([1, 1, 3])
stars = c1.selectbox("Star rating", ["Not given", 1, 2, 3, 4, 5], help="Used by the keyword rules, as in the pipeline. "
                     "If not given, the model's own sentiment decides whether the review counts as negative.")
likes = c2.number_input("Likes", min_value=0, value=0, step=1, help="Likes raise the urgency score a little.")
run = c3.button("Analyse", type="primary")

if not text.strip():
    st.info("Paste a review above or pick an example.")
    st.stop()

if run or "an_result" not in st.session_state or st.session_state.get("an_last") != (text, stars, likes):
    from core.model import analyse  # loads torch only when this page is used
    try:
        with st.spinner("Reading the review…"):
            res = analyse(text, None if stars == "Not given" else int(stars), int(likes))
    except Exception as e:  # e.g. model link not set yet
        st.error(f"The model could not be loaded: {e}. Check MODEL_URL in core/model.py or the MODEL_URL secret.")
        st.stop()
    st.session_state.an_result = res
    st.session_state.an_last = (text, stars, likes)
res = st.session_state.an_result


def highlight(t: str, words) -> str:
    ws = {w.lower() for w in words}
    return "".join(f"<mark>{esc(tok)}</mark>" if re.sub(r"^\W+|\W+$", "", tok.lower()) in ws else esc(tok)
                   for tok in re.split(r"(\s+)", t))


a, b, c = st.columns(3, gap="medium")
with a:
    bc = BAND_COLORS[res["band"]]
    acc = {"High": "97%", "Medium": "87%", "Low": "56%"}[res["band"]]
    st.markdown(f'<div class="rl-card"><span class="rl-muted" style="font-weight:600">SENTIMENT</span>'
                f'<div class="rl-big" style="color:{SENT_COLORS[res["sentiment"]]}">{res["sentiment"]}</div>'
                f'<div class="rl-meter" style="height:8px;margin:10px 0 6px"><i style="width:{res["confidence"] * 100:.0f}%;background:{bc}"></i></div>'
                f'<span class="rl-muted">{res["confidence"]:.3f} confidence · <b style="color:{bc}">{res["band"]} band</b> '
                f'({acc} correct on validation)</span></div>', unsafe_allow_html=True)
with b:
    uc = URG_COLORS[res["urgency"]]
    st.markdown(f'<div class="rl-card"><span class="rl-muted" style="font-weight:600">URGENCY</span>'
                f'<div class="rl-big" style="color:{uc}">{res["urgency"]}</div>'
                f'<p style="margin:8px 0 4px">{esc(nice_reason(res["urgency_reason"]))}</p>'
                f'<span class="rl-muted">Score {res["urgency_score"]:.2f} · model {res["urgency_model"]}, rules '
                f'{res["urgency_rules"]} → the higher one is used</span></div>', unsafe_allow_html=True)
with c:
    tags = chips(aspect_label(x) for x in res["aspects"]) if res["aspects"] else (
        '<span class="rl-chip">Too short to tag (under 3 words)</span>' if res["word_count"] < 3 else
        '<span class="rl-chip">None found</span>')
    st.markdown(f'<div class="rl-card"><span class="rl-muted" style="font-weight:600">ISSUES MENTIONED</span>'
                f'<div style="margin:10px 0">{tags}</div>'
                f'<span class="rl-muted">Model (tuned thresholds) or keyword match, 8 issue types</span></div>',
                unsafe_allow_html=True)

st.write("")
with st.container(border=True):
    st.markdown("**Words that drove the result**")
    st.markdown(f'<div class="rl-quote">{highlight(text, res["why_words"])}</div>', unsafe_allow_html=True)
    st.caption("Gradient × input attribution on the final sentiment class and the predicted urgency.")
    notes = []
    if res["truncated"]:
        notes.append("This review is longer than 64 tokens, so the model only read the first part.")
    if res["script"] in ("other_script", "symbols_or_digits_only"):
        notes.append("The text is not in Latin or Devanagari script, so the confidence band is set to Low.")
    for n in notes:
        st.warning(n)

with st.expander("Model details: all probabilities"):
    d1, d2 = st.columns(2)
    fig = go.Figure(go.Bar(x=list(res["sent_probs"].values()), y=[k.capitalize() for k in res["sent_probs"]], orientation="h",
                           marker_color=[SENT_COLORS[k] for k in res["sent_probs"]], texttemplate="%{x:.2f}"))
    fig.update_xaxes(range=[0, 1], title="Sentiment probability (after T = 0.61 and neutral bias)")
    d1.plotly_chart(style_fig(fig, 200), use_container_width=True)
    fig = go.Figure(go.Bar(x=list(res["urg_probs"].values()), y=list(res["urg_probs"]), orientation="h",
                           marker_color=[URG_COLORS[k] for k in res["urg_probs"]], texttemplate="%{x:.2f}"))
    fig.update_xaxes(range=[0, 1], title="Model urgency probability")
    d2.plotly_chart(style_fig(fig, 200), use_container_width=True)
    from core.model import SETTINGS
    thr = SETTINGS["aspect_thresholds"]
    names = [ASPECT_NAMES[k] for k in res["asp_probs"]]
    fig = go.Figure()
    fig.add_bar(x=names, y=list(res["asp_probs"].values()), name="Model probability",
                marker_color=[SAFFRON if res["asp_probs"][k] > thr[k] else "#E8D6BE" for k in res["asp_probs"]])
    fig.add_scatter(x=names, y=[thr[k] for k in res["asp_probs"]], mode="markers", name="Threshold (tuned on validation)",
                    marker=dict(symbol="line-ew", size=26, line=dict(width=3, color="#2B241C")))
    fig.update_yaxes(range=[0, 1])
    st.plotly_chart(style_fig(fig, 260), use_container_width=True)
    st.caption(f"Keyword matches: {', '.join(aspect_label(x) for x in res['aspects_keyword']) or 'none'} · "
               f"model above threshold: {', '.join(aspect_label(x) for x in res['aspects_model']) or 'none'}.")
