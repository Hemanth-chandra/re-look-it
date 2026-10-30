import re

import pandas as pd
import streamlit as st

from core.data import filtered, load_reviews
from core.theme import (ASPECT_NAMES, ASPECTS, CHILI, MUSTARD, SENT_COLORS, URG_COLORS, aspect_label, badge, chips,
                        dot, esc, nice_reason)

st.title("Priority queue")
st.markdown('<p class="rl-muted">Reviews that need a person. Highest urgency level first, then urgency score. The score '
            'rises with likes, so widely seen complaints come first.</p>', unsafe_allow_html=True)

df = filtered(load_reviews())
q = df[df["urgency"].isin(["High", "Medium"])]

c1, c2, c3, c4 = st.columns([1.2, 1.4, 2, 1])
levels = c1.pills("Urgency", ["High", "Medium"], selection_mode="multi", default=["High", "Medium"], key="q_levels")
issue = c2.selectbox("Issue", ["All issues"] + [ASPECT_NAMES[a] for a in ASPECTS], key="q_issue")
search = c3.text_input("Search review text", placeholder="e.g. expired, refund, atta", key="q_search")
unreplied = c4.toggle("Unreplied only", key="q_unreplied")

if levels:
    q = q[q["urgency"].isin(levels)]
if issue != "All issues":
    key = [k for k, v in ASPECT_NAMES.items() if v == issue][0]
    q = q[q["aspect_list"].map(lambda xs: key in xs)]
if search.strip():
    q = q[q["content"].fillna("").str.contains(re.escape(search.strip()), case=False)]
if unreplied:
    q = q[~q["has_reply"]]
q = q.sort_values(["urg_rank", "urgency_score", "thumbsUpCount"], ascending=False)

m1, m2, m3, m4 = st.columns(4)
m1.metric("In queue", f"{len(q):,}")
m2.metric("High", f"{(q['urgency'] == 'High').sum():,}")
m3.metric("Medium", f"{(q['urgency'] == 'Medium').sum():,}")
m4.metric("No reply from the app", f"{(~q['has_reply']).sum():,}")

if q.empty:
    st.info("No reviews match these filters. Clear the search or add an urgency level.")
    st.stop()

view = q.head(2000).copy()
view["Issues"] = view["aspect_list"].map(lambda xs: ", ".join(aspect_label(a) for a in xs))
view["Reason"] = view["urgency_reason"].map(nice_reason)
view["Replied"] = view["has_reply"]
table = view[["urgency", "urgency_score", "app", "date", "score", "content", "Reason", "Issues", "thumbsUpCount", "Replied"]]
table = table.rename(columns={"urgency": "Urgency", "urgency_score": "Score", "app": "App", "date": "Date",
                              "score": "Stars", "content": "Review", "thumbsUpCount": "Likes"})

left, right = st.columns([3, 2], gap="medium")
with left:
    event = st.dataframe(
        table, hide_index=True, use_container_width=True, height=560, on_select="rerun", selection_mode="single-row",
        key="q_table",
        column_config={
            "Score": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1, width="small"),
            "Date": st.column_config.DateColumn(format="D MMM"),
            "Stars": st.column_config.NumberColumn(format="%d★", width="small"),
            "Review": st.column_config.TextColumn(width="large"),
            "Likes": st.column_config.NumberColumn(width="small"),
            "Replied": st.column_config.CheckboxColumn(width="small"),
        })
    st.caption(f"Showing the top {len(view):,} of {len(q):,}. Click a row to read it in full."
               if len(q) > len(view) else "Click a row to read it in full.")
    st.download_button("Download this queue as CSV", q.drop(columns=["aspect_list", "week", "urg_rank"]).to_csv(index=False),
                       file_name="relookit_priority_queue.csv", mime="text/csv")

rows = event.selection.rows if event and event.selection else []
r = view.iloc[rows[0]] if rows else view.iloc[0]


def highlight(text: str, words) -> str:
    ws = {w.strip().lower() for w in words if w.strip()}
    out = []
    for tok in re.split(r"(\s+)", str(text)):
        core = re.sub(r"^\W+|\W+$", "", tok.lower())
        out.append(f"<mark>{esc(tok)}</mark>" if core and core in ws else esc(tok))
    return "".join(out)


with right:
    with st.container(border=True):
        why = [w for w in str(r["top_words"] or "").split(", ") if w]
        st.markdown(f'{badge(r["urgency"] + " urgency · " + format(r["urgency_score"], ".2f"), URG_COLORS[r["urgency"]])} '
                    f'&nbsp;{dot(r["app"])}<b>{esc(r["app"])}</b> · {r["date"]:%d %b %Y} · {"★" * int(r["score"])}'
                    f' · ▲ {int(r["thumbsUpCount"]):,} likes', unsafe_allow_html=True)
        st.markdown(f'<div class="rl-quote" style="margin:10px 0">{highlight(r["content"], why)}</div>',
                    unsafe_allow_html=True)
        st.markdown(f"**Why urgent:** {nice_reason(r['urgency_reason'])}")
        st.markdown(f'**Sentiment:** <span style="color:{SENT_COLORS.get(r["sentiment"], "#000")};font-weight:700">'
                    f'{r["sentiment"]}</span> · {r["sentiment_band"]} confidence ({r["sentiment_conf"]:.2f}) · '
                    f'model urgency {r["urgency_model"]}', unsafe_allow_html=True)
        st.markdown("**Why-words:** " + (" ".join(f"<mark>{esc(w)}</mark>" for w in why) or "—"), unsafe_allow_html=True)
        st.markdown("**Issues:** " + (chips(aspect_label(a) for a in r["aspect_list"]) or "None tagged"),
                    unsafe_allow_html=True)
        if r["has_reply"] and r["replyContent"]:
            st.markdown("**App's reply**" + (f" · after {r['reply_minutes']:.0f} min" if pd.notna(r["reply_minutes"]) and r["reply_minutes"] >= 0 else ""))
            st.markdown(f'<div class="rl-quote" style="font-size:14px">{esc(r["replyContent"])}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<b style="color:{CHILI}">No reply from the app yet.</b>', unsafe_allow_html=True)
