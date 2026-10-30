import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.data import filtered, load_csv, load_reviews, selected_apps
from core.theme import (APP_COLORS, ASPECT_NAMES, ASPECTS, BAND_COLORS, BASIL, CHILI, LINE, MUTED, NEUTRAL,
                        SENT_COLORS, dot, esc, style_fig)

st.title("Overview")
st.markdown('<p class="rl-muted">How customers feel about each app, week by week. Sentiment comes from the model, '
            'not the star rating.</p>', unsafe_allow_html=True)

df = filtered(load_reviews())
apps = selected_apps()
if df.empty:
    st.info("No reviews in this date range. Widen the dates in the sidebar.")
    st.stop()

# ---------- KPI cards ----------
cols = st.columns(len(apps))
for col, app in zip(cols, apps):
    d = df[df["app"] == app]
    if d.empty:
        col.markdown(f'<div class="rl-kpi"><div class="top"><span>{dot(app)}{app}</span></div>'
                     f'<span class="rl-muted">No reviews in range</span></div>', unsafe_allow_html=True)
        continue
    neg = (d["sentiment"] == "negative").mean() * 100
    high = (d["urgency"] == "High").mean() * 100
    col.markdown(
        f'<div class="rl-kpi"><div class="top"><span>{dot(app)}{app}</span><span class="rl-muted">★ {d["score"].mean():.2f}</span></div>'
        f'<div class="big">{neg:.1f}%<small>negative</small></div>'
        f'<div class="rl-meter"><i style="width:{min(neg * 2, 100):.1f}%;background:{APP_COLORS[app]}"></i></div>'
        f'<div class="foot"><span>{len(d):,} reviews</span><span>High urgency <b style="color:{CHILI}">{high:.1f}%</b></span></div></div>',
        unsafe_allow_html=True)

st.write("")

# ---------- weekly trend ----------
left, right = st.columns([2, 1], gap="medium")
with left:
    with st.container(border=True):
        metric = st.segmented_control("Show", ["Negative share", "High urgency share", "Review volume"],
                                      default="Negative share", key="ov_metric", label_visibility="collapsed")
        metric = metric or "Negative share"
        g = df.groupby(["week", "app"])
        weekly = pd.DataFrame({
            "Negative share": g["sentiment"].apply(lambda s: (s == "negative").mean() * 100),
            "High urgency share": g["urgency"].apply(lambda s: (s == "High").mean() * 100),
            "Review volume": g.size(),
        }).reset_index()
        st.markdown(f"#### Weekly {metric.lower()}")
        fig = go.Figure()
        for app in apps:
            w = weekly[weekly["app"] == app]
            fig.add_scatter(x=w["week"], y=w[metric], name=app, mode="lines+markers",
                            line=dict(color=APP_COLORS[app], width=2.5), marker=dict(size=6),
                            hovertemplate="%{x|Week of %d %b}<br>" + app + ": %{y:,.1f}" +
                                          ("%" if metric != "Review volume" else " reviews") + "<extra></extra>")
        fig.update_yaxes(ticksuffix="%" if metric != "Review volume" else "", rangemode="tozero")
        fig.update_xaxes(tickformat="%d %b")
        st.plotly_chart(style_fig(fig, 330), use_container_width=True)
        st.caption("Weeks start on Monday. The last week of September is partial.")

with right:
    with st.container(border=True):
        st.markdown("#### How sure is the model?")
        st.markdown('<p class="rl-muted">Confidence bands, cut-offs chosen on validation</p>', unsafe_allow_html=True)
        bands = load_csv("confidence_bands.csv")
        live = df["sentiment_band"].value_counts(normalize=True) * 100
        for _, b in bands.iterrows():
            share = live.get(b["band"], 0.0)
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;font-size:14px"><b>{b["band"]}</b>'
                f'<span class="rl-muted">{share:.0f}% of these reviews</span></div>'
                f'<div class="rl-meter" style="height:10px"><i style="width:{share:.1f}%;background:{BAND_COLORS[b["band"]]}"></i></div>'
                f'<p class="rl-muted" style="font-size:13px;margin:4px 0 12px">{b["val_accuracy"] * 100:.0f}% correct on validation</p>',
                unsafe_allow_html=True)
        st.caption("Calibration error (ECE) on validation fell from 0.156 to 0.039 after temperature scaling and the "
                   "neutral fix. Reviews in other scripts are always Low.")

# ---------- sentiment mix + issues ----------
c1, c2 = st.columns(2, gap="medium")
with c1:
    with st.container(border=True):
        st.markdown("#### Sentiment mix")
        mix = (df.groupby("app")["sentiment"].value_counts(normalize=True).mul(100).rename("share").reset_index())
        fig = go.Figure()
        for s in ["negative", "neutral", "positive"]:
            m = mix[mix["sentiment"] == s].set_index("app").reindex(apps)
            fig.add_bar(y=apps, x=m["share"], name=s.capitalize(), orientation="h", marker_color=SENT_COLORS[s],
                        text=m["share"].map(lambda v: f"{v:.0f}%" if v >= 6 else ""), textposition="inside",
                        hovertemplate="%{y}: %{x:.1f}% " + s + "<extra></extra>")
        fig.update_layout(barmode="stack")
        fig.update_xaxes(ticksuffix="%", range=[0, 100])
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(style_fig(fig, 260), use_container_width=True)

with c2:
    with st.container(border=True):
        st.markdown("#### What negative reviews mention")
        neg = df[(df["sentiment"] == "negative") & (df["aspect_list"].str.len() > 0)]
        if neg.empty:
            st.caption("No tagged negative reviews in this selection.")
        else:
            ex = neg[["app", "aspect_list"]].explode("aspect_list")
            tot = neg.groupby("app").size()
            rate = ex.groupby(["app", "aspect_list"]).size().div(tot, level="app").mul(100).rename("rate").reset_index()
            fig = go.Figure()
            for app in apps:
                r = rate[rate["app"] == app].set_index("aspect_list").reindex(ASPECTS).fillna(0)
                fig.add_bar(x=[ASPECT_NAMES[a] for a in ASPECTS], y=r["rate"], name=app, marker_color=APP_COLORS[app],
                            hovertemplate=app + " · %{x}: %{y:.1f}% of tagged complaints<extra></extra>")
            fig.update_layout(barmode="group")
            fig.update_yaxes(ticksuffix="%")
            st.plotly_chart(style_fig(fig, 260), use_container_width=True)
            st.caption("Share of negative reviews (3+ words, with an issue tagged) that mention each issue.")

# ---------- spike days ----------
with st.container(border=True):
    st.markdown("#### Spike days")
    st.markdown('<p class="rl-muted">Days well above the app\'s usual negative share. Details on the Competitor page.</p>',
                unsafe_allow_html=True)
    sp = load_csv("spikes_daily_flagged.csv", "model_tables")
    sp["date"] = pd.to_datetime(sp["date"])
    sp = sp[sp["app"].isin(apps)].sort_values("date")
    rows = ""
    for _, r in sp.iterrows():
        rows += (f'<div class="rl-row"><div class="rl-cal"><b>{r["date"].day}</b><span>{r["date"]:%b}</span></div>'
                 f'<div style="flex:1;min-width:0"><b>{dot(r["app"])}{esc(r["app"])} · {r["negative_share"] * 100:.0f}% negative '
                 f'vs {r["baseline"] * 100:.0f}% usual</b><br><span class="rl-muted">{esc(str(r["aspects_up"]).split(";")[0])} · '
                 f'+{int(r["extra_negative_reviews"])} negative reviews</span></div></div>')
    if "Blinkit" in apps:
        rows += '<p class="rl-muted" style="border-top:1px solid #EADFCD;padding-top:10px">Blinkit had no spike day this quarter.</p>'
    st.markdown(rows or '<p class="rl-muted">No spike days for the selected apps.</p>', unsafe_allow_html=True)
