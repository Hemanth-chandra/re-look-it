import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.data import load_csv, selected_apps
from core.theme import (APP_COLORS, ASPECT_NAMES, ASPECTS, CHILI, LINE, MUTED, NEUTRAL, SAFFRON, TEXT, dot, esc,
                        style_fig)

st.title("Competitor comparison")
st.markdown('<p class="rl-muted">Each app against the other three: overall, by complaint type, by day and by how it '
            'replies. Whole quarter (the date filter does not apply here, the confidence intervals were '
            'computed for the full period).</p>', unsafe_allow_html=True)

apps = selected_apps()
src = st.segmented_control("Sentiment source", ["Model", "Star ratings (1–2★ = negative)"], default="Model",
                           key="cmp_src") or "Model"
folder = "model_tables" if src == "Model" else "star_tables"
gaps = load_csv("competitor_gaps_bootstrap.csv", folder).set_index("app")
summ = load_csv("competitor_summary.csv", folder).set_index("app")
mix = load_csv("competitor_aspect_mix.csv", folder)

# ---------- headline ----------
z = gaps.loc["Zepto"]
other = "Star ratings" if src == "Model" else "the model"
alt = load_csv("competitor_gaps_bootstrap.csv", "star_tables" if src == "Model" else "model_tables").set_index("app").loc["Zepto"]
with st.container(border=True):
    a, b = st.columns([1, 2], gap="large")
    a.markdown(f'<span class="rl-muted" style="font-weight:600;letter-spacing:.06em">ZEPTO VS THE OTHER THREE</span>'
               f'<div class="rl-hero">{z["ratio"]:.2f}×</div>'
               f'<span class="rl-muted">as likely to be negative · 95% CI {z["ratio_ci_low"]:.2f}–{z["ratio_ci_high"]:.2f}</span>',
               unsafe_allow_html=True)
    b.markdown(f'<p style="font-size:17px;margin-top:8px">Zepto\'s negative share is <b>{z["negative_share"] * 100:.1f}%</b> '
               f'against <b>{z["other_three"] * 100:.1f}%</b> for the rest (+{z["gap_pp"]:.1f} pp). The gap holds in every '
               f'month and every week, so it is a steady level, not a few bad days.</p>'
               f'<p class="rl-muted">With {other} instead: {alt["ratio"]:.2f}× (CI {alt["ratio_ci_low"]:.2f}–{alt["ratio_ci_high"]:.2f}). '
               f'Both sources agree.</p>', unsafe_allow_html=True)

# ---------- summary table ----------
with st.container(border=True):
    st.markdown("#### Summary by app")
    t = pd.DataFrame({
        "App": apps,
        "Reviews": [int(summ.loc[a, "reviews"]) for a in apps],
        "Mean ★": [summ.loc[a, "mean_stars"] for a in apps],
        "Negative share": [gaps.loc[a, "negative_share"] * 100 for a in apps],
        "95% CI": [f'{gaps.loc[a, "ci_low"] * 100:.1f}–{gaps.loc[a, "ci_high"] * 100:.1f}%' for a in apps],
        "Gap vs others (pp)": [gaps.loc[a, "gap_pp"] for a in apps],
        "Ratio": [gaps.loc[a, "ratio"] for a in apps],
        "Negative, 8+ words": [summ.loc[a, "negative_share_long_reviews"] * 100 for a in apps],
    })
    st.dataframe(t, hide_index=True, use_container_width=True, column_config={
        "Mean ★": st.column_config.NumberColumn(format="%.2f"),
        "Negative share": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=50),
        "Gap vs others (pp)": st.column_config.NumberColumn(format="%+.1f"),
        "Ratio": st.column_config.NumberColumn(format="%.2f×"),
        "Negative, 8+ words": st.column_config.NumberColumn(format="%.1f%%"),
    })
    st.caption("Confidence intervals come from a bootstrap over the 92 days, because reviews on the same day share the "
               "same events. Long reviews are much more negative: unhappy customers write more.")

# ---------- aspect heatmap ----------
with st.container(border=True):
    h1, h2 = st.columns([2, 1])
    h1.markdown("#### What people complain about")
    h1.markdown('<p class="rl-muted">Negative reviews of 3+ words</p>', unsafe_allow_html=True)
    view = h2.segmented_control("Value", ["Complaint rate", "Gap vs other three"], default="Complaint rate",
                                key="cmp_heat", label_visibility="collapsed") or "Complaint rate"
    m = mix[mix["app"].isin(apps)]
    rate = m.pivot(index="app", columns="aspect", values="complaint_rate").reindex(index=apps, columns=ASPECTS) * 100
    gap = m.pivot(index="app", columns="aspect", values="gap_pp").reindex(index=apps, columns=ASPECTS)
    labels = [ASPECT_NAMES[a] for a in ASPECTS]
    if view == "Complaint rate":
        z_, txt, cs, zmin, zmax, fmt = rate.values, rate.map(lambda v: f"{v:.0f}%").values, \
            [[0, "#FFF8EE"], [1, "#B5531A"]], 0, 50, "%{z:.1f}%"
    else:
        z_, txt, cs, zmin, zmax, fmt = gap.values, gap.map(lambda v: f"{v:+.1f}").values, \
            [[0, "#2A6FC9"], [0.5, "#FFF8EE"], [1, CHILI]], -18, 18, "%{z:+.1f} pp"
    fig = go.Figure(go.Heatmap(z=z_, x=labels, y=apps, text=txt, texttemplate="%{text}", colorscale=cs, zmin=zmin,
                               zmax=zmax, xgap=3, ygap=3, colorbar=dict(thickness=10, outlinewidth=0),
                               hovertemplate="%{y} · %{x}: " + fmt + "<extra></extra>"))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, side="top")
    st.plotly_chart(style_fig(fig, 300), use_container_width=True)

    cols = st.columns(len(apps))
    for col, app in zip(cols, apps):
        g = gap.loc[app].sort_values(ascending=False)
        top, val = g.index[0], g.iloc[0]
        r = rate.loc[app, top]
        title = ASPECT_NAMES[top] if val >= 3 else "No clear weakness"
        body = (f"{r:.0f}% of complaints vs {r - val:.0f}% for the others (+{val:.1f} pp)" if val >= 3 else
                f"Below or close to the others on most issues; highest gap {ASPECT_NAMES[top].lower()} +{val:.1f} pp")
        col.markdown(f'<div class="rl-card"><span class="rl-muted" style="font-size:13px;font-weight:600">{dot(app)}{app} · '
                     f'biggest weakness</span><h4>{esc(title)}</h4><span class="rl-muted">{esc(body)}</span></div>',
                     unsafe_allow_html=True)

# ---------- daily spikes ----------
with st.container(border=True):
    st.markdown("#### Daily negative share and spike days")
    st.markdown('<p class="rl-muted">Spike = robust z ≥ 3 against the surrounding 14 days, at least +5 pp and 100 '
                'reviews that day. Model sentiment.</p>', unsafe_allow_html=True)
    daily = load_csv("spikes_daily_all.csv", "model_tables")
    daily["date"] = pd.to_datetime(daily["date"])
    flagged = load_csv("spikes_daily_flagged.csv", "model_tables")
    flagged["date"] = pd.to_datetime(flagged["date"])
    app = st.segmented_control("App", apps, default=("Zepto" if "Zepto" in apps else apps[0]), key="cmp_daily") or apps[0]
    d = daily[daily["app"] == app]
    s = d[d["is_spike"]]
    fig = go.Figure()
    fig.add_scatter(x=d["date"], y=d["baseline"] * 100, name="Usual level (14-day median)", mode="lines",
                    line=dict(color=MUTED, dash="dash", width=1.5), hovertemplate="%{x|%d %b}: usual %{y:.1f}%<extra></extra>")
    fig.add_scatter(x=d["date"], y=d["negative_share"] * 100, name="Daily negative share", mode="lines",
                    line=dict(color=APP_COLORS[app], width=2), customdata=d["reviews"],
                    hovertemplate="%{x|%d %b}: %{y:.1f}% negative · %{customdata:,} reviews<extra></extra>")
    fig.add_scatter(x=s["date"], y=s["negative_share"] * 100, name="Spike day", mode="markers",
                    marker=dict(color=CHILI, size=12, line=dict(color="white", width=2)),
                    hovertemplate="%{x|%d %b}: spike day, %{y:.1f}%<extra></extra>")
    fig.update_yaxes(ticksuffix="%", rangemode="tozero")
    fig.update_xaxes(tickformat="%d %b")
    st.plotly_chart(style_fig(fig, 320), use_container_width=True)
    f = flagged[flagged["app"] == app]
    if f.empty:
        st.caption(f"{app} had no spike day this quarter.")
    for _, r in f.iterrows():
        with st.expander(f"{r['date']:%d %b} · {r['negative_share'] * 100:.0f}% negative vs {r['baseline'] * 100:.0f}% usual "
                         f"· +{int(r['extra_negative_reviews'])} negative reviews"):
            st.markdown(f"**Issues up:** {r['aspects_up']}  \n**Main app version that day:** {r['top_version_that_day']}")
            for line in str(r["evidence_top_liked_negative_reviews"]).split(" || "):
                st.markdown(f'<div class="rl-quote" style="font-size:14px;margin-bottom:6px">{esc(line)}</div>',
                            unsafe_allow_html=True)

# ---------- versions ----------
with st.container(border=True):
    st.markdown("#### App versions")
    st.markdown('<p class="rl-muted">Each version (300+ reviews) against other versions on the same days, so outages '
                'that hit everyone cancel out. A bar right of zero means more negative than other versions.</p>',
                unsafe_allow_html=True)
    v = load_csv("spikes_versions.csv", "model_tables")
    v = v[v["app"] == app].sort_values("first_seen")
    fig = go.Figure(go.Scatter(
        x=v["gap_pp"], y=v["version"], mode="markers",
        marker=dict(color=APP_COLORS[app], size=10),
        error_x=dict(type="data", symmetric=False, array=v["gap_ci_high_pp"] - v["gap_pp"],
                     arrayminus=v["gap_pp"] - v["gap_ci_low_pp"], color=MUTED, thickness=1.5),
        customdata=np.stack([v["reviews"], v["first_seen"]], axis=1),
        hovertemplate="Version %{y}<br>Gap %{x:+.1f} pp<br>%{customdata[0]:,} reviews · first seen %{customdata[1]}<extra></extra>"))
    fig.add_vline(x=0, line_color=TEXT, line_width=1)
    fig.update_xaxes(ticksuffix=" pp", title="Negative share vs other versions on the same days")
    fig.update_yaxes(type="category", autorange="reversed")
    st.plotly_chart(style_fig(fig, max(220, 34 * len(v) + 80)), use_container_width=True)
    st.caption("No version is 5+ pp worse than others on the same days. Wording for the report: a version "
               "\"coincides with\" a change, never \"caused\" it, because early updaters differ from other users.")

# ---------- replies ----------
with st.container(border=True):
    st.markdown("#### How each app replies")
    r = load_csv("response_summary_by_app.csv").set_index("app")
    rr = load_csv("response_reply_rate_by_star.csv").set_index("score")
    notes = {"BigBasket": "Steady ~14 min replies at every star level",
             "Blinkit": "3★ reviews wait about 58 h; a praise template reaches 4% of complaints",
             "Instamart": "Fastest overall",
             "Zepto": "Replies only to complaints, with one email-redirect template and no apology"}
    t = pd.DataFrame({
        "App": apps,
        "Reply rate 1★": [rr.loc[1, a] * 100 for a in apps],
        "Reply rate 5★": [rr.loc[5, a] * 100 for a in apps],
        "Median reply (min)": [r.loc[a, "median_min"] for a in apps],
        "Within 1 h": [r.loc[a, "within_1h"] * 100 for a in apps],
        "Apologises (1–2★)": [r.loc[a, "apologise"] * 100 for a in apps],
        "Distinct replies": [int(r.loc[a, "distinct_masked"]) for a in apps],
        "What stands out": [notes[a] for a in apps],
    })
    st.dataframe(t, hide_index=True, use_container_width=True, column_config={
        "Reply rate 1★": st.column_config.NumberColumn(format="%.1f%%"),
        "Reply rate 5★": st.column_config.NumberColumn(format="%.1f%%"),
        "Median reply (min)": st.column_config.NumberColumn(format="%.0f"),
        "Within 1 h": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100),
        "Apologises (1–2★)": st.column_config.NumberColumn(format="%.1f%%"),
        "What stands out": st.column_config.TextColumn(width="large"),
    })
    w1 = load_csv("response_within_1h_by_star.csv").set_index("score")
    fig = go.Figure()
    for a in apps:
        fig.add_scatter(x=[f"{i}★" for i in w1.index], y=w1[a] * 100, name=a, mode="lines+markers",
                        line=dict(color=APP_COLORS[a], width=2.5), hovertemplate=a + " · %{x}: %{y:.0f}% within 1 h<extra></extra>")
    fig.update_yaxes(ticksuffix="%", range=[0, 102], title="Replied within 1 hour")
    st.plotly_chart(style_fig(fig, 280), use_container_width=True)
    st.caption("Reply speed excludes reviews edited after the reply. Star-based: replies do not depend on the model.")
