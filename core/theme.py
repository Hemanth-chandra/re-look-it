"""Warm "Kirana Store" theme: colours, CSS and small HTML helpers shared by every page."""
import html

import plotly.graph_objects as go
import streamlit as st

# Palette
BG = "#FAF6EF"        # warm sand page
CARD = "#FFFFFF"
SAND = "#F3EADB"      # secondary surface
LINE = "#EADFCD"
TEXT = "#2B241C"
MUTED = "#6B5E4E"
SAFFRON = "#BF5B16"   # main accent (buttons, selected)
SAFFRON_SOFT = "#F6E3CF"
MUSTARD = "#A97F12"   # Medium urgency
CHILI = "#C0392B"     # High urgency / negative
BASIL = "#3E7D3A"     # positive
NEUTRAL = "#8C7F6E"

APPS = ["BigBasket", "Blinkit", "Instamart", "Zepto"]
APP_COLORS = {"BigBasket": "#2A6FC9", "Blinkit": "#D9602A", "Instamart": "#14935F", "Zepto": "#C08A00"}
SENT_COLORS = {"negative": CHILI, "neutral": NEUTRAL, "positive": BASIL}
URG_COLORS = {"High": CHILI, "Medium": MUSTARD, "Low": NEUTRAL}
BAND_COLORS = {"High": BASIL, "Medium": MUSTARD, "Low": NEUTRAL}

ASPECTS = ["delivery_order", "refund_payment", "product_quality", "missing_wrong_item",
           "app_tech", "customer_support", "pricing_offers", "safety"]
ASPECT_NAMES = {
    "delivery_order": "Delivery / order", "refund_payment": "Refund / payment",
    "product_quality": "Product quality", "missing_wrong_item": "Missing / wrong item",
    "app_tech": "App / tech", "customer_support": "Customer support",
    "pricing_offers": "Pricing / offers", "safety": "Safety",
}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,700&display=swap');
h1, h2, h3 {{ font-family: 'Bricolage Grotesque', 'Source Sans Pro', sans-serif !important; letter-spacing: -0.01em; color: {TEXT}; }}
h1 {{ font-weight: 700 !important; }}
[data-testid="stSidebar"] {{ background: {SAND}; border-right: 1px solid {LINE}; }}
.rl-kpi {{ background:{CARD}; border:1px solid {LINE}; border-radius:14px; padding:14px 16px; display:flex; flex-direction:column; gap:6px; height:100%; }}
.rl-kpi .top {{ display:flex; justify-content:space-between; align-items:center; font-weight:600; }}
.rl-kpi .big {{ font-family:'Bricolage Grotesque',sans-serif; font-size:34px; font-weight:700; line-height:1.05; }}
.rl-kpi .big small {{ font-family:inherit; font-size:13px; font-weight:400; color:{MUTED}; margin-left:4px; }}
.rl-kpi .foot {{ display:flex; justify-content:space-between; font-size:13px; color:{MUTED}; gap:8px; }}
.rl-meter {{ height:6px; background:{SAND}; border-radius:3px; overflow:hidden; }}
.rl-meter i {{ display:block; height:100%; border-radius:3px; }}
.rl-dot {{ display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:7px; vertical-align:middle; }}
.rl-badge {{ display:inline-block; font-size:12px; font-weight:700; padding:2px 9px; border-radius:999px; border:1.5px solid currentColor; white-space:nowrap; }}
.rl-chip {{ display:inline-block; font-size:13px; padding:3px 10px; margin:2px 4px 2px 0; border-radius:999px; background:{SAND}; border:1px solid {LINE}; color:{TEXT}; }}
.rl-quote {{ background:{BG}; border:1px solid {LINE}; border-radius:10px; padding:12px 14px; font-size:16px; line-height:1.6; color:{TEXT}; }}
.rl-quote mark {{ background:#F9D9B8; color:inherit; border-radius:3px; padding:0 2px; box-shadow: inset 0 -2px 0 {SAFFRON}; }}
.rl-muted {{ color:{MUTED}; font-size:14px; }}
.rl-card {{ background:{CARD}; border:1px solid {LINE}; border-radius:14px; padding:16px 18px; height:100%; }}
.rl-card h4 {{ margin:0 0 4px 0; font-family:'Bricolage Grotesque',sans-serif; font-size:20px; }}
.rl-big {{ font-family:'Bricolage Grotesque',sans-serif; font-size:30px; font-weight:700; line-height:1.1; text-transform:capitalize; }}
.rl-hero {{ font-family:'Bricolage Grotesque',sans-serif; font-size:64px; font-weight:700; line-height:1; color:{APP_COLORS['Zepto']}; }}
.rl-row {{ display:flex; gap:12px; align-items:center; padding:10px 0; border-top:1px solid {LINE}; }}
.rl-cal {{ width:54px; flex:none; text-align:center; background:{SAND}; border-radius:10px; padding:5px 0; line-height:1.15; }}
.rl-cal b {{ font-family:'Bricolage Grotesque',sans-serif; font-size:19px; display:block; }}
.rl-cal span {{ font-size:12px; color:{MUTED}; }}
</style>
"""


def apply_theme():
    st.markdown(CSS, unsafe_allow_html=True)


def esc(s) -> str:
    return html.escape(str(s))


def dot(app: str) -> str:
    return f'<span class="rl-dot" style="background:{APP_COLORS.get(app, NEUTRAL)}"></span>'


def badge(text: str, color: str) -> str:
    return f'<span class="rl-badge" style="color:{color}">{esc(text)}</span>'


def chips(items) -> str:
    return "".join(f'<span class="rl-chip">{esc(i)}</span>' for i in items)


def aspect_label(key: str) -> str:
    return ASPECT_NAMES.get(key.strip(), key.strip())


def nice_reason(reason: str) -> str:
    """'Negative: delivery_order, safety' -> 'Negative: Delivery / order, Safety'."""
    r = str(reason or "")
    for k, v in ASPECT_NAMES.items():
        r = r.replace(k, v)
    return r


def style_fig(fig: go.Figure, height: int = 340) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=10, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT, size=13), hoverlabel=dict(bgcolor=CARD, font_color=TEXT, bordercolor=LINE),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title_text=""),
    )
    fig.update_xaxes(gridcolor=LINE, linecolor=LINE, zeroline=False)
    fig.update_yaxes(gridcolor=LINE, linecolor=LINE, zeroline=False)
    return fig
