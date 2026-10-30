"""Re-look-it: review intelligence for Indian quick-commerce apps (Streamlit entry point)."""
import streamlit as st

from core.data import load_reviews
from core.theme import APPS, apply_theme

st.set_page_config(page_title="Re-look-it", page_icon="🛒", layout="wide", initial_sidebar_state="expanded")
apply_theme()

pages = {
    "Insights": [
        st.Page("app_pages/overview.py", title="Overview", icon=":material/dashboard:", default=True),
        st.Page("app_pages/competitors.py", title="Competitor comparison", icon=":material/bar_chart:"),
    ],
    "Act on reviews": [
        st.Page("app_pages/queue.py", title="Priority queue", icon=":material/priority_high:"),
        st.Page("app_pages/analyse.py", title="Analyse a review", icon=":material/search:"),
    ],
    "Method": [
        st.Page("app_pages/about.py", title="About the model", icon=":material/model_training:"),
    ],
}
nav = st.navigation(pages)

df = load_reviews()
dmin, dmax = df["date"].min().date(), df["date"].max().date()

with st.sidebar:
    st.markdown("### Re-look-it")
    st.caption("Google Play reviews of four 10-minute grocery apps · 1 Jul – 30 Sep 2026")
    st.pills("Apps", APPS, selection_mode="multi", default=APPS, key="apps")
    st.slider("Dates", min_value=dmin, max_value=dmax, value=(dmin, dmax), format="D MMM", key="date_range")
    st.caption("Filters apply to Overview, Competitor comparison and Priority queue.")
    st.divider()
    st.caption("Model: multi-task Twitter-RoBERTa (v3) · sentiment, urgency, 8 issue types. "
               "Built by Rasee Singh and S Hemanth Chandra, IIITM Gwalior.")

nav.run()
