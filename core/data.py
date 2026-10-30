"""Cached data loaders. Everything the pages read comes through here."""
from pathlib import Path

import pandas as pd
import streamlit as st

from core.theme import APPS

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
URG_RANK = {"High": 2, "Medium": 1, "Low": 0}


@st.cache_data(show_spinner="Loading 195,875 reviews…")
def load_reviews() -> pd.DataFrame:
    df = pd.read_parquet(DATA / "app_data.parquet")
    for c in ["reviewId", "app", "content", "sentiment", "sentiment_band", "urgency", "urgency_model",
              "urgency_reason", "aspects", "top_words", "appVersion", "replyContent"]:
        df[c] = df[c].astype(object).where(df[c].notna(), None)
    df["at"] = pd.to_datetime(df["at"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["week"] = df["date"] - pd.to_timedelta(df["date"].dt.weekday, unit="D")
    df["urg_rank"] = df["urgency"].map(URG_RANK).fillna(0).astype(int)
    df["aspect_list"] = df["aspects"].map(lambda s: [a.strip() for a in s.split(",") if a.strip()] if s else [])
    return df


@st.cache_data
def load_csv(name: str, folder: str = "") -> pd.DataFrame:
    return pd.read_csv(DATA / folder / name)


def filtered(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the sidebar filters (apps + date range) stored in session state."""
    apps = st.session_state.get("apps") or APPS
    start, end = st.session_state.get("date_range", (df["date"].min().date(), df["date"].max().date()))
    m = df["app"].isin(apps) & (df["date"].dt.date >= start) & (df["date"].dt.date <= end)
    return df[m]


def selected_apps() -> list:
    return [a for a in APPS if a in (st.session_state.get("apps") or APPS)]
