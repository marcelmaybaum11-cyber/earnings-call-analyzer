"""AI Earnings Call Sentiment Analyzer - Streamlit app.

Paste an earnings call transcript, get a FinBERT sentiment breakdown plus a
predicted direction for the stock's 1-day and 5-day post-call returns, and
see where the transcript falls relative to the historical dataset.

Run from the project root:
    .venv\\Scripts\\streamlit run app.py
"""

import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from sentiment import FinbertScorer

DATASET_CSV = Path("data/processed/dataset.csv")
MODELS_DIR = Path("models")
HORIZONS = {"return_1d": "1 day", "return_5d": "5 days"}

st.set_page_config(page_title="Earnings Call Sentiment Analyzer", layout="wide")


@st.cache_resource(show_spinner="Loading FinBERT model...")
def get_scorer() -> FinbertScorer:
    """Load FinBERT once per server, not once per click."""
    return FinbertScorer()


@st.cache_resource
def get_models() -> dict:
    models = {}
    for horizon in HORIZONS:
        path = MODELS_DIR / f"logreg_{horizon}.joblib"
        if path.exists():
            models[horizon] = joblib.load(path)
    return models


@st.cache_data
def get_history() -> pd.DataFrame | None:
    if not DATASET_CSV.exists():
        return None
    df = pd.read_csv(DATASET_CSV)
    df["net_sentiment"] = df["positive"] - df["negative"]
    return df


st.title("AI Earnings Call Sentiment Analyzer")
st.caption(
    "FinBERT sentiment on earnings call transcripts, correlated with "
    "post-call stock returns."
)

transcript = st.text_area(
    "Paste an earnings call transcript:",
    height=250,
    placeholder="Good afternoon everyone, and thank you for joining us...",
)

if st.button("Analyze", type="primary") and transcript.strip():
    scores = get_scorer().score(transcript)
    net = scores["positive"] - scores["negative"]

    left, right = st.columns(2)

    with left:
        st.subheader("Sentiment breakdown")
        fig = px.bar(
            x=list(scores.values()),
            y=list(scores.keys()),
            orientation="h",
            color=list(scores.keys()),
            color_discrete_map={
                "positive": "#2ca02c",
                "negative": "#d62728",
                "neutral": "#7f7f7f",
            },
            labels={"x": "probability", "y": ""},
        )
        fig.update_layout(showlegend=False, xaxis_tickformat=".0%", height=260)
        st.plotly_chart(fig, use_container_width=True)
        st.metric("Net sentiment (positive - negative)", f"{net:+.1%}")

    with right:
        st.subheader("Predicted return direction")
        models = get_models()
        if not models:
            st.info(
                "No trained models found. Run "
                "`.venv\\Scripts\\python.exe src\\model.py` first."
            )
        for horizon, label in HORIZONS.items():
            if horizon not in models:
                continue
            model = models[horizon]
            features = [[scores["positive"], scores["negative"]]]
            prob_up = float(model.predict_proba(features)[0][1])
            direction = "UP" if prob_up >= 0.5 else "DOWN"
            confidence = prob_up if prob_up >= 0.5 else 1 - prob_up
            st.metric(
                f"{label} after the call",
                direction,
                f"{confidence:.0%} confidence",
                delta_color="normal" if direction == "UP" else "inverse",
            )
        st.caption(
            "Educational project - this is a statistical exercise, "
            "not investment advice."
        )

    history = get_history()
    if history is not None:
        st.subheader("Where this call sits in the historical data")
        tab1, tab5 = st.tabs([f"{label} return" for label in HORIZONS.values()])
        for tab, horizon in zip((tab1, tab5), HORIZONS):
            with tab:
                fig = px.scatter(
                    history.dropna(subset=[horizon]),
                    x="net_sentiment",
                    y=horizon,
                    color="ticker",
                    hover_data=["call_date"],
                    trendline="ols",
                    trendline_scope="overall",
                    labels={"net_sentiment": "net sentiment", horizon: "return"},
                )
                fig.add_vline(
                    x=net,
                    line_dash="dash",
                    line_color="black",
                    annotation_text="this transcript",
                )
                fig.update_layout(yaxis_tickformat=".0%", height=450)
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(
            "No historical dataset yet. Run "
            "`.venv\\Scripts\\python.exe src\\build_dataset.py` to create it."
        )
