"""Earnings Call Tone & Post-Call Return Analyzer - Streamlit app.

Paste or upload an earnings call transcript, get a FinBERT sentiment
breakdown with live processing details, a predicted direction for the
stock's 1-day and 5-day post-call returns, an optional price chart with a
statistical 5-day projection, and a view of where the call sits relative
to the historical dataset of 188 real earnings calls.

Run from the project root:
    .venv\\Scripts\\streamlit run app.py
"""

import sys
import time
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from fetch_returns import download_close_prices
from sentiment import MAX_CHUNK_TOKENS, FinbertScorer

DATASET_CSV = Path("data/processed/dataset.csv")
MODELS_DIR = Path("models")
FEATURED_DIR = Path("data/featured")
HORIZONS = {"return_1d": "1 day", "return_5d": "5 days"}

st.set_page_config(
    page_title="Earnings Call Tone Analyzer",
    page_icon="📈",
    layout="wide",
)

# Design polish: centered content column, gradient hero, pill-style tabs,
# full-width primary button. Kept in one place so it is easy to tweak.
st.markdown(
    """
    <style>
    .block-container { max-width: 1100px; padding-top: 2.2rem; }
    .hero-title {
        font-size: 2.7rem; font-weight: 800; line-height: 1.15;
        background: linear-gradient(90deg, #60a5fa, #34d399);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .hero-sub { color: #94a3b8; font-size: 1.02rem; margin-bottom: 1.4rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 0.5rem; }
    .stTabs [data-baseweb="tab"] {
        background: rgba(148, 163, 184, 0.12); border-radius: 999px;
        padding: 0.35rem 1.1rem;
    }
    .stTabs [aria-selected="true"] { background: #3b82f6; }
    .stTabs [aria-selected="true"] p { color: #ffffff; }
    .stTabs [data-baseweb="tab-highlight"],
    .stTabs [data-baseweb="tab-border"] { display: none; }
    .stButton > button[kind="primary"] {
        border-radius: 10px; padding: 0.55rem 1rem; font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------- caching


@st.cache_resource(show_spinner="Loading FinBERT (first time only)...")
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


@st.cache_data(ttl=3600, show_spinner=False)
def get_recent_prices(ticker: str) -> pd.Series:
    today = pd.Timestamp.today().normalize()
    return download_close_prices(ticker, today - pd.Timedelta(days=180), today)


@st.cache_data
def get_featured() -> pd.DataFrame | None:
    manifest = FEATURED_DIR / "manifest.csv"
    if not manifest.exists():
        return None
    return pd.read_csv(manifest)


# ----------------------------------------------------------------- sidebar

history = get_history()
models = get_models()

with st.sidebar:
    st.header("About this tool")
    st.markdown(
        "Scores the tone of an earnings call transcript with "
        "[FinBERT](https://huggingface.co/ProsusAI/finbert), a language "
        "model trained on financial text, then predicts the direction of "
        "the stock's post-call move with a model trained on real "
        "historical calls."
    )
    if history is not None:
        st.divider()
        st.subheader("Training data")
        c1, c2 = st.columns(2)
        c1.metric("Earnings calls", len(history))
        c2.metric("Companies", history["ticker"].nunique())
        st.caption(
            f"{history['call_date'].min()} to {history['call_date'].max()} - "
            "large-cap tech (AAPL, MSFT, NVDA, AMZN, GOOGL, AMD, INTC, "
            "CSCO, MU, ASML)."
        )
        st.divider()
        st.subheader("Validated performance")
        st.markdown(
            "- 1-day direction: **60.6%** CV accuracy (baseline 55.9%)\n"
            "- 5-day direction: **63.4%** CV accuracy (baseline 53.7%)\n"
            "- Sentiment-return correlation significant at p < 0.01"
        )
    st.divider()
    st.caption(
        "🌙 Dark theme is the default - switch to light mode via the "
        "menu (top right) → Settings → Theme."
    )
    st.caption(
        "Educational project - a statistical exercise, not investment advice."
    )

# ------------------------------------------------------------------- input

st.markdown(
    '<div class="hero-title">Earnings Call Tone Analyzer</div>'
    '<div class="hero-sub">FinBERT sentiment on any earnings call '
    "transcript, with a return-direction prediction validated on 188 real "
    "calls.</div>",
    unsafe_allow_html=True,
)

paste_tab, upload_tab, recent_tab = st.tabs(
    ["✏️ Paste text", "📂 Upload file", "⚡ Sample calls"]
)
with paste_tab:
    pasted = st.text_area(
        "Transcript text",
        height=220,
        placeholder="Good afternoon everyone, and thank you for joining us...",
        label_visibility="collapsed",
    )
with upload_tab:
    uploaded = st.file_uploader(
        "Drag & drop a transcript file (.txt / .md)", type=["txt", "md"]
    )

featured_row = None
featured_text = ""
featured_choice = "— select a call —"
with recent_tab:
    featured = get_featured()
    if featured is None:
        st.info(
            "No featured calls downloaded yet. Run "
            "`.venv\\Scripts\\python.exe src\\fetch_featured.py` first."
        )
    else:
        labels = [
            f"{r.company} ({r.ticker}) — {r.quarter} · {r.call_date}"
            for r in featured.itertuples()
        ]
        featured_choice = st.selectbox(
            "Pick a recent call - it is analyzed automatically:",
            ["— select a call —", *labels],
        )
        if featured_choice != "— select a call —":
            featured_row = featured.iloc[labels.index(featured_choice)]
            featured_text = (FEATURED_DIR / featured_row["file"]).read_text(
                encoding="utf-8", errors="replace"
            )
            st.caption(
                f"**{featured_row['company']} {featured_row['quarter']}** "
                f"({len(featured_text):,} characters) — source: "
                f"[The Motley Fool]({featured_row['source']})"
            )

transcript = ""
if featured_row is not None:
    transcript = featured_text
elif uploaded is not None:
    transcript = uploaded.read().decode("utf-8", errors="replace")
    st.caption(f"Loaded **{uploaded.name}** ({len(transcript):,} characters)")
elif pasted.strip():
    transcript = pasted

analyze = st.button("Analyze transcript", type="primary", use_container_width=True)

# The price chart needs a ticker, which we only know for sample calls.
ticker = str(featured_row["ticker"]) if featured_row is not None else ""

# A sample call analyzes itself as soon as it is picked - no extra click.
auto_run = (
    featured_row is not None
    and st.session_state.get("last_featured") != featured_choice
)
if auto_run:
    st.session_state["last_featured"] = featured_choice

# ---------------------------------------------------------------- analysis

if (analyze or auto_run) and transcript.strip():
    scorer = get_scorer()

    with st.status("Analyzing transcript...", expanded=True) as status:
        t0 = time.time()

        st.write("**Step 1 - Tokenizing** the transcript with BERT's tokenizer...")
        chunks = scorer._chunks(transcript)
        n_tokens = sum(len(c) for c in chunks)
        st.write(
            f"→ {len(transcript):,} characters → **{n_tokens:,} tokens**, "
            f"split into **{len(chunks)} chunks** of ≤{MAX_CHUNK_TOKENS} "
            "tokens (BERT's hard limit is 512 per pass)."
        )

        st.write("**Step 2 - Scoring** every chunk through FinBERT...")
        bar = st.progress(0.0, text=f"0 / {len(chunks)} chunks")
        scores = scorer.score(
            transcript,
            on_progress=lambda done, total: bar.progress(
                done / total, text=f"{done} / {total} chunks"
            ),
        )
        net = scores["positive"] - scores["negative"]
        st.write(
            "**Step 3 - Aggregating**: chunk scores combined with a "
            f"length-weighted average → net sentiment **{net:+.1%}**."
        )

        prices = None
        if ticker:
            st.write(f"**Step 4 - Fetching** {ticker} price history...")
            try:
                prices = get_recent_prices(ticker)
                st.write(f"→ {len(prices)} trading days retrieved.")
            except Exception as exc:
                st.write(f"→ ⚠️ could not fetch prices: {exc}")

        status.update(
            label=f"Analysis complete in {time.time() - t0:.1f}s",
            state="complete",
            expanded=False,
        )

    # ------------------------------------------------------------- results

    st.divider()
    left, mid, right = st.columns([1.2, 1, 1])

    with left:
        st.subheader("Sentiment breakdown")
        fig = px.bar(
            x=list(scores.values()),
            y=list(scores.keys()),
            orientation="h",
            color=list(scores.keys()),
            color_discrete_map={
                "positive": "#16a34a",
                "negative": "#dc2626",
                "neutral": "#94a3b8",
            },
            labels={"x": "probability", "y": ""},
        )
        fig.update_layout(
            showlegend=False,
            xaxis_tickformat=".0%",
            height=230,
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    with mid:
        st.subheader("Management tone")
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=net * 100,
                number={"suffix": "%", "font": {"size": 36}},
                gauge={
                    "axis": {"range": [-50, 50], "ticksuffix": "%"},
                    "bar": {"color": "#3b82f6"},
                    # translucent zones read well on dark and light themes
                    "steps": [
                        {"range": [-50, -10], "color": "rgba(220,38,38,0.35)"},
                        {"range": [-10, 10], "color": "rgba(148,163,184,0.25)"},
                        {"range": [10, 50], "color": "rgba(22,163,74,0.35)"},
                    ],
                },
            )
        )
        gauge.update_layout(height=230, margin=dict(l=20, r=20, t=20, b=0))
        st.plotly_chart(gauge, use_container_width=True)
        st.caption("Net sentiment = positive − negative probability.")

    with right:
        st.subheader("Predicted direction")
        if not models:
            st.info("No trained models found - run `src\\model.py` first.")
        for horizon, label in HORIZONS.items():
            if horizon not in models:
                continue
            prob_up = float(
                models[horizon].predict_proba(
                    [[scores["positive"], scores["negative"]]]
                )[0][1]
            )
            direction = "UP" if prob_up >= 0.5 else "DOWN"
            confidence = prob_up if prob_up >= 0.5 else 1 - prob_up
            st.metric(
                f"{label} after the call",
                f"{'📈' if direction == 'UP' else '📉'} {direction}",
                f"{confidence:.0%} confidence",
                delta_color="normal" if direction == "UP" else "inverse",
            )

    # ------------------------------------------------- price + projection

    if prices is not None and history is not None and "return_5d" in models:
        st.divider()
        st.subheader(f"{ticker} - recent price and 5-day statistical projection")

        prob_up_5d = float(
            models["return_5d"].predict_proba(
                [[scores["positive"], scores["negative"]]]
            )[0][1]
        )
        # Expected move = probability-weighted blend of the historical
        # average up-move and down-move after an earnings call.
        hist_5d = history["return_5d"].dropna()
        up_mean = hist_5d[hist_5d > 0].mean()
        down_mean = hist_5d[hist_5d <= 0].mean()
        expected_5d = prob_up_5d * up_mean + (1 - prob_up_5d) * down_mean
        band = hist_5d.std()

        last_close = float(prices.iloc[-1])
        last_date = prices.index[-1]
        future_dates = pd.bdate_range(
            last_date + pd.Timedelta(days=1), periods=5
        )
        steps = pd.Series(range(1, 6), index=future_dates) / 5.0
        proj_mid = last_close * (1 + expected_5d * steps)
        proj_hi = last_close * (1 + (expected_5d + band) * steps)
        proj_lo = last_close * (1 + (expected_5d - band) * steps)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=prices.index, y=prices.values,
                name="Close (last 6 months)", line={"color": "#94a3b8"},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[last_date, *future_dates], y=[last_close, *proj_hi],
                line={"width": 0}, showlegend=False, hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[last_date, *future_dates], y=[last_close, *proj_lo],
                fill="tonexty", fillcolor="rgba(59,130,246,0.15)",
                line={"width": 0}, name="±1σ historical range",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[last_date, *future_dates], y=[last_close, *proj_mid],
                name=f"Projection ({expected_5d:+.1%} expected)",
                line={"color": "#60a5fa", "dash": "dash"},
            )
        )
        fig.update_layout(
            height=420,
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", y=1.05),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"Projection = model's up-probability ({prob_up_5d:.0%}) blended "
            "with the average historical post-call up/down move; the shaded "
            "band is ±1 standard deviation of historical 5-day returns. "
            "A statistical illustration, not a forecast."
        )

    # ------------------------------------------------------------- history

    if history is not None:
        st.divider()
        st.subheader("Where this call sits in the historical data")
        tabs = st.tabs([f"{label} return" for label in HORIZONS.values()])
        for tab, horizon in zip(tabs, HORIZONS):
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
elif analyze:
    st.warning("Paste a transcript or upload a file first.")
