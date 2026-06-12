"""Earnings Call Tone & Post-Call Return Analyzer - Streamlit app.

Paste or upload an earnings call transcript, get a FinBERT sentiment
breakdown with live processing details, a predicted direction for the
stock's 1-day and 5-day post-call returns, an optional price chart with a
statistical 5-day projection, and a view of where the call sits relative
to the historical dataset of 188 real earnings calls.

Run from the project root:
    .venv\\Scripts\\streamlit run app.py
"""

import html
import json
import re
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from fetch_featured import download_transcript
from fetch_returns import download_close_prices
from fundamentals import get_fundamentals
from sentiment import MAX_CHUNK_TOKENS, FinbertScorer, aggregate

DATASET_CSV = Path("data/processed/dataset.csv")
MODELS_DIR = Path("models")
FEATURED_DIR = Path("data/featured")
HORIZONS = {"return_1d": "1 day", "return_5d": "5 days"}
GITHUB_URL = "https://github.com/marcelmaybaum11-cyber/earnings-call-analyzer"

# Transcripts usually announce the Q&A part with a heading like
# "Questions and Answers" or "Question-and-Answer Session".
QA_MARKER = re.compile(
    r"question[-\s]?(?:s)?[-\s]?(?:and|&)[-\s]?answer", re.IGNORECASE
)

# Words that signal hedging / uncertainty, in the spirit of the
# Loughran-McDonald financial sentiment word lists. A high density means
# management is qualifying its statements a lot.
HEDGING_WORDS = frozenset(
    """approximately assume assumed assumes believe believes anticipate
    anticipates caution cautious challenging could depend depending depends
    estimate estimated estimates expect expects headwind headwinds may might
    possibly probable probably risk riskier risks uncertain uncertainties
    uncertainty unclear unknown volatile volatility""".split()
)


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split a transcript into prepared remarks and Q&A, if possible.

    Earnings calls have two very different halves: a scripted opening
    statement and an unscripted Q&A with analysts. Comparing their tone is
    informative because the Q&A is harder to polish. If no Q&A heading is
    found (or it sits implausibly close to either end), the whole text is
    treated as one section.
    """
    match = QA_MARKER.search(text)
    if match and 0.05 * len(text) < match.start() < 0.95 * len(text):
        return [
            ("Prepared remarks", text[: match.start()]),
            ("Q&A session", text[match.start() :]),
        ]
    return [("Full call", text)]


def hedging_density(text: str) -> tuple[int, int, float]:
    """Return (hedge_count, word_count, hedges per 1,000 words)."""
    words = re.findall(r"[a-z']+", text.lower())
    hedges = sum(word in HEDGING_WORDS for word in words)
    per_1000 = hedges / len(words) * 1000 if words else 0.0
    return hedges, len(words), per_1000

st.set_page_config(
    page_title="Earnings Call Analyzer",
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
    .step-card {
        background: rgba(148, 163, 184, 0.10);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 14px; padding: 0.85rem 1rem;
        min-height: 8.2rem;
    }
    [data-testid="stExpander"] details {
        border-radius: 14px; border-color: rgba(148, 163, 184, 0.25);
    }
    .step-num {
        font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em;
        text-transform: uppercase; color: #60a5fa; margin-bottom: 0.15rem;
    }
    .step-title { font-weight: 700; margin-bottom: 0.15rem; }
    .step-text { color: #94a3b8; font-size: 0.85rem; line-height: 1.35; }
    .excerpt {
        border-left: 4px solid; border-radius: 0 10px 10px 0;
        background: rgba(148, 163, 184, 0.08);
        padding: 0.7rem 0.9rem; font-size: 0.9rem; font-style: italic;
        color: inherit;
    }
    .excerpt-pos { border-color: #16a34a; }
    .excerpt-neg { border-color: #dc2626; }
    .footer {
        text-align: center; color: #64748b; font-size: 0.88rem;
        margin-top: 3rem; padding-top: 1.2rem;
        border-top: 1px solid rgba(148, 163, 184, 0.2);
    }
    .footer a { color: #60a5fa; text-decoration: none; }
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


@st.cache_data(ttl=3600, show_spinner=False)
def get_fundamentals_cached(ticker: str) -> dict:
    return get_fundamentals(ticker)


@st.cache_data(show_spinner="Downloading transcript from the source...")
def get_featured_text(filename: str, source_url: str) -> str:
    """Read a sample transcript, downloading it on first use.

    Transcripts are not committed to the repo (they are (c) The Motley
    Fool), so a fresh deployment fetches them from the source on demand
    and keeps a local copy.
    """
    path = FEATURED_DIR / filename
    if path.exists():
        return path.read_text(encoding="utf-8", errors="replace")
    text = download_transcript(source_url)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    except OSError:
        pass  # read-only filesystem - st.cache_data still avoids refetching
    return text


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
    '<div class="hero-title">Earnings Call Analyzer</div>'
    '<div class="hero-sub">Drop in any earnings call transcript and get '
    "the full picture: management tone, how it shifts through the call, "
    "prepared remarks vs. Q&amp;A, hedging language, standout quotes, and "
    "a return-direction prediction validated on 188 real calls.</div>",
    unsafe_allow_html=True,
)

steps = [
    ("Step 1", "Read", "The transcript is split into BERT-sized pieces "
     "(512 tokens each) so no part of the call is skipped."),
    ("Step 2", "Score", "FinBERT, an AI model trained on financial text, "
     "rates every piece as positive, negative, or neutral."),
    ("Step 3", "Dissect", "Tone is tracked through the call: prepared "
     "remarks vs. Q&A, plus a count of hedging words."),
    ("Step 4", "Predict", "A model trained on 188 real calls turns the "
     "scores into an up/down call for the days after."),
]
step_cols = st.columns(4)
for col, (num, title, text) in zip(step_cols, steps):
    col.markdown(
        f'<div class="step-card"><div class="step-num">{num}</div>'
        f'<div class="step-title">{title}</div>'
        f'<div class="step-text">{text}</div></div>',
        unsafe_allow_html=True,
    )

st.write("")

with st.expander("🔍 What exactly happens under the hood?"):
    st.markdown(
        """
**The idea.** When executives discuss results, *how* they say things
carries information beyond the numbers: confident language tends to
accompany good quarters, hedged and defensive language bad ones. This tool
measures that tone objectively and tests whether it predicts the stock's
next move.

**Tone scoring.** [FinBERT](https://huggingface.co/ProsusAI/finbert) is a
BERT language model fine-tuned on financial text, so it knows that
*"headwinds"* is bad news and *"raised guidance"* is good news even though
neither contains an obviously emotional word. BERT can only read 512
tokens at a time, so the transcript is chunked, every chunk is scored, and
the scores are combined weighted by chunk length. The **net sentiment**
shown everywhere is simply *positive − negative* probability.

**The prediction.** A logistic regression was trained on 188 real
earnings calls (10 large-cap tech companies, 2016-2020): FinBERT scores in,
actual post-call returns out. In 5-fold cross-validation it called the
5-day direction correctly **63.4%** of the time vs. **53.7%** for always
guessing the majority class - a real but modest edge, exactly what you
would expect from tone alone.

**What it does not do.** It does not read the actual numbers (EPS, revenue,
guidance), so a beat with cautious language can confuse it - and tone
partly proxies for the surprise itself. Treat the prediction as a
statistical exercise, not investment advice.
        """
    )

st.write("")

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
            featured_text = get_featured_text(
                featured_row["file"], featured_row["source"]
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
        sections = split_sections(transcript)
        section_chunk_counts = [
            len(scorer._chunks(text)) for _, text in sections
        ]
        total_chunks = sum(section_chunk_counts)
        n_tokens = len(
            scorer.tokenizer.encode(transcript, add_special_tokens=False)
        )
        st.write(
            f"→ {len(transcript):,} characters → **{n_tokens:,} tokens**, "
            f"split into **{total_chunks} chunks** of ≤{MAX_CHUNK_TOKENS} "
            "tokens (BERT's hard limit is 512 per pass)."
        )
        if len(sections) == 2:
            st.write(
                "→ Found a Q&A heading - prepared remarks and the Q&A "
                "session will also be scored separately."
            )

        st.write("**Step 2 - Scoring** every chunk through FinBERT...")
        bar = st.progress(0.0, text=f"0 / {total_chunks} chunks")
        section_scores: dict[str, dict] = {}
        all_chunk_scores: list[dict] = []
        done_offset = 0
        for (name, text), n_chunks in zip(sections, section_chunk_counts):
            chunk_scores = scorer.score_chunks(
                text,
                on_progress=lambda done, _total, off=done_offset: bar.progress(
                    (off + done) / total_chunks,
                    text=f"{off + done} / {total_chunks} chunks",
                ),
            )
            section_scores[name] = aggregate(chunk_scores)
            all_chunk_scores.extend(chunk_scores)
            done_offset += n_chunks

        scores = aggregate(all_chunk_scores)
        net = scores["positive"] - scores["negative"]
        st.write(
            "**Step 3 - Aggregating**: chunk scores combined with a "
            f"length-weighted average → net sentiment **{net:+.1%}**."
        )
        hedge_count, n_words, hedge_per_1000 = hedging_density(transcript)

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

    # Up-probability per horizon, computed once and reused by the metric
    # cards, the prediction-vs-reality chart, and the downloadable report.
    predictions = {
        horizon: float(
            model.predict_proba(
                [[scores["positive"], scores["negative"]]]
            )[0][1]
        )
        for horizon, model in models.items()
    }

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
            if horizon not in predictions:
                continue
            prob_up = predictions[horizon]
            direction = "UP" if prob_up >= 0.5 else "DOWN"
            confidence = prob_up if prob_up >= 0.5 else 1 - prob_up
            st.metric(
                f"{label} after the call",
                f"{'📈' if direction == 'UP' else '📉'} {direction}",
                f"{confidence:.0%} confidence",
                delta_color="normal" if direction == "UP" else "inverse",
            )

    # --------------------------- tone through the call & language check

    st.divider()
    chunk_token_total = sum(c["tokens"] for c in all_chunk_scores)
    chart_col, stats_col = st.columns([1.8, 1])

    with chart_col:
        st.subheader("How tone moved through the call")
        if total_chunks >= 3:
            cum = 0
            positions, chunk_nets = [], []
            for c in all_chunk_scores:
                positions.append((cum + c["tokens"] / 2) / chunk_token_total * 100)
                chunk_nets.append(c["positive"] - c["negative"])
                cum += c["tokens"]
            fig = go.Figure(
                go.Scatter(
                    x=positions,
                    y=chunk_nets,
                    mode="lines+markers",
                    line={"color": "#60a5fa"},
                    fill="tozeroy",
                    fillcolor="rgba(96,165,250,0.12)",
                    hovertemplate="%{x:.0f}% through the call: "
                    "net %{y:+.0%}<extra></extra>",
                )
            )
            fig.add_hline(y=0, line_dash="dot", line_color="#94a3b8")
            if len(sections) == 2:
                qa_start_pct = (
                    sum(
                        c["tokens"]
                        for c in all_chunk_scores[: section_chunk_counts[0]]
                    )
                    / chunk_token_total
                    * 100
                )
                fig.add_vline(
                    x=qa_start_pct, line_dash="dash", line_color="#f59e0b"
                )
                fig.add_annotation(
                    x=qa_start_pct, y=1.06, yref="paper", showarrow=False,
                    text="Q&A starts", font={"color": "#f59e0b"},
                )
            fig.update_layout(
                height=320,
                margin=dict(l=0, r=0, t=30, b=0),
                xaxis_title="position in the call (%)",
                yaxis_title="net sentiment",
                yaxis_tickformat="+.0%",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Each point is one ~510-token chunk. Watch for tone fading "
                "late in the call - that is usually the Q&A getting tough."
            )
        else:
            st.info(
                "The transcript is too short for a tone timeline - it "
                "needs at least 3 chunks (~1,500 tokens)."
            )

    with stats_col:
        st.subheader("Language check")
        if len(sections) == 2:
            prep = section_scores["Prepared remarks"]
            qa = section_scores["Q&A session"]
            prep_net = prep["positive"] - prep["negative"]
            qa_net = qa["positive"] - qa["negative"]
            c1, c2 = st.columns(2)
            c1.metric("Prepared remarks", f"{prep_net:+.1%}")
            c2.metric(
                "Q&A session",
                f"{qa_net:+.1%}",
                f"{qa_net - prep_net:+.1%} vs script",
            )
            st.caption(
                "The opening statement is scripted and polished; the Q&A "
                "is not. A tone drop in the Q&A suggests the confidence "
                "lived mostly in the script."
            )
        c1, c2 = st.columns(2)
        c1.metric("Words", f"{n_words:,}")
        c2.metric(
            "Hedging density",
            f"{hedge_per_1000:.1f}",
            f"{hedge_count} hedges / 1,000 words",
            delta_color="off",
        )
        st.caption(
            "Hedging counts qualifiers like *approximately, expect, risk, "
            "uncertain*. The number itself matters less than comparing "
            "calls: a sudden jump vs. last quarter means management is "
            "qualifying far more statements than usual."
        )

    # ------------------------------------------------- standout passages

    if len(all_chunk_scores) >= 2:
        st.divider()
        st.subheader("Standout passages")
        best = max(all_chunk_scores, key=lambda c: c["positive"] - c["negative"])
        worst = min(all_chunk_scores, key=lambda c: c["positive"] - c["negative"])

        def excerpt(chunk: dict, limit: int = 460) -> str:
            text = " ".join(chunk["text"].split())
            if len(text) > limit:
                text = text[:limit].rsplit(" ", 1)[0] + " …"
            return html.escape(text)

        pos_col, neg_col = st.columns(2)
        with pos_col:
            st.markdown(
                "**🌤️ Most positive** "
                f"(net {best['positive'] - best['negative']:+.0%})"
            )
            st.markdown(
                f'<div class="excerpt excerpt-pos">{excerpt(best)}</div>',
                unsafe_allow_html=True,
            )
        with neg_col:
            st.markdown(
                "**⛈️ Most negative** "
                f"(net {worst['positive'] - worst['negative']:+.0%})"
            )
            st.markdown(
                f'<div class="excerpt excerpt-neg">{excerpt(worst)}</div>',
                unsafe_allow_html=True,
            )
        st.caption(
            "The chunks FinBERT scored highest and lowest. Punctuation can "
            "look slightly off because the text is reconstructed from "
            "BERT tokens."
        )

    # ------------------- prediction vs. reality around the call date

    if (
        prices is not None
        and featured_row is not None
        and "return_5d" in predictions
    ):
        st.divider()
        st.subheader(f"{ticker} — model prediction vs. what actually happened")
        call_ts = pd.Timestamp(featured_row["call_date"])

        prob_up_5d = predictions["return_5d"]
        # Expected move = probability-weighted blend of the historical
        # average post-call up-move and down-move.
        hist_5d = (
            history["return_5d"].dropna() if history is not None else pd.Series()
        )
        up_mean = hist_5d[hist_5d > 0].mean() if len(hist_5d) else 0.03
        down_mean = hist_5d[hist_5d <= 0].mean() if len(hist_5d) else -0.03
        expected_5d = prob_up_5d * up_mean + (1 - prob_up_5d) * down_mean

        before = prices.loc[:call_ts]
        if before.empty:
            st.info("No price data around the call date.")
        else:
            # Anchor at the last close on/before the call (calls are after
            # hours, so this is the price the market reacted from).
            base_idx = len(before) - 1
            base_price = float(prices.iloc[base_idx])
            base_date = prices.index[base_idx]

            # Uncertainty cone from the stock's own daily volatility,
            # widening with sqrt(time): roughly an 80% interval (z=1.28).
            daily_vol = float(prices.pct_change().std())
            n_after = min(5, len(prices) - base_idx - 1)
            cone_dates = prices.index[base_idx + 1 : base_idx + 1 + n_after]
            mid, hi, lo = [], [], []
            for t in range(1, n_after + 1):
                drift = expected_5d * t / 5
                spread = 1.28 * daily_vol * t**0.5
                mid.append(base_price * (1 + drift))
                hi.append(base_price * (1 + drift + spread))
                lo.append(base_price * (1 + drift - spread))

            actual_1d = (
                float(prices.iloc[base_idx + 1] / base_price - 1)
                if n_after >= 1 else None
            )
            actual_5d = (
                float(prices.iloc[base_idx + 5] / base_price - 1)
                if n_after >= 5 else None
            )
            predicted_up = prob_up_5d >= 0.5

            c1, c2, c3 = st.columns(3)
            c1.metric(
                "Model said (5-day)",
                "📈 UP" if predicted_up else "📉 DOWN",
                f"{(prob_up_5d if predicted_up else 1 - prob_up_5d):.0%} confidence",
                delta_color="off",
            )
            if actual_1d is not None:
                c2.metric("Actually, next day", f"{actual_1d:+.2%}")
            if actual_5d is not None:
                hit = (actual_5d > 0) == predicted_up
                c3.metric(
                    "Actually, 5 days",
                    f"{actual_5d:+.2%}",
                    "✅ direction hit" if hit else "❌ direction miss",
                    delta_color="normal" if hit else "inverse",
                )

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=prices.index, y=prices.values,
                    name="Close", line={"color": "#94a3b8"},
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[base_date, *cone_dates], y=[base_price, *hi],
                    line={"width": 0}, showlegend=False, hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[base_date, *cone_dates], y=[base_price, *lo],
                    fill="tonexty", fillcolor="rgba(96,165,250,0.18)",
                    line={"width": 0}, name="~80% expected range",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[base_date, *cone_dates], y=[base_price, *mid],
                    name=f"Model path ({expected_5d:+.1%} expected)",
                    line={"color": "#60a5fa", "dash": "dash"},
                )
            )
            fig.add_vline(x=base_date, line_dash="dot", line_color="#f59e0b")
            fig.add_annotation(
                x=base_date, y=1.02, yref="paper", showarrow=False,
                text="earnings call", font={"color": "#f59e0b"},
            )
            # Zoom to the weeks around the call; users can zoom out.
            fig.update_xaxes(
                range=[call_ts - pd.Timedelta(days=45),
                       min(call_ts + pd.Timedelta(days=20),
                           prices.index[-1] + pd.Timedelta(days=2))]
            )
            window = prices.loc[
                call_ts - pd.Timedelta(days=45):
                call_ts + pd.Timedelta(days=20)
            ]
            fig.update_yaxes(
                range=[window.min() * 0.96, window.max() * 1.04]
            )
            fig.update_layout(
                height=420,
                margin=dict(l=0, r=0, t=24, b=0),
                legend=dict(orientation="h", y=1.08),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "The dashed line is the path implied by the model's "
                f"up-probability ({prob_up_5d:.0%}) blended with average "
                "historical post-call moves; the cone widens with the "
                "stock's own daily volatility (~80% interval). The gray "
                "line is what the stock actually did."
            )

    # ------------------------------------- fundamentals and fair value

    if ticker:
        st.divider()
        st.subheader(f"{ticker} — key fundamentals & fair value")
        try:
            funda = get_fundamentals_cached(ticker)
        except Exception as exc:
            funda = None
            st.warning(f"Could not fetch fundamentals: {exc}")

        if funda:
            m = funda["metrics"]
            fv = funda["fair_value"]

            def fmt_big(x):
                if x is None:
                    return "n/a"
                for unit, div in (("T", 1e12), ("B", 1e9), ("M", 1e6)):
                    if abs(x) >= div:
                        return f"${x / div:,.2f}{unit}"
                return f"${x:,.0f}"

            def fmt_pct(x):
                return f"{x:.1%}" if x is not None else "n/a"

            def fmt_num(x, prefix=""):
                return f"{prefix}{x:,.2f}" if x is not None else "n/a"

            r1 = st.columns(4)
            r1[0].metric("Price", fmt_num(m["currentPrice"], "$"))
            r1[1].metric("Market cap", fmt_big(m["marketCap"]))
            r1[2].metric("P/E trailing", fmt_num(m["trailingPE"]))
            r1[3].metric("P/E forward", fmt_num(m["forwardPE"]))
            r2 = st.columns(4)
            r2[0].metric("Revenue growth", fmt_pct(m["revenueGrowth"]))
            r2[1].metric("Profit margin", fmt_pct(m["profitMargins"]))
            r2[2].metric("Return on equity", fmt_pct(m["returnOnEquity"]))
            r2[3].metric("Free cash flow", fmt_big(m["freeCashflow"]))

            current = fv["current"]
            if current and fv["analyst_mean"]:
                st.markdown("##### Fair value estimates")
                col_chart, col_nums = st.columns([2, 1])

                with col_chart:
                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(
                            x=[fv["analyst_low"], fv["analyst_high"]],
                            y=["", ""],
                            mode="lines",
                            line={"color": "rgba(148,163,184,0.6)", "width": 6},
                            name="Analyst range",
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=[fv["analyst_mean"]], y=[""],
                            mode="markers+text",
                            marker={"size": 16, "color": "#34d399"},
                            text=["mean target"], textposition="top center",
                            name="Analyst mean",
                        )
                    )
                    if fv["lynch"]:
                        fig.add_trace(
                            go.Scatter(
                                x=[fv["lynch"]], y=[""],
                                mode="markers+text",
                                marker={"size": 14, "color": "#a78bfa",
                                        "symbol": "diamond"},
                                text=["PEG=1 value"], textposition="bottom center",
                                name="Lynch fair value",
                            )
                        )
                    fig.add_trace(
                        go.Scatter(
                            x=[current], y=[""],
                            mode="markers+text",
                            marker={"size": 16, "color": "#60a5fa",
                                    "symbol": "line-ns", "line": {"width": 3}},
                            text=["current"], textposition="top center",
                            name="Current price",
                        )
                    )
                    fig.update_layout(
                        height=170, showlegend=False,
                        margin=dict(l=0, r=0, t=30, b=0),
                        xaxis={"tickprefix": "$"}, yaxis={"visible": False},
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col_nums:
                    upside = fv["analyst_mean"] / current - 1
                    st.metric(
                        f"Analyst mean target ({fv['analyst_count']} analysts)",
                        f"${fv['analyst_mean']:,.0f}",
                        f"{upside:+.1%} vs current",
                    )
                    if fv["lynch"]:
                        st.metric(
                            "Lynch fair value (PEG = 1)",
                            f"${fv['lynch']:,.0f}",
                            f"{fv['lynch'] / current - 1:+.1%} vs current",
                        )
                    if fv["recommendation"]:
                        st.caption(
                            "Street consensus: "
                            f"**{fv['recommendation'].replace('_', ' ').title()}**"
                        )
                st.caption(
                    "Fair value here is illustrative: the analyst range is "
                    "Yahoo's aggregate of published 12-month targets; the "
                    "Lynch estimate assumes a stock is fairly priced when "
                    "P/E equals earnings growth (PEG = 1, growth capped at "
                    "50%). Neither is investment advice."
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

    # -------------------------------------------------------------- export

    st.divider()
    report = {
        "generated": pd.Timestamp.now().isoformat(timespec="seconds"),
        "characters": len(transcript),
        "words": n_words,
        "tokens": n_tokens,
        "overall": {**scores, "net_sentiment": net},
        "sections": {
            name: {**s, "net_sentiment": s["positive"] - s["negative"]}
            for name, s in section_scores.items()
        },
        "hedging": {"count": hedge_count, "per_1000_words": hedge_per_1000},
        "predicted_probability_up": {
            HORIZONS[h]: p for h, p in predictions.items()
        },
        "tone_timeline_net": [
            round(c["positive"] - c["negative"], 4) for c in all_chunk_scores
        ],
    }
    st.download_button(
        "⬇️ Download this analysis (JSON)",
        json.dumps(report, indent=2),
        file_name="earnings_call_analysis.json",
        mime="application/json",
        use_container_width=True,
    )
elif analyze:
    st.warning("Paste a transcript or upload a file first.")

# -------------------------------------------------------------------- footer

st.markdown(
    '<div class="footer">Made by <b>Marcel Maybaum</b> · '
    f'<a href="{GITHUB_URL}" target="_blank">Source code on GitHub</a> · '
    'Powered by <a href="https://huggingface.co/ProsusAI/finbert" '
    'target="_blank">FinBERT</a> and Yahoo Finance · '
    "Educational project, not investment advice</div>",
    unsafe_allow_html=True,
)
