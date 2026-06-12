# Earnings Call Analyzer

Does management tone on an earnings call predict the stock's move in the
days that follow? This project scores earnings call transcripts with
[FinBERT](https://huggingface.co/ProsusAI/finbert) (a BERT model fine-tuned
on financial text) and tests whether sentiment correlates with the 1-day
and 5-day post-call returns.

A Streamlit app lets you paste any transcript and get a sentiment
breakdown, a predicted return direction, and a view of where the call sits
relative to the historical dataset.

**Try it live:** https://earnings-call-analyzer-marcel.streamlit.app/

> Educational project — a statistical exercise, not investment advice.

## How it works

1. **Data** — 188 real earnings call transcripts (AAPL, MSFT, NVDA, AMZN,
   GOOGL, AMD, INTC, CSCO, MU, ASML; 2016–2020) from the
   [jlh-ibm/earnings_call](https://huggingface.co/datasets/jlh-ibm/earnings_call)
   dataset, with stock prices from yfinance.
2. **Sentiment** — FinBERT outputs positive/negative/neutral probabilities.
   Transcripts exceed BERT's 512-token limit, so they are split into
   chunks, scored in batches, and combined with a length-weighted average.
3. **Returns** — earnings calls usually happen after market close, so the
   baseline is the close on the call date and returns are measured 1 and 5
   trading days forward.
4. **Model** — logistic regression on (positive, negative) probabilities
   predicting return *direction*, evaluated with 5-fold stratified
   cross-validation against the majority-class baseline.

## Project structure

```
├── data/
│   ├── raw/            # downloaded transcripts (not committed)
│   └── processed/      # dataset.csv: sentiment + returns per call
├── models/             # trained logistic regression models
├── src/
│   ├── sentiment.py    # FinBERT scorer with chunking + batching
│   ├── fetch_returns.py# post-call return calculation via yfinance
│   ├── build_dataset.py# scores all transcripts, builds dataset.csv
│   └── model.py        # correlation analysis + classifier training
├── app.py              # Streamlit app
└── requirements.txt
```

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Usage

```powershell
# 1. Build the dataset (downloads transcripts on first run, ~1.5h on CPU)
python src\build_dataset.py

# 2. Train models and see the correlation analysis
python src\model.py

# 3. Launch the app
streamlit run app.py
```

## Results

On 188 earnings calls (10 tech tickers, 2016–2020), management tone as
measured by FinBERT **does** correlate with post-call returns:

| Horizon | Spearman ρ (net sentiment vs return) | p-value | Baseline acc. | LogReg CV acc. |
|---|---|---|---|---|
| 1 day  | +0.246 | 0.001 | 55.9% | **60.6% ± 4.6%** |
| 5 days | +0.222 | 0.002 | 53.7% | **63.4% ± 9.9%** |

Both correlations are statistically significant (p < 0.01), and the
classifier beats the majority-class baseline at both horizons. The effect
is real but modest — tone explains a small part of post-earnings moves,
which is consistent with the academic literature on earnings call
sentiment.

Caveats: tech mega-caps only, 2016–2020 (a strong bull market), and no
control for the earnings surprise itself — sentiment likely proxies
partly for whether results beat or missed expectations.
