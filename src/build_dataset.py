"""Build the modeling dataset: FinBERT sentiment + post-call returns.

Reads every transcript under data/raw/data/transcripts/<TICKER>/, scores it
with FinBERT, joins it with the stock's 1-day and 5-day post-call returns,
and appends to data/processed/dataset.csv.

The job is resumable: each row is written to the CSV immediately, and on
restart any transcript already present in the CSV is skipped. Scoring 190
transcripts on CPU takes a while, so a crash or interruption should never
lose completed work.

Run from the project root:
    .venv\\Scripts\\python.exe src\\build_dataset.py
"""

import csv
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from fetch_returns import compute_post_call_returns, download_close_prices
from sentiment import FinbertScorer

TRANSCRIPTS_DIR = Path("data/raw/data/transcripts")
OUTPUT_CSV = Path("data/processed/dataset.csv")

FIELDS = [
    "ticker",
    "call_date",
    "transcript_path",
    "positive",
    "negative",
    "neutral",
    "return_1d",
    "return_5d",
]


def parse_filename(path: Path) -> tuple[str, pd.Timestamp]:
    """'2016-Apr-26-AAPL.txt' -> ('AAPL', Timestamp 2016-04-26)."""
    stem_parts = path.stem.split("-")  # [year, mon, day, ticker]
    ticker = stem_parts[3]
    date = pd.Timestamp("-".join(stem_parts[:3]))
    return ticker, date


def already_done() -> set[str]:
    """Transcript paths already present in the output CSV."""
    if not OUTPUT_CSV.exists():
        return set()
    return set(pd.read_csv(OUTPUT_CSV)["transcript_path"].astype(str))


def main() -> None:
    # Only files inside ticker subfolders (AAPL/, MSFT/, ...). The dataset
    # also ships train.txt/test.txt split indexes in the root - skip those.
    files = sorted(TRANSCRIPTS_DIR.glob("*/*.txt"))
    if not files:
        raise SystemExit(f"No transcripts found under {TRANSCRIPTS_DIR}")

    done = already_done()
    todo = [f for f in files if str(f) not in done]
    print(f"{len(files)} transcripts total, {len(done)} done, {len(todo)} to go.")
    if not todo:
        print("Nothing to do.")
        return

    print("Loading FinBERT...")
    scorer = FinbertScorer()

    # One price-history download per ticker (not per call) to stay fast
    # and avoid Yahoo rate limits.
    tickers = sorted({parse_filename(f)[0] for f in todo})
    dates = [parse_filename(f)[1] for f in files]
    start = min(dates) - pd.Timedelta(days=15)
    end = max(dates) + pd.Timedelta(days=20)
    print(f"Downloading price history for {len(tickers)} tickers...")
    histories = {t: download_close_prices(t, start, end) for t in tickers}

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not OUTPUT_CSV.exists()

    t0 = time.time()
    with OUTPUT_CSV.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()

        for i, path in enumerate(todo, 1):
            ticker, date = parse_filename(path)
            text = path.read_text(encoding="utf-8", errors="replace")

            scores = scorer.score(text)
            returns = compute_post_call_returns(histories[ticker], date)

            writer.writerow(
                {
                    "ticker": ticker,
                    "call_date": date.date(),
                    "transcript_path": str(path),
                    **scores,
                    **returns,
                }
            )
            fh.flush()  # row is safe on disk even if the process dies

            elapsed = time.time() - t0
            eta = elapsed / i * (len(todo) - i)
            print(
                f"[{i}/{len(todo)}] {ticker} {date.date()}  "
                f"pos={scores['positive']:.0%} neg={scores['negative']:.0%}  "
                f"(eta {eta/60:.0f} min)",
                flush=True,
            )

    print(f"\nDataset complete: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
