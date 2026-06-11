"""Download featured recent earnings call transcripts for the app.

Fetches a small curated set of recent big-company earnings call
transcripts from The Motley Fool, extracts the article text, and stores
them under data/featured/ together with a manifest.csv the app reads.

For personal/educational use - transcripts remain (c) The Motley Fool.

Run from the project root:
    .venv\\Scripts\\python.exe src\\fetch_featured.py
"""

import csv
from pathlib import Path

import requests
from bs4 import BeautifulSoup

FEATURED_DIR = Path("data/featured")

FEATURED_CALLS = [
    ("NVDA", "NVIDIA", "2026-05-20", "Q1 FY2027",
     "https://www.fool.com/earnings/call-transcripts/2026/05/20/nvidia-nvda-q1-2027-earnings-transcript/"),
    ("AAPL", "Apple", "2026-04-30", "Q2 FY2026",
     "https://www.fool.com/earnings/call-transcripts/2026/04/30/apple-aapl-q2-2026-earnings-call-transcript/"),
    ("MSFT", "Microsoft", "2026-04-29", "Q3 FY2026",
     "https://www.fool.com/earnings/call-transcripts/2026/04/29/microsoft-msft-q3-2026-earnings-transcript/"),
    ("AMZN", "Amazon", "2026-04-29", "Q1 2026",
     "https://www.fool.com/earnings/call-transcripts/2026/04/29/amazon-amzn-q1-2026-earnings-call-transcript/"),
    ("GOOGL", "Alphabet", "2026-04-29", "Q1 2026",
     "https://www.fool.com/earnings/call-transcripts/2026/04/29/alphabet-googl-q1-2026-earnings-call-transcript/"),
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    )
}


def extract_transcript(html: str) -> str:
    """Pull the transcript text out of a fool.com article page.

    Tries the known article container first; falls back to whichever
    parent element holds the most <p> text on the page.
    """
    soup = BeautifulSoup(html, "html.parser")

    container = soup.select_one("div.article-body")
    if container is None:
        candidates = {}
        for p in soup.find_all("p"):
            parent = p.parent
            candidates[parent] = candidates.get(parent, 0) + len(p.get_text())
        container = max(candidates, key=candidates.get)

    paragraphs = [p.get_text(" ", strip=True) for p in container.find_all("p")]
    return "\n\n".join(p for p in paragraphs if p)


def download_transcript(url: str) -> str:
    """Fetch one transcript page and return the extracted text."""
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    text = extract_transcript(resp.text)
    if len(text) < 5000:
        raise RuntimeError(
            f"Extracted only {len(text)} chars from {url} - page layout "
            "may have changed."
        )
    return text


def main() -> None:
    FEATURED_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for ticker, company, date, quarter, url in FEATURED_CALLS:
        print(f"Fetching {company} ({ticker}) {quarter}...")
        text = download_transcript(url)
        filename = f"{ticker}_{date}.txt"
        (FEATURED_DIR / filename).write_text(text, encoding="utf-8")
        rows.append(
            {"ticker": ticker, "company": company, "call_date": date,
             "quarter": quarter, "file": filename, "source": url}
        )
        print(f"  -> {len(text):,} chars saved to {filename}")

    with (FEATURED_DIR / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nManifest written: {FEATURED_DIR / 'manifest.csv'}")


if __name__ == "__main__":
    main()
