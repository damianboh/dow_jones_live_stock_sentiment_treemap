import os
import time
import random
import datetime as dt
from datetime import datetime

import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

import plotly.express as px


import nltk
nltk.downloader.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# dow jones ticker
tickers = [
    "V", "KO", "SHW", "PG", "TRV", "UNH", "VZ", "GS", "HD", "IBM",
    "JNJ", "JPM", "MCD", "MMM", "MRK", "NKE", "WMT", "CSCO", "AAPL",
    "AXP", "BA", "CAT", "CRM", "CVX", "DIS", "AMGN", "AMZN", "HON",
    "MSFT", "NVDA"
]

FINVIZ_QUOTE_URL = "https://finviz.com/quote.ashx?t={ticker}"

MIN_SLEEP_BETWEEN_TICKERS = 0.7
MAX_SLEEP_BETWEEN_TICKERS = 1.5


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
        "DNT": "1",
        "Upgrade-Insecure-Requests": "1",
    })
    return s

def looks_blocked(html_text: str) -> bool:
    """Heuristic: Finviz (or proxy/WAF) sometimes returns short/blocked pages."""
    if not html_text or len(html_text) < 4000:
        return True
    lowered = html_text.lower()
    block_markers = [
        "access denied", "forbidden", "captcha", "verify you are human",
        "unusual traffic", "temporarily unavailable"
    ]
    return any(m in lowered for m in block_markers)

def fetch_quote_soup(session: requests.Session, ticker: str, max_retries: int = 6) -> BeautifulSoup | None:
    url = FINVIZ_QUOTE_URL.format(ticker=ticker)
    last_err = None

    for attempt in range(1, max_retries + 1):
        try:
            r = session.get(url, timeout=25)
            status = r.status_code

            # If rate-limited / blocked, backoff and retry
            if status in (401, 403, 429, 503):
                sleep_s = (2 ** (attempt - 1)) + random.uniform(0.2, 1.5)
                print(f"[{ticker}] HTTP {status} (possible block). Backing off {sleep_s:.1f}s...")
                time.sleep(sleep_s)
                continue

            if status != 200:
                sleep_s = (2 ** (attempt - 1)) + random.uniform(0.2, 1.5)
                print(f"[{ticker}] HTTP {status}. Backing off {sleep_s:.1f}s...")
                time.sleep(sleep_s)
                continue

            if looks_blocked(r.text):
                sleep_s = (2 ** (attempt - 1)) + random.uniform(0.2, 1.5)
                print(f"[{ticker}] HTML looks blocked/partial. Backing off {sleep_s:.1f}s...")
                time.sleep(sleep_s)
                continue

            return BeautifulSoup(r.text, "html.parser")

        except requests.RequestException as e:
            last_err = e
            sleep_s = (2 ** (attempt - 1)) + random.uniform(0.2, 1.5)
            print(f"[{ticker}] Request error: {e}. Backing off {sleep_s:.1f}s...")
            time.sleep(sleep_s)

    print(f"[{ticker}] Failed to fetch after {max_retries} retries. Last error: {last_err}")
    return None


# Parsers (news + snapshot)
def parse_news_rows(soup: BeautifulSoup, ticker: str) -> list[list]:
    """
    Returns rows: [ticker, date_str, time_str, headline]
    Finviz uses:
      - "Today 10:15AM" OR
      - "Dec-19-25 10:15AM" OR
      - for subsequent rows same day: only "10:12AM" (date omitted)
    """
    rows_out = []
    table = soup.find(id="news-table")
    if not table:
        return rows_out

    last_date = None

    for tr in table.find_all("tr"):
        try:
            a = tr.find("a")
            td = tr.find("td")
            if not a or not td:
                continue

            headline = a.get_text(strip=True)
            stamp = td.get_text(" ", strip=True).split()

            if len(stamp) == 1:
                time_str = stamp[0]
                date_str = last_date
            else:
                date_str, time_str = stamp[0], stamp[1]
                last_date = date_str

            if date_str is None:
                continue

            rows_out.append([ticker, date_str, time_str, headline])
        except Exception:
            # never let a malformed row crash the run
            continue

    return rows_out

def parse_snapshot_info(soup: BeautifulSoup) -> dict:
    """
    Extracts Market Cap, Sector, Industry, Company, plus any other snapshot key/values.
    Returns a dict with safe defaults if missing.
    """
    data = {}

    # Snapshot table values
    snap_cells = soup.select(".snapshot-td2")
    for i, cell in enumerate(snap_cells):
        txt = cell.get_text(strip=True)
        if not txt:
            continue
        if i % 2 == 0:
            key = txt
        else:
            data[key] = txt

    # Sector / Industry (quote links)
    tabs = soup.select(".quote-links .tab-link")
    data["Sector"] = tabs[0].get_text(strip=True) if len(tabs) >= 1 else "Others"
    data["Industry"] = tabs[1].get_text(strip=True) if len(tabs) >= 2 else "Others"

    # Company
    company_el = soup.select_one("h2.quote-header_ticker-wrapper_company a")
    data["Company"] = company_el.get_text(strip=True) if company_el else "Name Not Found"

    # Normalize Market Cap to float dollars when possible
    mc = data.get("Market Cap")
    data["Market Cap"] = normalize_market_cap(mc)

    return data

def normalize_market_cap(mc) -> float:
    if mc is None:
        return np.nan
    if isinstance(mc, (int, float)):
        return float(mc)

    s = str(mc).strip()
    if s in ("", "-", "N/A"):
        return np.nan

    try:
        if s.endswith("B"):
            return float(s[:-1]) * 1e9
        if s.endswith("M"):
            return float(s[:-1]) * 1e6
        if s.endswith("K"):
            return float(s[:-1]) * 1e3
        return float(s.replace(",", ""))
    except Exception:
        return np.nan


def finviz_date_to_date(date_str: str) -> dt.date | None:
    """
    Finviz examples:
      "Today"
      "Dec-19-25" (most common)
      sometimes could be "Dec-19-2025" (rare)
    """
    if date_str is None:
        return None
    s = str(date_str).strip()

    if s.lower() == "today":
        return dt.date.today()

    # Try common Finviz formats
    for fmt in ("%b-%d-%y", "%b-%d-%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue

    # Last resort: let pandas/dateutil try (but controlled)
    try:
        return pd.to_datetime(s, errors="coerce").date()
    except Exception:
        return None

session = make_session()

parsed_news = []
companies = []
sectors = []
industries = []
market_caps = []
valid_tickers = [] 

for ticker in tickers:
    print(f"Fetching Finviz page for {ticker} ...")
    soup = fetch_quote_soup(session, ticker)

    if soup is None:
        # fallback placeholders
        info = {"Company": "Name Not Found", "Sector": "Others", "Industry": "Others", "Market Cap": np.nan}
        news_rows = []
    else:
        info = parse_snapshot_info(soup)
        news_rows = parse_news_rows(soup, ticker)

    # append info rows
    valid_tickers.append(ticker)
    companies.append(info.get("Company", "Name Not Found"))
    sectors.append(info.get("Sector", "Others"))
    industries.append(info.get("Industry", "Others"))
    market_caps.append(info.get("Market Cap", np.nan))

    # append news rows
    if news_rows:
        parsed_news.extend(news_rows)

    # random sleep
    time.sleep(random.uniform(MIN_SLEEP_BETWEEN_TICKERS, MAX_SLEEP_BETWEEN_TICKERS))

# Build news dataframe + VADER to score sentiment
columns = ["ticker", "date", "time", "headline"]
parsed_and_scored_news = pd.DataFrame(parsed_news, columns=columns)

# if no news rows at all (blocked), create empty structure to avoid crashing
if parsed_and_scored_news.empty:
    parsed_and_scored_news = pd.DataFrame(columns=columns + ["neg", "neu", "pos", "compound"])
else:
    # convert "Today"/"Dec-19-25" into actual date objects
    parsed_and_scored_news["date"] = parsed_and_scored_news["date"].apply(finviz_date_to_date)

    vader = SentimentIntensityAnalyzer()
    scores = parsed_and_scored_news["headline"].astype(str).apply(vader.polarity_scores).tolist()
    scores_df = pd.DataFrame(scores)

    parsed_and_scored_news = parsed_and_scored_news.join(scores_df)

# compute mean sentiment per ticker (ensure all tickers appear)
if "compound" in parsed_and_scored_news.columns and not parsed_and_scored_news.empty:
    mean_scores = (
        parsed_and_scored_news
        .groupby("ticker", as_index=False)[["neg", "neu", "pos", "compound"]]
        .mean()
    )
else:
    mean_scores = pd.DataFrame({
        "ticker": valid_tickers,
        "neg": np.nan,
        "neu": np.nan,
        "pos": np.nan,
        "compound": np.nan,
    })

# Build info dataframe
df_info = pd.DataFrame({
    "Company": companies,
    "Symbol": valid_tickers,
    "Sector": sectors,
    "Industry": industries,
    "Market Cap": market_caps,
})

# Merge info + sentiment
df = mean_scores.merge(df_info, left_on="ticker", right_on="Symbol", how="right")

# If a ticker had no news, compound may be NaN. Fill with 0 for coloring.
df["compound"] = df["compound"].fillna(0.0)
df["neg"] = df["neg"].fillna(0.0)
df["neu"] = df["neu"].fillna(0.0)
df["pos"] = df["pos"].fillna(0.0)

df = df.rename(columns={
    "compound": "Sentiment Score",
    "neg": "Negative",
    "neu": "Neutral",
    "pos": "Positive",
})

# If Market Cap missing, give small value so treemap can still render
df["Market Cap"] = df["Market Cap"].fillna(1.0)


# Treemap
fig = px.treemap(
    df,
    path=[px.Constant("Dow Jones"), "Sector", "Industry", "Symbol"],
    values="Market Cap",
    color="Sentiment Score",
    hover_data=["Company", "Negative", "Neutral", "Positive", "Sentiment Score"],
    color_continuous_scale=["#FF0000", "#000000", "#00FF00"],
    color_continuous_midpoint=0,
)

fig.data[0].customdata = df[["Company", "Negative", "Neutral", "Positive", "Sentiment Score"]].round(3)
fig.data[0].texttemplate = "%{label}<br>%{customdata[4]}"
fig.update_traces(textposition="middle center")
fig.update_layout(margin=dict(t=30, l=10, r=10, b=10), font_size=20)

# Write HTML
now = dt.datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
timezone_string = dt.datetime.now().astimezone().tzname()

out_html = "dow_jones_live_sentiment.html"

with open(out_html, "w", encoding="utf-8") as f:
    title = "<h1>Dow Jones Stock Sentiment Dashboard</h1>"
    updated = f"<h2>Last updated: {dt_string} (Timezone: {timezone_string})</h2>"
    description = (
        "This dashboard is updated every half an hour with sentiment analysis performed "
        "on latest scraped news headlines from the FinViz website.<br><br>"
    )
    code_links = (
        '<a href="https://medium.com/datadriveninvestor/use-github-actions-to-create-a-live-stock-sentiment-dashboard-online-580a08457650">'
        "Explanatory Article</a> | "
        '<a href="https://github.com/damianboh/dow_jones_live_stock_sentiment_treemap">Source Code</a>'
    )
    author = ' | Created by Damian Boh, check out my <a href="https://damianboh.github.io/">GitHub Page</a>'

    f.write(title + updated + description + code_links + author)
    f.write(fig.to_html(full_html=False, include_plotlyjs="cdn"))

print(f"Done. Wrote: {out_html}")
