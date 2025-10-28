"""Reddit scraping utilities using snscrape."""
from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd

from config import FAST_MODE, RAW_DIR, SCRAPE_CONFIG
from utils_io import log, save_parquet


def _run_snscrape(query: str, limit: int) -> List[dict]:
    cmd = [
        "python",
        "-m",
        "snscrape",
        "--jsonl",
        "--max-results",
        str(limit),
        "reddit-search",
        query,
    ]
    log(f"Running snscrape with query='{query}' and limit={limit}")
    proc = subprocess.run(cmd, capture_output=True, check=False, text=True)
    if proc.returncode != 0:
        log("snscrape failed; returning empty list")
        log(proc.stderr)
        return []
    rows = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
    log(f"snscrape returned {len(rows)} rows")
    return rows


def scrape_reddit(force: bool = False) -> Path:
    if FAST_MODE and not force:
        sample_path = Path(__file__).resolve().parent / "data" / "sample" / "sample_reddit.parquet"
        log(f"FAST_MODE=1 -> using sample data at {sample_path}")
        return sample_path

    rows = _run_snscrape(SCRAPE_CONFIG.reddit_query, SCRAPE_CONFIG.reddit_limit)
    if not rows:
        log("No rows scraped from Reddit")
        df = pd.DataFrame(columns=["platform", "title", "text"])
    else:
        df = pd.DataFrame(rows)
        df = df[
            [
                "id",
                "date",
                "title",
                "selftext",
                "url",
                "author",
                "subreddit",
                "score",
            ]
        ]
        df.rename(
            columns={
                "id": "post_id",
                "date": "created_utc",
                "selftext": "text",
            },
            inplace=True,
        )
        df["platform"] = "reddit"
        df["city"] = None
        df["created_utc"] = pd.to_datetime(df["created_utc"]).astype("datetime64[ms]")
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = RAW_DIR / f"reddit_{timestamp}.parquet"
    save_parquet(df, out_path)
    return out_path


if __name__ == "__main__":
    path = scrape_reddit()
    log(f"Reddit data saved to {path}")
