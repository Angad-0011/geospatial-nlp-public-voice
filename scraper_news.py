"""Simple news scraping module."""
from __future__ import annotations

import re
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Set

import pandas as pd
import requests
from bs4 import BeautifulSoup

from config import CITIES, FAST_MODE, HTTP_PROXY, HTTPS_PROXY, RAW_DIR, SCRAPE_CONFIG
from utils_io import log, save_parquet

SESSION = requests.Session()
if HTTP_PROXY:
    SESSION.proxies["http"] = HTTP_PROXY
if HTTPS_PROXY:
    SESSION.proxies["https"] = HTTPS_PROXY

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; geospatial-nlp/0.1)"}
CITY_REGEX = re.compile(r"|".join(re.escape(city) for city in CITIES), flags=re.IGNORECASE)


def _extract_links(url: str, html: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    links: List[str] = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("http") and any(source in href for source in SCRAPE_CONFIG.news_sources):
            links.append(href)
    return links


def _fetch(url: str) -> str:
    try:
        resp = SESSION.get(url, headers=HEADERS, timeout=10)
        if resp.status_code != 200:
            log(f"Skipping {url} status={resp.status_code}")
            return ""
        return resp.text
    except requests.RequestException as exc:
        log(f"Request failed for {url}: {exc}")
        return ""


def _extract_article(url: str, html: str) -> dict | None:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.find("title")
    paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
    text = " ".join(paragraphs)
    if not text or len(text) < 120:
        return None
    match = CITY_REGEX.search(text)
    city = match.group(0) if match else None
    return {
        "platform": "news",
        "city": city,
        "title": title.get_text(strip=True) if title else url,
        "text": text[:5000],
        "url": url,
    }


def scrape_news(force: bool = False) -> Path:
    if FAST_MODE and not force:
        sample_path = Path(__file__).resolve().parent / "data" / "sample" / "sample_news.parquet"
        log(f"FAST_MODE=1 -> using sample data at {sample_path}")
        return sample_path

    visited: Set[str] = set()
    articles: List[dict] = []
    queue: deque[str] = deque(SCRAPE_CONFIG.news_sources)

    while queue and len(articles) < SCRAPE_CONFIG.news_limit_per_source * len(SCRAPE_CONFIG.news_sources):
        url = queue.popleft()
        if url in visited:
            continue
        visited.add(url)
        html = _fetch(url)
        if not html:
            continue
        article = _extract_article(url, html)
        if article:
            articles.append(article)
        for link in _extract_links(url, html):
            if link not in visited and len(queue) < 1000:
                queue.append(link)
        if len(articles) % 20 == 0 and articles:
            log(f"Discovered {len(articles)} articles so far")

    df = pd.DataFrame(articles)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = RAW_DIR / f"news_{timestamp}.parquet"
    save_parquet(df, out_path)
    return out_path


if __name__ == "__main__":
    path = scrape_news()
    log(f"News data saved to {path}")
