"""Project configuration module."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

# Base directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
INDEX_DIR = BASE_DIR / "index"
LOGS_DIR = BASE_DIR / "logs"

for path in [DATA_DIR, RAW_DIR, PROC_DIR, MODELS_DIR, INDEX_DIR, LOGS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Feature flags and toggles
FAST_MODE = os.getenv("FAST_MODE", "1") not in {"0", "false", "False"}
ENABLE_BERTOPIC = os.getenv("ENABLE_BERTOPIC", "1") not in {"0", "false", "False"}
ENABLE_CAUSAL = os.getenv("ENABLE_CAUSAL", "1") not in {"0", "false", "False"}

# City list for tagging - focus on Indian metro areas
CITIES: List[str] = [
    "Mumbai",
    "Delhi",
    "Bengaluru",
    "Chennai",
    "Hyderabad",
    "Pune",
    "Kolkata",
    "Ahmedabad",
    "Jaipur",
    "Surat",
]

# Scraping configuration
@dataclass
class ScrapeConfig:
    reddit_query: str = os.getenv("REDDIT_QUERY", "city public services")
    reddit_limit: int = int(os.getenv("REDDIT_LIMIT", "200"))
    news_sources: List[str] = tuple(
        os.getenv(
            "NEWS_SOURCES",
            "https://www.thehindu.com,https://indianexpress.com,https://www.deccanherald.com",
        ).split(",")
    )
    news_limit_per_source: int = int(os.getenv("NEWS_LIMIT", "100"))


SCRAPE_CONFIG = ScrapeConfig()

# NLP models
SPACY_MODEL_SMALL = os.getenv("SPACY_MODEL_SMALL", "en_core_web_sm")
SPACY_MODEL_TRF = os.getenv("SPACY_MODEL_TRF", "en_core_web_trf")
SBERT_MODEL_NAME = os.getenv("SBERT_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

# Topic modeling
TOPIC_PARAMS: Dict[str, int] = {
    "num_topics": int(os.getenv("LDA_TOPICS", "10")),
    "passes": int(os.getenv("LDA_PASSES", "2")),
    "iterations": int(os.getenv("LDA_ITERATIONS", "50")),
}

# Proxy support
HTTP_PROXY = os.getenv("HTTP_PROXY")
HTTPS_PROXY = os.getenv("HTTPS_PROXY")

__all__ = [
    "BASE_DIR",
    "DATA_DIR",
    "RAW_DIR",
    "PROC_DIR",
    "MODELS_DIR",
    "INDEX_DIR",
    "FAST_MODE",
    "ENABLE_BERTOPIC",
    "ENABLE_CAUSAL",
    "CITIES",
    "SCRAPE_CONFIG",
    "SPACY_MODEL_SMALL",
    "SPACY_MODEL_TRF",
    "SBERT_MODEL_NAME",
    "TOPIC_PARAMS",
    "HTTP_PROXY",
    "HTTPS_PROXY",
]
