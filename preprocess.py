"""Data preprocessing pipeline."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List

import pandas as pd
import spacy
from nltk import download as nltk_download
from nltk.corpus import stopwords

from config import CITIES, FAST_MODE, PROC_DIR, RAW_DIR, SPACY_MODEL_SMALL
from utils_io import log, load_df, save_df


CLEAN_REGEX = re.compile(r"https?://\S+|www\.\S+")
PUNCT_REGEX = re.compile(r"[^A-Za-z0-9\s]")
CITY_REGEX = re.compile(r"|".join(re.escape(city) for city in CITIES), flags=re.IGNORECASE)

try:
    STOPWORDS = set(stopwords.words("english"))
except LookupError:  # pragma: no cover - executed in fresh envs
    nltk_download("stopwords")
    STOPWORDS = set(stopwords.words("english"))

_nlp = None


def get_nlp():
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load(SPACY_MODEL_SMALL, disable=["parser", "textcat"])
        except OSError as exc:  # fallback informative message
            log(
                "spaCy model missing. Run `python -m spacy download en_core_web_sm` or set SPACY_MODEL_SMALL."
            )
            raise exc
    return _nlp


def clean_text(text: str) -> str:
    text = CLEAN_REGEX.sub(" ", text)
    text = PUNCT_REGEX.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def lemmatize(text: str) -> str:
    nlp = get_nlp()
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if token.lemma_ and token.lemma_ not in STOPWORDS]
    return " ".join(tokens)


def guess_city(text: str, existing: str | None = None) -> str | None:
    if existing:
        return existing
    match = CITY_REGEX.search(text)
    if match:
        return match.group(0)
    doc = get_nlp()(text)
    for ent in doc.ents:
        if ent.label_ in {"GPE", "LOC"}:
            return ent.text
    return None


def preprocess_frames(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    records: List[pd.DataFrame] = []
    for frame in frames:
        if frame.empty:
            continue
        frame = frame.copy()
        frame["text"] = frame["text"].fillna("")
        frame["clean_text"] = frame["text"].apply(clean_text)
        frame["lemma_text"] = frame["clean_text"].apply(lemmatize)
        frame["city"] = frame.apply(lambda row: guess_city(row["clean_text"], row.get("city")), axis=1)
        records.append(frame)
    if not records:
        return pd.DataFrame(columns=["platform", "title", "lemma_text", "city"])
    df = pd.concat(records, ignore_index=True)
    log(f"Preprocessed {len(df):,} rows")
    return df


def preprocess_data(reddit_path: Path | None = None, news_path: Path | None = None) -> Path:
    if FAST_MODE and reddit_path is None and news_path is None:
        base = Path(__file__).resolve().parent / "data" / "sample"
        reddit_path = base / "sample_reddit.parquet"
        news_path = base / "sample_news.parquet"
        log("FAST_MODE=1 -> using sample parquet files for preprocessing")

    frames: List[pd.DataFrame] = []
    for path in filter(None, [reddit_path, news_path]):
        frames.append(load_df(path))

    if not frames:
        raw_files = sorted(RAW_DIR.glob("*.parquet"))
        if not raw_files:
            raise FileNotFoundError("No raw data available for preprocessing")
        frames = [load_df(path) for path in raw_files]

    df = preprocess_frames(frames)
    corpus_path = PROC_DIR / "corpus.parquet"
    save_df(df, corpus_path)
    small_path = PROC_DIR / "corpus_small.parquet"
    columns = [col for col in ["platform", "title", "lemma_text", "city", "url"] if col in df.columns]
    save_df(df[columns], small_path)
    return corpus_path


if __name__ == "__main__":
    out = preprocess_data()
    log(f"Preprocessed corpus saved to {out}")
