"""NER and sentiment enrichment."""
from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
import spacy
from nltk import download as nltk_download
from nltk.sentiment import SentimentIntensityAnalyzer

from config import PROC_DIR, SPACY_MODEL_SMALL
from utils_io import log, load_parquet, save_parquet

try:  # pragma: no cover
    nltk_download("vader_lexicon")
except Exception:
    pass

_nlp = None
_sia = SentimentIntensityAnalyzer()


def get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load(SPACY_MODEL_SMALL)
    return _nlp


def extract_entities(text: str) -> List[str]:
    doc = get_nlp()(text)
    return [ent.text for ent in doc.ents if ent.label_ in {"GPE", "LOC", "ORG", "PERSON", "LAW"}]


def compute_sentiment(text: str) -> float:
    return _sia.polarity_scores(text).get("compound", 0.0)


def enrich_corpus(corpus_path: Path | None = None) -> Path:
    if corpus_path is None:
        corpus_path = PROC_DIR / "corpus.parquet"
    if not corpus_path.exists():
        raise FileNotFoundError("Corpus not found. Run preprocess step first.")

    df = load_parquet(corpus_path)
    df["entities"] = df["clean_text"].fillna("").apply(extract_entities)
    df["sentiment"] = df["clean_text"].fillna("").apply(compute_sentiment)

    out_path = PROC_DIR / "corpus_ner_sent.parquet"
    save_parquet(df, out_path)
    return out_path


if __name__ == "__main__":
    path = enrich_corpus()
    log(f"NER + sentiment saved to {path}")
