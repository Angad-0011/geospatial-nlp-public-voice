"""End-to-end pipeline runner."""
from __future__ import annotations

from config import ENABLE_BERTOPIC, FAST_MODE
from scraper_reddit import scrape_reddit
from scraper_news import scrape_news
from preprocess import preprocess_data
from embeddings import build_embeddings
from topic_modeling import topic_pipeline
from ner_sentiment import enrich_corpus
from index_search import build_index
from utils_io import log


def run_pipeline() -> None:
    log("Starting pipeline")
    reddit_path = scrape_reddit()
    news_path = scrape_news()
    log(f"Reddit path: {reddit_path}")
    log(f"News path: {news_path}")

    corpus_path = preprocess_data(reddit_path, news_path)
    log(f"Corpus saved to {corpus_path}")

    corpus, embeddings = build_embeddings()

    topic_pipeline(embeddings if ENABLE_BERTOPIC else None)
    enrich_corpus(corpus_path)

    try:
        build_index()
    except Exception as exc:
        log(f"Index build skipped: {exc}")

    log("Pipeline completed")


if __name__ == "__main__":
    run_pipeline()
