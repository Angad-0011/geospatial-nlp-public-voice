"""Topic modeling (LDA + BERTopic)."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from config import ENABLE_BERTOPIC, FAST_MODE, MODELS_DIR, PROC_DIR, TOPIC_PARAMS
from utils_io import log, load_parquet, save_parquet

try:  # gensim imports
    from gensim.corpora.dictionary import Dictionary
    from gensim.models import LdaModel
except ImportError as exc:  # pragma: no cover
    Dictionary = None  # type: ignore
    LdaModel = None  # type: ignore
    _GENSIM_ERROR = exc
else:
    _GENSIM_ERROR = None

try:
    from bertopic import BERTopic
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP
    import hdbscan
except ImportError:  # pragma: no cover
    BERTopic = None  # type: ignore
    CountVectorizer = None  # type: ignore
    UMAP = None  # type: ignore
    hdbscan = None  # type: ignore


def _tokenize(text: str) -> list[str]:
    return [token for token in text.split() if token]


def run_lda(corpus_df: pd.DataFrame) -> Tuple[pd.DataFrame, Path]:
    if Dictionary is None or LdaModel is None:
        log("gensim not installed. Install via `pip install gensim`. Skipping LDA.")
        corpus_df["lda_topic"] = -1
        out_path = PROC_DIR / "corpus_lda.parquet"
        save_parquet(corpus_df, out_path)
        return corpus_df, out_path

    tokens = corpus_df["lemma_text"].fillna("").apply(_tokenize)
    dictionary = Dictionary(tokens)
    bow_corpus = [dictionary.doc2bow(text) for text in tokens]
    log(f"Training LDA with {TOPIC_PARAMS['num_topics']} topics on {len(tokens)} documents")
    lda_model = LdaModel(
        corpus=bow_corpus,
        id2word=dictionary,
        num_topics=TOPIC_PARAMS["num_topics"],
        passes=TOPIC_PARAMS["passes"],
        iterations=TOPIC_PARAMS["iterations"],
        random_state=42,
    )

    topics = [int(max(lda_model[bow], key=lambda x: x[1])[0]) if bow else -1 for bow in bow_corpus]
    corpus_df["lda_topic"] = topics

    MODELS_DIR.mkdir(exist_ok=True, parents=True)
    lda_model.save(str(MODELS_DIR / "lda_model"))
    dictionary.save(str(MODELS_DIR / "lda_dictionary.dict"))

    out_path = PROC_DIR / "corpus_lda.parquet"
    save_parquet(corpus_df, out_path)
    return corpus_df, out_path


def run_bertopic(corpus_df: pd.DataFrame, embeddings: np.ndarray | None = None) -> Path | None:
    if not ENABLE_BERTOPIC:
        log("ENABLE_BERTOPIC=0 -> skipping BERTopic")
        return None
    if FAST_MODE and len(corpus_df) > 200:
        log("FAST_MODE=1 and dataset >200 docs -> skipping BERTopic")
        return None
    if BERTopic is None:
        log("bertopic not installed. Install via `pip install bertopic`. Skipping BERTopic")
        return None
    if embeddings is None:
        emb_path = MODELS_DIR / "sbert_embeddings.npy"
        if not emb_path.exists():
            log("Embeddings file missing for BERTopic")
            return None
        embeddings = np.load(emb_path)

    topic_model = BERTopic(
        umap_model=UMAP(random_state=42, n_neighbors=15, n_components=5, min_dist=0.0),
        hdbscan_model=hdbscan.HDBSCAN(min_cluster_size=5, prediction_data=True),
        vectorizer_model=CountVectorizer(stop_words="english"),
        verbose=False,
    )
    topics, _ = topic_model.fit_transform(corpus_df["lemma_text"].tolist(), embeddings)
    corpus_df["bertopic"] = topics

    MODELS_DIR.mkdir(exist_ok=True, parents=True)
    topic_model.save(MODELS_DIR / "bertopic_model")

    out_path = PROC_DIR / "corpus_bertopic.parquet"
    save_parquet(corpus_df, out_path)
    return out_path


def topic_pipeline(embeddings: np.ndarray | None = None) -> tuple[pd.DataFrame, Path | None]:
    corpus_path = PROC_DIR / "corpus.parquet"
    if not corpus_path.exists():
        raise FileNotFoundError("corpus.parquet missing. Run preprocessing first.")
    corpus = load_parquet(corpus_path)
    corpus, lda_path = run_lda(corpus)
    bertopic_path = run_bertopic(corpus.copy(), embeddings)
    return corpus, bertopic_path or lda_path


if __name__ == "__main__":
    topic_pipeline()
