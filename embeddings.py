"""SBERT embeddings generation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import FAST_MODE, MODELS_DIR, PROC_DIR, SBERT_MODEL_NAME
from utils_io import log, load_parquet

try:  # optional heavy import guard
    from sentence_transformers import SentenceTransformer
except ImportError as exc:  # pragma: no cover
    SentenceTransformer = None  # type: ignore
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12
    return matrix / norms


def build_embeddings(force: bool = False) -> tuple[pd.DataFrame, np.ndarray]:
    corpus_path = PROC_DIR / "corpus.parquet"
    if not corpus_path.exists():
        raise FileNotFoundError("Processed corpus not found. Run preprocess step first.")

    corpus = load_parquet(corpus_path)
    texts = corpus["lemma_text"].fillna("").tolist()

    if SentenceTransformer is None:
        log("sentence-transformers not installed. Install via `pip install sentence-transformers`. Skipping embeddings.")
        embeddings = np.zeros((len(texts), 1), dtype=np.float32)
    else:
        model = SentenceTransformer(SBERT_MODEL_NAME)
        batch_size = 16 if FAST_MODE else 64
        log(f"Encoding {len(texts)} documents with SBERT model {SBERT_MODEL_NAME}")
        embeddings = model.encode(texts, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=False)
        embeddings = _normalize(embeddings.astype(np.float32))

    models_dir = MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)
    emb_path = models_dir / "sbert_embeddings.npy"
    np.save(emb_path, embeddings)
    log(f"Saved embeddings to {emb_path}")

    slim_path = MODELS_DIR / "sbert_embeddings_small.npy"
    np.save(slim_path, embeddings[: min(50, len(embeddings))])
    log(f"Saved slim embeddings to {slim_path}")

    small_corpus_path = PROC_DIR / "corpus_small.parquet"
    if corpus_path.exists() and not small_corpus_path.exists():
        corpus[["platform", "title", "lemma_text", "city", "url"]].to_parquet(small_corpus_path, index=False)

    return corpus, embeddings


if __name__ == "__main__":
    build_embeddings()
