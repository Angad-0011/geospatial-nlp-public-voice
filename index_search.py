"""FAISS cosine index and CLI search."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from config import INDEX_DIR, MODELS_DIR, PROC_DIR, SBERT_MODEL_NAME
from utils_io import log, load_parquet

try:  # pragma: no cover
    import faiss
except ImportError:
    faiss = None  # type: ignore

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # type: ignore


INDEX_PATH = INDEX_DIR / "sbert.cosine.faiss"


def _load_embeddings() -> Tuple[np.ndarray, pd.DataFrame]:
    emb_path = MODELS_DIR / "sbert_embeddings.npy"
    if not emb_path.exists():
        raise FileNotFoundError("Embeddings not found. Run embeddings.py first.")
    corpus_path = PROC_DIR / "corpus.parquet"
    if not corpus_path.exists():
        raise FileNotFoundError("Corpus not found. Run preprocess.py first.")
    embeddings = np.load(emb_path).astype(np.float32)
    corpus = load_parquet(corpus_path)
    return embeddings, corpus


def build_index() -> Path:
    if faiss is None:
        raise ImportError("faiss is not installed. Install via `pip install faiss-cpu`.")
    embeddings, _ = _load_embeddings()
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    log(f"FAISS index with {index.ntotal} vectors saved to {INDEX_PATH}")
    return INDEX_PATH


def load_index() -> "faiss.Index":
    if faiss is None:
        raise ImportError("faiss is not installed. Install via `pip install faiss-cpu`.")
    if not INDEX_PATH.exists():
        log("Index not found; building one now.")
        build_index()
    return faiss.read_index(str(INDEX_PATH))


def encode_query(query: str) -> np.ndarray:
    if SentenceTransformer is None:
        raise ImportError("sentence-transformers missing; install via `pip install sentence-transformers`.")
    model = SentenceTransformer(SBERT_MODEL_NAME)
    vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    return vec.astype(np.float32)


def search(query: str, city: str | None, top_k: int) -> pd.DataFrame:
    _, corpus = _load_embeddings()
    if faiss is None:
        raise ImportError("faiss is not installed; cannot perform search.")
    index = load_index()
    query_vec = encode_query(query)
    scores, indices = index.search(query_vec, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        row = corpus.iloc[int(idx)]
        if city and row.get("city") and city.lower() not in str(row.get("city", "")).lower():
            continue
        results.append({
            "score": float(score),
            "city": row.get("city"),
            "title": row.get("title"),
            "url": row.get("url"),
        })
    return pd.DataFrame(results)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Semantic search over city discourse")
    parser.add_argument("--build", action="store_true", help="Build the FAISS index")
    parser.add_argument("--query", type=str, help="Query text")
    parser.add_argument("--city", type=str, default=None, help="Optional city filter")
    parser.add_argument("--k", type=int, default=5, help="Top k results")
    args = parser.parse_args(argv)

    if args.build:
        try:
            build_index()
        except Exception as exc:
            log(f"Failed to build index: {exc}")
            return 1
        return 0

    if not args.query:
        parser.error("--query required unless --build is used")

    try:
        results = search(args.query, args.city, args.k)
    except Exception as exc:
        log(f"Search failed: {exc}")
        return 1

    if results.empty:
        print("No results found")
        return 0

    output = results.head(args.k)
    print(output[["score", "city", "title", "url"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
