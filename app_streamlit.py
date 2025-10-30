"""Streamlit UI for semantic search."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

from config import INDEX_DIR, MODELS_DIR, PROC_DIR, SBERT_MODEL_NAME
from utils_io import log

try:
    import faiss
except ImportError:
    faiss = None  # type: ignore

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # type: ignore


@st.cache_resource(show_spinner=False)
def load_index():
    index_path = INDEX_DIR / "sbert.cosine.faiss"
    if faiss is None or not index_path.exists():
        return None
    return faiss.read_index(str(index_path))


@st.cache_resource(show_spinner=False)
def load_model():
    if SentenceTransformer is None:
        return None
    return SentenceTransformer(SBERT_MODEL_NAME)


@st.cache_data(show_spinner=False)
def load_corpus() -> pd.DataFrame:
    path = PROC_DIR / "corpus.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def encode_query(model, query: str) -> Optional[np.ndarray]:
    if model is None:
        return None
    return model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)


st.set_page_config(page_title="City Discourse Explorer", layout="wide")
st.title("City Discourse Explorer")

corpus = load_corpus()
model = load_model()
index = load_index()

if corpus.empty:
    st.warning("Processed corpus not found. Run the pipeline first.")
else:
    st.success(f"Loaded corpus with {len(corpus)} documents")

query = st.text_input("Search query", "waste management infrastructure")
city_filter = st.selectbox("City filter", ["All"] + sorted({c for c in corpus.get("city", []) if c}), index=0)
top_k = st.slider("Top K", min_value=5, max_value=50, value=10, step=5)

if st.button("Search"):
    if index is None or model is None:
        st.error("Missing FAISS index or SBERT model. Run the pipeline and build the index.")
    else:
        query_vec = encode_query(model, query)
        if query_vec is None:
            st.error("Unable to encode query. Install sentence-transformers.")
        else:
            scores, indices = index.search(query_vec, top_k)
            rows = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:
                    continue
                row = corpus.iloc[int(idx)]
                if city_filter != "All" and city_filter.lower() not in str(row.get("city", "")).lower():
                    continue
                rows.append({
                    "Score": float(score),
                    "City": row.get("city"),
                    "Title": row.get("title"),
                    "URL": row.get("url"),
                })
            if not rows:
                st.info("No results found")
            else:
                st.dataframe(pd.DataFrame(rows))

st.markdown(
    "---
Need to (re)build artifacts? Run `make run` for sample pipeline and `python index_search.py --build` to rebuild the index."
)
