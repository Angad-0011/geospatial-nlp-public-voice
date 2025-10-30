import numpy as np
import pytest

faiss = pytest.importorskip("faiss")


def test_faiss_cosine_index():
    vectors = np.array([
        [1.0, 0.0, 0.0],
        [0.9, 0.1, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.7, 0.2, 0.1],
    ], dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    index = faiss.IndexFlatIP(3)
    index.add(vectors)
    query = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    query /= np.linalg.norm(query, axis=1, keepdims=True)
    scores, indices = index.search(query, 1)
    assert indices[0][0] == 0
    assert pytest.approx(scores[0][0], 1e-5) == 1.0
