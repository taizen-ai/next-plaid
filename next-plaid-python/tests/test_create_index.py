import json

import numpy as np
import pytest

next_plaid = pytest.importorskip("next_plaid")


def _docs(n: int, tokens: int, dim: int, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    docs = []
    for _ in range(n):
        x = rng.standard_normal((tokens, dim)).astype(np.float32)
        docs.append(x / np.linalg.norm(x, axis=1, keepdims=True))
    return docs


def test_create_index_from_precomputed_centroids(tmp_path):
    docs = _docs(n=64, tokens=12, dim=128)
    rng = np.random.default_rng(1)
    centroids = rng.standard_normal((32, 128)).astype(np.float32)
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)

    next_plaid.create_index(docs, centroids, str(tmp_path), nbits=4, seed=42, force_cpu=True)

    meta = json.loads((tmp_path / "metadata.json").read_text())
    assert meta["num_documents"] == 64
    assert np.load(tmp_path / "centroids.npy").shape == (32, 128)
    assert (tmp_path / "ivf.npy").exists()
    assert not (tmp_path / "metadata.db").exists()


def test_module_exposes_only_create_index():
    public = {name for name in dir(next_plaid) if not name.startswith("_")}
    assert "create_index" in public
    assert "create_metadata" not in public
