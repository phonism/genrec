"""
Residual Quantized K-Means (RQ-KMeans) for Semantic ID Generation.

A simpler alternative to RQ-VAE that directly applies residual K-means clustering
on item embeddings without an encoder/decoder. Used by OpenOneRec.

Key differences from RQ-VAE:
    - No neural network (encoder/decoder) — just K-means on raw embeddings
    - Much faster to fit (minutes vs hours of training)
    - Larger codebook (8192 vs 256) compensates for simpler quantization
"""
import numpy as np
import torch
from collections import Counter
from typing import List, NamedTuple
from sklearn.cluster import KMeans, MiniBatchKMeans


class RqKmeansOutput(NamedTuple):
    """RQ-KMeans assignment output."""
    sem_ids: np.ndarray   # [N, n_layers] int array of codebook indices


class RqKmeans:
    """
    Residual Quantized K-Means.

    Fits n_layers of K-means sequentially on residuals:
        Layer 0: K-means on original embeddings
        Layer i: K-means on residual from layer i-1

    Args:
        n_layers: Number of quantization layers (default 3)
        codebook_size: Number of centroids per layer (default 8192)
        random_state: Random seed for reproducibility
        max_iter: Max iterations for KMeans
        batch_size: Mini-batch size when backend="minibatch"
        backend: "full" uses sklearn KMeans (k-means++, n_init>1); "minibatch"
                 uses MiniBatchKMeans (legacy, large catalogs); "auto" picks
                 "full" when N*K < 5e7 else "minibatch".
        n_init: Number of KMeans restarts to pick best inertia (default 10).
        dead_reinit: If True, reseed unused centroids to farthest residual
                     points and run one extra Lloyd step. Mitigates dead clusters.
    """

    def __init__(
        self,
        n_layers: int = 3,
        codebook_size: int = 8192,
        random_state: int = 42,
        max_iter: int = 100,
        batch_size: int = 4096,
        backend: str = "auto",
        n_init: int = 10,
        dead_reinit: bool = True,
    ):
        self.n_layers = n_layers
        self.codebook_size = codebook_size
        self.random_state = random_state
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.backend = backend
        self.n_init = n_init
        self.dead_reinit = dead_reinit
        self.centroids: List[np.ndarray] = []  # [n_layers] each [codebook_size, D]

    def fit(self, embeddings: np.ndarray) -> "RqKmeans":
        """
        Fit RQ-KMeans on embeddings.

        Args:
            embeddings: [N, D] float array

        Returns:
            self
        """
        self.centroids = []
        residual = embeddings.copy()
        N = len(embeddings)

        backend = self.backend
        if backend == "auto":
            backend = "full" if N * self.codebook_size < 5e7 else "minibatch"

        for layer in range(self.n_layers):
            print(f"Fitting layer {layer}/{self.n_layers} [{backend}], "
                  f"N={N}, K={self.codebook_size}, "
                  f"residual norm: {np.linalg.norm(residual, axis=1).mean():.4f}")

            if backend == "full":
                kmeans = KMeans(
                    n_clusters=self.codebook_size,
                    init="k-means++",
                    n_init=self.n_init,
                    max_iter=self.max_iter,
                    random_state=self.random_state,
                    verbose=0,
                )
            else:
                kmeans = MiniBatchKMeans(
                    n_clusters=self.codebook_size,
                    random_state=self.random_state,
                    max_iter=self.max_iter,
                    batch_size=min(self.batch_size, N),
                    n_init=self.n_init,
                    verbose=0,
                )
            kmeans.fit(residual)
            centroids = kmeans.cluster_centers_.copy()
            codes = kmeans.predict(residual)

            # Dead cluster reinit: reseed any cluster that got 0 assignments
            # using the residual point currently farthest from its centroid.
            if self.dead_reinit:
                used = set(codes.tolist())
                dead = [c for c in range(self.codebook_size) if c not in used]
                if dead:
                    print(f"  Layer {layer}: reinit {len(dead)} dead clusters")
                    err = ((residual - centroids[codes]) ** 2).sum(axis=1)
                    far_idx = np.argsort(-err)
                    for i, c in enumerate(dead):
                        centroids[c] = residual[far_idx[i % N]]
                    # Reassign with patched centroids (one extra Lloyd step)
                    x_sq = (residual ** 2).sum(axis=1, keepdims=True)
                    c_sq = (centroids ** 2).sum(axis=1, keepdims=True).T
                    dist = x_sq + c_sq - 2 * residual @ centroids.T
                    codes = dist.argmin(axis=1)

            self.centroids.append(centroids)
            residual = residual - centroids[codes]

            # Per-layer usage report
            cnt = Counter(codes.tolist())
            used_count = len(cnt)
            top_freq = max(cnt.values())
            print(f"  Layer {layer} usage: {used_count}/{self.codebook_size} "
                  f"({100*used_count/self.codebook_size:.1f}%), "
                  f"max_freq={top_freq} ({100*top_freq/N:.1f}% of items)")

        print(f"Final residual norm: {np.linalg.norm(residual, axis=1).mean():.4f}")
        return self

    def assign(self, embeddings: np.ndarray) -> RqKmeansOutput:
        """
        Assign semantic IDs to embeddings using fitted centroids.

        Args:
            embeddings: [N, D] float array

        Returns:
            RqKmeansOutput with sem_ids [N, n_layers]
        """
        assert len(self.centroids) == self.n_layers, "Must call fit() first"

        N = embeddings.shape[0]
        sem_ids = np.zeros((N, self.n_layers), dtype=np.int64)
        residual = embeddings.copy()

        for layer in range(self.n_layers):
            # Compute distances to centroids: [N, codebook_size]
            # ||x - c||^2 = ||x||^2 + ||c||^2 - 2*x@c.T
            x_sq = (residual ** 2).sum(axis=1, keepdims=True)  # [N, 1]
            c_sq = (self.centroids[layer] ** 2).sum(axis=1, keepdims=True).T  # [1, K]
            dist = x_sq + c_sq - 2 * residual @ self.centroids[layer].T
            codes = dist.argmin(axis=1)
            sem_ids[:, layer] = codes
            residual = residual - self.centroids[layer][codes]

        return RqKmeansOutput(sem_ids=sem_ids)

    def save(self, path: str) -> None:
        """Save centroids to file."""
        torch.save({
            'n_layers': self.n_layers,
            'codebook_size': self.codebook_size,
            'centroids': [c.copy() for c in self.centroids],
        }, path)
        print(f"Saved RQ-KMeans to {path}")

    @classmethod
    def load(cls, path: str) -> "RqKmeans":
        """Load centroids from file."""
        # Backwards-compat: ckpts saved with numpy 2.x reference `numpy._core` in
        # the pickle stream, which doesn't exist in numpy 1.x. Alias them.
        if not hasattr(np, "_core"):
            import sys as _sys
            _sys.modules.setdefault("numpy._core", np.core)
            _sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
            _sys.modules.setdefault("numpy._core.numeric", np.core.numeric)
        state = torch.load(path, map_location='cpu', weights_only=False)
        obj = cls(
            n_layers=state['n_layers'],
            codebook_size=state['codebook_size'],
        )
        obj.centroids = state['centroids']
        print(f"Loaded RQ-KMeans from {path} "
              f"(layers={obj.n_layers}, codebook={obj.codebook_size}, "
              f"dim={obj.centroids[0].shape[1]})")
        return obj

    def collision_rate(self, embeddings: np.ndarray) -> dict:
        """
        Compute collision statistics.

        Returns:
            dict with 'total', 'unique', 'collision_rate', 'max_collision'
        """
        output = self.assign(embeddings)
        sem_ids = output.sem_ids  # [N, n_layers]

        # Convert to string keys for uniqueness check
        keys = ["-".join(str(c) for c in row) for row in sem_ids]
        unique = len(set(keys))
        total = len(keys)

        # Count max collision group size
        from collections import Counter
        counts = Counter(keys)
        max_collision = max(counts.values())

        return {
            'total': total,
            'unique': unique,
            'collision_rate': 1 - unique / total,
            'max_collision': max_collision,
        }
