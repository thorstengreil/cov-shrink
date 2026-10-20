"""
Blockbuster Constant Variance Covariance Shrinkage (BB)

Description:
    Linear shrinkage estimator that determines a given number of groups
    endogenously (via clustering) and applies CVC shrinkage within each group.

Reference:
    De Nard, G. (2022). Oops! I shrunk the sample covariance matrix again:
    Blockbuster meets shrinkage. Journal of Financial Econometrics, 20(4), 569–611.
    https://doi.org/10.1093/jjfinec/nbaa020

Implementation:
    Custom implementation developed for the working paper:
    Greil, T., & Grunow, M. (2025). "Improving Scenario Generation in Supply Chain
    Planning: The Power of Covariance Shrinkage". Available at SSRN 5765865.
    http://dx.doi.org/10.2139/ssrn.5765865

License:
    MIT License (see repository root LICENSE file).
    Copyright (c) 2026 Thorsten Greil (thorstengreil).
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

try:
    from src.ext import ext
except (ImportError, ModuleNotFoundError):
    try:
        from .ext import ext
    except (ImportError, ModuleNotFoundError):
        from ext import ext

try:
    from sklearn.cluster import KMeans
    _HAS_SKLEARN_KMEANS = True
except ImportError:
    _HAS_SKLEARN_KMEANS = False
    KMeans = None


def _kmeans_numpy(X: np.ndarray, n_clusters: int, random_state: int = 0, max_iter: int = 100) -> np.ndarray:
    """Fallback pure NumPy K-means implementation when scikit-learn is not available."""
    rng = np.random.default_rng(random_state)
    n_samples = X.shape[0]
    if n_clusters >= n_samples:
        return np.arange(n_samples)

    # K-means++ initialization
    centers = [X[rng.integers(0, n_samples)]]
    for _ in range(1, n_clusters):
        dist_sq = np.min([np.sum((X - c) ** 2, axis=1) for c in centers], axis=0)
        prob = dist_sq / np.sum(dist_sq) if np.sum(dist_sq) > 0 else np.ones(n_samples) / n_samples
        next_idx = rng.choice(n_samples, p=prob)
        centers.append(X[next_idx])
    centers = np.array(centers)

    labels = np.zeros(n_samples, dtype=int)
    for _ in range(max_iter):
        # Assign closest center
        dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        new_labels = np.argmin(dists, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        # Recompute centers
        for k in range(n_clusters):
            mask = labels == k
            if np.any(mask):
                centers[k] = np.mean(X[mask], axis=0)

    return labels


def blockbuster_clustering(
    X: Union[pd.DataFrame, np.ndarray],
    n_clusters: int,
    random_state: int = 0,
) -> Dict[int, List[Any]]:
    """
    Cluster variables endogenously via spectral embedding and K-means (De Nard, 2022).

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray of shape (n_samples, n_features)
        Input observations matrix.
    n_clusters : int
        Number of clusters (groups) K.
    random_state : int, default=0
        Random seed for K-means initialization reproducibility.

    Returns
    -------
    groups_dict : dict
        Mapping from cluster ID (0 to K-1) to list of variable column names or indices.
    """
    is_dataframe = isinstance(X, pd.DataFrame)
    if is_dataframe:
        orig_columns = list(X.columns)
        X_arr = X.to_numpy(dtype=float)
    else:
        orig_columns = None
        X_arr = np.asarray(X, dtype=float)

    n_features = X_arr.shape[1]
    K = max(1, min(n_clusters, n_features))

    if K == 1:
        keys = orig_columns if orig_columns is not None else list(range(n_features))
        return {0: keys}

    # Step 1: Sample covariance matrix
    X_centered = X_arr - np.mean(X_arr, axis=0)
    sample_cov = np.dot(X_centered.T, X_centered) / (X_arr.shape[0] - 1)
    sample_cov = (sample_cov + sample_cov.T) / 2.0

    # Step 2: Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(sample_cov)

    # Get the K largest eigenvalues and their corresponding eigenvectors
    U = eigenvectors[:, -K:]

    # Step 3: Standardize each row of U by its Euclidean (L2) norm
    row_norms = np.linalg.norm(U, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1.0
    U_norm = U / row_norms

    # Step 4: Apply K-means to the rows of U_norm
    if _HAS_SKLEARN_KMEANS:
        kmeans = KMeans(n_clusters=K, random_state=random_state, n_init=10).fit(U_norm)
        labels = kmeans.labels_
    else:
        labels = _kmeans_numpy(U_norm, n_clusters=K, random_state=random_state)

    # Group assignments dictionary
    result: Dict[int, List[Any]] = {k: [] for k in range(K)}
    col_names = orig_columns if orig_columns is not None else list(range(n_features))
    for col_name, cluster_id in zip(col_names, labels):
        result[cluster_id].append(col_name)

    return result


def bb(
    X: Union[pd.DataFrame, np.ndarray],
    n_clusters: int = 2,
    random_state: int = 0,
    assume_centered: bool = False,
    use_sample_covariance: bool = True,
    return_shrinkage: bool = False,
) -> Union[pd.DataFrame, np.ndarray, Tuple[Union[pd.DataFrame, np.ndarray], float]]:
    """
    Blockbuster Constant Variance Covariance (BB) shrinkage estimator (De Nard, 2022).

    Endogenously clusters variables into a predefined number of groups K via
    spectral decomposition of the sample covariance matrix, then applies CVC
    shrinkage within and across clusters.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray of shape (n_samples, n_features)
        Input observations matrix.
    n_clusters : int, default=2
        Number of variable groups K to cluster endogenously.
    random_state : int, default=0
        Random state seed for K-means initialization.
    assume_centered : bool, default=False
        If True, data will not be centered before covariance computation.
    use_sample_covariance : bool, default=True
        If True, uses (n - 1) degrees of freedom (ddof=1). If False, ddof=0.
    return_shrinkage : bool, default=False
        If True, returns a tuple of (shrunk_cov, shrinkage_intensity).

    Returns
    -------
    shrunk_cov : pd.DataFrame or np.ndarray of shape (n_features, n_features)
        The shrunken covariance matrix estimate. Returns pd.DataFrame if input
        X is a DataFrame, otherwise np.ndarray.
    shrinkage : float (optional)
        The optimal shrinkage intensity delta in [0, 1]. Only returned if
        return_shrinkage=True.
    """
    # Cluster variables endogenously
    groups_dict = blockbuster_clustering(X, n_clusters=n_clusters, random_state=random_state)

    # Apply grouped CVC shrinkage
    return ext(
        X,
        groups=groups_dict,
        assume_centered=assume_centered,
        use_sample_covariance=use_sample_covariance,
        return_shrinkage=return_shrinkage,
    )


# Uppercase alias
BB = bb


if __name__ == '__main__':
    print("Testing Blockbuster CVC Shrinkage (BB)...")
    np.random.seed(42)
    N_samples, p_vars = 60, 20
    col_names = [f"Var_{i+1}" for i in range(p_vars)]
    sim_data = pd.DataFrame(
        np.random.randn(N_samples, p_vars),
        columns=col_names,
    )

    cov_matrix, shrinkage_val = bb(sim_data, n_clusters=3, return_shrinkage=True)
    print(f"Input shape:         ({N_samples}, {p_vars})")
    print(f"Output shape:        {cov_matrix.shape}")
    print(f"Shrinkage intensity: {shrinkage_val:.6f}")
    print(f"Trace:               {np.trace(cov_matrix.to_numpy()):.6f}")
    print(f"Condition #:         {np.linalg.cond(cov_matrix.to_numpy()):.4f}")
    print("BB self-test completed successfully.")
