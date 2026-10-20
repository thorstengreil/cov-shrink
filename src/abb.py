"""
Adaptive Blockbuster Constant Variance Covariance Shrinkage (ABB)

Description:
    Linear shrinkage estimator extending Blockbuster (BB) by determining both
    the optimal number of groups and the group assignments endogenously from data,
    then applying CVC shrinkage within each group.

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

from typing import Optional, Tuple, Union
import numpy as np
import pandas as pd

try:
    from src.bb import bb, blockbuster_clustering
except (ImportError, ModuleNotFoundError):
    try:
        from .bb import bb, blockbuster_clustering
    except (ImportError, ModuleNotFoundError):
        from bb import bb, blockbuster_clustering

try:
    from sklearn.neighbors import KernelDensity
    _HAS_SKLEARN_KDE = True
except ImportError:
    _HAS_SKLEARN_KDE = False
    KernelDensity = None


def blockbuster_nr_of_clusters(
    X: Union[pd.DataFrame, np.ndarray],
    verbose: bool = False,
) -> int:
    """
    Endogenously determine the optimal number of variable clusters (De Nard, 2022).

    Applies a Gaussian kernel density estimator to the standardized sample
    eigenvalues of the sample covariance matrix and counts the number of local
    minima M to set K = M + 1.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray of shape (n_samples, n_features)
        Input observations matrix.
    verbose : bool, default=False
        If True, prints diagnostic information about the detected clusters.

    Returns
    -------
    K : int
        The endogenously determined number of clusters (groups).
    """
    if isinstance(X, pd.DataFrame):
        X_arr = X.to_numpy(dtype=float)
    else:
        X_arr = np.asarray(X, dtype=float)

    n_features = X_arr.shape[1]
    if n_features <= 1:
        return 1

    # Step 1: Sample covariance matrix and eigenvalues
    X_centered = X_arr - np.mean(X_arr, axis=0)
    sample_cov = np.dot(X_centered.T, X_centered) / (X_arr.shape[0] - 1)
    sample_cov = (sample_cov + sample_cov.T) / 2.0

    eigenvalues, _ = np.linalg.eigh(sample_cov)

    # Step 2: Standardize the sorted sample eigenvalues (descending)
    sorted_eigenvalues = np.sort(eigenvalues)[::-1]
    norm_val = np.linalg.norm(sorted_eigenvalues)
    if norm_val == 0:
        return 1
    norm_sorted_eigenvalues = sorted_eigenvalues / norm_val

    # Step 3: Kernel density estimation with Silverman's rule of thumb bandwidth
    sigma = np.std(norm_sorted_eigenvalues)
    N = len(norm_sorted_eigenvalues)
    if sigma == 0 or N <= 2:
        return 1

    bandwidth = (4.0 * (sigma ** 5) / (3.0 * N)) ** (1.0 / 5.0)
    if bandwidth <= 0:
        bandwidth = 0.1

    if _HAS_SKLEARN_KDE:
        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(
            norm_sorted_eigenvalues[:, np.newaxis]
        )
        log_dens = kde.score_samples(norm_sorted_eigenvalues[:, np.newaxis])
    else:
        # Exact closed-form Gaussian KDE log density in pure NumPy
        diff = (norm_sorted_eigenvalues[:, None] - norm_sorted_eigenvalues[None, :]) / bandwidth
        dens = np.sum(np.exp(-0.5 * (diff ** 2)), axis=1) / (N * bandwidth * np.sqrt(2.0 * np.pi))
        dens = np.maximum(dens, 1e-300)
        log_dens = np.log(dens)

    # Find local minima in the log-density
    local_minima_indices = (np.diff(np.sign(np.diff(log_dens))) > 0).nonzero()[0]
    M = len(local_minima_indices)
    K = max(1, min(M + 1, n_features))

    if verbose:
        print(f"Number of blockbuster clusters: {K}")
        if K == 1:
            print("Single group detected (no distinct subgroup structure).")

    return K


def abb(
    X: Union[pd.DataFrame, np.ndarray],
    random_state: int = 0,
    assume_centered: bool = False,
    use_sample_covariance: bool = True,
    return_shrinkage: bool = False,
    verbose: bool = False,
) -> Union[pd.DataFrame, np.ndarray, Tuple[Union[pd.DataFrame, np.ndarray], float]]:
    """
    Adaptive Blockbuster Constant Variance Covariance (ABB) shrinkage estimator (De Nard, 2022).

    Extends Blockbuster (BB) by determining both the optimal number of groups K
    and the group assignments endogenously from data, then applying CVC shrinkage
    within and across groups.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray of shape (n_samples, n_features)
        Input observations matrix.
    random_state : int, default=0
        Random state seed for K-means initialization.
    assume_centered : bool, default=False
        If True, data will not be centered before covariance computation.
    use_sample_covariance : bool, default=True
        If True, uses (n - 1) degrees of freedom (ddof=1). If False, ddof=0.
    return_shrinkage : bool, default=False
        If True, returns a tuple of (shrunk_cov, shrinkage_intensity).
    verbose : bool, default=False
        If True, prints clustering diagnostic messages.

    Returns
    -------
    shrunk_cov : pd.DataFrame or np.ndarray of shape (n_features, n_features)
        The shrunken covariance matrix estimate. Returns pd.DataFrame if input
        X is a DataFrame, otherwise np.ndarray.
    shrinkage : float (optional)
        The optimal shrinkage intensity delta in [0, 1]. Only returned if
        return_shrinkage=True.
    """
    # Determine optimal number of clusters endogenously
    K = blockbuster_nr_of_clusters(X, verbose=verbose)

    # Apply Blockbuster CVC shrinkage with K clusters
    return bb(
        X,
        n_clusters=K,
        random_state=random_state,
        assume_centered=assume_centered,
        use_sample_covariance=use_sample_covariance,
        return_shrinkage=return_shrinkage,
    )


# Uppercase alias
ABB = abb


if __name__ == '__main__':
    print("Testing Adaptive Blockbuster CVC Shrinkage (ABB)...")
    np.random.seed(42)
    N_samples, p_vars = 60, 20
    col_names = [f"Var_{i+1}" for i in range(p_vars)]
    sim_data = pd.DataFrame(
        np.random.randn(N_samples, p_vars),
        columns=col_names,
    )

    detected_clusters = blockbuster_nr_of_clusters(sim_data, verbose=True)
    cov_matrix, shrinkage_val = abb(sim_data, return_shrinkage=True, verbose=False)
    print(f"Input shape:         ({N_samples}, {p_vars})")
    print(f"Detected clusters:   {detected_clusters}")
    print(f"Output shape:        {cov_matrix.shape}")
    print(f"Shrinkage intensity: {shrinkage_val:.6f}")
    print(f"Trace:               {np.trace(cov_matrix.to_numpy()):.6f}")
    print(f"Condition #:         {np.linalg.cond(cov_matrix.to_numpy()):.4f}")
    print("ABB self-test completed successfully.")
