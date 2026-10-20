"""
Constant Correlation Covariance Shrinkage (CC)

Description:
    Linear shrinkage estimator assuming all pairwise correlations are identical,
    i.e., all off-diagonal correlations are equal.

Reference:
    Ledoit, O., & Wolf, M. (2003a). Honey, I shrunk the sample covariance matrix.
    UPF Economics and Business Working Paper No. 691, Available at SSRN 433840.
    https://dx.doi.org/10.2139/ssrn.433840
    (See also: Ledoit, O., & Wolf, M. (2003b). Improved estimation of the covariance
    matrix of stock returns with an application to portfolio selection. Journal of
    Empirical Finance, 10(5), 603–621. https://doi.org/10.1016/S0927-5398(03)00007-0)

Implementation:
    Custom implementation developed for the working paper:
    Greil, T., & Grunow, M. (2025). "Improving Scenario Generation in Supply Chain
    Planning: The Power of Covariance Shrinkage". Available at SSRN 5765865.
    http://dx.doi.org/10.2139/ssrn.5765865

License:
    MIT License (see repository root LICENSE file).
    Copyright (c) 2026 Thorsten Greil (thorstengreil).
"""

import math
from typing import Optional, Tuple, Union
import numpy as np
import pandas as pd


def cc(
    Y: Union[pd.DataFrame, np.ndarray],
    k: Optional[int] = None,
    return_shrinkage: bool = False,
) -> Union[pd.DataFrame, np.ndarray, Tuple[Union[pd.DataFrame, np.ndarray], float]]:
    """
    Constant Correlation (CC) covariance shrinkage estimator (Ledoit & Wolf, 2003a, 2003b).

    Shrinks the sample covariance matrix toward a target matrix with individual sample
    variances on the diagonal and constant average pairwise correlation on off-diagonals.

    Parameters
    ----------
    Y : pd.DataFrame or np.ndarray of shape (N, p)
        Data matrix with N observations on p random variables.
    k : int, float, or None, optional
        De-meaning specification:
        - If k is None, np.nan, or absent: de-means data by default and adjusts
          the effective sample size (n = N - 1, k = 1).
        - If k = 0: no de-meaning takes place (n = N).
        - If k = 1: data Y is assumed already de-meaned (n = N - 1).
    return_shrinkage : bool, default=False
        If True, returns a tuple of (shrunk_cov, shrinkage_intensity).

    Returns
    -------
    shrunk_cov : pd.DataFrame or np.ndarray of shape (p, p)
        The regularized constant correlation covariance matrix. Returns a
        pd.DataFrame if input Y is a DataFrame, otherwise np.ndarray.
    shrinkage : float (optional)
        The optimal shrinkage intensity in [0, 1]. Only returned if
        return_shrinkage=True.
    """
    is_dataframe = isinstance(Y, pd.DataFrame)
    if is_dataframe:
        orig_columns = Y.columns
        Y_arr = Y.to_numpy(dtype=float).copy()
    else:
        orig_columns = None
        Y_arr = np.asarray(Y, dtype=float).copy()

    N, p = Y_arr.shape
    if N < 2:
        raise ValueError("At least 2 observations are required.")

    # Default de-meaning setting
    if k is None or (isinstance(k, float) and math.isnan(k)):
        Y_arr = Y_arr - np.mean(Y_arr, axis=0)
        k = 1

    n = N - k  # Effective sample size
    if n <= 0:
        raise ValueError("Effective sample size (N - k) must be positive.")

    # Sample covariance matrix S
    sample = np.dot(Y_arr.T, Y_arr) / n
    sample = (sample + sample.T) / 2.0  # Ensure symmetry

    # Compute shrinkage target
    samplevar = np.diag(sample)
    sqrtvar = np.sqrt(np.maximum(samplevar, 1e-12))
    sqrtvar_mat = np.outer(sqrtvar, sqrtvar)

    # Sample correlation matrix
    corr = sample / sqrtvar_mat
    # Mean off-diagonal correlation
    rBar = (np.sum(corr) - p) / (p * (p - 1)) if p > 1 else 0.0

    target = rBar * sqrtvar_mat
    np.fill_diagonal(target, samplevar)

    # Estimate pi parameter (Ledoit & Wolf, 2003b)
    Y2 = Y_arr ** 2
    sample2 = np.dot(Y2.T, Y2) / n
    piMat = sample2 - (sample ** 2)
    pihat = float(np.sum(piMat))

    # Estimate gamma parameter (Ledoit & Wolf, 2003b)
    gammahat = float(np.sum((sample - target) ** 2))

    # Diagonal part of rho parameter
    rho_diag = float(np.sum(np.diag(piMat)))

    # Off-diagonal part of rho parameter
    term1 = np.dot((Y_arr ** 3).T, Y_arr) / n
    term2 = np.outer(samplevar, np.ones(p)) * sample
    thetaMat = term1 - term2
    np.fill_diagonal(thetaMat, 0.0)

    inv_sqrtvar_mat = np.outer(1.0 / sqrtvar, sqrtvar)
    rho_off = float(rBar * np.sum(inv_sqrtvar_mat * thetaMat))

    # Compute shrinkage intensity
    rhohat = rho_diag + rho_off
    if gammahat == 0:
        shrinkage = 1.0
    else:
        kappahat = (pihat - rhohat) / gammahat
        shrinkage = float(max(0.0, min(1.0, kappahat / n)))

    # Convex combination
    result = (1.0 - shrinkage) * sample + shrinkage * target
    result = (result + result.T) / 2.0  # Symmetry

    if is_dataframe:
        result_df = pd.DataFrame(result, index=orig_columns, columns=orig_columns)
        if return_shrinkage:
            return result_df, shrinkage
        return result_df

    if return_shrinkage:
        return result, shrinkage
    return result


# Uppercase alias
CC = cc


if __name__ == '__main__':
    print("Testing Constant Correlation Shrinkage (CC)...")
    np.random.seed(42)
    N_samples, p_vars = 60, 20
    sim_data = pd.DataFrame(
        np.random.randn(N_samples, p_vars),
        columns=[f"Var_{i+1}" for i in range(p_vars)],
    )

    cov_matrix, shrinkage_val = cc(sim_data, return_shrinkage=True)
    print(f"Input shape:         ({N_samples}, {p_vars})")
    print(f"Output shape:        {cov_matrix.shape}")
    print(f"Shrinkage intensity: {shrinkage_val:.6f}")
    print(f"Trace:               {np.trace(cov_matrix.to_numpy()):.6f}")
    print(f"Condition #:         {np.linalg.cond(cov_matrix.to_numpy()):.4f}")
    print("CC self-test completed successfully.")
