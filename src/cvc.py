"""
Constant Variance Covariance Shrinkage (CVC)

Description:
    Linear shrinkage estimator assuming constant mean variance on the diagonal
    and constant mean covariance on the off-diagonal for the shrinkage target.

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


def cvc(
    X: Union[pd.DataFrame, np.ndarray],
    assume_centered: bool = False,
    use_sample_covariance: bool = True,
    return_shrinkage: bool = False,
) -> Union[pd.DataFrame, np.ndarray, Tuple[Union[pd.DataFrame, np.ndarray], float]]:
    """
    Constant Variance Covariance (CVC) shrinkage estimator (De Nard, 2022).

    Shrinks the sample covariance matrix toward a target with constant mean
    variance on the diagonal and constant mean covariance on the off-diagonal.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray of shape (n_samples, n_features)
        Data matrix with observations in rows and variables in columns.
    assume_centered : bool, default=False
        If True, data will not be centered before covariance computation.
    use_sample_covariance : bool, default=True
        If True, sample covariance uses (n - 1) degrees of freedom (ddof=1).
        If False, uses n degrees of freedom (ddof=0).
    return_shrinkage : bool, default=False
        If True, returns a tuple of (shrunk_cov, shrinkage_intensity).

    Returns
    -------
    shrunk_cov : pd.DataFrame or np.ndarray of shape (n_features, n_features)
        The shrunken covariance matrix estimate. Returns a pd.DataFrame if input
        X is a DataFrame, otherwise np.ndarray.
    shrinkage : float (optional)
        The optimal shrinkage intensity delta in [0, 1]. Only returned if
        return_shrinkage=True.
    """
    is_dataframe = isinstance(X, pd.DataFrame)
    if is_dataframe:
        orig_columns = X.columns
        X_arr = X.to_numpy(dtype=float)
    else:
        orig_columns = None
        X_arr = np.asarray(X, dtype=float)

    T, N = X_arr.shape
    if T < 2:
        raise ValueError("At least 2 observations are required.")

    # Center data if required
    if not assume_centered:
        X_centered = X_arr - np.mean(X_arr, axis=0)
    else:
        X_centered = X_arr

    # Compute sample covariance matrix S
    ddof = 1 if use_sample_covariance else 0
    cov_input = np.dot(X_centered.T, X_centered) / (T - ddof)
    cov_input = (cov_input + cov_input.T) / 2.0

    # Average variance: phi
    phi = float(np.trace(cov_input) / N)

    # Average off-diagonal covariance: nu
    if N > 1:
        off_diag_sum = np.sum(cov_input) - np.trace(cov_input)
        nu = float(off_diag_sum / (N * (N - 1)))
    else:
        nu = 0.0

    # Shrinkage target: F = phi * I + nu * (1 1^T - I)
    off_diagonal_matrix = np.ones((N, N)) - np.eye(N)
    shrink_target = phi * np.eye(N) + nu * off_diagonal_matrix

    # xi squared = || F - S ||_F^2
    xi_squared = float(np.sum((shrink_target - cov_input) ** 2))

    # tau squared
    sum_sq_diff = 0.0
    for t in range(T):
        xt = X_centered[t, :]
        xt_outer = np.outer(xt, xt)
        sum_sq_diff += np.sum((xt_outer - cov_input) ** 2)

    sample_var_bound = sum_sq_diff / (T ** 2)
    tau_squared = float(min(sample_var_bound, xi_squared))

    # Optimal shrinkage intensity: delta_opt
    if xi_squared == 0:
        shrinkage = 1.0
    else:
        shrinkage = float(min(1.0, max(0.0, tau_squared / xi_squared)))

    # Shrunk covariance matrix
    result = (1.0 - shrinkage) * cov_input + shrinkage * shrink_target
    result = (result + result.T) / 2.0

    if is_dataframe:
        result_df = pd.DataFrame(result, index=orig_columns, columns=orig_columns)
        if return_shrinkage:
            return result_df, shrinkage
        return result_df

    if return_shrinkage:
        return result, shrinkage
    return result


# Uppercase alias
CVC = cvc


if __name__ == '__main__':
    print("Testing Constant Variance Covariance Shrinkage (CVC)...")
    np.random.seed(42)
    N_samples, p_vars = 60, 20
    sim_data = pd.DataFrame(
        np.random.randn(N_samples, p_vars),
        columns=[f"Var_{i+1}" for i in range(p_vars)],
    )

    cov_matrix, shrinkage_val = cvc(sim_data, return_shrinkage=True)
    print(f"Input shape:         ({N_samples}, {p_vars})")
    print(f"Output shape:        {cov_matrix.shape}")
    print(f"Shrinkage intensity: {shrinkage_val:.6f}")
    print(f"Trace:               {np.trace(cov_matrix.to_numpy()):.6f}")
    print(f"Condition #:         {np.linalg.cond(cov_matrix.to_numpy()):.4f}")
    print("CVC self-test completed successfully.")
