"""
Identity Matrix-Based Covariance Shrinkage (IM) / Ledoit-Wolf (2004)

Description:
    Linear shrinkage estimator shrinking toward a scaled identity matrix,
    assuming uncorrelated variables with equal variances.

Academic Reference:
    Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for large-dimensional
    covariance matrices. Journal of Multivariate Analysis, 88(2), 365–411.
    https://doi.org/10.1016/S0047-259X(03)00096-4

Implementation:
    Custom implementation developed for:
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


def im(
    X: Union[pd.DataFrame, np.ndarray],
    assume_centered: bool = False,
    use_sample_covariance: bool = True,
    return_shrinkage: bool = False,
) -> Union[pd.DataFrame, np.ndarray, Tuple[Union[pd.DataFrame, np.ndarray], float]]:
    """
    Identity Matrix-based (IM) covariance shrinkage estimator (Ledoit & Wolf, 2004).

    Shrinks the sample covariance matrix toward a scaled identity matrix F = mu * I,
    where mu = Tr(S) / p is the average sample variance.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray of shape (n_samples, n_features)
        Input observations matrix with samples in rows and variables in columns.
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
        The regularized covariance matrix estimate. Returns pd.DataFrame
        if input X is a DataFrame, otherwise np.ndarray.
    shrinkage : float (optional)
        The optimal shrinkage intensity in [0, 1]. Only returned if
        return_shrinkage=True.
    """
    is_dataframe = isinstance(X, pd.DataFrame)
    if is_dataframe:
        orig_columns = X.columns
        X_arr = X.to_numpy(dtype=float)
    else:
        orig_columns = None
        X_arr = np.asarray(X, dtype=float)

    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(1, -1)

    T, n = X_arr.shape
    if T < 2:
        raise ValueError("At least 2 observations are required for covariance estimation.")

    # Center data if required
    if not assume_centered:
        X_centered = X_arr - np.mean(X_arr, axis=0)
    else:
        X_centered = X_arr

    # Sample covariance matrix S
    ddof = 1 if use_sample_covariance else 0
    cov_input = np.dot(X_centered.T, X_centered) / (T - ddof)
    cov_input = (cov_input + cov_input.T) / 2.0

    # Shrinkage target: average variance * I
    avg_var = float(np.mean(np.diag(cov_input)))
    shrink_target = avg_var * np.eye(n)

    # Optimal shrinkage intensity (Ledoit & Wolf, 2004)
    beta_square = 0.0
    for t in range(T):
        xt = X_centered[t, :]
        beta_square += float(np.sum((np.outer(xt, xt) - cov_input) ** 2))
    beta_square /= (T ** 2)

    delta_square = float(np.sum((cov_input - shrink_target) ** 2))
    if delta_square == 0:
        shrinkage = 1.0
    else:
        shrinkage = float(min(1.0, max(0.0, beta_square / delta_square)))

    # Convex combination: (1 - shrinkage) * S + shrinkage * F
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


class IM:
    """
    Identity Matrix-based (Ledoit-Wolf, 2004) estimator class interface.
    """

    def __init__(
        self,
        store_precision: bool = True,
        assume_centered: bool = False,
        use_sample_covariance: bool = True,
    ):
        self.store_precision = store_precision
        self.assume_centered = assume_centered
        self.use_sample_covariance = use_sample_covariance
        self.covariance_: Optional[np.ndarray] = None
        self.precision_: Optional[np.ndarray] = None
        self.shrinkage_: Optional[float] = None
        self.location_: Optional[np.ndarray] = None

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        """Fit the IM covariance model to X."""
        X_arr = X.to_numpy(dtype=float) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
        if self.assume_centered:
            self.location_ = np.zeros(X_arr.shape[1])
        else:
            self.location_ = np.mean(X_arr, axis=0)

        cov, shrinkage = im(
            X_arr,
            assume_centered=self.assume_centered,
            use_sample_covariance=self.use_sample_covariance,
            return_shrinkage=True,
        )
        self.covariance_ = cov
        self.shrinkage_ = shrinkage
        if self.store_precision:
            self.precision_ = np.linalg.pinv(cov)
        return self


if __name__ == '__main__':
    print("Testing Identity Matrix-Based Shrinkage (IM / Ledoit-Wolf 2004)...")
    np.random.seed(42)
    N_samples, p_vars = 60, 20
    sim_data = pd.DataFrame(
        np.random.randn(N_samples, p_vars),
        columns=[f"Var_{i+1}" for i in range(p_vars)],
    )

    cov_matrix, shrinkage_val = im(sim_data, return_shrinkage=True)
    print(f"Input shape:         ({N_samples}, {p_vars})")
    print(f"Output shape:        {cov_matrix.shape}")
    print(f"Shrinkage intensity: {shrinkage_val:.6f}")
    print(f"Trace:               {np.trace(cov_matrix.to_numpy()):.6f}")
    print(f"Condition #:         {np.linalg.cond(cov_matrix.to_numpy()):.4f}")
    print("IM self-test completed successfully.")
