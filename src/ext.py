"""
Constant Variance Covariance with External Information (EXT)

Description:
    Linear shrinkage estimator that uses external information to partition
    variables into predefined groups and applies CVC shrinkage within each group.

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
    from src.cvc import cvc
except (ImportError, ModuleNotFoundError):
    try:
        from .cvc import cvc
    except (ImportError, ModuleNotFoundError):
        from cvc import cvc


def ext(
    X: Union[pd.DataFrame, np.ndarray],
    groups: Optional[Dict[Any, List[Any]]] = None,
    assume_centered: bool = False,
    use_sample_covariance: bool = True,
    return_shrinkage: bool = False,
) -> Union[pd.DataFrame, np.ndarray, Tuple[Union[pd.DataFrame, np.ndarray], float]]:
    """
    CVC with External Information (EXT) shrinkage estimator (De Nard, 2022).

    Partitions variables using external metadata (e.g. product hierarchies or
    domain categories) into predefined groups and applies CVC shrinkage within
    and across groups.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray of shape (n_samples, n_features)
        Data matrix with observations in rows and variables in columns.
    groups : dict, optional
        Dictionary mapping group identifiers to lists of variable names (if X is
        a pd.DataFrame) or column integer indices. If None, all variables are
        treated as a single group (equivalent to CVC).
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
    is_dataframe = isinstance(X, pd.DataFrame)
    if is_dataframe:
        orig_columns = list(X.columns)
        X_arr = X.to_numpy(dtype=float)
    else:
        orig_columns = None
        X_arr = np.asarray(X, dtype=float)

    T, n = X_arr.shape
    if T < 2:
        raise ValueError("At least 2 observations are required.")

    # Convert groups to list of integer indices
    if groups is None:
        idx_groups = {0: list(range(n))}
    else:
        idx_groups = {}
        for g_name, var_list in groups.items():
            converted_indices = []
            for v in var_list:
                if isinstance(v, int) and (orig_columns is None or v not in orig_columns):
                    idx = v
                else:
                    idx = orig_columns.index(v) if orig_columns is not None else int(v)
                converted_indices.append(idx)
            idx_groups[g_name] = converted_indices

    # Center data if required
    if not assume_centered:
        X_centered = X_arr - np.mean(X_arr, axis=0)
    else:
        X_centered = X_arr

    # Compute sample covariance matrix S
    ddof = 1 if use_sample_covariance else 0
    sample_cov = np.dot(X_centered.T, X_centered) / (T - ddof)
    sample_cov = (sample_cov + sample_cov.T) / 2.0

    # Build grouped shrinkage target F
    shrinkage_target = np.zeros((n, n), dtype=float)

    # Step 1: Average variance and covariances within each group
    for g_name, indices in idx_groups.items():
        if len(indices) == 0:
            continue
        sub_matrix = sample_cov[np.ix_(indices, indices)]
        avg_var = float(np.mean(np.diag(sub_matrix)))

        for i in indices:
            shrinkage_target[i, i] = avg_var

        if len(indices) > 1:
            off_diag_mask = ~np.eye(len(indices), dtype=bool)
            avg_cov = float(np.mean(sub_matrix[off_diag_mask]))
            for i in indices:
                for j in indices:
                    if i != j:
                        shrinkage_target[i, j] = avg_cov

    # Step 2: Cross-group average covariance for pairwise group combinations
    group_names = list(idx_groups.keys())
    for i_g in range(len(group_names)):
        for j_g in range(i_g + 1, len(group_names)):
            g1_indices = idx_groups[group_names[i_g]]
            g2_indices = idx_groups[group_names[j_g]]
            if len(g1_indices) == 0 or len(g2_indices) == 0:
                continue

            sub_matrix = sample_cov[np.ix_(g1_indices, g2_indices)]
            avg_cross_cov = float(np.mean(sub_matrix))

            for i in g1_indices:
                for j in g2_indices:
                    shrinkage_target[i, j] = avg_cross_cov
                    shrinkage_target[j, i] = avg_cross_cov

    # Obtain shrinkage intensity from CVC (De Nard, 2022)
    _, shrinkage = cvc(
        X_centered,
        assume_centered=True,
        use_sample_covariance=use_sample_covariance,
        return_shrinkage=True,
    )

    # Convex combination
    result = (1.0 - shrinkage) * sample_cov + shrinkage * shrinkage_target
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
EXT = ext


if __name__ == '__main__':
    print("Testing CVC with External Information (EXT)...")
    np.random.seed(42)
    N_samples, p_vars = 60, 20
    col_names = [f"Var_{i+1}" for i in range(p_vars)]
    sim_data = pd.DataFrame(
        np.random.randn(N_samples, p_vars),
        columns=col_names,
    )

    # Group variables into two distinct external product categories
    sample_groups = {
        "Group_A": col_names[:10],
        "Group_B": col_names[10:],
    }

    cov_matrix, shrinkage_val = ext(sim_data, groups=sample_groups, return_shrinkage=True)
    print(f"Input shape:         ({N_samples}, {p_vars})")
    print(f"Output shape:        {cov_matrix.shape}")
    print(f"Shrinkage intensity: {shrinkage_val:.6f}")
    print(f"Trace:               {np.trace(cov_matrix.to_numpy()):.6f}")
    print(f"Condition #:         {np.linalg.cond(cov_matrix.to_numpy()):.4f}")
    print("EXT self-test completed successfully.")
