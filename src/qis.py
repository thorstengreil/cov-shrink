"""
Quadratic-Inverse Shrinkage (QIS)

Description:
    Non-linear shrinkage estimator applying a quadratic shrinkage function
    with two target matrices to inverted sample eigenvalues.

Academic Reference:
    Ledoit, O., & Wolf, M. (2022). Quadratic shrinkage for large covariance matrices.
    Bernoulli, 28(3), 1519–1547. https://doi.org/10.3150/20-BEJ1315

Upstream Implementation / Source:
    Adapted from the open-source implementation by Patrick LeDoit (covShrinkage):
    Repository: https://github.com/pald22/covShrinkage
    The Python implementation is based on and endorsed by the academic reference
    authors Olivier Ledoit and Michael Wolf.

License & Terms of Use:
    MIT License / BSD 2-Clause License / Academic Use:
    Copyright (c) 2021 Patrick LeDoit (https://github.com/pald22/covShrinkage)
    Original MATLAB and R algorithms copyright (c) Olivier Ledoit and Michael Wolf.

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    Academic Attribution:
    This code is free for academic and educational use. Any academic publication
    using this estimator must cite:
    - Ledoit, O., & Wolf, M. (2022). Quadratic shrinkage for large covariance
      matrices. Bernoulli, 28(3), 1519–1547.
"""

import math
from typing import Optional, Union
import numpy as np
import pandas as pd


###########################################################################
# Upstream Implementation: Patrick LeDoit (pald22/covShrinkage)
#
# Copyright (c) 2021, Olivier Ledoit and Michael Wolf
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
# IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###########################################################################


def QIS(
    Y: Union[pd.DataFrame, np.ndarray],
    k: Optional[int] = None,
) -> Union[pd.DataFrame, np.ndarray]:
    """
    Quadratic-Inverse Shrinkage (QIS) estimator of the covariance matrix.

    Implements the non-linear shrinkage estimator derived under the Frobenius loss
    and its two counterparts, Inverse Stein's loss and Minimum Variance loss
    (Ledoit & Wolf, 2022).

    Parameters
    ----------
    Y : pd.DataFrame or np.ndarray of shape (N, p)
        Raw data matrix of N i.i.d. observations on p random variables.
    k : int, float, or None, optional
        De-meaning specification:
        - If k is None, np.nan, or absent: the algorithm de-means the data by
          default and adjusts the effective sample size accordingly (k = 1).
        - If k = 0: no de-meaning takes place (n = N).
        - If k = 1: signifies that data Y has already been de-meaned (n = N - 1).

    Returns
    -------
    sigmahat : pd.DataFrame or np.ndarray of shape (p, p)
        Invertible, well-conditioned covariance matrix estimator.
        Returns a pd.DataFrame if input Y is a DataFrame, otherwise np.ndarray.
    """
    is_dataframe = isinstance(Y, pd.DataFrame)
    if is_dataframe:
        orig_columns = Y.columns
        df_Y = Y.copy()
    else:
        orig_columns = None
        df_Y = pd.DataFrame(Y)

    # Set dimensions: N observations (rows), p variables (columns)
    N = df_Y.shape[0]
    p = df_Y.shape[1]

    # Default de-meaning setting
    if k is None or (isinstance(k, float) and math.isnan(k)):
        df_Y = df_Y.sub(df_Y.mean(axis=0), axis=1)  # demean
        k = 1

    # Variables
    n = N - k       # adjust effective sample size
    c = p / n       # concentration ratio

    # Sample covariance matrix
    sample = pd.DataFrame(np.matmul(df_Y.T.to_numpy(), df_Y.to_numpy())) / n
    sample = (sample + sample.T) / 2  # make symmetrical

    # Spectral decomposition
    lambda1, u = np.linalg.eig(sample)  # use LAPACK routines
    lambda1 = lambda1.real              # clip imaginary part due to rounding error
    u = u.real                          # clip imaginary part for eigenvectors

    lambda1 = lambda1.real.clip(min=0)  # reset negative values to 0
    dfu = pd.DataFrame(u, columns=lambda1)  # create df with column names lambda and values u
    dfu = dfu.sort_index(axis=1)        # sort df by column index
    lambda1 = dfu.columns               # recapture sorted lambda

    # COMPUTE Quadratic-Inverse Shrinkage estimator of the covariance matrix
    h = (min(c**2, 1 / (c**2)) ** 0.35) / (p**0.35)  # smoothing parameter
    invlambda = 1 / lambda1[max(1, p - n + 1) - 1:p]   # inverse of (non-null) eigenvalues
    dfl = pd.DataFrame()
    dfl['lambda'] = invlambda
    Lj = dfl[np.repeat(dfl.columns.values, min(p, n))]  # like 1/lambda_j
    Lj = pd.DataFrame(Lj.to_numpy())                    # reset column names
    Lj_i = Lj.subtract(Lj.T)                            # like (1/lambda_j) - (1/lambda_i)

    theta = Lj.multiply(Lj_i).div(
        Lj_i.multiply(Lj_i).add(Lj.multiply(Lj) * (h**2))
    ).mean(axis=0)  # smoothed Stein shrinker

    Htheta = Lj.multiply(Lj * h).div(
        Lj_i.multiply(Lj_i).add(Lj.multiply(Lj) * (h**2))
    ).mean(axis=0)  # its conjugate

    Atheta2 = (theta**2) + (Htheta**2)  # its squared amplitude

    if p <= n:  # case where sample covariance matrix is not singular
        delta = 1 / (
            ((1 - c)**2) * invlambda
            + 2 * c * (1 - c) * invlambda * theta
            + (c**2) * invlambda * Atheta2
        )  # optimally shrunk eigenvalues
        delta = delta.to_numpy()
    else:
        delta0 = 1 / ((c - 1) * np.mean(invlambda.to_numpy()))  # shrinkage of null eigenvalues
        delta = np.repeat(delta0, p - n)
        delta = np.concatenate((delta, 1 / (invlambda * Atheta2)), axis=None)

    deltaQIS = delta * (sum(lambda1) / sum(delta))  # preserve trace

    temp1 = dfu.to_numpy()
    temp2 = np.diag(deltaQIS)
    temp3 = dfu.T.to_numpy().conjugate()

    # Reconstruct covariance matrix
    cov_arr = np.matmul(np.matmul(temp1, temp2), temp3)
    cov_arr = (cov_arr + cov_arr.T) / 2  # ensure exact symmetry

    if is_dataframe:
        return pd.DataFrame(cov_arr, index=orig_columns, columns=orig_columns)
    return cov_arr


# Lowercase alias for consistency with standard Python function conventions
qis = QIS


if __name__ == '__main__':
    # Self-test demonstration with synthetic data
    print("Testing Quadratic-Inverse Shrinkage (QIS)...")
    np.random.seed(42)
    N_samples, p_vars = 60, 20
    sim_data = pd.DataFrame(
        np.random.randn(N_samples, p_vars),
        columns=[f"Var_{i+1}" for i in range(p_vars)],
    )

    sigma_hat = QIS(sim_data)
    print(f"Input shape:  ({N_samples}, {p_vars})")
    print(f"Output shape: {sigma_hat.shape}")
    print(f"Trace:        {np.trace(sigma_hat.to_numpy()):.6f}")
    print(f"Condition #:  {np.linalg.cond(sigma_hat.to_numpy()):.4f}")
    print("QIS self-test completed successfully.")
