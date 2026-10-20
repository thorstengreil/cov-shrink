"""
Oracle Approximating Shrinkage (OAS)

Description:
    Linear shrinkage estimator shrinking toward a scaled identity matrix with a
    shrinkage intensity tailored to small sample sizes under Gaussian assumptions.

Academic Reference:
    Chen, Y., Wiesel, A., Eldar, Y. C., & Hero, A. O. (2010). Shrinkage algorithms
    for MMSE covariance estimation. IEEE Transactions on Signal Processing, 58(10),
    5016–5029. https://doi.org/10.1109/TSP.2010.2053029

Package Reference:
    Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O.,
    Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A.,
    Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, E. (2011). Scikit-learn:
    Machine learning in Python. Journal of Machine Learning Research, 12, 2825–2830.
    https://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html

Source / Upstream Implementation:
    Scikit-learn: sklearn.covariance.OAS / sklearn.covariance.oas
    https://scikit-learn.org/stable/modules/generated/sklearn.covariance.OAS.html

License:
    BSD 3-Clause License (New BSD License).
    Copyright (c) 2007-2024 The scikit-learn developers. All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice, this
      list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.

    * Neither the name of the copyright holder nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
    OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from typing import Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.covariance import OAS as SklearnOAS
from sklearn.covariance import oas as sklearn_oas


def oas(
    X: Union[pd.DataFrame, np.ndarray],
    assume_centered: bool = False,
    return_shrinkage: bool = False,
) -> Union[pd.DataFrame, np.ndarray, Tuple[Union[pd.DataFrame, np.ndarray], float]]:
    """
    Oracle Approximating Shrinkage (OAS) covariance estimator.

    Computes the OAS linear shrinkage estimator (Chen et al., 2010) directly
    via ``sklearn.covariance.oas``.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray of shape (n_samples, n_features)
        Input data matrix with observations in rows and variables in columns.
    assume_centered : bool, default=False
        If True, data will not be centered before covariance computation.
    return_shrinkage : bool, default=False
        If True, returns a tuple of (covariance_matrix, shrinkage_intensity).

    Returns
    -------
    shrunk_cov : pd.DataFrame or np.ndarray of shape (n_features, n_features)
        Regularized OAS covariance matrix estimate.
        Returns pd.DataFrame if input X is a DataFrame, otherwise np.ndarray.
    shrinkage : float (optional)
        The optimal shrinkage intensity rho in [0, 1]. Only returned if
        return_shrinkage=True.
    """
    is_dataframe = isinstance(X, pd.DataFrame)
    if is_dataframe:
        orig_columns = X.columns
        X_arr = X.to_numpy()
    else:
        orig_columns = None
        X_arr = np.asarray(X)

    cov, shrinkage = sklearn_oas(X_arr, assume_centered=assume_centered)

    if is_dataframe:
        cov = pd.DataFrame(cov, index=orig_columns, columns=orig_columns)

    if return_shrinkage:
        return cov, shrinkage
    return cov


class OAS:
    """
    Oracle Approximating Shrinkage (OAS) estimator class interface.

    Wraps ``sklearn.covariance.OAS``.
    """

    def __init__(self, store_precision: bool = True, assume_centered: bool = False):
        self.store_precision = store_precision
        self.assume_centered = assume_centered
        self.covariance_: Optional[np.ndarray] = None
        self.precision_: Optional[np.ndarray] = None
        self.shrinkage_: Optional[float] = None
        self.location_: Optional[np.ndarray] = None

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        """Fit the OAS covariance model to X."""
        sklearn_estimator = SklearnOAS(
            store_precision=self.store_precision,
            assume_centered=self.assume_centered,
        )
        sklearn_estimator.fit(X, y)
        self.covariance_ = sklearn_estimator.covariance_
        self.precision_ = sklearn_estimator.precision_
        self.shrinkage_ = sklearn_estimator.shrinkage_
        self.location_ = sklearn_estimator.location_
        return self


if __name__ == '__main__':
    # Self-test demonstration with synthetic data
    print("Testing Oracle Approximating Shrinkage (OAS)...")
    np.random.seed(42)
    N_samples, p_vars = 60, 20
    sim_data = pd.DataFrame(
        np.random.randn(N_samples, p_vars),
        columns=[f"Var_{i+1}" for i in range(p_vars)],
    )

    cov_matrix, shrinkage_val = oas(sim_data, return_shrinkage=True)
    print(f"Input shape:        ({N_samples}, {p_vars})")
    print(f"Output shape:       {cov_matrix.shape}")
    print(f"Shrinkage intensity: {shrinkage_val:.6f}")
    print(f"Trace:              {np.trace(cov_matrix.to_numpy()):.6f}")
    print(f"Condition #:        {np.linalg.cond(cov_matrix.to_numpy()):.4f}")
    print("OAS self-test completed successfully.")
