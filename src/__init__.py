"""
Covariance Shrinkage Estimators.

Accompanying code for:
"Improving Scenario Generation in Supply Chain Planning: The Power of Covariance Shrinkage"
by Thorsten Greil and Martin Grunow (European Journal of Operational Research).
"""

from src.im import IM, im
from src.cc import CC, cc
from src.oas import OAS, oas
from src.cvc import CVC, cvc
from src.ext import EXT, ext
from src.bb import BB, bb
from src.abb import ABB, abb
from src.qis import QIS, qis

__all__ = [
    # Identity Matrix-Based (Ledoit & Wolf, 2004)
    "IM",
    "im",
    # Constant Correlation (Ledoit & Wolf, 2003a)
    "CC",
    "cc",
    # Oracle Approximating Shrinkage (Chen et al., 2010; scikit-learn)
    "OAS",
    "oas",
    # Constant Variance Covariance (De Nard, 2022)
    "CVC",
    "cvc",
    # CVC with External Information (De Nard, 2022)
    "EXT",
    "ext",
    # Blockbuster CVC (De Nard, 2022)
    "BB",
    "bb",
    # Adaptive Blockbuster CVC (De Nard, 2022)
    "ABB",
    "abb",
    # Quadratic-Inverse Shrinkage (Ledoit & Wolf, 2022)
    "QIS",
    "qis",
]
