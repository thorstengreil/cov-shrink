# Improving Scenario Generation in Supply Chain Planning: The Power of Covariance Shrinkage

Supplementary electronic material accompanying the manuscript:

> **Greil, T., & Grunow, M. (2025).** *Improving Scenario Generation in Supply Chain Planning: The Power of Covariance Shrinkage.* Available at SSRN 5765865. http://dx.doi.org/10.2139/ssrn.5765865.

---

## Overview

This repository provides supplementary code and computational results:

1. **Covariance Shrinkage Estimators (`src/`)**: Python implementations and interfaces for the eight linear and non-linear covariance shrinkage estimators evaluated in the paper.
2. **Normalized Instance-Level Results (`output/`)**: Detailed numerical results across 76 instances and 3,800 simulation runs.

---

## Repository Structure

```text
cov-shrink/
├── LICENSE                          # MIT License for repository & custom implementations
├── README.md                        # Documentation, citations, and licensing guide
├── requirements.txt                 # Python dependencies
├── output/
│   ├── README.md                    # Data dictionary, schema definitions, and replication guide
│   └── instance_results.csv         # Normalized results for 3,800 runs across 76 instances
└── src/
    ├── __init__.py                  # Package initializer
    ├── im.py                        # Identity Matrix-based (Ledoit & Wolf, 2004)
    ├── cc.py                        # Constant Correlation (Ledoit & Wolf, 2003a)
    ├── oas.py                       # Oracle Approximating Shrinkage (Chen et al., 2010; scikit-learn)
    ├── cvc.py                       # Constant Variance Covariance (De Nard, 2022)
    ├── ext.py                       # CVC with External Information (De Nard, 2022)
    ├── bb.py                        # Blockbuster CVC (De Nard, 2022)
    ├── abb.py                       # Adaptive Blockbuster CVC (De Nard, 2022)
    └── qis.py                       # Quadratic-Inverse Shrinkage (Ledoit & Wolf, 2022; Patrick LeDoit)
```

---

## Installation & Dependencies

Dependencies can be installed via `pip`:

```bash
pip install -r requirements.txt
```

* **Core dependencies:** `numpy`, `pandas`, `scipy`.
* **OAS estimator:** Requires `scikit-learn` (`sklearn.covariance.OAS`).

---

## Covariance Shrinkage Estimators

The table below summarizes the eight estimators deployed in the study (see Table 2 in the manuscript), their theoretical targets, sources, and applicable licenses:

| Abbreviation | Estimator Name | Type | Shrinkage Target / Description | Source / Implementation | License | Primary Reference |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **IM** | Identity Matrix-based | Linear | Scaled identity matrix $F = \mu I$, assuming uncorrelated variables with equal variance. | Custom implementation | MIT | [Ledoit & Wolf (2004)](https://doi.org/10.1016/S0047-259X(03)00096-4) |
| **CC** | Constant Correlation | Linear | Equal pairwise correlations; all off-diagonal correlations equal the mean correlation. | Custom implementation | MIT | [Ledoit & Wolf (2003a)](https://dx.doi.org/10.2139/ssrn.433840) |
| **OAS** | Oracle Approximating Shrinkage | Linear | Scaled identity matrix with shrinkage intensity tailored to small sample sizes under Gaussian assumptions. | `scikit-learn` (`sklearn.covariance.OAS`) | BSD 3-Clause | [Chen et al. (2010)](https://doi.org/10.1109/TSP.2010.2053029) |
| **CVC** | Constant Variance Covariance | Linear | Constant mean variance on diagonal, constant mean covariance on off-diagonal. | Custom implementation | MIT | [De Nard (2022)](https://doi.org/10.1093/jjfinec/nbaa020) |
| **EXT** | CVC with External Information | Linear | Uses external domain metadata (e.g., product hierarchies) to group variables; applies CVC within groups. | Custom implementation | MIT | [De Nard (2022)](https://doi.org/10.1093/jjfinec/nbaa020) |
| **BB** | Blockbuster CVC | Linear | Endogenously clusters variables into a predefined number of groups; applies CVC within groups. | Custom implementation | MIT | [De Nard (2022)](https://doi.org/10.1093/jjfinec/nbaa020) |
| **ABB** | Adaptive Blockbuster CVC | Linear | Extends BB by determining both the optimal number of clusters and cluster assignments endogenously. | Custom implementation | MIT | [De Nard (2022)](https://doi.org/10.1093/jjfinec/nbaa020) |
| **QIS** | Quadratic-Inverse Shrinkage | Non-linear | Applies a quadratic shrinkage formula with two target matrices to inverted sample eigenvalues. | Adapted from `pald22/covShrinkage` | MIT | [Ledoit & Wolf (2022)](https://doi.org/10.3150/20-BEJ1315) |

---

## Licensing and Attribution

To prevent licensing ambiguity, the licenses governing each component are clarified below:

### 1. Repository Code & Custom Implementations (MIT License)
The repository as a whole and all custom estimator implementations (`im.py`, `cc.py`, `cvc.py`, `ext.py`, `bb.py`, `abb.py`) are licensed under the **MIT License**:
* Copyright &copy; 2026 Thorsten Greil.
* See [LICENSE](LICENSE) for the full text.

### 2. External Package: Scikit-learn / OAS (`oas.py`) (BSD 3-Clause License)
* Estimator **OAS** is available via the `scikit-learn` Python package ([`sklearn.covariance.OAS`](https://scikit-learn.org/stable/modules/generated/sklearn.covariance.OAS.html)).
* `scikit-learn` is licensed under the **BSD 3-Clause License** (Copyright &copy; 2007–2024 The scikit-learn developers).
* If using `scikit-learn`, please cite [Pedregosa et al. (2011)](https://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html) and [Chen et al. (2010)](https://doi.org/10.1109/TSP.2010.2053029).

### 3. External Implementation: covShrinkage / QIS (`qis.py`) (MIT License)
* Estimator **QIS** is adapted from the open-source Python implementation by Patrick LeDoit ([`pald22/covShrinkage`](https://github.com/pald22/covShrinkage)), based on and endorsed by Olivier Ledoit and Michael Wolf.
* Licensed under the **MIT License** (Copyright &copy; 2022 Patrick LeDoit; original MATLAB and R algorithms copyright &copy; Olivier Ledoit and Michael Wolf).
* Free for academic and educational use with citation of [Ledoit & Wolf (2022)](https://doi.org/10.3150/20-BEJ1315).

---

## Computational Results (`output/`)

The file [`output/instance_results.csv`](output/instance_results.csv) provides normalized computational performance data across all experimental configurations (see [`output/README.md`](output/README.md) for the complete schema and data dictionary):

* **Figure 4 Instances:** 4 base instances evaluated over 50 rolling-horizon runs each (200 total runs), detailing normalized profit improvements and computational runtimes.
* **Figure 5 Instances:** 24 additional instances evaluated over 50 runs each (1,200 total runs), analyzing performance across varying cumulative demand volume thresholds $\alpha$.
* **Table 3 Instances:** 48 additional instances evaluated over 50 runs each (2,400 total runs), reporting normalized profit improvements and standard deviations across diverse problem characteristics.
* **Total Scope:** **3,800 individual computational runs** across **76 distinct instances**.

These normalized metrics allow reproducing all aggregate statistical analyses, tables, and plots presented in the paper without compromising proprietary demand records.

---

## Academic References

* **Chen, Y., Wiesel, A., Eldar, Y. C., & Hero, A. O. (2010).** Shrinkage algorithms for MMSE covariance estimation. *IEEE Transactions on Signal Processing*, 58(10), 5016–5029. [https://doi.org/10.1109/TSP.2010.2053029](https://doi.org/10.1109/TSP.2010.2053029)
* **De Nard, G. (2022).** Oops! I shrunk the sample covariance matrix again: Blockbuster meets shrinkage. *Journal of Financial Econometrics*, 20(4), 569–611. [https://doi.org/10.1093/jjfinec/nbaa020](https://doi.org/10.1093/jjfinec/nbaa020)
* **Ledoit, O., & Wolf, M. (2003a).** Honey, I shrunk the sample covariance matrix. *UPF Economics and Business Working Paper No. 691*, SSRN 433840. [https://dx.doi.org/10.2139/ssrn.433840](https://dx.doi.org/10.2139/ssrn.433840)
* **Ledoit, O., & Wolf, M. (2003b).** Improved estimation of the covariance matrix of stock returns with an application to portfolio selection. *Journal of Empirical Finance*, 10(5), 603–621. [https://doi.org/10.1016/S0927-5398(03)00007-0](https://doi.org/10.1016/S0927-5398(03)00007-0)
* **Ledoit, O., & Wolf, M. (2004).** A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis*, 88(2), 365–411. [https://doi.org/10.1016/S0047-259X(03)00096-4](https://doi.org/10.1016/S0047-259X(03)00096-4)
* **Ledoit, O., & Wolf, M. (2022).** Quadratic shrinkage for large covariance matrices. *Bernoulli*, 28(3), 1519–1547. [https://doi.org/10.3150/20-BEJ1315](https://doi.org/10.3150/20-BEJ1315)
* **Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, E. (2011).** Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830. [https://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html](https://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html)
