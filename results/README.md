# Computational Results Data Dictionary (`instance_results.csv`)

This folder contains the normalized computational performance data accompanying the study:

> **Greil, T., & Grunow, M. (2025).** *Improving Scenario Generation in Supply Chain Planning: The Power of Covariance Shrinkage.* Available at SSRN 5765865. http://dx.doi.org/10.2139/ssrn.5765865.

The primary dataset is stored in [`instance_results.csv`](instance_results.csv). It contains **3,800 individual computational simulation runs** across **76 distinct problem instances** evaluated over rolling-horizon replanning intervals.

---

## 1. Schema and Column Definitions

| Column Name | Data Type | Nullable | Example | Description & Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| `instance_id` | Integer | No | `1` | Unique numerical identifier for the experimental configuration (1 to 76). |
| `run_id` | Integer | No | `1` | Index of the rolling-horizon simulation run (1 to 50) within each configuration. |
| `estimator` | String | No | `'OAS'` | Covariance estimation method or benchmark deployed (see [Estimator Codes](#3-estimator-codes)). |
| `num_scenarios` | Integer | No | `50` | Number of scenario samples generated for the two-stage stochastic program ($n \in \{10, 25, 50, 100\}$). |
| `alpha` | Float | Yes | `0.5` | Share of cumulative demand volume $\alpha \in [0.1, 0.8]$ prioritized for pairwise correlation estimation. Blank for uncorrelated benchmarks. |
| `profit_improvement` | Float | No | `0.0094` | Realized relative profit improvement vs. the deterministic single-scenario benchmark (expressed as a decimal fraction, e.g., `0.0094` = +0.94%). |
| `runtime_minutes` | Float | Yes | `10.2` | Computational optimization solve time in minutes. Recorded for Figure 4 scalability instances; blank otherwise. |
| `figure_4` | Integer | No | `1` | Binary flag (`1` or `0`): Indicates whether this run is evaluated in **Figure 4** (Scenario Size Scalability). |
| `figure_5` | Integer | No | `1` | Binary flag (`1` or `0`): Indicates whether this run is evaluated in **Figure 5** (Uncorrelated vs. Correlated Benchmarks across $\alpha$). |
| `table_3` | Integer | No | `1` | Binary flag (`1` or `0`): Indicates whether this run is evaluated in **Table 3** (Inferential Statistics & Estimator Comparison). |

---

## 2. Mapping to Paper Figures and Tables

To avoid redundant data duplication, instances evaluated across multiple elements (such as the uncorrelated baselines and sample covariance benchmarks) appear exactly once as distinct rows, flagged via the indicator columns `figure_4`, `figure_5`, and `table_3`:

| Paper Element | Filter Condition | Instances | Runs per Instance | Total Rows | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Figure 4** | `figure_4 == 1` | 4 | 50 | **200** | Scalability of uncorrelated scenarios ($n \in \{10, 25, 50, 100\}$), comparing profit improvements and solve runtimes. |
| **Figure 5** | `figure_5 == 1` | 27 | 50 | **1,350** | Sensitivity analysis across varying demand volume cutoffs $\alpha \in \{0.1, \dots, 0.8\}$ for $n \in \{10, 25, 50\}$, plus uncorrelated baselines. |
| **Table 3** | `table_3 == 1` | 56 | 50 | **2,800** | Statistical comparison of all 8 covariance shrinkage estimators against uncorrelated ($U$) and sample correlation ($S$) benchmarks across $\alpha \in \{0.3, 0.5, 0.7\}$ and $n \in \{25, 50\}$. |

### Instance Count Summary:
- **Base Uncorrelated Instances (Figure 4):** 4 instances (200 runs)
- **Additional Correlated (No Shrinkage) Instances (Figure 5):** 24 instances (1,200 runs)
- **Additional Shrinkage Estimator Instances (Table 3):** 48 instances (2,400 runs)
- **Total Unique Scope:** **76 distinct instances** $\times 50\text{ runs} =$ **3,800 total rows**

---

## 3. Estimator Codes

| Code | Full Name | Type | Shrinkage Target / Model | Source Reference |
| :--- | :--- | :--- | :--- | :--- |
| `Uncorrelated` | Uncorrelated Benchmark | Benchmark | Zero off-diagonal covariances | Deterministic / Sample Variances |
| `Sample` | Sample Covariance | Benchmark | Empirical sample covariance matrix (no shrinkage regularizer) | Sample covariance |
| `IM` | Identity Matrix-based | Linear | Scaled identity matrix $F = \mu I$ | Ledoit & Wolf (2004) |
| `CC` | Constant Correlation | Linear | Target with equal pairwise off-diagonal correlations | Ledoit & Wolf (2003a) |
| `OAS` | Oracle Approximating Shrinkage | Linear | Scaled identity with MMSE intensity | Chen et al. (2010); `scikit-learn` |
| `CVC` | Constant Variance Covariance | Linear | Constant variance on diagonal, constant covariance on off-diagonal | De Nard (2022) |
| `EXT` | CVC with External Information | Linear | Grouped CVC based on product/domain hierarchy | De Nard (2022) |
| `BB` | Blockbuster CVC | Linear | Endogenously clustered groups ($K$ clusters) | De Nard (2022) |
| `ABB` | Adaptive Blockbuster CVC | Linear | Endogenously determined cluster count ($K$) and groupings | De Nard (2022) |
| `QIS` | Quadratic-Inverse Shrinkage | Non-linear | Non-linear quadratic eigenvalue regularizer | Ledoit & Wolf (2022) |

---

## 4. Usage & Replication Examples (Python)

```python
import pandas as pd

# Load dataset
df = pd.read_csv("output/instance_results.csv")

# 1. Filter Figure 4 runs (Scalability analysis)
fig4_df = df[df["figure_4"] == 1]
fig4_summary = fig4_df.groupby("num_scenarios").agg(
    mean_profit=("profit_improvement", lambda x: f"{x.mean()*100:.2f}%"),
    mean_runtime_min=("runtime_minutes", "mean")
)
print("Figure 4 Summary:")
print(fig4_summary)

# 2. Filter Figure 5 runs (Uncorrelated vs. correlated across alpha thresholds)
fig5_df = df[df["figure_5"] == 1]
fig5_summary = fig5_df.groupby(["num_scenarios", "estimator", "alpha"], dropna=False).agg(
    mean_profit=("profit_improvement", lambda x: f"{x.mean()*100:.2f}%"),
    std_profit=("profit_improvement", lambda x: f"{x.std()*100:.2f}%")
).reset_index()
print("\nFigure 5 Summary (First 10 rows):")
print(fig5_summary.head(10))

# 3. Filter Table 3 runs (Estimator comparison)
t3_df = df[df["table_3"] == 1]
t3_summary = t3_df.groupby(["num_scenarios", "alpha", "estimator"])["profit_improvement"].agg(
    Mean=lambda x: f"{x.mean()*100:.2f}%",
    Std=lambda x: f"{x.std()*100:.2f}%"
).reset_index()
print("\nTable 3 Summary (First 10 rows):")
print(t3_summary.head(10))
```
