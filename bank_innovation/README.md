# Bank Innovation Clustering

An unsupervised learning pipeline for identifying distinct innovation patterns across U.S. commercial banks using FDIC Call Report data (2010–2021).

## Overview

The pipeline applies **UMAP** dimensionality reduction paired with **HDBSCAN** density-based clustering to first-to-last-year change scores of 20 size-independent financial ratios, segmented by three bank asset-size tiers:

| Tier   | Threshold          | Typical Count |
|--------|--------------------|---------------|
| Small  | < $1 B assets      | ~3,600 banks  |
| Medium | $1 B – $10 B       | ~630 banks    |
| Large  | ≥ $10 B            | ~120 banks    |

Change scores (last year value − first year value) eliminate temporal autocorrelation that would otherwise cause clustering to recover institutional identity rather than innovation trajectories.

## Package Structure

```
bank_innovation/
├── __init__.py          # Public API exports
├── data_prep.py         # Column merging, sticky tiers, change scores
├── ratios.py            # 20 financial ratio calculations
├── clustering.py        # UMAP + HDBSCAN pipeline & parameter tuning
├── analysis.py          # ANOVA feature importance, cluster profiling
└── visualization.py     # Publication-ready UMAP scatter plots
```

## Installation

```bash
pip install -r requirements.txt
```

Then place the `bank_innovation/` directory on your Python path or install in development mode.

## Quick Start

```python
import logging
import pandas as pd
from bank_innovation import (
    merge_split_columns,
    calculate_ratios,
    calculate_additional_innovation_ratios,
    assign_sticky_bank_tiers,
    calculate_innovation_change_scores,
    cluster_by_tier,
    comprehensive_tuning,
    cluster_with_best_params,
    assign_cluster_names,
    analyze_innovation_change_clusters,
    create_clustering_visualization,
    print_clustering_summary,
)

# Configure logging (replaces print statements)
logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")

# 1. Load raw data
data = pd.read_csv("wrds_bank_data_MERGED.csv")
df = data.copy()
df["report_date"] = pd.to_datetime(df["rssd9999"], errors="coerce")
df["year"] = df["report_date"].dt.year
df["quarter"] = df["report_date"].dt.quarter

# 2. Merge split RCFD/RCON columns
df = merge_split_columns(df)

# 3. Calculate all 20 financial ratios
df = calculate_ratios(df)
df = calculate_additional_innovation_ratios(df)

# 4. Select innovation features and drop NAs
innovation_features = [
    "tech_investment_ratio", "nib_deposit_ratio", "service_charge_intensity",
    "efficiency_ratio", "nonint_income_pct", "loans_to_assets",
    "equity_to_assets", "deposits_to_assets", "roa", "roe",
    "nontrans_deposits_pct", "digital_revenue_ratio", "non_branch_revenue_pct",
    "loan_yield", "securities_to_assets", "expense_per_salary_dollar",
    "occupancy_intensity", "chargeoff_rate", "provision_intensity",
    "asset_growth_capacity",
]

identifiers = ["rssd9001", "rssd9999", "rssd9017", "year", "quarter", "total_assets"]
cols = [c for c in identifiers + innovation_features if c in df.columns]
df_slim = df[cols].dropna(subset=innovation_features)

# 5. Assign sticky bank tiers
df_slim = assign_sticky_bank_tiers(df_slim)

# 6. Aggregate to bank-year and compute change scores
bank_year = df_slim.groupby(["rssd9017", "year", "bank_tier"])[innovation_features].mean().reset_index()
df_changes = calculate_innovation_change_scores(bank_year, innovation_features, min_years=9)

# 7. Cluster
change_cols = [c for c in df_changes.columns if c.endswith("_change")]
df_changes = cluster_by_tier(df_changes, change_cols)

# 8. Analyse & visualise
print_clustering_summary(df_changes)
for tier in ["Large", "Medium", "Small"]:
    analyze_innovation_change_clusters(df_changes, tier, change_cols)
fig = create_clustering_visualization(df_changes)
```

## Logging

All modules use Python's `logging` library under the `bank_innovation.*` namespace. Configure the root logger or per-module loggers as needed:

```python
logging.getLogger("bank_innovation").setLevel(logging.DEBUG)   # verbose
logging.getLogger("bank_innovation").setLevel(logging.WARNING)  # quiet
```

## Key Methodological Notes

- **Sticky tiers** prevent banks from oscillating between size categories quarter-to-quarter by requiring 3 consecutive quarters in a new tier before reassignment.
- **Change scores** (first-to-last year) produce one observation per bank, eliminating temporal autocorrelation in the quarterly panel data.
- **Parameter tuning** uses a combined score favouring ~4 clusters, low noise, and balanced cluster sizes.
- **Colorblind-friendly palette** (Wong 2011) is used in all visualisations.

## Author

Justin Dorval
