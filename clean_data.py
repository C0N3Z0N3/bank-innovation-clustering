"""
clean_data.py
=============

End-to-end pipeline that transforms the raw WRDS Call Report CSV into a
clustering-ready dataset with one row per bank and 20 change-score features.

Usage
-----
    python clean_data.py

Outputs
-------
    bank_innovation_clean.csv
        One row per qualifying bank containing identifiers, bank tier,
        observation metadata, and 20 ``<ratio>_change`` columns ready for
        UMAP + HDBSCAN clustering.
"""

import logging
import sys

import pandas as pd

from bank_innovation.data_prep import (
    merge_split_columns,
    assign_sticky_bank_tiers,
    calculate_innovation_change_scores,
)
from bank_innovation.ratios import (
    calculate_ratios,
    calculate_additional_innovation_ratios,
)

# ── Configuration ────────────────────────────────────────────────────────────

INPUT_PATH = "data/wrds_bank_data_MERGED.csv"
OUTPUT_PATH = "data/bank_innovation_clean.csv"
MIN_YEARS = 3  # minimum years of data required per bank

IDENTIFIERS = [
    "rssd9001", "rssd9999", "rssd9017", "year", "quarter", "total_assets",
]

INNOVATION_FEATURES = [
    # Core ratios (11)
    "tech_investment_ratio",
    "nib_deposit_ratio",
    "service_charge_intensity",
    "efficiency_ratio",
    "nonint_income_pct",
    "loans_to_assets",
    "equity_to_assets",
    "deposits_to_assets",
    "roa",
    "roe",
    "nontrans_deposits_pct",
    # Additional ratios (9)
    "digital_revenue_ratio",
    "non_branch_revenue_pct",
    "loan_yield",
    "securities_to_assets",
    "expense_per_salary_dollar",
    "occupancy_intensity",
    "chargeoff_rate",
    "provision_intensity",
    "asset_growth_capacity",
]

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Pipeline ─────────────────────────────────────────────────────────────────

def main() -> None:
    # Step 1: Load raw data
    logger.info("Loading raw data from %s", INPUT_PATH)
    data = pd.read_csv(INPUT_PATH)
    logger.info("Loaded %d rows × %d columns", *data.shape)

    df = data.copy()
    df["report_date"] = pd.to_datetime(df["rssd9999"], errors="coerce")
    df["year"] = df["report_date"].dt.year
    df["quarter"] = df["report_date"].dt.quarter

    # Step 2: Merge RCFD/RCON split columns
    df = merge_split_columns(df)

    # Step 3: Calculate all 20 financial ratios
    df = calculate_ratios(df)
    df = calculate_additional_innovation_ratios(df)

    # Step 4: Select identifiers + innovation features, drop NAs
    existing_ids = [c for c in IDENTIFIERS if c in df.columns]
    existing_feats = [c for c in INNOVATION_FEATURES if c in df.columns]

    logger.info(
        "Feature selection — %d identifiers, %d innovation features",
        len(existing_ids), len(existing_feats),
    )

    df_slim = df[existing_ids + existing_feats].copy()

    before = len(df_slim)
    df_slim = df_slim.dropna(subset=existing_feats)
    logger.info(
        "Dropped NAs: %d → %d rows (%.1f%% retained)",
        before, len(df_slim), len(df_slim) / before * 100,
    )

    # Step 5: Assign sticky bank tiers
    df_slim = assign_sticky_bank_tiers(
        df_slim, asset_col="total_assets", min_consecutive_quarters=3,
    )

    # Step 6: Aggregate quarterly observations to bank-year means
    bank_year = (
        df_slim
        .groupby(["rssd9017", "year", "bank_tier"])[existing_feats]
        .mean()
        .reset_index()
    )
    logger.info("Bank-year aggregated: %d observations", len(bank_year))

    # Step 7: Calculate year-over-year change scores
    df_changes = calculate_innovation_change_scores(
        bank_year, existing_feats, min_years=MIN_YEARS,
    )

    change_cols = [c for c in df_changes.columns if c.endswith("_change")]
    logger.info("Change-score features: %d", len(change_cols))

    # Step 8: Write output
    df_changes.to_csv(OUTPUT_PATH, index=False)
    logger.info(
        "Wrote %d YoY observations (%d banks) × %d columns to %s",
        len(df_changes), df_changes["rssd9017"].nunique(),
        len(df_changes.columns), OUTPUT_PATH,
    )

    # Summary
    logger.info("── Tier summary ──")
    for tier in ["Small", "Medium", "Large"]:
        tier_rows = df_changes[df_changes["bank_tier"] == tier]
        logger.info(
            "  %s: %d observations (%d banks)",
            tier, len(tier_rows), tier_rows["rssd9017"].nunique(),
        )


if __name__ == "__main__":
    main()