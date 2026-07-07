"""
Data Preparation
=================

Functions for loading, cleaning, and transforming raw FDIC Call Report data
into a clustering-ready dataset.  Handles the RCFD/RCON column split that
occurred around 2011, assigns sticky bank-size tiers, aggregates quarterly
observations to bank-year level, and computes first-to-last-year change
scores that eliminate temporal autocorrelation.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column-merge pairs: (rcfd_col, rcon_col, merged_name)
# ---------------------------------------------------------------------------
SPLIT_PAIRS: List[Tuple[str, str, str]] = [
    ("rcfd2_rcfd2170", "rcon2_rcon2170", "total_assets"),
    ("rcfd2_rcfd2122", "rcon2_rcon2122", "total_loans"),
    ("rcfd2_rcfd3210", "rcon2_rcon3210", "total_equity"),
    ("rcfd1_rcfd3123", "rcon1_rcon3123", "allowance_loan_losses"),
    ("rcfd1_rcfd1590", "rcon1_rcon1590", "agricultural_loans"),
    ("rcfd1_rcfd1754", "rcon1_rcon1754", "htm_securities"),
    ("rcfd1_rcfd1773", "rcon1_rcon1773", "afs_securities"),
    ("rcfd1_rcfd2150", "rcon2_rcon2150", "goodwill"),
    ("rcfd1_rcfd0081", "rcon2_rcon0081", "cash_items_process"),
    ("rcfd2_rcfd1420", "rcon2_rcon1420", "farmland_loans"),
    ("rcfd2_rcfd1460", "rcon2_rcon1460", "multifamily_loans"),
]

# ---------------------------------------------------------------------------
# Bank-tier thresholds (assets reported in thousands on Call Reports)
# ---------------------------------------------------------------------------
SMALL_THRESHOLD: int = 1_000_000       # < $1 B
MEDIUM_THRESHOLD: int = 10_000_000     # $1 B – $10 B


def merge_split_columns(
    df: pd.DataFrame,
    split_pairs: Optional[List[Tuple[str, str, str]]] = None,
) -> pd.DataFrame:
    """Merge RCFD and RCON columns that split during the 2011 reporting transition.

    RCFD (consolidated) columns contain data for earlier periods while RCON
    (domestic) columns cover later periods.  For each pair the merged column
    takes the RCFD value when available, back-filling with RCON.  Original
    source columns are dropped after merging.

    :param df: Raw Call Report DataFrame.
    :type df: pandas.DataFrame
    :param split_pairs: Column triplets to merge.  Each triplet is
        ``(rcfd_column, rcon_column, merged_name)``.  Defaults to the
        package-level ``SPLIT_PAIRS`` constant when *None*.
    :type split_pairs: list of tuple(str, str, str), optional
    :returns: DataFrame with merged columns replacing the original pairs.
    :rtype: pandas.DataFrame

    .. note::
       The function operates on a copy of *df* only when columns are dropped;
       the caller should reassign the return value.
    """
    if split_pairs is None:
        split_pairs = SPLIT_PAIRS

    logger.info("Merging split RCFD/RCON columns (2011 transition)")

    merged_count = 0
    for rcfd_col, rcon_col, new_name in split_pairs:
        rcfd_exists = rcfd_col in df.columns
        rcon_exists = rcon_col in df.columns

        if rcfd_exists and rcon_exists:
            df[new_name] = df[rcfd_col].fillna(df[rcon_col])
            rcfd_n = df[rcfd_col].notna().sum()
            rcon_n = df[rcon_col].notna().sum()
            total_n = df[new_name].notna().sum()
            logger.debug(
                "%s | RCFD: %d | RCON: %d | Total: %d",
                new_name, rcfd_n, rcon_n, total_n,
            )
            df = df.drop(columns=[rcfd_col, rcon_col])
            merged_count += 1

        elif rcfd_exists:
            df[new_name] = df[rcfd_col]
            df = df.drop(columns=[rcfd_col])
            logger.warning("%s — only RCFD exists, renamed", new_name)
            merged_count += 1

        elif rcon_exists:
            df[new_name] = df[rcon_col]
            df = df.drop(columns=[rcon_col])
            logger.warning("%s — only RCON exists, renamed", new_name)
            merged_count += 1

        else:
            logger.warning("%s — neither column found, skipping", new_name)

    logger.info("Successfully merged/renamed %d split column pairs", merged_count)
    return df


def prepare_clustering_features(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str]]:
    """Select universal-coverage features and retain bank identifiers.

    Assembles three groups of columns — raw RIAD/RCON fields with near-
    universal coverage (< 2 % missing), merged RCFD/RCON columns, and
    calculated financial ratios — then drops any rows with missing feature
    values.

    :param df: DataFrame that has already been through
        :func:`merge_split_columns` and ratio calculation.
    :type df: pandas.DataFrame
    :returns: A two-element tuple:

        * **df_clean** — DataFrame retaining only identifier and feature
          columns, with incomplete rows removed.
        * **existing_features** — List of feature column names present in
          the cleaned DataFrame.
    :rtype: tuple(pandas.DataFrame, list of str)

    .. seealso:: :func:`merge_split_columns`, :func:`ratios.calculate_ratios`
    """
    logger.info("Preparing clustering features — universal coverage only")

    # RIAD columns — universal coverage (< 2 % missing)
    riad_universal = [
        "riad4010", "riad4012", "riad4020", "riad4073", "riad4074", "riad4079",
        "riad4080", "riad4092", "riad4093", "riad4107", "riad4115", "riad4135",
        "riad4150", "riad4180", "riad4217", "riad4230", "riad4266", "riad4267",
        "riad4300", "riad4301", "riad4302", "riad4313", "riad4340", "riad4356",
        "riad4415", "riad4435", "riad4436", "riad4460", "riad4470", "riad4498",
        "riad4499", "riad4507", "riad4508", "riad4518", "riad4605", "riad4608",
        "riad4628", "riad4635", "riad4638", "riad4644", "riad4769",
    ]

    # RCON columns — universal coverage, no split
    rcon_universal = [
        "rcon2_rcon2200",  # Total deposits
        "rcon2_rcon2202",  # Transaction accounts
        "rcon2_rcon2215",  # Nontransaction accounts
        "rcon2_rcon6631",  # NIB deposits
        "rcon1_rcon1766",  # C&I loans
    ]

    # Merged columns (from split pairs)
    merged_columns = [
        "total_assets", "total_loans", "total_equity",
        "allowance_loan_losses", "agricultural_loans",
        "htm_securities", "afs_securities", "goodwill",
        "cash_items_process", "farmland_loans", "multifamily_loans",
    ]

    # Calculated ratios (< 2 % missing)
    ratios_clean = [
        "tech_investment_ratio", "nib_deposit_ratio",
        "service_charge_intensity", "efficiency_ratio",
        "nonint_income_pct", "loans_to_assets", "equity_to_assets",
        "deposits_to_assets", "roa", "roe", "nontrans_deposits_pct",
    ]

    identifiers = ["rssd9001", "rssd9999", "rssd9017", "year", "quarter"]

    existing_identifiers = [c for c in identifiers if c in df.columns]
    missing_ids = set(identifiers) - set(existing_identifiers)
    if missing_ids:
        logger.warning("Missing identifier columns: %s", missing_ids)

    feature_cols = riad_universal + rcon_universal + merged_columns + ratios_clean
    existing_features = [c for c in feature_cols if c in df.columns]
    missing_features = set(feature_cols) - set(existing_features)

    logger.info(
        "Feature selection — expected: %d, found: %d, missing: %d",
        len(feature_cols), len(existing_features), len(missing_features),
    )
    if missing_features:
        logger.debug("Missing features: %s", missing_features)

    all_cols = existing_identifiers + existing_features
    df_subset = df[all_cols].copy()

    features_only = df_subset[existing_features]
    total_missing = features_only.isna().sum().sum()
    total_cells = features_only.shape[0] * features_only.shape[1]
    missing_pct = (total_missing / total_cells) * 100
    logger.info(
        "Missing values in features: %d (%.2f%%)", total_missing, missing_pct,
    )

    before_rows = len(df_subset)
    complete_mask = features_only.notna().all(axis=1)
    df_clean = df_subset[complete_mask].copy()
    logger.info(
        "Row retention after NA drop: %d → %d (%.1f%%)",
        before_rows, len(df_clean), len(df_clean) / before_rows * 100,
    )

    logger.info(
        "Final dataset — %d observations, %d identifiers, %d features",
        len(df_clean), len(existing_identifiers), len(existing_features),
    )
    return df_clean, existing_features


def assign_sticky_bank_tiers(
    df: pd.DataFrame,
    asset_col: str = "total_assets",
    min_consecutive_quarters: int = 3,
) -> pd.DataFrame:
    """Assign bank-size tiers with hysteresis to prevent quarterly oscillation.

    A bank's tier changes only after its raw asset-based tier has been
    consistently different for *min_consecutive_quarters* consecutive
    quarters.  Thresholds follow industry-standard boundaries:

    * **Small** — total assets < $1 B
    * **Medium** — $1 B ≤ total assets < $10 B
    * **Large** — total assets ≥ $10 B

    :param df: DataFrame containing at least ``rssd9017``, ``year``,
        ``quarter``, and the column named by *asset_col*.
    :type df: pandas.DataFrame
    :param asset_col: Name of the total-assets column (values in thousands).
    :type asset_col: str
    :param min_consecutive_quarters: Number of consecutive quarters a bank
        must remain in a new raw tier before the sticky assignment changes.
    :type min_consecutive_quarters: int
    :returns: DataFrame with a new ``bank_tier`` column and the temporary
        ``raw_tier`` column removed.
    :rtype: pandas.DataFrame
    """
    logger.info(
        "Assigning sticky bank tiers (threshold: %d consecutive quarters)",
        min_consecutive_quarters,
    )

    def _raw_tier(assets: float) -> Optional[str]:
        if pd.isna(assets):
            return None
        if assets < SMALL_THRESHOLD:
            return "Small"
        if assets < MEDIUM_THRESHOLD:
            return "Medium"
        return "Large"

    df = df.sort_values(["rssd9017", "year", "quarter"]).copy()
    df["raw_tier"] = df[asset_col].apply(_raw_tier)
    df["bank_tier"] = None

    tier_changes = 0
    total_banks = df["rssd9017"].nunique()

    for bank_id in df["rssd9017"].unique():
        bank_mask = df["rssd9017"] == bank_id
        bank_data = df.loc[bank_mask]

        current_tier = bank_data["raw_tier"].iloc[0]
        df.loc[bank_mask, "bank_tier"] = current_tier

        consecutive_count = 0
        potential_new_tier = None

        for idx in bank_data.index[1:]:
            raw = df.loc[idx, "raw_tier"]

            if raw != current_tier:
                if raw == potential_new_tier:
                    consecutive_count += 1
                else:
                    potential_new_tier = raw
                    consecutive_count = 1

                if consecutive_count >= min_consecutive_quarters:
                    current_tier = potential_new_tier
                    tier_changes += 1
                    consecutive_count = 0
                    potential_new_tier = None
            else:
                consecutive_count = 0
                potential_new_tier = None

            df.loc[idx, "bank_tier"] = current_tier

    logger.info("Processed %d banks with %d tier changes", total_banks, tier_changes)

    tier_counts = df.groupby("bank_tier")["rssd9017"].nunique()
    for tier in ["Small", "Medium", "Large"]:
        if tier in tier_counts.index:
            count = tier_counts[tier]
            pct = count / total_banks * 100
            logger.info("  %s: %d banks (%.1f%%)", tier, count, pct)

    df = df.drop(columns=["raw_tier"])
    return df


def calculate_innovation_change_scores(
    df: pd.DataFrame,
    feature_list: List[str],
    min_years: int = 3,
) -> pd.DataFrame:
    """Compute year-over-year change scores for each consecutive year pair.

    For every bank with at least *min_years* of annual observations the
    function calculates ``year_t+1 − year_t`` for each feature across all
    consecutive year pairs, producing a multi-row-per-bank panel suitable
    for pooled or per-year clustering.

    :param df: Bank-year DataFrame with columns ``rssd9017``, ``year``,
        ``bank_tier``, and all columns listed in *feature_list*.
    :type df: pandas.DataFrame
    :param feature_list: Column names to compute change scores for.
    :type feature_list: list of str
    :param min_years: Minimum number of distinct years a bank must have
        to be included.  Banks with fewer years are excluded entirely.
    :type min_years: int
    :returns: DataFrame with one row per consecutive year pair per
        qualifying bank.  Contains metadata columns (``rssd9017``,
        ``bank_tier``, ``year_from``, ``year_to``, ``years_observed``)
        and ``<feature>_change`` for each feature.
    :rtype: pandas.DataFrame
    """
    logger.info("Calculating YoY innovation change scores (min_years=%d)", min_years)

    bank_changes: List[Dict] = []
    banks_excluded = 0

    for bank_id in df["rssd9017"].unique():
        bank_data = df[df["rssd9017"] == bank_id].sort_values("year")

        if len(bank_data) < min_years:
            banks_excluded += 1
            continue

        for i in range(len(bank_data) - 1):
            year_from = bank_data.iloc[i]
            year_to = bank_data.iloc[i + 1]

            # Only pair consecutive years — skip gaps
            if int(year_to["year"]) - int(year_from["year"]) != 1:
                continue

            changes: Dict = {
                "rssd9017": bank_id,
                "bank_tier": year_to["bank_tier"],
                "year_from": int(year_from["year"]),
                "year_to": int(year_to["year"]),
                "years_observed": len(bank_data),
            }

            for feat in feature_list:
                if feat in bank_data.columns:
                    changes[f"{feat}_change"] = year_to[feat] - year_from[feat]

            bank_changes.append(changes)

    df_changes = pd.DataFrame(bank_changes)

    n_banks = df_changes["rssd9017"].nunique()
    logger.info(
        "Produced %d YoY observations from %d banks, excluded %d banks",
        len(df_changes), n_banks, banks_excluded,
    )

    tier_counts = df_changes["bank_tier"].value_counts()
    for tier in ["Small", "Medium", "Large"]:
        if tier in tier_counts.index:
            logger.info("  %s: %d observations", tier, tier_counts[tier])

    change_cols = [c for c in df_changes.columns if c.endswith("_change")]
    missing_pct = (
        df_changes[change_cols].isna().sum().sum()
        / (len(df_changes) * len(change_cols))
        * 100
    )
    logger.info("Missing values in change scores: %.2f%%", missing_pct)

    return df_changes