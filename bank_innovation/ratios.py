"""
Financial Ratio Calculation
============================

Functions for computing 20 size-independent financial ratios from raw
Call Report fields.  The ratios serve as innovation and efficiency proxies
and fall into six categories:

1. **Core Innovation** — tech investment, NIB deposits, service charges
2. **Efficiency** — efficiency ratio, noninterest income share
3. **Balance Sheet** — loan, equity, and deposit composition
4. **Profitability** — ROA, ROE
5. **Deposit Mix** — nontransaction deposit share
6. **Additional Innovation / Efficiency** — digital revenue, non-branch
   revenue, loan yield, securities intensity, expense leverage, occupancy,
   credit quality, and capital efficiency
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Element-wise division replacing inf/−inf with NaN.

    :param numerator: Numerator series.
    :type numerator: pandas.Series
    :param denominator: Denominator series.
    :type denominator: pandas.Series
    :returns: Quotient with infinities replaced by ``NaN``.
    :rtype: pandas.Series
    """
    result = numerator / denominator
    return result.replace([np.inf, -np.inf], np.nan)


def calculate_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate the first 11 core financial ratios.

    Expects the DataFrame to contain merged asset/equity/loan columns
    produced by :func:`data_prep.merge_split_columns` as well as raw
    RIAD and RCON fields.

    :param df: DataFrame with merged columns and raw Call Report fields.
    :type df: pandas.DataFrame
    :returns: Copy of *df* augmented with the following ratio columns:

        * ``tech_investment_ratio`` — (RIAD 4092 / total assets) × 1 000
        * ``nib_deposit_ratio`` — (NIB deposits / total deposits) × 100
        * ``service_charge_intensity`` — (RIAD 4080 / total deposits) × 1 000
        * ``efficiency_ratio`` — (noninterest expense / total revenue) × 100
        * ``nonint_income_pct`` — (noninterest income / total revenue) × 100
        * ``loans_to_assets`` — (total loans / total assets) × 100
        * ``equity_to_assets`` — (total equity / total assets) × 100
        * ``deposits_to_assets`` — (total deposits / total assets) × 100
        * ``roa`` — (net income / total assets) × 100
        * ``roe`` — (net income / total equity) × 100
        * ``nontrans_deposits_pct`` — (nontransaction deposits / total
          deposits) × 100
    :rtype: pandas.DataFrame
    """
    logger.info("Calculating core financial ratios (11 ratios)")
    ratios = df.copy()

    # === Core Innovation (3) ===
    ratios["tech_investment_ratio"] = (
        _safe_divide(df["riad4092"], df["total_assets"]) * 1000
    )
    ratios["nib_deposit_ratio"] = (
        _safe_divide(df["rcon2_rcon6631"], df["rcon2_rcon2200"]) * 100
    )
    ratios["service_charge_intensity"] = (
        _safe_divide(df["riad4080"], df["rcon2_rcon2200"]) * 1000
    )

    # === Efficiency (2) ===
    total_revenue = df["riad4074"] + df["riad4079"]
    ratios["efficiency_ratio"] = _safe_divide(df["riad4093"], total_revenue) * 100
    ratios["nonint_income_pct"] = _safe_divide(df["riad4079"], total_revenue) * 100

    # === Balance Sheet (3) ===
    ratios["loans_to_assets"] = (
        _safe_divide(df["total_loans"], df["total_assets"]) * 100
    )
    ratios["equity_to_assets"] = (
        _safe_divide(df["total_equity"], df["total_assets"]) * 100
    )
    ratios["deposits_to_assets"] = (
        _safe_divide(df["rcon2_rcon2200"], df["total_assets"]) * 100
    )

    # === Profitability (2) ===
    ratios["roa"] = _safe_divide(df["riad4340"], df["total_assets"]) * 100
    ratios["roe"] = _safe_divide(df["riad4340"], df["total_equity"]) * 100

    # === Deposit Mix (1) ===
    ratios["nontrans_deposits_pct"] = (
        _safe_divide(df["rcon2_rcon2215"], df["rcon2_rcon2200"]) * 100
    )

    logger.info("Created 11 core ratios")
    return ratios


def calculate_additional_innovation_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate 9 additional innovation and efficiency ratios.

    These supplement the core ratios with metrics capturing digital revenue
    channels, operational leverage, credit quality, and capital efficiency.

    :param df: DataFrame that has already passed through
        :func:`calculate_ratios`.
    :type df: pandas.DataFrame
    :returns: Copy of *df* augmented with the following ratio columns:

        * ``digital_revenue_ratio`` — (credit-card fees / total revenue)
          × 100
        * ``non_branch_revenue_pct`` — ((noninterest income − service
          charges) / total revenue) × 100
        * ``loan_yield`` — (interest income on loans / total loans) × 100
        * ``securities_to_assets`` — ((HTM + AFS securities) / total
          assets) × 100
        * ``expense_per_salary_dollar`` — noninterest expense / salaries
        * ``occupancy_intensity`` — (occupancy expense / total assets)
          × 1 000
        * ``chargeoff_rate`` — (total charge-offs / total loans) × 100
        * ``provision_intensity`` — (provision for loan losses / total
          loans) × 100
        * ``asset_growth_capacity`` — (total equity / total assets) × 100
    :rtype: pandas.DataFrame
    """
    logger.info("Calculating additional innovation/efficiency ratios (9 ratios)")
    ratios = df.copy()

    total_revenue = df["riad4074"] + df["riad4079"]

    ratios["digital_revenue_ratio"] = (
        _safe_divide(df["riad4415"], total_revenue) * 100
    )
    ratios["non_branch_revenue_pct"] = (
        _safe_divide(df["riad4079"] - df["riad4080"], total_revenue) * 100
    )
    ratios["loan_yield"] = (
        _safe_divide(df["riad4107"], df["total_loans"]) * 100
    )
    ratios["securities_to_assets"] = (
        _safe_divide(
            df["htm_securities"] + df["afs_securities"], df["total_assets"]
        )
        * 100
    )
    ratios["expense_per_salary_dollar"] = _safe_divide(
        df["riad4093"], df["riad4135"]
    )
    ratios["occupancy_intensity"] = (
        _safe_divide(df["riad4115"], df["total_assets"]) * 1000
    )
    ratios["chargeoff_rate"] = (
        _safe_divide(df["riad4635"], df["total_loans"]) * 100
    )
    ratios["provision_intensity"] = (
        _safe_divide(df["riad4230"], df["total_loans"]) * 100
    )
    ratios["asset_growth_capacity"] = (
        _safe_divide(df["total_equity"], df["total_assets"]) * 100
    )

    logger.info("Created 9 additional innovation/efficiency ratios")
    return ratios
