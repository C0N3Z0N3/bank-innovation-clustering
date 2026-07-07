"""
Cluster Migration Analysis
============================

Tools for tracking how banks move between innovation clusters over time.
Operates on the clustered YoY panel where each bank has a cluster
assignment for every consecutive year pair.

Key concepts:

* **Migration event** — a bank's cluster assignment changes from one year
  pair to the next (e.g., cluster 1 in 2014→2015, cluster 3 in 2015→2016).
* **Migration matrix** — a transition count / probability matrix showing
  how often banks move from cluster *i* to cluster *j*.
* **Migration profile** — the ratio changes that accompanied a specific
  bank's transition between clusters.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_bank_cluster_history(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Pivot the clustered YoY panel into a bank × year cluster-assignment table.

    Each row represents one bank, and columns ``cluster_<year>`` contain
    the cluster assignment for the year pair ending in that year (i.e.
    the ``year_to`` value).

    :param df: Clustered YoY DataFrame with columns ``rssd9017``,
        ``bank_tier``, ``year_to``, and ``innovation_cluster``.
    :type df: pandas.DataFrame
    :returns: Wide-format DataFrame indexed by ``rssd9017`` with columns
        ``bank_tier`` and ``cluster_<year>`` for each year present.
    :rtype: pandas.DataFrame
    """
    logger.info("Building bank cluster history")

    # Use year_to as the reference year for each assignment
    pivot = df.pivot_table(
        index="rssd9017",
        columns="year_to",
        values="innovation_cluster",
        aggfunc="first",
    )

    # Rename columns to cluster_YYYY
    pivot.columns = [f"cluster_{int(y)}" for y in pivot.columns]

    # Add bank_tier (use the most recent assignment)
    tier_map = (
        df.sort_values("year_to")
        .drop_duplicates("rssd9017", keep="last")
        .set_index("rssd9017")["bank_tier"]
    )
    pivot["bank_tier"] = tier_map

    pivot = pivot.reset_index()

    n_banks = len(pivot)
    cluster_cols = [c for c in pivot.columns if c.startswith("cluster_")]
    logger.info(
        "Built history for %d banks across %d years",
        n_banks, len(cluster_cols),
    )

    return pivot


def detect_migrations(
    df: pd.DataFrame,
    cluster_history: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Identify every cluster migration event in the dataset.

    A migration occurs when a bank's cluster assignment changes between
    consecutive year pairs.  Returns one row per migration event with
    the bank ID, tier, years involved, source and destination clusters,
    and the change-score values that accompanied the transition.

    :param df: Clustered YoY DataFrame with columns ``rssd9017``,
        ``bank_tier``, ``year_from``, ``year_to``, ``innovation_cluster``,
        and change-score columns.
    :type df: pandas.DataFrame
    :param cluster_history: Pre-computed output of
        :func:`build_bank_cluster_history`.  If *None*, it will be
        computed internally.
    :type cluster_history: pandas.DataFrame, optional
    :returns: DataFrame with one row per migration event containing:

        * ``rssd9017`` — bank identifier
        * ``bank_tier`` — tier at time of migration
        * ``year_from`` — year the bank was in the source cluster
        * ``year_to`` — year the bank moved to the destination cluster
        * ``cluster_from`` — source cluster ID
        * ``cluster_to`` — destination cluster ID
        * all ``_change`` columns from the destination year pair
    :rtype: pandas.DataFrame
    """
    logger.info("Detecting cluster migration events")

    change_cols = [c for c in df.columns if c.endswith("_change")]

    # Sort so consecutive rows for the same bank are adjacent
    df_sorted = df.sort_values(["rssd9017", "year_to"]).copy()

    migrations: List[Dict] = []

    for bank_id, bank_data in df_sorted.groupby("rssd9017"):
        if len(bank_data) < 2:
            continue

        rows = bank_data.reset_index(drop=True)
        for i in range(len(rows) - 1):
            curr = rows.iloc[i]
            nxt = rows.iloc[i + 1]

            cluster_from = curr["innovation_cluster"]
            cluster_to = nxt["innovation_cluster"]

            if cluster_from != cluster_to:
                event: Dict = {
                    "rssd9017": bank_id,
                    "bank_tier": nxt["bank_tier"],
                    "year_from": int(curr["year_to"]),
                    "year_to": int(nxt["year_to"]),
                    "cluster_from": int(cluster_from),
                    "cluster_to": int(cluster_to),
                }

                # Attach the change scores from the destination year
                for col in change_cols:
                    event[col] = nxt[col]

                migrations.append(event)

    df_migrations = pd.DataFrame(migrations)

    if len(df_migrations) == 0:
        logger.warning("No migration events detected")
        return df_migrations

    n_banks = df_migrations["rssd9017"].nunique()
    logger.info(
        "Detected %d migration events across %d banks",
        len(df_migrations), n_banks,
    )

    # Summary by tier
    for tier in ["Small", "Medium", "Large"]:
        tier_m = df_migrations[df_migrations["bank_tier"] == tier]
        if len(tier_m) > 0:
            tier_banks = tier_m["rssd9017"].nunique()
            logger.info(
                "  %s: %d events (%d banks)",
                tier, len(tier_m), tier_banks,
            )

    return df_migrations


def compute_migration_matrix(
    df_migrations: pd.DataFrame,
    tier: Optional[str] = None,
    normalize: bool = True,
) -> pd.DataFrame:
    """Build a cluster-to-cluster transition matrix.

    :param df_migrations: Migration events DataFrame as returned by
        :func:`detect_migrations`.
    :type df_migrations: pandas.DataFrame
    :param tier: If provided, restrict to migrations within this tier.
        If *None*, use all migrations.
    :type tier: str, optional
    :param normalize: If *True*, return row-normalised transition
        probabilities (each row sums to 1).  If *False*, return raw
        counts.
    :type normalize: bool
    :returns: Square DataFrame where index is ``cluster_from`` and
        columns are ``cluster_to``.
    :rtype: pandas.DataFrame
    """
    data = df_migrations.copy()
    if tier is not None:
        data = data[data["bank_tier"] == tier]

    if len(data) == 0:
        logger.warning("No migrations to build matrix from")
        return pd.DataFrame()

    matrix = pd.crosstab(
        data["cluster_from"],
        data["cluster_to"],
    )

    if normalize and matrix.sum(axis=1).sum() > 0:
        matrix = matrix.div(matrix.sum(axis=1), axis=0)

    label = "probability" if normalize else "count"
    logger.info(
        "Migration matrix (%s%s): %d x %d",
        f"{tier} " if tier else "", label,
        matrix.shape[0], matrix.shape[1],
    )

    return matrix


def compute_migration_rates_by_year(
    df: pd.DataFrame,
    df_migrations: pd.DataFrame,
    tier: Optional[str] = None,
) -> pd.DataFrame:
    """Calculate the fraction of banks that changed clusters each year.

    :param df: Full clustered YoY DataFrame (to get total bank counts
        per year).
    :type df: pandas.DataFrame
    :param df_migrations: Migration events DataFrame.
    :type df_migrations: pandas.DataFrame
    :param tier: Restrict to a single tier, or *None* for all.
    :type tier: str, optional
    :returns: DataFrame with columns ``year``, ``total_banks``,
        ``migrating_banks``, and ``migration_rate``.
    :rtype: pandas.DataFrame
    """
    obs = df.copy()
    mig = df_migrations.copy()

    if tier is not None:
        obs = obs[obs["bank_tier"] == tier]
        mig = mig[mig["bank_tier"] == tier]

    # Total unique banks active in each year_to
    total_by_year = (
        obs.groupby("year_to")["rssd9017"]
        .nunique()
        .rename("total_banks")
    )

    # Migrating banks per year_to
    mig_by_year = (
        mig.groupby("year_to")["rssd9017"]
        .nunique()
        .rename("migrating_banks")
    )

    rates = pd.DataFrame({"total_banks": total_by_year}).join(
        mig_by_year, how="left"
    )
    rates["migrating_banks"] = rates["migrating_banks"].fillna(0).astype(int)
    rates["migration_rate"] = rates["migrating_banks"] / rates["total_banks"]
    rates = rates.reset_index().rename(columns={"year_to": "year"})

    logger.info(
        "Migration rates%s: mean=%.1f%%, range=%.1f%%–%.1f%%",
        f" ({tier})" if tier else "",
        rates["migration_rate"].mean() * 100,
        rates["migration_rate"].min() * 100,
        rates["migration_rate"].max() * 100,
    )

    return rates


def profile_migration_drivers(
    df_migrations: pd.DataFrame,
    cluster_from: int,
    cluster_to: int,
    tier: Optional[str] = None,
    top_n: int = 5,
) -> pd.DataFrame:
    """Identify which ratio changes are most distinctive for a specific transition.

    Compares the mean change scores of banks that migrated from
    *cluster_from* to *cluster_to* against all other migrations, ranking
    features by absolute difference.

    :param df_migrations: Migration events DataFrame with change-score
        columns.
    :type df_migrations: pandas.DataFrame
    :param cluster_from: Source cluster ID.
    :type cluster_from: int
    :param cluster_to: Destination cluster ID.
    :type cluster_to: int
    :param tier: Restrict to a specific tier, or *None* for all.
    :type tier: str, optional
    :param top_n: Number of top distinguishing features to return.
    :type top_n: int
    :returns: DataFrame with columns ``feature``, ``migration_mean``,
        ``other_mean``, ``difference`` sorted by absolute difference
        descending.
    :rtype: pandas.DataFrame
    """
    data = df_migrations.copy()
    if tier is not None:
        data = data[data["bank_tier"] == tier]

    change_cols = [c for c in data.columns if c.endswith("_change")]

    target_mask = (
        (data["cluster_from"] == cluster_from)
        & (data["cluster_to"] == cluster_to)
    )
    target = data[target_mask]
    others = data[~target_mask]

    if len(target) == 0:
        logger.warning(
            "No migrations found from cluster %d to %d", cluster_from, cluster_to,
        )
        return pd.DataFrame()

    rows = []
    for col in change_cols:
        mig_mean = target[col].mean()
        oth_mean = others[col].mean() if len(others) > 0 else 0.0
        rows.append({
            "feature": col,
            "migration_mean": mig_mean,
            "other_mean": oth_mean,
            "difference": mig_mean - oth_mean,
        })

    result = (
        pd.DataFrame(rows)
        .assign(abs_diff=lambda x: x["difference"].abs())
        .sort_values("abs_diff", ascending=False)
        .drop(columns="abs_diff")
        .head(top_n)
        .reset_index(drop=True)
    )

    logger.info(
        "Migration %d -> %d (%d events): top driver = %s (diff=%.4f)",
        cluster_from, cluster_to, len(target),
        result.iloc[0]["feature"] if len(result) > 0 else "N/A",
        result.iloc[0]["difference"] if len(result) > 0 else 0.0,
    )

    return result


def summarize_bank_trajectories(
    cluster_history: pd.DataFrame,
    tier: Optional[str] = None,
) -> pd.DataFrame:
    """Summarise each bank's cluster trajectory as a sequence string.

    Produces a compact view showing each bank's cluster path across years
    and counts total migrations.

    :param cluster_history: Wide-format bank × year table from
        :func:`build_bank_cluster_history`.
    :type cluster_history: pandas.DataFrame
    :param tier: Restrict to a specific tier, or *None* for all.
    :type tier: str, optional
    :returns: DataFrame with columns ``rssd9017``, ``bank_tier``,
        ``trajectory`` (e.g. ``"1→1→3→3→1"``), ``n_migrations``,
        and ``n_years``.
    :rtype: pandas.DataFrame
    """
    data = cluster_history.copy()
    if tier is not None:
        data = data[data["bank_tier"] == tier]

    cluster_cols = sorted(
        [c for c in data.columns if c.startswith("cluster_")]
    )

    rows = []
    for _, bank in data.iterrows():
        assignments = []
        for col in cluster_cols:
            val = bank[col]
            if pd.notna(val):
                assignments.append(str(int(val)))

        trajectory = "→".join(assignments) if assignments else ""

        # Count transitions
        n_mig = 0
        for i in range(len(assignments) - 1):
            if assignments[i] != assignments[i + 1]:
                n_mig += 1

        rows.append({
            "rssd9017": bank["rssd9017"],
            "bank_tier": bank["bank_tier"],
            "trajectory": trajectory,
            "n_migrations": n_mig,
            "n_years": len(assignments),
        })

    result = pd.DataFrame(rows)

    # Summary stats
    if len(result) > 0:
        n_movers = (result["n_migrations"] > 0).sum()
        n_stayers = (result["n_migrations"] == 0).sum()
        avg_mig = result["n_migrations"].mean()
        logger.info(
            "Trajectories%s: %d banks — %d movers (%.1f%%), "
            "%d stayers, avg migrations=%.2f",
            f" ({tier})" if tier else "",
            len(result), n_movers, n_movers / len(result) * 100,
            n_stayers, avg_mig,
        )

    return result
