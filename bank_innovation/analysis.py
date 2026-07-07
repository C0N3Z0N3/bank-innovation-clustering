"""
Cluster Analysis
=================

Statistical tools for evaluating clustering results.  Includes ANOVA-based
feature-importance ranking, cluster profiling by mean change values, and
asset-size independence checks.

Supports year-range filtering via ``year_start`` / ``year_end`` parameters
and single-cluster isolation via the ``cluster_id`` parameter.
"""

import logging
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def _filter_by_years(df, year_start, year_end):
    """Apply year-range filtering to a DataFrame.

    Filters rows based on the YoY year-pair columns produced by
    :func:`data_prep.calculate_innovation_change_scores`.  When
    *year_start* is given, only rows whose ``year_from`` >= *year_start*
    are kept.  When *year_end* is given, only rows whose ``year_to`` <=
    *year_end* are kept.  If the expected columns are missing a warning
    is logged and the DataFrame is returned unmodified.

    :param df: DataFrame to filter.
    :type df: pandas.DataFrame
    :param year_start: Earliest ``year_from`` to include (inclusive), or
        *None* to skip lower-bound filtering.
    :type year_start: int or None
    :param year_end: Latest ``year_to`` to include (inclusive), or *None*
        to skip upper-bound filtering.
    :type year_end: int or None
    :returns: Filtered copy of *df*.
    :rtype: pandas.DataFrame
    """
    filtered = df.copy()

    if year_start is not None:
        if "year_from" in filtered.columns:
            filtered = filtered[filtered["year_from"] >= year_start]
        else:
            logger.warning(
                "year_start=%d requested but 'year_from' column not found; "
                "skipping lower-bound filter",
                year_start,
            )

    if year_end is not None:
        if "year_to" in filtered.columns:
            filtered = filtered[filtered["year_to"] <= year_end]
        else:
            logger.warning(
                "year_end=%d requested but 'year_to' column not found; "
                "skipping upper-bound filter",
                year_end,
            )

    if year_start is not None or year_end is not None:
        logger.info(
            "Year filter applied (start=%s, end=%s): %d → %d rows",
            year_start, year_end, len(df), len(filtered),
        )

    return filtered


def analyze_cluster_by_size_changes(
    df: pd.DataFrame,
    tier: str,
    cluster_id: Optional[int] = None,
    year_start: Optional[int] = None,
    year_end: Optional[int] = None,
    compare_to_rest: bool = True,
) -> Optional[dict]:
    """Check whether clusters are driven by asset size rather than innovation.

    Performs a one-way ANOVA on asset values across non-noise clusters.  A
    statistically significant result (p < 0.001) suggests that clustering
    may be confounded by institutional size.

    When *cluster_id* is provided the analysis is restricted to that single
    cluster.  If *compare_to_rest* is ``True`` (default) the ANOVA compares
    the selected cluster against all other non-noise clusters combined;
    otherwise only a descriptive profile of the selected cluster is returned.

    :param df: Clustered change-score DataFrame with ``bank_tier``,
        ``innovation_cluster``, and optionally ``total_assets_last`` or
        ``total_assets_first`` columns.
    :type df: pandas.DataFrame
    :param tier: Tier to analyse (``'Large'``, ``'Medium'``, or
        ``'Small'``).
    :type tier: str
    :param cluster_id: If provided, restrict analysis to this single cluster.
        Pass *None* (default) to analyse all clusters together.
    :type cluster_id: int or None
    :param year_start: If provided, only include observations whose
        ``year_from`` is >= this value.
    :type year_start: int or None
    :param year_end: If provided, only include observations whose
        ``year_to`` is <= this value.
    :type year_end: int or None
    :param compare_to_rest: When *cluster_id* is set, whether to run an
        ANOVA comparing the selected cluster against the rest (``True``) or
        just return descriptive statistics (``False``).  Ignored when
        *cluster_id* is ``None``.
    :type compare_to_rest: bool
    :returns: Dictionary with keys ``f_stat``, ``p_value``, and
        ``size_driven`` (bool) if ANOVA was performed.  When only a
        profile is requested, returns a dictionary with ``cluster_id``,
        ``n``, ``mean``, ``median``, ``min``, ``max``.  Returns *None* if
        asset columns are unavailable.
    :rtype: dict or None
    """
    # Apply year filter first
    df = _filter_by_years(df, year_start, year_end)

    label = f"{tier} banks"
    if cluster_id is not None:
        label += f", cluster {cluster_id}"
    logger.info("Cluster vs asset-size analysis — %s", label)

    tier_data = df[df["bank_tier"] == tier].copy()
    all_clusters = sorted(
        c for c in tier_data["innovation_cluster"].unique() if c != -1
    )

    # Locate an asset-size column
    asset_cols = [c for c in tier_data.columns if "total_assets" in c.lower()]
    if not asset_cols:
        logger.warning("No asset-size column found; skipping analysis")
        return None

    if "total_assets_last" in tier_data.columns:
        asset_col = "total_assets_last"
    elif "total_assets_first" in tier_data.columns:
        asset_col = "total_assets_first"
    else:
        asset_col = asset_cols[0]

    logger.info("Using asset column: %s", asset_col)

    # --- Single-cluster mode ---
    if cluster_id is not None:
        if cluster_id not in tier_data["innovation_cluster"].values:
            logger.warning(
                "Cluster %d not found in %s tier", cluster_id, tier
            )
            return None

        cdata = tier_data[tier_data["innovation_cluster"] == cluster_id]
        assets = cdata[asset_col]

        profile = {
            "cluster_id": cluster_id,
            "n": len(cdata),
            "mean": assets.mean(),
            "median": assets.median(),
            "min": assets.min(),
            "max": assets.max(),
        }
        logger.info(
            "  Cluster %d — n=%d, mean=%.0f, median=%.0f, min=%.0f, max=%.0f",
            cluster_id, profile["n"], profile["mean"],
            profile["median"], profile["min"], profile["max"],
        )

        if not compare_to_rest:
            return profile

        # Compare selected cluster vs rest
        rest_clusters = [c for c in all_clusters if c != cluster_id]
        if not rest_clusters:
            logger.warning("No other clusters to compare against")
            return profile

        rest_data = tier_data[
            tier_data["innovation_cluster"].isin(rest_clusters)
        ]
        f_stat, p_value = stats.f_oneway(
            assets.values, rest_data[asset_col].values
        )
        size_driven = p_value < 0.001

        if size_driven:
            logger.warning(
                "Cluster %d differs significantly from rest by asset size "
                "(F=%.4f, p=%.6f)",
                cluster_id, f_stat, p_value,
            )
        else:
            logger.info(
                "Cluster %d has similar asset distribution to rest "
                "(F=%.4f, p=%.6f)",
                cluster_id, f_stat, p_value,
            )

        profile.update({
            "f_stat": f_stat,
            "p_value": p_value,
            "size_driven": size_driven,
        })
        return profile

    # --- All-clusters mode (original behaviour) ---
    for cluster in all_clusters:
        cdata = tier_data[tier_data["innovation_cluster"] == cluster]
        assets = cdata[asset_col]
        logger.debug(
            "  Cluster %d — n=%d, mean=%.0f, median=%.0f",
            cluster, len(cdata), assets.mean(), assets.median(),
        )

    if -1 in tier_data["innovation_cluster"].values:
        noise = tier_data[tier_data["innovation_cluster"] == -1]
        logger.debug(
            "  Noise — n=%d, mean=%.0f, median=%.0f",
            len(noise), noise[asset_col].mean(), noise[asset_col].median(),
        )

    result: Optional[dict] = None
    if len(all_clusters) > 1:
        cluster_assets = [
            tier_data[tier_data["innovation_cluster"] == c][asset_col].values
            for c in all_clusters
        ]
        f_stat, p_value = stats.f_oneway(*cluster_assets)
        size_driven = p_value < 0.001

        if size_driven:
            logger.warning(
                "Clusters differ significantly by asset size "
                "(F=%.4f, p=%.6f) — may be size-driven",
                f_stat, p_value,
            )
        else:
            logger.info(
                "Clusters have similar asset distributions "
                "(F=%.4f, p=%.6f)",
                f_stat, p_value,
            )

        result = {"f_stat": f_stat, "p_value": p_value, "size_driven": size_driven}

    return result


def analyze_innovation_change_clusters(
    df: pd.DataFrame,
    tier: str,
    change_feature_cols: List[str],
    cluster_id: Optional[int] = None,
    year_start: Optional[int] = None,
    year_end: Optional[int] = None,
    compare_to_rest: bool = True,
    top_n: int = 10,
    profile_n: int = 5,
) -> Optional[pd.DataFrame]:
    """Rank features by discriminating power and profile each cluster.

    Uses one-way ANOVA F-statistics to identify which change-score features
    most strongly differentiate the non-noise clusters, then prints mean
    change values for the top features.

    When *cluster_id* is provided the analysis focuses on that single
    cluster.  If *compare_to_rest* is ``True`` the ANOVA compares the
    selected cluster against all other non-noise clusters combined; if
    ``False`` only a descriptive profile (mean/std per feature) is returned
    with no F-statistics.

    :param df: Clustered change-score DataFrame.
    :type df: pandas.DataFrame
    :param tier: Bank tier to analyse.
    :type tier: str
    :param change_feature_cols: List of change-score column names.
    :type change_feature_cols: list of str
    :param cluster_id: If provided, restrict analysis to this single cluster.
        Pass *None* (default) to analyse all clusters together.
    :type cluster_id: int or None
    :param year_start: If provided, only include observations whose
        ``year_from`` is >= this value.
    :type year_start: int or None
    :param year_end: If provided, only include observations whose
        ``year_to`` is <= this value.
    :type year_end: int or None
    :param compare_to_rest: When *cluster_id* is set, whether to run an
        ANOVA comparing the selected cluster against the rest (``True``) or
        just return a descriptive profile (``False``).  Ignored when
        *cluster_id* is ``None``.
    :type compare_to_rest: bool
    :param top_n: Number of top-ranked features to report.
    :type top_n: int
    :param profile_n: Number of top features for which to print per-cluster
        mean/std tables.
    :type profile_n: int
    :returns: DataFrame with columns ``feature``, ``f_stat``, ``p_value``
        sorted by F-statistic descending (all-clusters and compare_to_rest
        modes), or DataFrame with columns ``feature``, ``mean``, ``std``
        (profile-only mode), or *None* if no clusters exist.
    :rtype: pandas.DataFrame or None
    """
    # Apply year filter first
    df = _filter_by_years(df, year_start, year_end)

    label = f"{tier} banks"
    if cluster_id is not None:
        label += f", cluster {cluster_id}"
    logger.info("Innovation change cluster analysis — %s", label)

    tier_data = df[df["bank_tier"] == tier].copy()
    all_clusters = sorted(
        c for c in tier_data["innovation_cluster"].unique() if c != -1
    )

    if not all_clusters:
        logger.warning("No clusters found for tier %s", tier)
        return None

    # --- Single-cluster, profile-only mode ---
    if cluster_id is not None and not compare_to_rest:
        if cluster_id not in tier_data["innovation_cluster"].values:
            logger.warning(
                "Cluster %d not found in %s tier", cluster_id, tier
            )
            return None

        cdata = tier_data[tier_data["innovation_cluster"] == cluster_id]
        logger.info(
            "Profiling cluster %d (%d observations)", cluster_id, len(cdata)
        )

        profile_rows = []
        for feature in change_feature_cols:
            if feature not in cdata.columns:
                continue
            profile_rows.append({
                "feature": feature,
                "mean": cdata[feature].mean(),
                "std": cdata[feature].std(),
            })

        profile_df = pd.DataFrame(profile_rows)

        # Log the profile
        for _, row in profile_df.iterrows():
            logger.info(
                "  %-45s mean=%10.4f  std=%10.4f",
                row["feature"], row["mean"], row["std"],
            )

        # Interpretation labels
        profile_parts = []
        overall_means = {
            feat: tier_data[feat].mean()
            for feat in change_feature_cols
            if feat in tier_data.columns
        }
        # Sort by absolute deviation from tier mean to find distinguishing features
        deviations = []
        for _, row in profile_df.iterrows():
            feat = row["feature"]
            if feat in overall_means and overall_means[feat] != 0:
                deviations.append(
                    (feat, abs(row["mean"] - overall_means[feat]))
                )
            else:
                deviations.append((feat, abs(row["mean"])))
        deviations.sort(key=lambda x: x[1], reverse=True)

        for feat, _ in deviations[:3]:
            cluster_mean = cdata[feat].mean()
            overall_mean = overall_means.get(feat, 0)

            if cluster_mean > overall_mean * 1.2:
                direction = "HIGH INCREASE"
            elif cluster_mean < overall_mean * 0.8 and overall_mean < 0:
                direction = "HIGH DECREASE"
            elif abs(cluster_mean) < abs(overall_mean) * 0.5:
                direction = "STABLE"
            else:
                direction = "MODERATE"

            profile_parts.append(
                f"{feat.replace('_change', '')}: {direction}"
            )

        logger.info(
            "  Cluster %d (%d observations) — %s",
            cluster_id, len(cdata), " | ".join(profile_parts),
        )

        return profile_df

    # --- Single-cluster with ANOVA vs rest ---
    if cluster_id is not None and compare_to_rest:
        if cluster_id not in tier_data["innovation_cluster"].values:
            logger.warning(
                "Cluster %d not found in %s tier", cluster_id, tier
            )
            return None

        cdata = tier_data[tier_data["innovation_cluster"] == cluster_id]
        rest_clusters = [c for c in all_clusters if c != cluster_id]

        if not rest_clusters:
            logger.warning("No other clusters to compare against")
            return None

        rest_data = tier_data[
            tier_data["innovation_cluster"].isin(rest_clusters)
        ]

        logger.info(
            "Comparing cluster %d (%d obs) vs rest (%d obs)",
            cluster_id, len(cdata), len(rest_data),
        )

        importance_rows = []
        for feature in change_feature_cols:
            if feature not in tier_data.columns:
                continue
            f_stat, p_value = stats.f_oneway(
                cdata[feature].values, rest_data[feature].values
            )
            importance_rows.append({
                "feature": feature,
                "f_stat": f_stat,
                "p_value": p_value.round(6),
            })

        importance_df = (
            pd.DataFrame(importance_rows)
            .sort_values("f_stat", ascending=False)
            .reset_index(drop=True)
        )

        # Log top-N features
        for _, row in importance_df.head(top_n).iterrows():
            sig = (
                "***" if row["p_value"] < 0.001
                else "**" if row["p_value"] < 0.01
                else "*" if row["p_value"] < 0.05
                else ""
            )
            logger.info(
                "  %-45s F=%10.2f  p=%12.6f %s",
                row["feature"], row["f_stat"], row["p_value"], sig,
            )

        # Profile the selected cluster vs rest for top features
        for _, row in importance_df.head(profile_n).iterrows():
            feat = row["feature"]
            display_name = feat.replace("_change", "")
            lines = [f"  {display_name}:"]
            lines.append(
                f"    Cluster {cluster_id} (n={len(cdata)}): "
                f"mean={cdata[feat].mean():.4f}, "
                f"std={cdata[feat].std():.4f}"
            )
            lines.append(
                f"    Rest (n={len(rest_data)}): "
                f"mean={rest_data[feat].mean():.4f}, "
                f"std={rest_data[feat].std():.4f}"
            )
            logger.info("\n".join(lines))

        return importance_df

    # --- All-clusters mode (original behaviour) ---
    importance_rows = []
    for feature in change_feature_cols:
        if feature not in tier_data.columns:
            continue
        cluster_values = [
            tier_data[tier_data["innovation_cluster"] == c][feature].values
            for c in all_clusters
        ]
        if len(all_clusters) > 1:
            f_stat, p_value = stats.f_oneway(*cluster_values)
            importance_rows.append(
                {"feature": feature, "f_stat": f_stat, "p_value": p_value.round(6)}
            )

    importance_df = (
        pd.DataFrame(importance_rows)
        .sort_values("f_stat", ascending=False)
        .reset_index(drop=True)
    )

    # Log top-N features
    for _, row in importance_df.head(top_n).iterrows():
        sig = (
            "***" if row["p_value"] < 0.001
            else "**" if row["p_value"] < 0.01
            else "*" if row["p_value"] < 0.05
            else ""
        )
        logger.info(
            "  %-45s F=%10.2f  p=%12.6f %s",
            row["feature"], row["f_stat"], row["p_value"], sig,
        )

    # Per-cluster profiles for top features
    for _, row in importance_df.head(profile_n).iterrows():
        feat = row["feature"]
        display_name = feat.replace("_change", "")
        lines = [f"  {display_name}:"]
        for cluster in all_clusters:
            cdata = tier_data[tier_data["innovation_cluster"] == cluster]
            lines.append(
                f"    Cluster {cluster} (n={len(cdata)}): "
                f"mean={cdata[feat].mean():.4f}, "
                f"std={cdata[feat].std():.4f}"
            )
        if -1 in tier_data["innovation_cluster"].values:
            noise = tier_data[tier_data["innovation_cluster"] == -1]
            lines.append(
                f"    Noise (n={len(noise)}): "
                f"mean={noise[feat].mean():.4f}, "
                f"std={noise[feat].std():.4f}"
            )
        logger.info("\n".join(lines))

    # Cluster interpretation labels
    for cluster in all_clusters:
        cdata = tier_data[tier_data["innovation_cluster"] == cluster]
        profile_parts = []
        for _, frow in importance_df.head(3).iterrows():
            feat = frow["feature"]
            cluster_mean = cdata[feat].mean()
            overall_mean = tier_data[feat].mean()

            if cluster_mean > overall_mean * 1.2:
                direction = "HIGH INCREASE"
            elif cluster_mean < overall_mean * 0.8 and overall_mean < 0:
                direction = "HIGH DECREASE"
            elif abs(cluster_mean) < abs(overall_mean) * 0.5:
                direction = "STABLE"
            else:
                direction = "MODERATE"

            profile_parts.append(
                f"{feat.replace('_change', '')}: {direction}"
            )

        logger.info(
            "  Cluster %d (%d observations) — %s",
            cluster, len(cdata), " | ".join(profile_parts),
        )

    return importance_df