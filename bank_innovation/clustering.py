"""
Clustering Pipeline
====================

UMAP dimensionality reduction paired with HDBSCAN density-based clustering,
applied independently to each bank-size tier.  Operates on a pooled panel
of year-over-year change scores (multiple rows per bank) to define stable
cluster boundaries that can be used for year-by-year assignment and
migration tracking.

Includes a comprehensive parameter-tuning grid search and a function for
applying the best-found configuration.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import umap
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import QuantileTransformer, StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default UMAP / HDBSCAN parameters per tier
# Calibrated to pooled YoY observation counts:
#   Large ~1,200 obs, Medium ~6,000 obs, Small ~50,000 obs
# ---------------------------------------------------------------------------
DEFAULT_PARAMS: Dict[str, Dict[str, Any]] = {
    "Large": {
        "umap_n_neighbors": 15,
        "umap_min_dist": 0.1,
        "hdb_min_cluster_size": 15,
        "hdb_min_samples": 3,
        "hdb_method": "eom",
    },
    "Medium": {
        "umap_n_neighbors": 15,
        "umap_min_dist": 0.1,
        "hdb_min_cluster_size": 30,
        "hdb_min_samples": 5,
        "hdb_method": "eom",
    },
    "Small": {
        "umap_n_neighbors": 20,
        "umap_min_dist": 0.1,
        "hdb_min_cluster_size": 50,
        "hdb_min_samples": 10,
        "hdb_method": "eom",
    },
}


def cluster_by_tier(
    df_changes: pd.DataFrame,
    change_feature_cols: List[str],
    params: Optional[Dict[str, Dict[str, Any]]] = None,
    scaler: Optional[Any] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Run UMAP + HDBSCAN clustering on pooled YoY observations per tier.

    For each tier present in ``df_changes['bank_tier']`` the function
    scales the change-score features, reduces to two dimensions with
    UMAP, and clusters with HDBSCAN.  Because the input contains multiple
    rows per bank (one per year pair), the resulting clusters represent
    *types of annual innovation behaviour* rather than *types of bank*.
    A single bank may belong to different clusters in different years.

    :param df_changes: YoY change-score DataFrame produced by
        :func:`data_prep.calculate_innovation_change_scores`.  Must
        contain ``bank_tier``, ``year_from``, ``year_to``, ``rssd9017``,
        and all columns in *change_feature_cols*.
    :type df_changes: pandas.DataFrame
    :param change_feature_cols: Column names of the change-score features
        (typically ending in ``_change``).
    :type change_feature_cols: list of str
    :param params: Per-tier parameter dictionaries.  Each dictionary must
        contain keys ``umap_n_neighbors``, ``umap_min_dist``,
        ``hdb_min_cluster_size``, ``hdb_min_samples``, and ``hdb_method``.
        Defaults to :data:`DEFAULT_PARAMS` when *None*.
    :type params: dict, optional
    :param scaler: Scikit-learn scaler instance.  Defaults to
        ``QuantileTransformer(output_distribution='uniform')`` when *None*,
        which handles the heavy-tailed YoY change-score distributions
        better than StandardScaler.
    :type scaler: sklearn transformer, optional
    :param random_state: Random seed for UMAP reproducibility.
    :type random_state: int
    :returns: *df_changes* augmented with ``innovation_cluster``,
        ``umap_1``, and ``umap_2``.
    :rtype: pandas.DataFrame
    """
    if params is None:
        params = DEFAULT_PARAMS

    df_changes = df_changes.copy()
    df_changes["innovation_cluster"] = -1
    df_changes["umap_1"] = np.nan
    df_changes["umap_2"] = np.nan

    by_tier = df_changes.groupby("bank_tier")

    embedding_store: Dict[str, np.ndarray] = {}
    cluster_store: Dict[str, np.ndarray] = {}

    for tier, tier_data in by_tier:
        n_banks = tier_data["rssd9017"].nunique()
        logger.info(
            "Processing %s: %d YoY observations (%d unique banks)",
            tier, len(tier_data), n_banks,
        )

        p = params.get(tier, params.get("Small"))

        if scaler is not None:
            tier_scaler = scaler
        else:
            tier_scaler = QuantileTransformer(
                n_quantiles=min(len(tier_data), 1000),
                output_distribution="uniform",
                random_state=random_state,
            )
        scaled = tier_scaler.fit_transform(tier_data[change_feature_cols])

        reducer = umap.UMAP(
            n_neighbors=p["umap_n_neighbors"],
            min_dist=p["umap_min_dist"],
            metric="euclidean",
            random_state=random_state,
        )
        emb = reducer.fit_transform(scaled)
        embedding_store[tier] = emb

        clusterer = HDBSCAN(
            min_cluster_size=p["hdb_min_cluster_size"],
            min_samples=p["hdb_min_samples"],
            cluster_selection_method=p["hdb_method"],
            copy=True,
        )
        labels = clusterer.fit_predict(emb)
        cluster_store[tier] = labels

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()
        logger.info(
            "  %s — %d clusters, %d noise (%.1f%%)",
            tier, n_clusters, n_noise, n_noise / len(tier_data) * 100,
        )

    # Write results back
    for tier, tier_data in by_tier:
        df_changes.loc[tier_data.index, "innovation_cluster"] = cluster_store[tier]
        df_changes.loc[tier_data.index, "umap_1"] = embedding_store[tier][:, 0]
        df_changes.loc[tier_data.index, "umap_2"] = embedding_store[tier][:, 1]

    logger.info("Clustering complete for all tiers")
    return df_changes


# ---------------------------------------------------------------------------
# Parameter-tuning grid search
# ---------------------------------------------------------------------------

def _build_search_grid(
    tier_name: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return UMAP and HDBSCAN parameter grids appropriate for *tier_name*.

    Grids are calibrated to actual pooled YoY observation counts per tier
    (Large ~1,200, Medium ~6,000, Small ~50,000).  Includes both ``eom``
    and ``leaf`` selection methods to find the best cluster granularity.

    :param tier_name: One of ``'Large'``, ``'Medium'``, ``'Small'``.
    :type tier_name: str
    :returns: A tuple of (umap_configs, hdbscan_configs).
    :rtype: tuple(list of dict, list of dict)
    """
    if tier_name == "Large":
        # ~1,200 observations — need smaller cluster sizes
        umap_configs = [
            {"n_neighbors": 10, "min_dist": 0.0},
            {"n_neighbors": 15, "min_dist": 0.0},
            {"n_neighbors": 15, "min_dist": 0.1},
            {"n_neighbors": 20, "min_dist": 0.1},
        ]
        hdbscan_configs = [
            {"min_cluster_size": 8, "min_samples": 2, "method": "eom"},
            {"min_cluster_size": 10, "min_samples": 2, "method": "eom"},
            {"min_cluster_size": 12, "min_samples": 3, "method": "eom"},
            {"min_cluster_size": 15, "min_samples": 3, "method": "eom"},
            {"min_cluster_size": 20, "min_samples": 5, "method": "eom"},
            {"min_cluster_size": 8, "min_samples": 2, "method": "leaf"},
            {"min_cluster_size": 10, "min_samples": 2, "method": "leaf"},
            {"min_cluster_size": 12, "min_samples": 3, "method": "leaf"},
            {"min_cluster_size": 15, "min_samples": 3, "method": "leaf"},
            {"min_cluster_size": 20, "min_samples": 5, "method": "leaf"},
        ]
    elif tier_name == "Medium":
        # ~6,000 observations
        umap_configs = [
            {"n_neighbors": 10, "min_dist": 0.0},
            {"n_neighbors": 15, "min_dist": 0.0},
            {"n_neighbors": 15, "min_dist": 0.1},
            {"n_neighbors": 20, "min_dist": 0.1},
            {"n_neighbors": 30, "min_dist": 0.1},
        ]
        hdbscan_configs = [
            {"min_cluster_size": 15, "min_samples": 3, "method": "eom"},
            {"min_cluster_size": 20, "min_samples": 3, "method": "eom"},
            {"min_cluster_size": 25, "min_samples": 5, "method": "eom"},
            {"min_cluster_size": 30, "min_samples": 5, "method": "eom"},
            {"min_cluster_size": 40, "min_samples": 5, "method": "eom"},
            {"min_cluster_size": 15, "min_samples": 3, "method": "leaf"},
            {"min_cluster_size": 20, "min_samples": 3, "method": "leaf"},
            {"min_cluster_size": 25, "min_samples": 5, "method": "leaf"},
            {"min_cluster_size": 30, "min_samples": 5, "method": "leaf"},
            {"min_cluster_size": 40, "min_samples": 7, "method": "leaf"},
        ]
    else:  # Small
        # ~50,000 observations
        umap_configs = [
            {"n_neighbors": 10, "min_dist": 0.0},
            {"n_neighbors": 15, "min_dist": 0.0},
            {"n_neighbors": 15, "min_dist": 0.1},
            {"n_neighbors": 20, "min_dist": 0.1},
            {"n_neighbors": 30, "min_dist": 0.1},
        ]
        hdbscan_configs = [
            {"min_cluster_size": 30, "min_samples": 5, "method": "eom"},
            {"min_cluster_size": 40, "min_samples": 5, "method": "eom"},
            {"min_cluster_size": 50, "min_samples": 10, "method": "eom"},
            {"min_cluster_size": 70, "min_samples": 10, "method": "eom"},
            {"min_cluster_size": 100, "min_samples": 15, "method": "eom"},
            {"min_cluster_size": 30, "min_samples": 5, "method": "leaf"},
            {"min_cluster_size": 40, "min_samples": 5, "method": "leaf"},
            {"min_cluster_size": 50, "min_samples": 10, "method": "leaf"},
            {"min_cluster_size": 70, "min_samples": 15, "method": "leaf"},
            {"min_cluster_size": 100, "min_samples": 15, "method": "leaf"},
        ]

    return umap_configs, hdbscan_configs


def comprehensive_tuning(
    data: pd.DataFrame,
    features: List[str],
    tier_name: str,
    scaler: Optional[Any] = None,
    random_state: int = 42,
    target_clusters: int = 4,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Grid-search over UMAP and HDBSCAN hyper-parameters for one tier.

    For each UMAP configuration the embedding is computed once, then every
    HDBSCAN configuration is evaluated against that embedding.  A combined
    score penalises noise percentage, deviation from *target_clusters*, and
    imbalanced cluster sizes.

    :param data: Tier-filtered subset of the YoY change-score DataFrame.
    :type data: pandas.DataFrame
    :param features: Change-score column names to cluster on.
    :type features: list of str
    :param tier_name: Human-readable tier label (``'Large'``, ``'Medium'``,
        or ``'Small'``).
    :type tier_name: str
    :param scaler: Scikit-learn scaler instance.  Defaults to
        ``QuantileTransformer(output_distribution='uniform')`` when *None*.
    :type scaler: sklearn transformer, optional
    :param random_state: Random seed for UMAP.
    :type random_state: int
    :param target_clusters: Ideal number of clusters for scoring.
    :type target_clusters: int
    :returns: A tuple of:

        * **best** — Dictionary of the best-scoring configuration including
          keys ``umap_n_neighbors``, ``umap_min_dist``,
          ``hdb_min_cluster_size``, ``hdb_min_samples``, ``hdb_method``,
          ``n_clusters``, ``noise_pct``, ``embedding``, ``labels``, and
          ``clusterer``.
        * **results_df** — DataFrame of all tested configurations sorted
          by ``combined_score`` ascending.
    :rtype: tuple(dict, pandas.DataFrame)
    """
    logger.info(
        "Comprehensive tuning for %s (%d YoY observations)",
        tier_name, len(data),
    )

    if scaler is not None:
        scaled_data = scaler.fit_transform(data[features])
    else:
        qt = QuantileTransformer(
            n_quantiles=min(len(data), 1000),
            output_distribution="uniform",
            random_state=random_state,
        )
        scaled_data = qt.fit_transform(data[features])
    umap_configs, hdbscan_configs = _build_search_grid(tier_name)

    logger.info(
        "Testing %d UMAP x %d HDBSCAN = %d configurations",
        len(umap_configs), len(hdbscan_configs),
        len(umap_configs) * len(hdbscan_configs),
    )

    results: List[Dict[str, Any]] = []
    best_overall: Optional[Dict[str, Any]] = None
    best_score = float("inf")

    for i, umap_params in enumerate(umap_configs):
        logger.info(
            "UMAP config %d/%d: n_neighbors=%d, min_dist=%.2f",
            i + 1, len(umap_configs),
            umap_params["n_neighbors"], umap_params["min_dist"],
        )

        reducer = umap.UMAP(
            n_neighbors=umap_params["n_neighbors"],
            min_dist=umap_params["min_dist"],
            metric="euclidean",
            random_state=random_state,
        )
        embedding = reducer.fit_transform(scaled_data)

        for hdb_params in hdbscan_configs:
            clusterer = HDBSCAN(
                min_cluster_size=hdb_params["min_cluster_size"],
                min_samples=hdb_params["min_samples"],
                cluster_selection_method=hdb_params["method"],
                copy=True,
            )
            labels = clusterer.fit_predict(embedding)

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = (labels == -1).sum()
            noise_pct = n_noise / len(data) * 100

            cluster_sizes = [
                int((labels == c).sum()) for c in set(labels) if c != -1
            ]
            min_sz = min(cluster_sizes) if cluster_sizes else 0
            max_sz = max(cluster_sizes) if cluster_sizes else 0
            avg_sz = float(np.mean(cluster_sizes)) if cluster_sizes else 0.0

            cluster_score = abs(n_clusters - target_clusters) * 10
            # Graduated balance penalty — harsher as the dominant cluster grows
            if max_sz > len(data) * 0.9:
                balance_penalty = 30
            elif max_sz > len(data) * 0.7:
                balance_penalty = 20
            elif max_sz > len(data) * 0.5:
                balance_penalty = 10
            else:
                balance_penalty = 0
            combined_score = noise_pct + cluster_score + balance_penalty

            result: Dict[str, Any] = {
                "umap_n_neighbors": umap_params["n_neighbors"],
                "umap_min_dist": umap_params["min_dist"],
                "hdb_min_cluster_size": hdb_params["min_cluster_size"],
                "hdb_min_samples": hdb_params["min_samples"],
                "hdb_method": hdb_params["method"],
                "n_clusters": n_clusters,
                "noise_count": n_noise,
                "noise_pct": noise_pct,
                "min_cluster_size": min_sz,
                "max_cluster_size": max_sz,
                "avg_cluster_size": avg_sz,
                "combined_score": combined_score,
                "embedding": embedding,
                "labels": labels,
                "clusterer": clusterer,
            }
            results.append(result)

            if combined_score < best_score:
                best_score = combined_score
                best_overall = result

        logger.info(
            "  Best so far: %d clusters, %.1f%% noise",
            best_overall["n_clusters"], best_overall["noise_pct"],
        )

    results_df = pd.DataFrame(results).sort_values("combined_score")

    best = results_df.iloc[0].to_dict()
    logger.info(
        "Recommended — UMAP(n=%d, d=%.2f), HDBSCAN(cs=%d, ms=%d, %s) "
        "-> %d clusters, %.1f%% noise",
        best["umap_n_neighbors"], best["umap_min_dist"],
        best["hdb_min_cluster_size"], best["hdb_min_samples"],
        best["hdb_method"], best["n_clusters"], best["noise_pct"],
    )

    return best, results_df


def cluster_with_best_params(
    df: pd.DataFrame,
    tier_name: str,
    change_feature_cols: List[str],
    tuning_results: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    """Apply the best tuning configuration to a single tier.

    Re-uses the embedding and labels already computed during
    :func:`comprehensive_tuning`, so no additional UMAP/HDBSCAN fitting
    is performed.

    :param df: Full YoY change-score DataFrame (all tiers).
    :type df: pandas.DataFrame
    :param tier_name: Tier to cluster (``'Large'``, ``'Medium'``, or
        ``'Small'``).
    :type tier_name: str
    :param change_feature_cols: Change-score column names (used only for
        logging; the actual features come from the stored tuning result).
    :type change_feature_cols: list of str
    :param tuning_results: Dictionary keyed by tier name whose values are
        dicts with a ``'best_config'`` entry as returned by
        :func:`comprehensive_tuning`.
    :type tuning_results: dict
    :returns: Tier-filtered DataFrame with ``innovation_cluster``,
        ``umap_1``, and ``umap_2`` populated from the best configuration.
    :rtype: pandas.DataFrame

    :raises KeyError: If *tier_name* is not present in *tuning_results*.
    """
    tier_data = df[df["bank_tier"] == tier_name].copy()

    if tier_name not in tuning_results:
        logger.error("No tuning results for %s", tier_name)
        raise KeyError(f"No tuning results for tier '{tier_name}'")

    best = tuning_results[tier_name]["best_config"]

    n_banks = tier_data["rssd9017"].nunique()
    logger.info(
        "Clustering %s with optimized params — "
        "%d observations (%d banks), "
        "UMAP(n=%s, d=%s), HDBSCAN(cs=%s, ms=%s, %s)",
        tier_name, len(tier_data), n_banks,
        best["umap_n_neighbors"], best["umap_min_dist"],
        best["hdb_min_cluster_size"], best["hdb_min_samples"],
        best["hdb_method"],
    )

    tier_data.loc[:, "innovation_cluster"] = best["labels"]
    tier_data.loc[:, "umap_1"] = best["embedding"][:, 0]
    tier_data.loc[:, "umap_2"] = best["embedding"][:, 1]

    n_clusters = len(set(best["labels"])) - (1 if -1 in best["labels"] else 0)
    n_noise = int((best["labels"] == -1).sum())
    logger.info(
        "  %s — %d clusters, %d noise (%.1f%%)",
        tier_name, n_clusters, n_noise, best["noise_pct"],
    )

    return tier_data


def assign_cluster_names(
    df: pd.DataFrame,
    cluster_names: Dict[str, Dict[int, str]],
) -> Tuple[pd.DataFrame, Dict[str, Dict[int, str]]]:
    """Map numeric cluster IDs to human-readable innovation-pattern names.

    Unlike the original first-to-last approach, there are no default names
    because cluster meanings will differ after pooled YoY clustering.  The
    caller must supply the mapping after inspecting cluster profiles.

    :param df: Clustered DataFrame with ``bank_tier`` and
        ``innovation_cluster`` columns.
    :type df: pandas.DataFrame
    :param cluster_names: Nested mapping ``{tier: {cluster_id: name}}``.
    :type cluster_names: dict
    :returns: A tuple of:

        * **df** — Copy of the input with ``cluster_name`` and
          ``cluster_full_label`` columns added.
        * **cluster_names** — The mapping that was applied.
    :rtype: tuple(pandas.DataFrame, dict)
    """
    df = df.copy()

    df["cluster_name"] = df.apply(
        lambda row: cluster_names.get(row["bank_tier"], {}).get(
            row["innovation_cluster"], "Unknown"
        ),
        axis=1,
    )
    df["cluster_full_label"] = df.apply(
        lambda row: f"{row['bank_tier']} - {row['cluster_name']}",
        axis=1,
    )

    logger.info("Assigned cluster names across %d tiers", len(cluster_names))
    return df, cluster_names