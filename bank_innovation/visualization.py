"""
Visualization
==============

Publication-ready plotting functions for UMAP clustering results.
All figures use a colorblind-friendly palette and are saved at 300 DPI
for academic presentations.
"""

import logging
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Colorblind-friendly palette (Wong 2011)
CLUSTER_COLORS: List[str] = [
    "#E69F00", "#56B4E9", "#009E73", "#F0E442",
    "#0072B2", "#D55E00", "#CC79A7", "#999999",
]
NOISE_COLOR: str = "#CCCCCC"

TIER_TITLES = {
    "Large": "Large Banks (>$10B assets)",
    "Medium": "Medium Banks ($1B–$10B assets)",
    "Small": "Small Banks (<$1B assets)",
}


def create_clustering_visualization(
    df_changes: pd.DataFrame,
    save_path: str = "viz/bank_clustering_visualization.png",
    figsize: tuple = (20, 6),
) -> plt.Figure:
    """Create a 3-panel UMAP scatter plot showing all tiers side by side.

    :param df_changes: Clustered DataFrame with ``bank_tier``,
        ``innovation_cluster``, ``umap_1``, and ``umap_2`` columns.
    :type df_changes: pandas.DataFrame
    :param save_path: File path for the saved figure.  Set to *None* to
        skip saving.
    :type save_path: str or None
    :param figsize: Figure dimensions ``(width, height)`` in inches.
    :type figsize: tuple of float
    :returns: The matplotlib Figure object.
    :rtype: matplotlib.figure.Figure
    """
    logger.info("Creating combined 3-panel clustering visualization")

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(
        "Bank Innovation Clustering by Tier (UMAP + HDBSCAN)",
        fontsize=16, fontweight="bold", y=0.98,
    )

    tiers = ["Large", "Medium", "Small"]

    for idx, tier in enumerate(tiers):
        ax = axes[idx]
        tier_data = df_changes[df_changes["bank_tier"] == tier].copy()

        clusters = sorted(
            c for c in tier_data["innovation_cluster"].unique() if c != -1
        )
        n_clusters = len(clusters)
        n_noise = (tier_data["innovation_cluster"] == -1).sum()
        n_total = len(tier_data)

        # Noise layer (background)
        noise_data = tier_data[tier_data["innovation_cluster"] == -1]
        if len(noise_data) > 0:
            ax.scatter(
                noise_data["umap_1"], noise_data["umap_2"],
                c=NOISE_COLOR, s=20, alpha=0.3,
                label=f"Noise (n={len(noise_data)})",
                edgecolors="none",
            )

        # Cluster layers
        for ci, cluster in enumerate(clusters):
            cdata = tier_data[tier_data["innovation_cluster"] == cluster]
            color = CLUSTER_COLORS[ci % len(CLUSTER_COLORS)]
            ax.scatter(
                cdata["umap_1"], cdata["umap_2"],
                c=color, s=40, alpha=0.7,
                label=f"Cluster {cluster} (n={len(cdata)})",
                edgecolors="white", linewidths=0.5,
            )

        ax.set_title(
            f"{TIER_TITLES.get(tier, tier)}\n"
            f"{n_clusters} clusters, {n_noise} noise "
            f"({n_noise / n_total * 100:.1f}%)",
            fontsize=12, fontweight="bold",
        )
        ax.set_xlabel("UMAP Dimension 1", fontsize=11)
        ax.set_ylabel("UMAP Dimension 2", fontsize=11)
        ax.legend(loc="best", fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.2, linestyle="--")
        ax.set_axisbelow(True)
        ax.set_facecolor("#FAFAFA")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        logger.info("Saved combined visualization to %s", save_path)

    return fig


def create_individual_tier_plots(
    df_changes: pd.DataFrame,
    output_dir: str = "./",
    figsize: tuple = (10, 8),
) -> None:
    """Create separate high-resolution plots for each tier.

    Useful for presentations where each tier occupies its own slide.

    :param df_changes: Clustered DataFrame with ``bank_tier``,
        ``innovation_cluster``, ``umap_1``, and ``umap_2`` columns.
    :type df_changes: pandas.DataFrame
    :param output_dir: Directory in which to save the PNG files.  File
        names follow the pattern ``clustering_<tier>_banks.png``.
    :type output_dir: str
    :param figsize: Figure dimensions ``(width, height)`` in inches.
    :type figsize: tuple of float
    :returns: *None*
    """
    logger.info("Creating individual tier plots in %s", output_dir)

    tiers = ["Large", "Medium", "Small"]

    for tier in tiers:
        fig, ax = plt.subplots(figsize=figsize)
        tier_data = df_changes[df_changes["bank_tier"] == tier].copy()

        clusters = sorted(
            c for c in tier_data["innovation_cluster"].unique() if c != -1
        )
        n_noise = (tier_data["innovation_cluster"] == -1).sum()
        n_total = len(tier_data)

        # Noise
        noise_data = tier_data[tier_data["innovation_cluster"] == -1]
        if len(noise_data) > 0:
            ax.scatter(
                noise_data["umap_1"], noise_data["umap_2"],
                c=NOISE_COLOR, s=30, alpha=0.3,
                label=f"Noise (n={len(noise_data)})",
                edgecolors="none",
            )

        # Clusters
        for ci, cluster in enumerate(clusters):
            cdata = tier_data[tier_data["innovation_cluster"] == cluster]
            color = CLUSTER_COLORS[ci % len(CLUSTER_COLORS)]
            ax.scatter(
                cdata["umap_1"], cdata["umap_2"],
                c=color, s=50, alpha=0.7,
                label=f"Cluster {cluster} (n={len(cdata)})",
                edgecolors="white", linewidths=0.8,
            )

        ax.set_title(
            f"{TIER_TITLES.get(tier, tier)} — Innovation Clusters\n"
            f"{len(clusters)} clusters identified, "
            f"{n_noise} noise points ({n_noise / n_total * 100:.1f}%)",
            fontsize=14, fontweight="bold", pad=20,
        )
        ax.set_xlabel("UMAP Dimension 1", fontsize=12)
        ax.set_ylabel("UMAP Dimension 2", fontsize=12)
        ax.legend(loc="best", fontsize=10, framealpha=0.95, shadow=True)
        ax.grid(True, alpha=0.2, linestyle="--")
        ax.set_axisbelow(True)
        ax.set_facecolor("#FAFAFA")

        filename = f"{output_dir}clustering_{tier.lower()}_banks.png"
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches="tight", facecolor="white")
        logger.info("Saved %s banks plot: %s", tier, filename)
        plt.close()

    logger.info("All individual tier plots saved")


def print_clustering_summary(df_changes: pd.DataFrame) -> None:
    """Log a text summary of clustering results for every tier.

    :param df_changes: Clustered DataFrame with ``bank_tier`` and
        ``innovation_cluster`` columns.
    :type df_changes: pandas.DataFrame
    :returns: *None*
    """
    logger.info("=" * 60)
    logger.info("CLUSTERING RESULTS SUMMARY")
    logger.info("=" * 60)

    for tier in ["Large", "Medium", "Small"]:
        tier_data = df_changes[df_changes["bank_tier"] == tier]
        clusters = [
            c for c in tier_data["innovation_cluster"].unique() if c != -1
        ]
        n_clusters = len(clusters)
        n_noise = (tier_data["innovation_cluster"] == -1).sum()
        n_total = len(tier_data)
        n_clustered = n_total - n_noise
        n_banks = tier_data["rssd9017"].nunique() if "rssd9017" in tier_data.columns else 0

        logger.info(
            "%s Banks — observations: %d (%d unique banks), clusters: %d, "
            "clustered: %d (%.1f%%), noise: %d (%.1f%%)",
            tier, n_total, n_banks, n_clusters,
            n_clustered, n_clustered / n_total * 100,
            n_noise, n_noise / n_total * 100,
        )

        for cluster in sorted(clusters):
            count = (tier_data["innovation_cluster"] == cluster).sum()
            logger.info(
                "    Cluster %d: %d banks (%.1f%%)",
                cluster, count, count / n_total * 100,
            )


def create_year_pair_plots(
    df_changes: pd.DataFrame,
    output_dir: str = "viz/",
    figsize: tuple = (20, 6),
    year_pairs: Optional[List[tuple]] = None,
) -> List[str]:
    """Create a 3-panel UMAP scatter plot for each year pair, saved as individual PNGs.

    Each figure shows Large, Medium, and Small banks side by side for a
    single year-over-year transition (e.g. 2014→2015).  Cluster colours
    are consistent across all year pairs so that the same cluster ID always
    appears in the same colour.

    :param df_changes: Clustered YoY DataFrame with columns ``bank_tier``,
        ``innovation_cluster``, ``umap_1``, ``umap_2``, ``year_from``,
        and ``year_to``.
    :type df_changes: pandas.DataFrame
    :param output_dir: Directory in which to save the PNG files.  Files
        are named ``clustering_<year_from>_<year_to>.png``.
    :type output_dir: str
    :param figsize: Figure dimensions ``(width, height)`` in inches.
    :type figsize: tuple of float
    :param year_pairs: Specific year pairs to plot as a list of
        ``(year_from, year_to)`` tuples.  If *None*, all year pairs
        present in the data are plotted.
    :type year_pairs: list of tuple(int, int), optional
    :returns: List of file paths for the saved figures.
    :rtype: list of str
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    # Determine which year pairs to plot
    if year_pairs is None:
        all_pairs = (
            df_changes[["year_from", "year_to"]]
            .drop_duplicates()
            .sort_values(["year_from", "year_to"])
        )
        # Only include consecutive year pairs (year_to - year_from == 1)
        consecutive_mask = (all_pairs["year_to"] - all_pairs["year_from"]) == 1
        pairs = all_pairs[consecutive_mask].values.tolist()

        n_skipped = (~consecutive_mask).sum()
        if n_skipped > 0:
            logger.info(
                "Skipped %d non-consecutive year pairs", n_skipped,
            )
    else:
        pairs = list(year_pairs)

    logger.info(
        "Creating per-year-pair visualizations for %d year pairs", len(pairs),
    )

    # Determine global cluster set for consistent colours
    all_clusters = sorted(
        c for c in df_changes["innovation_cluster"].unique() if c != -1
    )
    cluster_color_map = {
        c: CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
        for i, c in enumerate(all_clusters)
    }

    tiers = ["Large", "Medium", "Small"]
    saved_paths = []

    for year_from, year_to in pairs:
        year_from = int(year_from)
        year_to = int(year_to)

        pair_data = df_changes[
            (df_changes["year_from"] == year_from)
            & (df_changes["year_to"] == year_to)
        ]

        if len(pair_data) == 0:
            logger.warning(
                "No data for year pair %d→%d, skipping", year_from, year_to,
            )
            continue

        fig, axes = plt.subplots(1, 3, figsize=figsize)
        fig.suptitle(
            f"Bank Innovation Clustering — {year_from}→{year_to} Transition",
            fontsize=16, fontweight="bold", y=0.98,
        )

        for idx, tier in enumerate(tiers):
            ax = axes[idx]
            tier_data = pair_data[pair_data["bank_tier"] == tier]

            clusters = sorted(
                c for c in tier_data["innovation_cluster"].unique() if c != -1
            )
            n_noise = (tier_data["innovation_cluster"] == -1).sum()
            n_total = len(tier_data)

            if n_total == 0:
                ax.set_title(
                    f"{TIER_TITLES.get(tier, tier)}\nNo data",
                    fontsize=12, fontweight="bold",
                )
                ax.set_facecolor("#FAFAFA")
                continue

            # Noise layer
            noise_data = tier_data[tier_data["innovation_cluster"] == -1]
            if len(noise_data) > 0:
                ax.scatter(
                    noise_data["umap_1"], noise_data["umap_2"],
                    c=NOISE_COLOR, s=20, alpha=0.3,
                    label=f"Noise (n={len(noise_data)})",
                    edgecolors="none",
                )

            # Cluster layers — use global colour map
            for cluster in clusters:
                cdata = tier_data[tier_data["innovation_cluster"] == cluster]
                color = cluster_color_map.get(
                    cluster, CLUSTER_COLORS[cluster % len(CLUSTER_COLORS)]
                )
                ax.scatter(
                    cdata["umap_1"], cdata["umap_2"],
                    c=color, s=40, alpha=0.7,
                    label=f"Cluster {cluster} (n={len(cdata)})",
                    edgecolors="white", linewidths=0.5,
                )

            noise_pct = n_noise / n_total * 100 if n_total > 0 else 0
            ax.set_title(
                f"{TIER_TITLES.get(tier, tier)}\n"
                f"{len(clusters)} clusters, {n_noise} noise "
                f"({noise_pct:.1f}%)",
                fontsize=12, fontweight="bold",
            )
            ax.set_xlabel("UMAP Dimension 1", fontsize=11)
            ax.set_ylabel("UMAP Dimension 2", fontsize=11)
            ax.legend(loc="best", fontsize=9, framealpha=0.9)
            ax.grid(True, alpha=0.2, linestyle="--")
            ax.set_axisbelow(True)
            ax.set_facecolor("#FAFAFA")

        plt.tight_layout()

        filename = os.path.join(
            output_dir, f"clustering_{year_from}_{year_to}.png"
        )
        plt.savefig(filename, dpi=300, bbox_inches="tight", facecolor="white")
        saved_paths.append(filename)
        logger.info("Saved year pair %d→%d: %s", year_from, year_to, filename)
        plt.close()

    logger.info(
        "All per-year-pair plots complete — %d figures saved to %s",
        len(saved_paths), output_dir,
    )

    return saved_paths