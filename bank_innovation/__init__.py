"""
Bank Innovation Clustering Package
====================================

An unsupervised learning pipeline for identifying distinct innovation patterns
across U.S. commercial banks using FDIC Call Report data (2010–2021).

The pipeline applies UMAP dimensionality reduction paired with HDBSCAN
clustering to year-over-year change scores of 20 size-independent financial
ratios, pooled across all years and segmented by bank asset-size tier.
Stable cluster definitions enable year-by-year tracking of individual bank
cluster membership and migration analysis.

Modules
-------
data_prep
    Data loading, RCFD/RCON column merging, feature selection, sticky tier
    assignment, and year-over-year change score calculation.
ratios
    Calculation of 20 financial ratios serving as innovation and efficiency
    proxies.
clustering
    UMAP + HDBSCAN clustering pipeline on pooled YoY data, parameter
    tuning, optimized re-clustering, and cluster naming.
analysis
    Statistical analysis of clustering results including ANOVA-based feature
    importance, cluster profiling, and asset-size independence checks.
visualization
    Publication-ready UMAP scatter plots (combined 3-panel and individual
    tier figures) and text summaries of clustering results.
migration
    Cluster migration tracking: bank-level cluster histories, transition
    matrices, migration rate time series, and migration driver profiling.
"""

__version__ = "2.0.0"
__author__ = "Justin Dorval"

from bank_innovation.data_prep import (
    merge_split_columns,
    prepare_clustering_features,
    assign_sticky_bank_tiers,
    calculate_innovation_change_scores,
)
from bank_innovation.ratios import (
    calculate_ratios,
    calculate_additional_innovation_ratios,
)
from bank_innovation.clustering import (
    cluster_by_tier,
    comprehensive_tuning,
    cluster_with_best_params,
    assign_cluster_names,
)
from bank_innovation.analysis import (
    analyze_cluster_by_size_changes,
    analyze_innovation_change_clusters,
)
from bank_innovation.visualization import (
    create_clustering_visualization,
    create_individual_tier_plots,
    print_clustering_summary,
    create_year_pair_plots, 
)
from bank_innovation.migration import (
    build_bank_cluster_history,
    detect_migrations,
    compute_migration_matrix,
    compute_migration_rates_by_year,
    profile_migration_drivers,
    summarize_bank_trajectories,
)

from bank_innovation.driver_analysis import (
    run_driver_analysis,           
    build_yearly_driver_table,     
    aggregate_driver_frequency,
    plot_yearly_driver_bars,
    plot_driver_timeseries,
    rank_stability,
    print_rank_stability,
)