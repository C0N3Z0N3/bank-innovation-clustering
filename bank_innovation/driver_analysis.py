"""
bank_innovation/driver_analysis.py
====================================
Migration-driver time-series analysis for the bank innovation project.

Builds on :mod:`bank_innovation.migration` to answer the question:
*Which ratio changes most consistently drive banks to switch innovation
clusters, and how has that changed year over year?*

Public API
----------
build_yearly_driver_table(df, df_migrations, top_n_drivers, tiers)
    Long-form panel: one row per (year, tier, transition, rank, feature).

aggregate_driver_frequency(driver_long, tier, score_col)
    Collapse to a year × feature % share matrix — ready for modelling.

plot_yearly_driver_bars(pivot_pct, tier, chart_type, ax, figsize)
    Stacked or grouped bar chart of driver composition per year.

plot_driver_timeseries(pivot_pct, tier, top_n, rolling_window, figsize)
    Per-feature line charts with rolling mean + optional STL trend.

print_rank_stability(pivot_pct, tier, top_n)
    Console table: mean rank, rank volatility, peak year per feature.

run_driver_analysis(df, df_migrations, tiers, top_n_drivers,
                    rolling_window, save_dir)
    One-shot entry point that runs everything and optionally saves outputs.
"""

from __future__ import annotations

import logging
import textwrap
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

# statsmodels is optional — STL trend lines are skipped gracefully if absent
try:
    from statsmodels.tsa.seasonal import STL
    _HAS_STL = True
except ImportError:
    _HAS_STL = False

logger = logging.getLogger(__name__)

# ── internal helpers ──────────────────────────────────────────────────────────

_COLORS = plt.cm.tab20.colors   # 20 distinct colours, reused cyclically


def _color(i: int):
    return _COLORS[i % len(_COLORS)]


def _short(feat: str, wrap: int = 14) -> str:
    """Strip *_change* suffix and line-wrap for axis labels."""
    s = feat.replace("_change", "").replace("_", " ")
    return "\n".join(textwrap.wrap(s, wrap))


# ── 1. Build the raw driver panel ────────────────────────────────────────────

def build_yearly_driver_table(
    df: pd.DataFrame,
    df_migrations: pd.DataFrame,
    top_n_drivers: int = 5,
    tiers: Optional[List[Optional[str]]] = None,
) -> pd.DataFrame:
    """Build a long-form panel of migration drivers by year.

    For every year and every observed cluster transition, calls
    :func:`bank_innovation.migration.profile_migration_drivers` and
    records the top-*N* features together with their rank and event-count
    weight.

    :param df: Full clustered YoY DataFrame (used only to satisfy the
        migration function signature; pass ``df_changes_optimized``).
    :type df: pandas.DataFrame
    :param df_migrations: Migration events as returned by
        :func:`bank_innovation.migration.detect_migrations`.
    :type df_migrations: pandas.DataFrame
    :param top_n_drivers: How many top features to capture per transition.
    :type top_n_drivers: int
    :param tiers: List of tier labels to slice by.  ``None`` in the list
        means "all tiers combined".  Defaults to
        ``[None, "Large", "Medium", "Small"]``.
    :type tiers: list, optional
    :returns: Long-form DataFrame with columns:

        * ``year`` — year_to value
        * ``tier`` — "All", "Large", "Medium", or "Small"
        * ``cluster_from``, ``cluster_to`` — transition endpoints
        * ``n_events`` — how many banks made this transition that year
        * ``rank`` — 1 = most distinctive feature for this transition
        * ``feature`` — ratio change column name
        * ``mean_diff`` — migration mean minus other-migrations mean
        * ``rank_weight`` — *top_n* − rank + 1  (top feature = highest)
        * ``weighted_score`` — rank_weight × n_events
    :rtype: pandas.DataFrame
    """
    # Import here to avoid circular imports inside the package
    from bank_innovation.migration import profile_migration_drivers

    if tiers is None:
        tiers = ["Large", "Medium", "Small"]

    years = sorted(df_migrations["year_to"].unique())
    records = []

    for tier in tiers:
        tier_label = tier 
        data = (
            df_migrations
            if tier is None
            else df_migrations[df_migrations["bank_tier"] == tier]
        )

        for year in years:
            year_mig = data[data["year_to"] == year]
            if year_mig.empty:
                continue

            transitions = (
                year_mig.groupby(["cluster_from", "cluster_to"])
                .size()
                .reset_index(name="n_events")
            )

            for _, row in transitions.iterrows():
                cf  = int(row["cluster_from"])
                ct  = int(row["cluster_to"])
                n_e = int(row["n_events"])

                # Use the full migrations df for stable mean estimates
                drivers = profile_migration_drivers(
                    df_migrations,
                    cluster_from=cf,
                    cluster_to=ct,
                    tier=tier,
                    top_n=top_n_drivers,
                )
                if drivers.empty:
                    continue

                for rank_idx, driver_row in drivers.iterrows():
                    weight = top_n_drivers - rank_idx   # top feature = highest
                    records.append({
                        "year":           year,
                        "tier":           tier_label,
                        "cluster_from":   cf,
                        "cluster_to":     ct,
                        "n_events":       n_e,
                        "rank":           rank_idx + 1,
                        "feature":        driver_row["feature"],
                        "mean_diff":      driver_row["difference"],
                        "rank_weight":    weight,
                        "weighted_score": weight * n_e,
                    })

    result = pd.DataFrame(records)
    logger.info(
        "Driver table: %d rows | %d years | %d tier slices",
        len(result),
        result["year"].nunique() if not result.empty else 0,
        result["tier"].nunique() if not result.empty else 0,
    )
    return result


# ── 2. Aggregate to year × feature matrix ────────────────────────────────────

def aggregate_driver_frequency(
    driver_long: pd.DataFrame,
    tier: str = "Large",
    score_col: str = "weighted_score",
) -> pd.DataFrame:
    """Collapse the long driver panel into a year × feature % share matrix.

    Each row sums to 100 so years with very different migration volumes
    are directly comparable.

    :param driver_long: Output of :func:`build_yearly_driver_table`.
    :type driver_long: pandas.DataFrame
    :param tier: Which tier slice to use.  Must match a value in the
        ``tier`` column (``"Large"``).
    :type tier: str
    :param score_col: Column to aggregate.  ``"weighted_score"``
        (default) weights by rank × event count.  ``"rank_weight"``
        ignores event volume.
    :type score_col: str
    :returns: DataFrame with ``year`` as index and one column per
        feature, values are row-normalised percentages.
    :rtype: pandas.DataFrame
    """
    sub = driver_long[driver_long["tier"] == tier]
    if sub.empty:
        logger.warning("No data for tier '%s'", tier)
        return pd.DataFrame()

    agg = (
        sub.groupby(["year", "feature"])[score_col]
        .sum()
        .reset_index()
    )
    pivot = agg.pivot(index="year", columns="feature", values=score_col).fillna(0)
    pivot.columns.name = None

    # Row-normalise → percentage
    row_sums = pivot.sum(axis=1)
    pivot_pct = pivot.div(row_sums, axis=0) * 100

    logger.info(
        "Driver frequency matrix [%s]: %d years × %d features",
        tier, len(pivot_pct), len(pivot_pct.columns),
    )
    return pivot_pct


# ── 3. Bar charts ─────────────────────────────────────────────────────────────

def plot_yearly_driver_bars(
    pivot_pct: pd.DataFrame,
    tier: str = "Large",
    chart_type: str = "stacked",
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (18, 7),
) -> plt.Figure:
    """Bar chart of migration driver composition by year.

    :param pivot_pct: Output of :func:`aggregate_driver_frequency`.
    :type pivot_pct: pandas.DataFrame
    :param tier: Label used in the chart title.
    :type tier: str
    :param chart_type: ``"stacked"`` (full % share stack)
    :type chart_type: str
    :param ax: Existing axes to draw on.  If *None*, a new figure is
        created.
    :type ax: matplotlib.axes.Axes, optional
    :param figsize: Figure size when creating a new figure.
    :type figsize: tuple
    :returns: The figure containing the chart.
    :rtype: matplotlib.figure.Figure
    """
    features = list(pivot_pct.columns)
    years    = list(pivot_pct.index)
    colors   = {f: _color(i) for i, f in enumerate(features)}

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    
    bottom = np.zeros(len(years))
    for feat in features:
        vals = pivot_pct[feat].values
        ax.bar(
            years, vals, bottom=bottom,
            label=_short(feat), color=colors[feat],
            edgecolor="white", linewidth=0.4,
        )
        bottom += vals

    if chart_type == "stacked": 
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.set_ylabel("% share of driver signal", fontsize=11)
        ax.set_title(
            f"Migration Driver Composition by Year  [{tier} tier]",
            fontsize=13, fontweight="bold", pad=12,
        )
        ax.legend(
            loc="upper left", bbox_to_anchor=(1.01, 1),
            fontsize=8, frameon=True,
            title="Feature", title_fontsize=9, ncol=1,
        )


    ax.set_xlabel("Year", fontsize=11)
    ax.set_xticks(years)
    ax.set_xticklabels(years, rotation=0)
    ax.spines[["top", "right"]].set_visible(False)

    if standalone:
        fig.tight_layout()

    return fig


# ── 4. Time-series line charts ────────────────────────────────────────────────

def plot_driver_timeseries(
    pivot_pct: pd.DataFrame,
    tier: str = "All",
    top_n: int = 6,
    rolling_window: int = 3,
    figsize: tuple = (14, 9),
) -> plt.Figure:
    """Per-feature line charts with rolling mean and optional STL trend.

    :param pivot_pct: Output of :func:`aggregate_driver_frequency`.
    :type pivot_pct: pandas.DataFrame
    :param tier: Label used in the chart title.
    :type tier: str
    :param top_n: Number of top features (by total signal) to plot.
    :type top_n: int
    :param rolling_window: Window size for the centred rolling mean.
        Requires at least this many data points.
    :type rolling_window: int
    :param figsize: Figure size.
    :type figsize: tuple
    :returns: The figure.
    :rtype: matplotlib.figure.Figure
    """
    top_feats = pivot_pct.sum().nlargest(top_n).index.tolist()
    years     = list(pivot_pct.index)

    fig, axes = plt.subplots(
        top_n, 1, figsize=figsize, sharex=True,
        gridspec_kw={"hspace": 0.08},
    )
    if top_n == 1:
        axes = [axes]

    for ax, feat in zip(axes, top_feats):
        series = pivot_pct[feat]
        color  = _color(top_feats.index(feat))

        # Raw annual signal
        ax.plot(
            years, series.values,
            marker="o", ms=5, color=color,
            alpha=0.55, linewidth=1.2, label="annual",
        )

        # Centred rolling mean
        if len(years) >= rolling_window:
            roll = series.rolling(rolling_window, center=True).mean()
            ax.plot(
                years, roll.values, color=color, linewidth=2.4,
                label=f"{rolling_window}yr rolling mean",
            )

        # STL trend (requires statsmodels; needs ≥ 7 observations)
        if _HAS_STL and len(years) >= 7:
            try:
                stl = STL(series, period=3, robust=True).fit()
                ax.plot(
                    years, stl.trend.values, color=color,
                    linewidth=1.5, linestyle="--", alpha=0.8,
                    label="STL trend",
                )
            except Exception as exc:
                logger.debug("STL failed for %s: %s", feat, exc)

        ax.set_ylabel("% share", fontsize=8)
        ax.set_ylim(bottom=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", labelsize=8)

        label = feat.replace("_change", "").replace("_", " ")
        ax.text(
            0.01, 0.82, label,
            transform=ax.transAxes,
            fontsize=9, fontweight="bold", color=color, va="top",
        )

        if feat == top_feats[0]:
            ax.legend(loc="upper right", fontsize=7, frameon=False)

    axes[-1].set_xlabel("Year", fontsize=10)
    axes[-1].set_xticks(years)
    axes[-1].set_xticklabels(years, rotation=0)

    fig.suptitle(
        f"Feature Importance Trends in Migration Drivers  [{tier} tier]",
        fontsize=13, fontweight="bold", y=1.002,
    )
    fig.tight_layout()
    return fig


# ── 5. Rank stability table ───────────────────────────────────────────────────

def rank_stability(
    pivot_pct: pd.DataFrame,
    top_n: int = 8,
) -> pd.DataFrame:
    """Return a rank-stability summary DataFrame.

    Ranks each feature within each year (1 = highest % share), then
    summarises consistency across years.

    :param pivot_pct: Output of :func:`aggregate_driver_frequency`.
    :type pivot_pct: pandas.DataFrame
    :param top_n: Rows to return (sorted by mean rank ascending).
    :type top_n: int
    :returns: DataFrame with columns ``mean_rank``, ``rank_std``,
        ``mean_pct_share``, ``peak_year``.
    :rtype: pandas.DataFrame
    """
    ranks = pivot_pct.rank(axis=1, ascending=False)
    summary = pd.DataFrame({
        "mean_rank":       ranks.mean(),
        "rank_std":        ranks.std(),
        "mean_pct_share":  pivot_pct.mean(),
        "peak_year":       pivot_pct.idxmax(),
    }).sort_values("mean_rank")

    return summary.head(top_n)


def print_rank_stability(
    pivot_pct: pd.DataFrame,
    tier: str = "All",
    top_n: int = 8,
) -> pd.DataFrame:
    """Print a formatted rank-stability table and return the DataFrame.

    :param pivot_pct: Output of :func:`aggregate_driver_frequency`.
    :type pivot_pct: pandas.DataFrame
    :param tier: Label used in the printed header.
    :type tier: str
    :param top_n: Number of features to show.
    :type top_n: int
    :returns: The stability summary (same as :func:`rank_stability`).
    :rtype: pandas.DataFrame
    """
    result = rank_stability(pivot_pct, top_n=top_n)
    print(f"\n{'═'*72}")
    print(f"  RANK STABILITY — {tier} tier  "
          f"(lower mean_rank = more influential)")
    print(f"{'═'*72}")
    with pd.option_context(
        "display.float_format", "{:.2f}".format,
        "display.max_colwidth", 40,
        "display.width", 120,
    ):
        print(result.to_string())
    print()
    return result


# ── 6. One-shot entry point ───────────────────────────────────────────────────

def run_driver_analysis(
    df: pd.DataFrame,
    df_migrations: pd.DataFrame,
    tiers: Optional[List[Optional[str]]] = None,
    top_n_drivers: int = 5,
    rolling_window: int = 3,
    save_dir: Optional[str] = None,
) -> dict:
    """Run the full migration-driver analysis pipeline.

    Calls every function in this module in sequence, renders all charts,
    prints stability tables, and optionally saves figures and the driver
    panel CSV.

    :param df: Full clustered YoY DataFrame (``df_changes_optimized``).
    :type df: pandas.DataFrame
    :param df_migrations: Migration events from
        :func:`bank_innovation.migration.detect_migrations`.
    :type df_migrations: pandas.DataFrame
    :param tiers: Tier slices to analyse.  ``None`` inside the list
        means "all tiers combined".
    :type tiers: list, optional
    :param top_n_drivers: Top-N features to capture per transition.
    :type top_n_drivers: int
    :param rolling_window: Window for rolling mean in time-series plots.
    :type rolling_window: int
    :param save_dir: If provided, figures are saved as PNGs and the
        driver panel is saved as CSV inside this directory.
    :type save_dir: str, optional
    :returns: Dictionary with keys:

        * ``driver_long``    — raw long-form panel
        * ``pct_by_tier``    — dict mapping tier label → pivot_pct
        * ``stability``      — dict mapping tier label → stability DataFrame
        * ``figures``        — list of all matplotlib Figure objects
    :rtype: dict
    """
    if tiers is None:
        tiers = ["Large", "Medium", "Small"]

    tier_labels = [t if t is not None else "All" for t in tiers]

    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    # ── Step 1: build raw driver panel ───────────────────────────────────────
    logger.info("Building yearly driver table …")
    driver_long = build_yearly_driver_table(
        df, df_migrations,
        top_n_drivers=top_n_drivers,
        tiers=tiers,
    )

    pct_by_tier   = {}
    stability     = {}
    figures       = []

    for tier_label in tier_labels:
        pct = aggregate_driver_frequency(driver_long, tier=tier_label)
        if pct.empty:
            continue
        pct_by_tier[tier_label] = pct

        
        # ── Step 2: stacked bar chart ─────────────────────────────────────
        fig_stack = plot_yearly_driver_bars(pct, tier=tier_label,
                                            chart_type="stacked")
        figures.append(fig_stack)
        plt.show()
        if save_dir:
            fig_stack.savefig(
                Path(save_dir) / f"driver_stacked_{tier_label}.png",
                dpi=150, bbox_inches="tight",
            )
            plt.close(fig_stack)
        


        # ── Step 4: time-series line chart ────────────────────────────────
        if len(pct) >= 2:
            fig_ts = plot_driver_timeseries(
                pct, tier=tier_label,
                top_n=6, rolling_window=rolling_window,
            )
            figures.append(fig_ts)
            plt.show()
            if save_dir:
                fig_ts.savefig(
                    Path(save_dir) / f"driver_timeseries_{tier_label}.png",
                    dpi=150, bbox_inches="tight",
                )
                plt.close(fig_ts)

        # ── Step 5: rank stability ────────────────────────────────────────
        stab = print_rank_stability(pct, tier=tier_label)
        stability[tier_label] = stab

    # ── Step 6: save driver panel CSV ────────────────────────────────────────
    if save_dir and "All" in pct_by_tier:
        csv_path = Path(save_dir) / "migration_driver_panel.csv"
        pct_by_tier["All"].to_csv(csv_path)
        logger.info("Driver panel saved → %s", csv_path)

    logger.info("run_driver_analysis complete.")
    return {
        "driver_long":  driver_long,
        "pct_by_tier":  pct_by_tier,
        "stability":    stability,
        "figures":      figures,
    }
