#!/usr/bin/env python3
##
# @file visualize.py
# @brief RISC-V microarchitecture research platform visualizer.
#
# Reads CSV output from rvsim sweeps and generates plots showing how
# architecture parameters affect performance metrics.
#
# Supported visualizations include:
# - Bar charts for single-parameter sweeps
# - Heatmaps for two-parameter sweeps
# - Pareto-front scatter plots for multi-objective analysis
# - Parallel coordinates plots for high-dimensional exploration
# 
# Metric columns are detected automatically from known metric names;
# all other columns are treated as architectural parameters.
#
# Pareto-optimal configurations are highlighted automatically when
# Pareto objectives are specified.
#
# @par Usage
# @code{.sh}
# # Generate plots using default metrics
# python tools/visualize.py results.csv
#
# # Select specific metrics
# python tools/visualize.py results.csv \
#     --metric ipc branch_accuracy l1_hit_rate
#
# # Specify Pareto optimization objectives
# python tools/visualize.py results.csv \
#     --pareto ipc l1_hit_rate
# 
# # Write plots to a custom directory
# python tools/visualize.py results.csv \
#     --output plots/
# @endcode
#

import argparse
import csv
import sys
from pathlib import Path
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colormaps
import numpy as np

# ── Metric metadata ─────────────────────────────────────────────────────

METRIC_COLS: set[str] = {
    "cycles", "instructions", "ipc", "stalls", "flushes",
    "l1_hit_rate", "l1_misses",
    "l1i_hit_rate", "l1i_misses",
    "l2_hit_rate", "l2_misses",
    "l3_hit_rate", "l3_misses",
    "branches", "mispredicts", "branch_accuracy",
}

# Higher is better for these; lower is better for the rest
HIGHER_IS_BETTER: set[str] = {
        "ipc", "l1_hit_rate", "l1i_hit_rate", "l2_hit_rate", "l3_hit_rate", "branch_accuracy",
}

# ── CSV loading ─────────────────────────────────────────────────────────

type ParamValue = str
type MetricValue = float
type CsvRow = dict[str, ParamValue | MetricValue]
type CsvData = list[CsvRow]


##
# @brief Load and classify sweep-result CSV data.
#
# Reads CSV output generated from rvsim parameter sweeps and
# separates columns into:
# - architectural parameters, and
# - performance metrics.
#
# Columns listed in @c METRIC_COLS are parsed as floating-point
# metrics. All other columns are treated as parameter values.
#
# Rows containing invalid metric values are skipped.
#
# @param path  Path to CSV input file.
#
# @return Tuple containing:
# - parsed configuration data,
# - parameter column names,
# - metric column names.
#
# @throws ValueError
# Raised if the CSV file is empty.
#
def load_csv(path: Path) -> tuple[CsvData, list[str], list[str]]:
    """Load CSV, auto-detect parameter vs metric columns."""
    with path.open(newline="", encoding="utf-8") as f:
        reader: csv.DictReader = csv.DictReader(f)
        rows: list[dict[str, str]] = list(reader)

    if not rows:
        raise ValueError(f"'{path}' is empty")

    headers: list[str] = list(rows[0].keys())

    # Classify columns as parameters or metrics
    param_cols: list[str] = []
    metric_cols: list[str] = []
    for h in headers:
        if h in METRIC_COLS:
            metric_cols.append(h)
        else:
            param_cols.append(h)

    # Parse values
    data: list[CsvRow] = []
    for row in rows:
        entry: CsvRow = {}
        for p in param_cols:
            entry[p] = row[p]
        try:
            for m in metric_cols:
                entry[m] = float(row[m])
        except (ValueError, TypeError):
            continue
        data.append(entry)

    return data, param_cols, metric_cols


# ── Pareto front ────────────────────────────────────────────────────────

##
# @brief Return whether configuration @p b dominates @p a.
#
# A configuration dominates another if it is:
# - at least as good in all objectives, and
# - strictly better in at least one objective.
#
# Metrics listed in @c HIGHER_IS_BETTER are maximized;
# all others are minimized.
#
# @param a           Candidate dominated configuration.
# @param b           Candidate dominating configuration.
# @param objectives  Optimization objectives.
#
# @return @c True if @p b dominates @p a.
#
def is_dominated(a: CsvRow, b: CsvRow, objectives: list[str]) -> bool:
    at_least_as_good = True
    strictly_better = False
    
    for obj in objectives:
        av = float(a[obj])
        bv = float(b[obj])

        if obj in HIGHER_IS_BETTER:
            if bv < av: at_least_as_good = False
            if bv > av: strictly_better = True
        else:
            if bv > av: at_least_as_good = False
            if bv < av: strictly_better = True
    
    return at_least_as_good and strictly_better


##
# @brief Compute the Pareto-optimal subset of configurations.
#
# Performs an O(n²) pairwise dominance check across all
# configurations.
#
# @param data        Configuration dataset.
# @param objectives  Optimization objectives.
#
# @return Indices of Pareto-optimal configurations.
#
def pareto_front(data: CsvData, objectives: list[str]) -> list[int]:
    n = len(data)
    is_pareto: list[bool] = [True] * n

    for i in range(n):
        if not is_pareto[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            if is_dominated(data[i], data[j], objectives):
                is_pareto[i] = False
                break

    return [i for i, keep in enumerate(is_pareto) if keep]


# ── Plotting helpers ────────────────────────────────────────────────────

##
# @brief Save and close a matplotlib figure.
#
# @param fig         Figure to save.
# @param output_dir  Output directory.
# @param filename    Output filename.
#
# @return Path to saved figure.
#
def _save_figure(fig: matplotlib.figure.Figure,
                 output_dir: Path, filename: str) -> Path:
    path = output_dir / filename

    fig.tight_layout()
    fig.savefig(path, dpi=150)

    plt.close(fig)

    return path


##
# @brief Return sorted unique categorical values.
#
# @param values  Input categorical values.
#
# @return Sorted unique values.
#
def _sorted_categories(values: list[ParamValue]) -> list[ParamValue]:
    return sorted(set(values), key=_sort_key)

##
# @brief Normalize numeric values to the range [0, 1].
#
# @param values  Input values.
#
# @return Normalized values.
#
def _normalize(values: list[float]) -> list[float]:
    vmin = min(values)
    vmax = max(values)

    if vmax == vmin:
        return [0.5] * len(values)

    return [
        (v - vmin) / (vmax - vmin)
        for v in values
    ]


# ── Plot: bar chart (1 parameter) ───────────────────────────────────────

##
# @brief Generate a bar chart for one parameter and one metric.
#
# Groups configurations by parameter value and plots the
# mean metric value with standard deviation error bars.
#
# @param data        Configuration dataset.
# @param param       Parameter column.
# @param metric      Metric column.
# @param output_dir  Output directory.
#
# @return Path to generated plot.
#
def plot_bars(data: CsvData, param: str, metric: str, output_dir: Path) -> Path:
    groups: dict[str, list[float]] = {}
    
    for row in data:
        key = str(row[param])
        groups.setdefault(key, []).append(float(row[metric]))

    labels = _sorted_categories(list(groups.keys()))

    means = [np.mean(groups[label])
             for label in labels]
    stds = [np.std(groups[label]) if len(groups[label]) > 1 else 0.0
            for label in labels]

    fig, ax = plt.subplots(figsize=(8, 5))
    
    x = np.arange(len(labels))

    ax.bar(
        x,
        means,
        yerr=stds,
        capsize=4,
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")

    ax.set_xlabel(param)
    ax.set_ylabel(metric)

    ax.set_title(f"{metric} vs {param}")

    ax.grid(axis="y", alpha=0.3)

    return _save_figure(fig, output_dir, f"bar_{param}_{metric}.png")


# ── Heatmap plot (2 parameters) ─────────────────────────────────────────

##
# @brief Generate a heatmap for two parameters and one metric.
#
# Parameter values define the heatmap axes and metric values
# define cell color.
#
# @param data        Configuration dataset.
# @param param1      First parameter axis.
# @param param2      Second parameter axis.
# @param metric      Metric to visualize.
# @param output_dir  Output directory.
#
# @return Path to generated plot.
#
def plot_heatmap(data: CsvData, param1: str, param2: str,
                 metric: str, output_dir: Path) -> Path:
    vals1 = _sorted_categories([str(d[param1]) for d in data])
    vals2 = _sorted_categories([str(d[param2]) for d in data])

    idx1 = {value: i for i, value in enumerate(vals1)}
    idx2 = {value: i for i, value in enumerate(vals2)}

    grid = np.full((len(vals2), len(vals1)), np.nan)

    for row in data:
        i = idx2[str(row[param2])]
        j = idx1[str(row[param1])]

        grid[i, j] = float(row[metric])

    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(
        grid,
        cmap="viridis",
        aspect="auto",
        origin="lower",
    )

    ax.set_xticks(range(len(vals1)))
    ax.set_xticklabels(vals1, rotation=45, ha="right")

    ax.set_yticks(range(len(vals2)))
    ax.set_yticklabels(vals2)

    ax.set_xlabel(param1)
    ax.set_ylabel(param2)

    ax.set_title(metric)

    fig.colorbar(im, ax=ax, label=metric)

    mean = np.nanmean(grid)

    # Annotate cells
    for i in range(len(vals2)):
        for j in range(len(vals1)):
            value = grid[i, j]

            if np.isnan(value):
                continue

            ax.text(
                j,
                i,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if value < mean else "black",
            )

    return _save_figure(
        fig,
        output_dir,
        f"heatmap_{param1}_{param2}_{metric}.png"
    )


# ── Scatter plot with Pareto front ──────────────────────────────────────

##
# @brief Generate a Pareto scatter plot for two metrics.
#
# All configurations are plotted in metric space, with
# Pareto-optimal points highlighted.
#
# Points are optionally colored by the first parameter.
#
# @param data        Configuration dataset.
# @param metric_x    X-axis metric.
# @param metric_y    Y-axis metric.
# @param param_cols  Parameter columns.
# @param pareto_idx  Pareto-optimal indices.
# @param output_dir  Output directory.
#
# @return Path to generated plot.
#
def plot_pareto_scatter(data: CsvData, metric_x: str, metric_y: str,
                        param_cols: list[str], pareto_idx: list[int],
                        output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6))

    points = [
        (
            float(row[metric_x]),
            float(row[metric_y]),
        )
        for row in data
    ]

    # Color by first parameter if available
    if param_cols:
        param = param_cols[0]

        categories = _sorted_categories([str(row[param]) for row in data])
        
        cmap = colormaps['tab10'].resampled(len(categories))

        for ci, category in enumerate(categories):
            xs: list[float] = []
            ys: list[float] = []

            for row, (x, y) in zip(data, points):
                if str(row[param]) == category:
                    xs.append(x)
                    ys.append(y)

            ax.scatter(
                xs,
                ys,
                label=f"{param}={category}",
                alpha=0.6,
                s=40,
                c=[cmap(ci)],
            )
    else:
        xs = [x for x, _ in points]
        ys = [y for y, _ in points]

        ax.scatter(
            xs,
            ys,
            alpha=0.6,
            s=40,
        )

    # Highlight Pareto front
    pareto_x = [points[i][0] for i in pareto_idx]
    pareto_y = [points[i][0] for i in pareto_idx]
    
    ax.scatter(
        pareto_x,
        pareto_y,
        edgecolors="red",
        facecolors="none",
        linewidths=2,
        s=120,
        label="Pareto optimal",
        zorder=5,
    )

    ax.set_xlabel(metric_x)
    ax.set_ylabel(metric_y)

    ax.set_title(f"Pareto front: {metric_x} vs {metric_y}")

    ax.grid(alpha=0.3)

    ax.legend(
        fontsize=8,
        loc="best",
    )

    fig.tight_layout()

    return _save_figure(
        fig,
        output_dir,
        f"pareto_{metric_x}_{metric_y}.png",
    )


# ── Parallel coordinates plot ───────────────────────────────────────────


##
# @brief Generate a parallel coordinates plot.
#
# Parameters and metrics are normalized onto parallel axes.
# Pareto-optimal configurations are highlighted.
#
# @param data         Configuration dataset.
# @param param_cols   Parameter columns.
# @param metric_cols  Metric columns.
# @param pareto_idx   Pareto-optimal indices.
# @param output_dir   Output directory.
#
# @return Path to generated plot, or @c None if insufficient axes exist.
#
def plot_parallel(data: CsvData, param_cols: list[str], metric_cols: list[str],
                  pareto_idx: list[int], output_dir: Path) -> Path | None:
    axes = param_cols + metric_cols

    if len(axes) < 2:
        return None

    fig, ax = plt.subplots(figsize=(max(10, len(axes) * 1.5), 6))

    normalized_axes: list[list[float]] = []

    for column in axes:
        if column in METRIC_COLS:
            values = [float(row[column])
                      for row in data]
            
            normalized_axes.append(_normalize(values))
        else:
            categories = _sorted_categories([str(row[column])
                                             for row in data])

            cat_map = {category: i / max(1, len(categories) - 1)
                       for i, category in enumerate(categories)}

            normalized_axes.append([cat_map[str(row[column])]
                                    for row in data])

    x = np.arange(len(axes))

    pareto_set = set(pareto_idx)

    for i in range(len(data)):
        ys = [normalized_axes[j][i]
              for j in range(len(axes))]

        is_pareto_point = i in pareto_set

        ax.plot(
            c="red" if is_pareto_point else "lightgray",
            alpha=0.9 if is_pareto_point else 0.3,
            linewidth=2.0 if is_pareto_point else 0.8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(axes, rotation=45, ha="right")

    ax.set_ylabel("Normalized value")
    ax.set_title("Parallel coordinates (red = Pareto optimal)")

    ax.set_ylim(-0.05, 1.05)
    ax.grid(axis="x", alpha=0.3)

    return _save_figure(
        fig,
        output_dir,
        "parallel_coordinates.png"
    )


# ── Sorting helper ──────────────────────────────────────────────────────

##
# @brief Numeric-aware sorting helper.
#
# Sorts values such that numeric quantities sort naturally:
# - 8K < 16K < 32K
# - 1M > 512K
#
# Non-numeric values are sorted lexicographically
# after numeric values.
#
# Supported suffixes:
# - K = 1024
# - M = 1024 * 1024
#
# @param value  Input value.
#
# @return Sort key tuple.
#
def _sort_key(value: object) -> tuple[int, float, str]:
    s = str(value).strip().upper()
    
    multiplier = 1.0
    
    if s.endswith("K"):
        multiplier = 1024.0
        s = s[:-1]
    elif s.endswith("M"):
        multiplier = 1024.0 * 1024.0
        s = s[:-1]
    try:
        numeric = float(s) * multiplier

        return (0, numeric, "")
    except ValueError:
        return (1, 0.0, s)


# ── Helpers for main() ──────────────────────────────────────────────────

##
# @brief Record and report a generated plot.
#
# @param generated  Generated plot list.
# @param path       Generated file path.
#
def _record_plot(generated: list[Path], path: Path) -> None:
    generated.append(path)
    print(f"  Generated '{path}'")

##
# @brief Select default metrics present in the dataset.
#
# @param metric_cols  Available metrics.
# @param preferred    Preferred metrics.
#
# @return Selected metrics.
#
def _select_metrics(metric_cols: list[str], preferred: list[str]) -> list[str]:
    selected = [metric
                for metric in preferred
                if metric in metric_cols]

    return selected or metric_cols[:3]


# ── Main ────────────────────────────────────────────────────────────────

##
# @brief Entry point.
#
# Parses command-line arguments, generates plots,
# and reports Pareto-optimal configurations.
#
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize rvsim architecture sweep results.")

    parser.add_argument("csv",
                        help="CSV file from rvsim sweep")

    parser.add_argument("--metric",
                        "-m",
                        nargs="*",
                        help="Metrics to plot")

    parser.add_argument("--pareto",
                        "-p",
                        nargs="*",
                        help="Pareto optimization objectives")
    
    parser.add_argument("--output",
                        "-o",
                        default="plots",
                        help="Output directory")

    args = parser.parse_args()

    csv_path = Path(args.csv)

    data, param_cols, metric_cols = load_csv(csv_path)
    
    print(f"Loaded {len(data)} configurations, "
          f"{len(param_cols)} parameters, "
          f"{len(metric_cols)} metrics")

    print(f"  Parameters: {', '.join(param_cols)}")
    print(f"  Metrics:    {', '.join(metric_cols)}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select metrics to plot
    plot_metrics = (args.metric
                    or _select_metrics(metric_cols, ["ipc", "l1_hit_rate", "cycles"]))

    # Pareto objectives
    pareto_objectives = (args.pareto
                         or _select_metrics(metric_cols, ["ipc", "l1_hit_rate"]))

    pareto_idx = (pareto_front(data, pareto_objectives)
                  if pareto_objectives else [])

    print(f"  Pareto front: "
          f"{len(pareto_idx)} / {len(data)} configurations "
          f"(objectives: {', '.join(pareto_objectives)})")

    generated: list[Path] = []

    # ── Single-parameter bar charts ──────────────────────

    for param in param_cols:
        for metric in plot_metrics:
            path = plot_bars(data, param, metric, output_dir)
            
            _record_plot(generated, path)

    # ── Two-parameter heatmaps ───────────────────────────

    if len(param_cols) == 2:
        for metric in plot_metrics:
            path = plot_heatmap(data, param_cols[0], param_cols[1],
                                metric, output_dir)

            _record_plot(generated, path)


    # ── Pareto scatter plots ─────────────────────────────

    for metric_x, metric_y in combinations(pareto_objectives, 2):
        path = plot_pareto_scatter(data, metric_x, metric_y,
                                   param_cols, pareto_idx, output_dir)
    
        _record_plot(generated, path)

    # ── Parellel coordinates ─────────────────────────────

    if len(param_cols) + len(plot_metrics) >= 3:
        path = plot_parallel(data, param_cols, plot_metrics,
                             pareto_idx, output_dir)
        
        if path is not None:
            _record_plot(generated, path)

    # ── Summary ──────────────────────────────────────────

    print(f"\n{len(generated)} plots saved to {output_dir}/")

    if pareto_idx:
        print(f"\nPareto-optimal configurations ({', '.join(pareto_objectives)}):")

        for i in pareto_idx:
            row = data[i]
            
            params = ", ".join(f"{p}={row[p]}" for p in param_cols)
            
            metrics = ", ".join(f"{m}={row[m]:.4f}" for m in pareto_objectives)
            
            print(f"  {params} -> {metrics}")


if __name__ == "__main__":
    main()
