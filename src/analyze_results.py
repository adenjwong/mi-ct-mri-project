from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_summary_tables(df: pd.DataFrame, outdir: Path) -> None:
    """
    Save mean/std/count tables grouped by metric and bins.
    """
    grouped = df.groupby(["metric_name", "bins"])

    summary = grouped.agg(
        runs=("run_name", "count"),
        mean_posthoc_mi=("posthoc_mi", "mean"),
        std_posthoc_mi=("posthoc_mi", "std"),
        mean_posthoc_nmi=("posthoc_nmi", "mean"),
        std_posthoc_nmi=("posthoc_nmi", "std"),
        mean_metric_improvement=("metric_improvement", "mean"),
        std_metric_improvement=("metric_improvement", "std"),
        mean_iterations=("iterations_recorded", "mean"),
        std_iterations=("iterations_recorded", "std"),
    ).reset_index()

    summary.to_csv(outdir / "summary_stats.csv", index=False)


def add_success_column(df: pd.DataFrame, threshold_fraction: float = 0.95) -> pd.DataFrame:
    """
    Define success relative to the best posthoc MI within each metric_name group.
    """
    df = df.copy()
    success_flags = []

    for metric_name, group in df.groupby("metric_name"):
        best = group["posthoc_mi"].max()
        threshold = threshold_fraction * best
        group_flags = group["posthoc_mi"] >= threshold
        success_flags.extend(group_flags.tolist())

    df["success"] = success_flags
    return df


def save_success_table(df: pd.DataFrame, outdir: Path) -> None:
    success_table = (
        df.groupby(["metric_name", "bins"])["success"]
        .mean()
        .mul(100.0)
        .reset_index()
        .rename(columns={"success": "success_rate_percent"})
    )
    success_table.to_csv(outdir / "success_rates.csv", index=False)


def plot_boxplot(df: pd.DataFrame, value_col: str, ylabel: str, title: str, outpath: Path) -> None:
    plt.figure(figsize=(8, 5))

    labels = []
    data = []

    grouped = df.groupby(["metric_name", "bins"])
    for (metric_name, bins), group in grouped:
        labels.append(f"{metric_name}\n{bins} bins")
        data.append(group[value_col].dropna().values)

    plt.boxplot(data, tick_labels=labels)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_success_rate(df: pd.DataFrame, outpath: Path) -> None:
    plt.figure(figsize=(8, 5))

    grouped = (
        df.groupby(["metric_name", "bins"])["success"]
        .mean()
        .mul(100.0)
        .reset_index()
    )

    for metric_name in grouped["metric_name"].unique():
        sub = grouped[grouped["metric_name"] == metric_name].sort_values("bins")
        plt.plot(sub["bins"], sub["success"], marker="o", label=metric_name)

    plt.xlabel("Histogram bins")
    plt.ylabel("Success rate (%)")
    plt.title("Success Rate vs Bin Size")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_metric_improvement(df: pd.DataFrame, outpath: Path) -> None:
    plt.figure(figsize=(8, 5))

    grouped = (
        df.groupby(["metric_name", "bins"])["metric_improvement"]
        .mean()
        .reset_index()
    )

    for metric_name in grouped["metric_name"].unique():
        sub = grouped[grouped["metric_name"] == metric_name].sort_values("bins")
        plt.plot(sub["bins"], sub["metric_improvement"], marker="o", label=metric_name)

    plt.xlabel("Histogram bins")
    plt.ylabel("Mean metric improvement")
    plt.title("Mean Metric Improvement vs Bin Size")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_iterations(df: pd.DataFrame, outpath: Path) -> None:
    plt.figure(figsize=(8, 5))

    grouped = (
        df.groupby(["metric_name", "bins"])["iterations_recorded"]
        .mean()
        .reset_index()
    )

    for metric_name in grouped["metric_name"].unique():
        sub = grouped[grouped["metric_name"] == metric_name].sort_values("bins")
        plt.plot(sub["bins"], sub["iterations_recorded"], marker="o", label=metric_name)

    plt.xlabel("Histogram bins")
    plt.ylabel("Mean iterations")
    plt.title("Mean Iterations to Convergence vs Bin Size")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_csv", required=True, help="Path to summary.csv from experiments.py")
    parser.add_argument("--outdir", required=True, help="Directory to save analysis outputs")
    parser.add_argument("--success_threshold", type=float, default=0.95)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    ensure_dir(outdir)

    df = pd.read_csv(args.summary_csv)
    df = add_success_column(df, threshold_fraction=args.success_threshold)

    df.to_csv(outdir / "summary_with_success.csv", index=False)

    save_summary_tables(df, outdir)
    save_success_table(df, outdir)

    plot_boxplot(
        df,
        value_col="posthoc_mi",
        ylabel="Post-hoc MI",
        title="Distribution of Final MI Values",
        outpath=outdir / "boxplot_posthoc_mi.png",
    )

    plot_boxplot(
        df,
        value_col="posthoc_nmi",
        ylabel="Post-hoc NMI",
        title="Distribution of Final NMI Values",
        outpath=outdir / "boxplot_posthoc_nmi.png",
    )

    plot_success_rate(df, outdir / "success_rate_vs_bins.png")
    plot_metric_improvement(df, outdir / "metric_improvement_vs_bins.png")
    plot_iterations(df, outdir / "iterations_vs_bins.png")

    print(f"Saved analysis outputs to: {outdir}")


if __name__ == "__main__":
    main()