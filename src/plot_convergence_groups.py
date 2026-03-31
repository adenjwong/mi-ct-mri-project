from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_dir", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    curves = []

    for run_dir in experiment_dir.iterdir():
        if not run_dir.is_dir():
            continue

        summary_json = run_dir / "summary.json"
        curve_csv = run_dir / "metric_curve.csv"

        if not summary_json.exists() or not curve_csv.exists():
            continue

        summary = pd.read_json(summary_json, typ="series")
        curve = pd.read_csv(curve_csv)
        curve["metric_name"] = summary["metric_name"]
        curve["bins"] = summary["bins"]
        curve["seed"] = summary["seed"]
        curves.append(curve)

    if not curves:
        print("No curves found.")
        return

    all_curves = pd.concat(curves, ignore_index=True)

    for metric_name in all_curves["metric_name"].unique():
        plt.figure(figsize=(8, 5))
        sub_metric = all_curves[all_curves["metric_name"] == metric_name]

        for bins in sorted(sub_metric["bins"].unique()):
            sub = sub_metric[sub_metric["bins"] == bins]
            grouped = sub.groupby("iteration")["metric_value"].mean().reset_index()
            plt.plot(grouped["iteration"], grouped["metric_value"], label=f"{bins} bins")

        plt.xlabel("Iteration")
        plt.ylabel("Metric value")
        plt.title(f"Mean Convergence Curves: {metric_name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / f"convergence_{metric_name}.png", dpi=200)
        plt.close()

    print(f"Saved convergence plots to {outdir}")


if __name__ == "__main__":
    main()