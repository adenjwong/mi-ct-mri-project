from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd

import SimpleITK as sitk
from tqdm import tqdm

from load_data import load_fixed_moving
from preprocess import preprocess_ct_mri
from register_rigid import (
    resample_registered_image,
    run_rigid_registration,
    save_results_json,
    save_transform,
)
from evaluate import summarize_registration
from visualize import save_metric_curve, save_overlay_figure
from roi_utils import crop_center_fraction


def _write_summary_csv(rows: List[Dict[str, Any]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    fieldnames = sorted(rows[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run rigid CT-MRI MI registration experiments.")
    parser.add_argument("--ct", required=True, help="Path to fixed CT image")
    parser.add_argument("--mri", required=True, help="Path to moving MRI image")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--metrics", nargs="+", default=["mattes_mi", "joint_hist_mi"])
    parser.add_argument("--bins", nargs="+", type=int, default=[16, 32, 64])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--sampling_percentage", type=float, default=0.2)
    parser.add_argument("--learning_rate", type=float, default=1.0)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--normalization", type=str, default="0_1", choices=["0_1", "zscore", "none"])
    parser.add_argument("--match_grid", action="store_true")
    parser.add_argument("--perturb_init", action="store_true")
    parser.add_argument("--use_roi", action="store_true")
    parser.add_argument("--roi_frac_x", type=float, default=0.5)
    parser.add_argument("--roi_frac_y", type=float, default=0.5)
    parser.add_argument("--roi_frac_z", type=float, default=0.5)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fixed_raw, moving_raw = load_fixed_moving(args.ct, args.mri)
    fixed, moving = preprocess_ct_mri(
        fixed_raw,
        moving_raw,
        normalization=args.normalization,
        match_grid=args.match_grid,
    )
    
    if args.use_roi:
        identity = sitk.Transform(fixed.GetDimension(), sitk.sitkIdentity)

        # First put moving onto the fixed image grid
        moving_resampled = sitk.Resample(
            moving,
            fixed,
            identity,
            sitk.sitkLinear,
            0.0,
            moving.GetPixelID(),
        )

        # Crop only the fixed image
        fixed = crop_center_fraction(
            fixed,
            frac_x=args.roi_frac_x,
            frac_y=args.roi_frac_y,
            frac_z=args.roi_frac_z,
        )

        # Now resample the moving image directly onto the cropped fixed ROI
        moving = sitk.Resample(
            moving_resampled,
            fixed,
            identity,
            sitk.sitkLinear,
            0.0,
            moving.GetPixelID(),
        )

        print("ROI mode enabled")
        print("Fixed ROI size:", fixed.GetSize())
        print("Fixed ROI spacing:", fixed.GetSpacing())
        print("Moving ROI size:", moving.GetSize())
        print("Moving ROI spacing:", moving.GetSpacing())

    summary_rows: List[Dict[str, Any]] = []

    run_configs = [
        (metric_name, bins, seed)
        for metric_name in args.metrics
        for bins in args.bins
        for seed in args.seeds
    ]

    for metric_name, bins, seed in tqdm(run_configs, desc="Experiment runs"):
        run_name = f"{metric_name}_bins{bins}_seed{seed}"
        run_dir = outdir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        transform, reg_results = run_rigid_registration(
            fixed=fixed,
            moving=moving,
            metric_name=metric_name,
            bins=bins,
            sampling_percentage=args.sampling_percentage,
            learning_rate=args.learning_rate,
            number_of_iterations=args.iterations,
            perturb_init=args.perturb_init,
            seed=seed,
        )
        
        curve_df = pd.DataFrame({
            "iteration": list(range(len(reg_results["iteration_metric_values"]))),
            "metric_value": reg_results["iteration_metric_values"],
            "metric_name": metric_name,
            "bins": bins,
            "seed": seed,
        })
        curve_df.to_csv(run_dir / "metric_curve.csv", index=False)

        registered = resample_registered_image(fixed, moving, transform)
        sitk.WriteImage(registered, str(run_dir / "registered_mri.nii.gz"))
        save_transform(transform, run_dir / "final_transform.tfm")
        save_results_json(reg_results, run_dir / "registration_results.json")

        summary = summarize_registration(fixed, registered, bins=bins)

        save_metric_curve(
            reg_results["iteration_metric_values"],
            run_dir / "metric_curve.png",
            title=f"{metric_name}, bins={bins}, seed={seed}",
        )
        # save_overlay_figure(
        #     fixed,
        #     moving,
        #     registered,
        #     run_dir / "overlay.png",
        #     axis=0,
        # )

        metric_values = reg_results["iteration_metric_values"]
        initial_metric = metric_values[0] if len(metric_values) > 0 else None
        final_metric = metric_values[-1] if len(metric_values) > 0 else None
        metric_improvement = None
        if initial_metric is not None and final_metric is not None:
            metric_improvement = final_metric - initial_metric

        row = {
            "run_name": run_name,
            "metric_name": metric_name,
            "bins": bins,
            "seed": seed,
            "sampling_percentage": args.sampling_percentage,
            "learning_rate": args.learning_rate,
            "iterations_requested": args.iterations,
            "iterations_recorded": len(metric_values),
            "initial_metric": initial_metric,
            "optimizer_final_metric": reg_results["final_metric_value"],
            "final_metric_curve_value": final_metric,
            "metric_improvement": metric_improvement,
            "posthoc_mi": summary["posthoc_mi"],
            "posthoc_nmi": summary["posthoc_nmi"],
            "stop_condition": reg_results["stop_condition"],
            "final_tx_p0": reg_results["final_parameters"][0] if len(reg_results["final_parameters"]) > 0 else None,
            "final_tx_p1": reg_results["final_parameters"][1] if len(reg_results["final_parameters"]) > 1 else None,
            "final_tx_p2": reg_results["final_parameters"][2] if len(reg_results["final_parameters"]) > 2 else None,
            "final_tx_p3": reg_results["final_parameters"][3] if len(reg_results["final_parameters"]) > 3 else None,
            "final_tx_p4": reg_results["final_parameters"][4] if len(reg_results["final_parameters"]) > 4 else None,
            "final_tx_p5": reg_results["final_parameters"][5] if len(reg_results["final_parameters"]) > 5 else None,
            "use_roi": args.use_roi,
            "roi_frac_x": args.roi_frac_x if args.use_roi else None,
            "roi_frac_y": args.roi_frac_y if args.use_roi else None,
            "roi_frac_z": args.roi_frac_z if args.use_roi else None,
        }
        summary_rows.append(row)

        with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(row, f, indent=2)

        print(f"Finished {run_name}")

    _write_summary_csv(summary_rows, outdir / "summary.csv")
    print(f"Saved summary to: {outdir / 'summary.csv'}")


if __name__ == "__main__":
    main()