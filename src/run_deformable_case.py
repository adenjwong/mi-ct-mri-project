from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import SimpleITK as sitk

from load_data import load_fixed_moving, print_image_info
from preprocess import preprocess_ct_mri
from register_rigid import (
    run_rigid_registration,
    resample_registered_image,
    save_transform,
    save_results_json,
)
from register_deformable import (
    run_bspline_registration,
    compose_transforms,
)
from evaluate import summarize_registration
from visualize import save_metric_curve


def main() -> None:
    parser = argparse.ArgumentParser(description="Run rigid + deformable CT-MRI registration.")
    parser.add_argument("--ct", required=True)
    parser.add_argument("--mri", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--metric", default="mattes_mi", choices=["mattes_mi", "joint_hist_mi"])
    parser.add_argument("--bins", type=int, default=32)
    parser.add_argument("--rigid_iterations", type=int, default=200)
    parser.add_argument("--deformable_iterations", type=int, default=75)
    parser.add_argument("--sampling_percentage", type=float, default=0.2)
    parser.add_argument("--normalization", type=str, default="0_1", choices=["0_1", "zscore", "none"])
    parser.add_argument("--match_grid", action="store_true")
    parser.add_argument("--perturb_init", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mesh_x", type=int, default=4)
    parser.add_argument("--mesh_y", type=int, default=4)
    parser.add_argument("--mesh_z", type=int, default=3)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fixed_raw, moving_raw = load_fixed_moving(args.ct, args.mri)
    print_image_info("Fixed CT", fixed_raw)
    print_image_info("Moving MRI", moving_raw)

    fixed, moving = preprocess_ct_mri(
        fixed_raw,
        moving_raw,
        normalization=args.normalization,
        match_grid=args.match_grid,
    )

    # Stage 1: rigid
    rigid_transform, rigid_results = run_rigid_registration(
        fixed=fixed,
        moving=moving,
        metric_name=args.metric,
        bins=args.bins,
        sampling_percentage=args.sampling_percentage,
        number_of_iterations=args.rigid_iterations,
        perturb_init=args.perturb_init,
        seed=args.seed,
    )

    rigid_registered = resample_registered_image(fixed, moving, rigid_transform)
    rigid_summary = summarize_registration(fixed, rigid_registered, bins=args.bins)

    sitk.WriteImage(rigid_registered, str(outdir / "rigid_registered_mri.nii.gz"))
    save_transform(rigid_transform, outdir / "rigid_transform.tfm")
    save_results_json(rigid_results, outdir / "rigid_results.json")

    rigid_curve = pd.DataFrame({
        "iteration": list(range(len(rigid_results["iteration_metric_values"]))),
        "metric_value": rigid_results["iteration_metric_values"],
        "stage": "rigid",
    })
    rigid_curve.to_csv(outdir / "rigid_metric_curve.csv", index=False)

    save_metric_curve(
        rigid_results["iteration_metric_values"],
        outdir / "rigid_metric_curve.png",
        title=f"Rigid: {args.metric}, bins={args.bins}",
    )

    # Stage 2: deformable refinement
    deformable_transform, deformable_results = run_bspline_registration(
        fixed=fixed,
        moving=moving,
        initial_transform=rigid_transform,
        metric_name=args.metric,
        bins=args.bins,
        sampling_percentage=args.sampling_percentage,
        number_of_iterations=args.deformable_iterations,
        mesh_size=(args.mesh_x, args.mesh_y, args.mesh_z),
    )

    composite_transform = compose_transforms(rigid_transform, deformable_transform)
    deformable_registered = resample_registered_image(fixed, moving, composite_transform)
    deformable_summary = summarize_registration(fixed, deformable_registered, bins=args.bins)

    sitk.WriteImage(deformable_registered, str(outdir / "deformable_registered_mri.nii.gz"))
    save_transform(deformable_transform, outdir / "deformable_transform.tfm")
    save_results_json(deformable_results, outdir / "deformable_results.json")

    deformable_curve = pd.DataFrame({
        "iteration": list(range(len(deformable_results["iteration_metric_values"]))),
        "metric_value": deformable_results["iteration_metric_values"],
        "stage": "deformable",
    })
    deformable_curve.to_csv(outdir / "deformable_metric_curve.csv", index=False)

    save_metric_curve(
        deformable_results["iteration_metric_values"],
        outdir / "deformable_metric_curve.png",
        title=f"Deformable: {args.metric}, bins={args.bins}",
    )

    summary = {
        "metric_name": args.metric,
        "bins": args.bins,
        "seed": args.seed,
        "mesh_size": [args.mesh_x, args.mesh_y, args.mesh_z],
        "rigid_posthoc_mi": rigid_summary["posthoc_mi"],
        "rigid_posthoc_nmi": rigid_summary["posthoc_nmi"],
        "deformable_posthoc_mi": deformable_summary["posthoc_mi"],
        "deformable_posthoc_nmi": deformable_summary["posthoc_nmi"],
        "rigid_metric_final": rigid_results["final_metric_value"],
        "deformable_metric_final": deformable_results["final_metric_value"],
        "rigid_stop_condition": rigid_results["stop_condition"],
        "deformable_stop_condition": deformable_results["stop_condition"],
    }

    with open(outdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Rigid summary:", rigid_summary)
    print("Deformable summary:", deformable_summary)
    print("Saved deformable experiment to:", outdir)


if __name__ == "__main__":
    main()