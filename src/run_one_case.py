from __future__ import annotations

import argparse
from pathlib import Path

import SimpleITK as sitk

from load_data import load_fixed_moving, print_image_info
from preprocess import preprocess_ct_mri
from register_rigid import (
    resample_registered_image,
    run_rigid_registration,
    save_results_json,
    save_transform,
)
from evaluate import summarize_registration
from visualize import save_metric_curve, save_overlay_figure, can_generate_overlay


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one rigid CT-MRI registration.")
    parser.add_argument("--ct", required=True)
    parser.add_argument("--mri", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--metric", default="mattes_mi", choices=["mattes_mi", "joint_hist_mi"])
    parser.add_argument("--bins", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--sampling_percentage", type=float, default=0.2)
    parser.add_argument("--normalization", type=str, default="0_1", choices=["0_1", "zscore", "none"])
    parser.add_argument("--match_grid", action="store_true")
    parser.add_argument("--perturb_init", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
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

    transform, reg_results = run_rigid_registration(
        fixed=fixed,
        moving=moving,
        metric_name=args.metric,
        bins=args.bins,
        sampling_percentage=args.sampling_percentage,
        number_of_iterations=args.iterations,
        perturb_init=args.perturb_init,
        seed=args.seed,
    )

    registered = resample_registered_image(fixed, moving, transform)

    sitk.WriteImage(registered, str(outdir / "registered_mri.nii.gz"))
    save_transform(transform, outdir / "final_transform.tfm")
    save_results_json(reg_results, outdir / "registration_results.json")

    summary = summarize_registration(fixed, registered, bins=args.bins)

    save_metric_curve(
        reg_results["iteration_metric_values"],
        outdir / "metric_curve.png",
        title=f"{args.metric}, bins={args.bins}",
    )
    if can_generate_overlay(fixed, moving) and can_generate_overlay(fixed, registered):
        save_overlay_figure(
            fixed,
            moving,
            registered,
            outdir / "overlay.png",
            axis=0,
        )
    else:
        print("Skipping overlay (shape mismatch or incompatible dims)")

    print("Registration complete.")
    print(summary)


if __name__ == "__main__":
    main()