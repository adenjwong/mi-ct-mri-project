from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm


def _make_initial_transform(
    fixed: sitk.Image,
    moving: sitk.Image,
    transform_type: str = "euler3d",
) -> sitk.Transform:
    """
    Create an initial centered rigid transform.
    """
    if transform_type.lower() == "euler3d":
        tx = sitk.Euler3DTransform()
    elif transform_type.lower() == "versor":
        tx = sitk.VersorRigid3DTransform()
    else:
        raise ValueError(f"Unsupported transform_type: {transform_type}")

    initial_transform = sitk.CenteredTransformInitializer(
        fixed,
        moving,
        tx,
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    return initial_transform


def _perturb_transform(
    transform: sitk.Transform,
    translation_std_mm: float = 5.0,
    rotation_std_deg: float = 3.0,
    seed: int | None = None,
) -> sitk.Transform:
    """
    Add random perturbations to a rigid transform.
    Assumes Euler3DTransform-like parameterization:
    (rx, ry, rz, tx, ty, tz), with rotations in radians.
    """
    rng = np.random.default_rng(seed)

    tx = sitk.Euler3DTransform(transform)
    params = list(tx.GetParameters())

    rotation_std_rad = np.deg2rad(rotation_std_deg)
    params[0] += float(rng.normal(0.0, rotation_std_rad))
    params[1] += float(rng.normal(0.0, rotation_std_rad))
    params[2] += float(rng.normal(0.0, rotation_std_rad))
    params[3] += float(rng.normal(0.0, translation_std_mm))
    params[4] += float(rng.normal(0.0, translation_std_mm))
    params[5] += float(rng.normal(0.0, translation_std_mm))

    tx.SetParameters(tuple(params))
    return tx


def _configure_metric(
    registration_method: sitk.ImageRegistrationMethod,
    metric_name: str,
    bins: int,
    sampling_percentage: float,
) -> None:
    """
    Configure the similarity metric.
    """
    metric_name = metric_name.lower()

    if metric_name == "mattes_mi":
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=bins)
    elif metric_name == "joint_hist_mi":
        registration_method.SetMetricAsJointHistogramMutualInformation(
            numberOfHistogramBins=bins,
            varianceForJointPDFSmoothing=1.5,
        )
    else:
        raise ValueError(
            f"Unsupported metric_name: {metric_name}. "
            "Use 'mattes_mi' or 'joint_hist_mi'."
        )

    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(sampling_percentage)
    registration_method.SetInterpolator(sitk.sitkLinear)


def _configure_optimizer(
    registration_method: sitk.ImageRegistrationMethod,
    learning_rate: float = 1.0,
    min_step: float = 1e-4,
    number_of_iterations: int = 200,
    relaxation_factor: float = 0.5,
) -> None:
    """
    Configure a gradient-descent optimizer.
    """
    registration_method.SetOptimizerAsRegularStepGradientDescent(
        learningRate=learning_rate,
        minStep=min_step,
        numberOfIterations=number_of_iterations,
        relaxationFactor=relaxation_factor,
        gradientMagnitudeTolerance=1e-8,
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()


def run_rigid_registration(
    fixed: sitk.Image,
    moving: sitk.Image,
    metric_name: str = "mattes_mi",
    bins: int = 32,
    sampling_percentage: float = 0.2,
    learning_rate: float = 1.0,
    min_step: float = 1e-4,
    number_of_iterations: int = 200,
    transform_type: str = "euler3d",
    perturb_init: bool = False,
    perturb_translation_std_mm: float = 5.0,
    perturb_rotation_std_deg: float = 3.0,
    seed: int | None = None,
) -> Tuple[sitk.Transform, Dict[str, Any]]:
    """
    Run rigid registration and collect per-iteration metric values.
    """
    registration_method = sitk.ImageRegistrationMethod()

    _configure_metric(registration_method, metric_name, bins, sampling_percentage)
    _configure_optimizer(
        registration_method,
        learning_rate=learning_rate,
        min_step=min_step,
        number_of_iterations=number_of_iterations,
    )

    initial_transform = _make_initial_transform(fixed, moving, transform_type=transform_type)

    if perturb_init:
        initial_transform = _perturb_transform(
            initial_transform,
            translation_std_mm=perturb_translation_std_mm,
            rotation_std_deg=perturb_rotation_std_deg,
            seed=seed,
        )

    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    metric_values: List[float] = []
    optimizer_positions: List[List[float]] = []

    pbar = tqdm(total=number_of_iterations, desc=f"Rigid {metric_name}, bins={bins}", leave=False)

    def _iteration_callback() -> None:
        metric_values.append(float(registration_method.GetMetricValue()))
        optimizer_positions.append(
            [float(x) for x in registration_method.GetOptimizerPosition()]
        )
        pbar.update(1)
        if len(metric_values) > 0:
            pbar.set_postfix(metric=f"{metric_values[-1]:.5f}")

    registration_method.AddCommand(sitk.sitkIterationEvent, _iteration_callback)

    try:
        final_transform = registration_method.Execute(fixed, moving)
    finally:
        pbar.close()

    results: Dict[str, Any] = {
        "metric_name": metric_name,
        "bins": bins,
        "sampling_percentage": sampling_percentage,
        "learning_rate": learning_rate,
        "min_step": min_step,
        "number_of_iterations": number_of_iterations,
        "stop_condition": registration_method.GetOptimizerStopConditionDescription(),
        "final_metric_value": float(registration_method.GetMetricValue()),
        "final_parameters": [float(x) for x in final_transform.GetParameters()],
        "iteration_metric_values": metric_values,
        "optimizer_positions": optimizer_positions,
    }

    return final_transform, results


def resample_registered_image(
    fixed: sitk.Image,
    moving: sitk.Image,
    transform: sitk.Transform,
    default_value: float = 0.0,
) -> sitk.Image:
    """
    Resample moving image into fixed image space.
    """
    registered = sitk.Resample(
        moving,
        fixed,
        transform,
        sitk.sitkLinear,
        default_value,
        moving.GetPixelID(),
    )
    return registered


def save_transform(transform: sitk.Transform, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteTransform(transform, str(path))


def save_results_json(results: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)