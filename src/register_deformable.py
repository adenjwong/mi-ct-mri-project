from __future__ import annotations

from typing import Any, Dict, List, Sequence

import SimpleITK as sitk
from tqdm import tqdm


def create_bspline_transform(
    fixed: sitk.Image,
    mesh_size: Sequence[int] | None = None,
    order: int = 3,
) -> sitk.BSplineTransform:
    """
    Create a B-spline transform initialized over the fixed image domain.
    Automatically supports 2D and 3D images.
    """
    dim = fixed.GetDimension()

    if mesh_size is None:
        mesh_size = (4, 4) if dim == 2 else (4, 4, 3)

    if len(mesh_size) != dim:
        raise ValueError(
            f"mesh_size length {len(mesh_size)} does not match image dimension {dim}"
        )

    return sitk.BSplineTransformInitializer(
        image1=fixed,
        transformDomainMeshSize=list(mesh_size),
        order=order,
    )


def run_bspline_registration(
    fixed: sitk.Image,
    moving: sitk.Image,
    initial_transform: sitk.Transform | None = None,
    metric_name: str = "mattes_mi",
    bins: int = 32,
    sampling_percentage: float = 0.2,
    learning_rate: float = 1.0,
    min_step: float = 1e-4,
    number_of_iterations: int = 75,
    mesh_size: Sequence[int] | None = None,
) -> tuple[sitk.Transform, Dict[str, Any]]:
    """
    Run deformable B-spline registration, optionally using a rigid transform
    as the moving initial transform.
    """
    if fixed.GetDimension() != moving.GetDimension():
        raise ValueError(
            f"Fixed and moving dimensions do not match: "
            f"{fixed.GetDimension()} vs {moving.GetDimension()}"
        )

    registration_method = sitk.ImageRegistrationMethod()

    metric_name = metric_name.lower()
    if metric_name == "mattes_mi":
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=bins)
    elif metric_name == "joint_hist_mi":
        registration_method.SetMetricAsJointHistogramMutualInformation(
            numberOfHistogramBins=bins,
            varianceForJointPDFSmoothing=1.5,
        )
    else:
        raise ValueError(f"Unsupported metric_name: {metric_name}")

    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(sampling_percentage)
    registration_method.SetInterpolator(sitk.sitkLinear)

    bspline_tx = create_bspline_transform(fixed=fixed, mesh_size=mesh_size, order=3)

    if initial_transform is not None:
        registration_method.SetMovingInitialTransform(initial_transform)

    registration_method.SetInitialTransform(bspline_tx, inPlace=False)

    registration_method.SetOptimizerAsRegularStepGradientDescent(
        learningRate=learning_rate,
        minStep=min_step,
        numberOfIterations=number_of_iterations,
        relaxationFactor=0.5,
        gradientMagnitudeTolerance=1e-8,
    )

    registration_method.SetOptimizerScales(
        [1.0] * len(bspline_tx.GetParameters())
    )

    metric_values: List[float] = []
    optimizer_positions: List[List[float]] = []

    pbar = tqdm(
        total=number_of_iterations,
        desc=f"Deformable {metric_name}, bins={bins}",
        leave=False,
    )

    def _iteration_callback() -> None:
        metric_values.append(float(registration_method.GetMetricValue()))
        optimizer_positions.append(
            [float(x) for x in registration_method.GetOptimizerPosition()]
        )
        pbar.update(1)
        if metric_values:
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
        "mesh_size": list(mesh_size) if mesh_size is not None else None,
        "stop_condition": registration_method.GetOptimizerStopConditionDescription(),
        "final_metric_value": float(registration_method.GetMetricValue()),
        "final_parameters": [float(x) for x in final_transform.GetParameters()],
        "iteration_metric_values": metric_values,
        "optimizer_positions": optimizer_positions,
    }

    return final_transform, results


def compose_transforms(
    rigid_transform: sitk.Transform,
    deformable_transform: sitk.Transform,
) -> sitk.CompositeTransform:
    """
    Compose rigid transform followed by deformable transform.
    Automatically supports 2D and 3D.
    """
    dim = rigid_transform.GetDimension()
    if deformable_transform.GetDimension() != dim:
        raise ValueError(
            f"Transform dimensions do not match: "
            f"{dim} vs {deformable_transform.GetDimension()}"
        )

    composite = sitk.CompositeTransform(dim)
    composite.AddTransform(rigid_transform)
    composite.AddTransform(deformable_transform)
    return composite