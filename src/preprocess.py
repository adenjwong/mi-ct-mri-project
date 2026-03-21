from __future__ import annotations

from typing import Tuple

import numpy as np
import SimpleITK as sitk


def clamp_intensity(image: sitk.Image, lower: float, upper: float) -> sitk.Image:
    """
    Clamp intensities to a fixed range.
    Useful for CT to reduce outlier influence.
    """
    return sitk.Clamp(image, image.GetPixelID(), lowerBound=lower, upperBound=upper)


def normalize_to_0_1(image: sitk.Image) -> sitk.Image:
    """
    Normalize image intensities to [0, 1].
    """
    arr = sitk.GetArrayViewFromImage(image).astype(np.float32)
    min_val = float(arr.min())
    max_val = float(arr.max())
    if np.isclose(max_val - min_val, 0.0):
        out = sitk.Image(image)
        out = sitk.Cast(out, sitk.sitkFloat32)
        out = out * 0.0
        return out
    norm = (image - min_val) / (max_val - min_val)
    return sitk.Cast(norm, sitk.sitkFloat32)


def zscore_normalize(image: sitk.Image) -> sitk.Image:
    """
    Z-score normalize intensities.
    """
    arr = sitk.GetArrayFromImage(image).astype(np.float32)
    mean = float(arr.mean())
    std = float(arr.std())
    if np.isclose(std, 0.0):
        out = sitk.Image(image)
        out = sitk.Cast(out, sitk.sitkFloat32)
        out = out * 0.0
        return out
    out = (image - mean) / std
    return sitk.Cast(out, sitk.sitkFloat32)


def resample_to_reference(
    moving: sitk.Image,
    reference: sitk.Image,
    interpolator: int = sitk.sitkLinear,
    default_value: float = 0.0,
) -> sitk.Image:
    """
    Resample moving image onto the reference image grid using identity transform.
    Useful for quick grid harmonization before registration.
    """
    identity = sitk.Transform(reference.GetDimension(), sitk.sitkIdentity)
    return sitk.Resample(
        moving,
        reference,
        identity,
        interpolator,
        default_value,
        moving.GetPixelID(),
    )


def preprocess_ct_mri(
    fixed_ct: sitk.Image,
    moving_mri: sitk.Image,
    ct_clamp: Tuple[float, float] | None = (-1000.0, 2000.0),
    normalization: str = "0_1",
    match_grid: bool = False,
) -> Tuple[sitk.Image, sitk.Image]:
    """
    Basic preprocessing:
    - optional CT clamping
    - intensity normalization
    - optional resampling of moving image to fixed grid

    normalization options:
    - "0_1"
    - "zscore"
    - "none"
    """
    fixed = sitk.Image(fixed_ct)
    moving = sitk.Image(moving_mri)

    if ct_clamp is not None:
        fixed = clamp_intensity(fixed, ct_clamp[0], ct_clamp[1])

    if normalization == "0_1":
        fixed = normalize_to_0_1(fixed)
        moving = normalize_to_0_1(moving)
    elif normalization == "zscore":
        fixed = zscore_normalize(fixed)
        moving = zscore_normalize(moving)
    elif normalization == "none":
        pass
    else:
        raise ValueError(f"Unsupported normalization: {normalization}")

    if match_grid:
        moving = resample_to_reference(moving, fixed)

    return fixed, moving