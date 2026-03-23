from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk


def save_metric_curve(metric_values: Sequence[float], out_path: str | Path, title: str = "Metric vs Iteration") -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.plot(metric_values)
    plt.xlabel("Iteration")
    plt.ylabel("Metric value")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _extract_middle_slice(image: sitk.Image, axis: int = 0) -> np.ndarray:
    """
    Returns a middle slice from a 3D image as a 2D numpy array.
    axis=0 means sagittal in array order [z, y, x] after GetArrayFromImage.
    """
    arr = sitk.GetArrayFromImage(image)

    if arr.ndim != 3:
        raise ValueError("Expected a 3D image.")

    if axis == 0:
        idx = arr.shape[2] // 2
        slice_2d = arr[:, :, idx]
    elif axis == 1:
        idx = arr.shape[1] // 2
        slice_2d = arr[:, idx, :]
    elif axis == 2:
        idx = arr.shape[0] // 2
        slice_2d = arr[idx, :, :]
    else:
        raise ValueError("axis must be 0, 1, or 2.")

    return slice_2d


def save_overlay_figure(
    fixed: sitk.Image,
    moving_before: sitk.Image,
    moving_after: sitk.Image,
    out_path: str | Path,
    axis: int = 0,
) -> None:
    """
    Save a 3-panel figure:
    - fixed slice
    - before overlay
    - after overlay
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fixed_slice = _extract_middle_slice(fixed, axis=axis)
    before_slice = _extract_middle_slice(moving_before, axis=axis)
    after_slice = _extract_middle_slice(moving_after, axis=axis)

    def _normalize(a: np.ndarray) -> np.ndarray:
        a = a.astype(np.float32)
        if np.isclose(a.max() - a.min(), 0.0):
            return np.zeros_like(a)
        return (a - a.min()) / (a.max() - a.min())

    fixed_slice = _normalize(fixed_slice)
    before_slice = _normalize(before_slice)
    after_slice = _normalize(after_slice)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(fixed_slice, cmap="gray")
    plt.title("Fixed (CT)")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    overlay_before = np.dstack([fixed_slice, before_slice, np.zeros_like(fixed_slice)])
    plt.imshow(overlay_before)
    plt.title("Before Registration")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    overlay_after = np.dstack([fixed_slice, after_slice, np.zeros_like(fixed_slice)])
    plt.imshow(overlay_after)
    plt.title("After Registration")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()