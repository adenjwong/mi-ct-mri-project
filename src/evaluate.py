from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import SimpleITK as sitk


def _safe_histogram(arr: np.ndarray, bins: int, value_range: Tuple[float, float] | None = None) -> np.ndarray:
    hist, _ = np.histogram(arr, bins=bins, range=value_range)
    hist = hist.astype(np.float64)
    hist /= max(hist.sum(), 1.0)
    return hist


def _safe_joint_histogram(
    arr1: np.ndarray,
    arr2: np.ndarray,
    bins: int,
    range1: Tuple[float, float] | None = None,
    range2: Tuple[float, float] | None = None,
) -> np.ndarray:
    hist2d, _, _ = np.histogram2d(arr1, arr2, bins=bins, range=[range1, range2])
    hist2d = hist2d.astype(np.float64)
    hist2d /= max(hist2d.sum(), 1.0)
    return hist2d


def entropy_from_probabilities(p: np.ndarray, eps: float = 1e-12) -> float:
    p = p[p > 0]
    return float(-np.sum(p * np.log(p + eps)))


def compute_mi_from_images(
    image1: sitk.Image,
    image2: sitk.Image,
    bins: int = 32,
) -> float:
    arr1 = sitk.GetArrayViewFromImage(image1).ravel().astype(np.float64)
    arr2 = sitk.GetArrayViewFromImage(image2).ravel().astype(np.float64)

    range1 = (float(arr1.min()), float(arr1.max()))
    range2 = (float(arr2.min()), float(arr2.max()))

    p1 = _safe_histogram(arr1, bins=bins, value_range=range1)
    p2 = _safe_histogram(arr2, bins=bins, value_range=range2)
    p12 = _safe_joint_histogram(arr1, arr2, bins=bins, range1=range1, range2=range2)

    h1 = entropy_from_probabilities(p1)
    h2 = entropy_from_probabilities(p2)
    h12 = entropy_from_probabilities(p12.ravel())

    mi = h1 + h2 - h12
    return float(mi)


def compute_nmi_from_images(
    image1: sitk.Image,
    image2: sitk.Image,
    bins: int = 32,
) -> float:
    """
    Post-hoc normalized mutual information:
        NMI = (H(X) + H(Y)) / H(X,Y)
    """
    arr1 = sitk.GetArrayViewFromImage(image1).ravel().astype(np.float64)
    arr2 = sitk.GetArrayViewFromImage(image2).ravel().astype(np.float64)

    range1 = (float(arr1.min()), float(arr1.max()))
    range2 = (float(arr2.min()), float(arr2.max()))

    p1 = _safe_histogram(arr1, bins=bins, value_range=range1)
    p2 = _safe_histogram(arr2, bins=bins, value_range=range2)
    p12 = _safe_joint_histogram(arr1, arr2, bins=bins, range1=range1, range2=range2)

    h1 = entropy_from_probabilities(p1)
    h2 = entropy_from_probabilities(p2)
    h12 = entropy_from_probabilities(p12.ravel())

    if np.isclose(h12, 0.0):
        return float("nan")

    nmi = (h1 + h2) / h12
    return float(nmi)


def dice_score(mask1: sitk.Image, mask2: sitk.Image) -> float:
    """
    Dice score for binary masks.
    """
    arr1 = sitk.GetArrayViewFromImage(mask1) > 0
    arr2 = sitk.GetArrayViewFromImage(mask2) > 0

    intersection = np.logical_and(arr1, arr2).sum()
    denom = arr1.sum() + arr2.sum()

    if denom == 0:
        return 1.0

    return float(2.0 * intersection / denom)


def centroid_mm_from_mask(mask: sitk.Image) -> np.ndarray:
    """
    Compute centroid in physical space from a binary mask.
    """
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(mask > 0)

    if not stats.HasLabel(1):
        raise ValueError("Mask does not contain foreground label 1.")

    centroid = np.array(stats.GetCentroid(1), dtype=np.float64)
    return centroid


def centroid_distance_mm(mask1: sitk.Image, mask2: sitk.Image) -> float:
    c1 = centroid_mm_from_mask(mask1)
    c2 = centroid_mm_from_mask(mask2)
    return float(np.linalg.norm(c1 - c2))


def summarize_registration(
    fixed: sitk.Image,
    registered_moving: sitk.Image,
    bins: int = 32,
) -> Dict[str, float]:
    """
    Basic post-hoc intensity-based summary.
    """
    mi = compute_mi_from_images(fixed, registered_moving, bins=bins)
    nmi = compute_nmi_from_images(fixed, registered_moving, bins=bins)

    return {
        "posthoc_mi": float(mi),
        "posthoc_nmi": float(nmi),
    }