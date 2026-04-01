from __future__ import annotations
from typing import Tuple
import SimpleITK as sitk


def crop_center_fraction(
    image: sitk.Image,
    frac_x: float = 0.5,
    frac_y: float = 0.5,
    frac_z: float = 0.5,
) -> sitk.Image:
    """
    Crop a centered ROI by keeping the given fraction of the image size
    in each dimension.
    """
    size = image.GetSize()

    new_size = [
        max(1, int(size[0] * frac_x)),
        max(1, int(size[1] * frac_y)),
        max(1, int(size[2] * frac_z)),
    ]

    start = [
        max(0, (size[0] - new_size[0]) // 2),
        max(0, (size[1] - new_size[1]) // 2),
        max(0, (size[2] - new_size[2]) // 2),
    ]

    roi = sitk.RegionOfInterest(image, size=new_size, index=start)
    return roi


def crop_with_index_size(
    image: sitk.Image,
    index: Tuple[int, int, int],
    size: Tuple[int, int, int],
) -> sitk.Image:
    return sitk.RegionOfInterest(image, size=list(size), index=list(index))