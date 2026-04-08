from __future__ import annotations

from pathlib import Path
from typing import Tuple

import SimpleITK as sitk


from pathlib import Path
import SimpleITK as sitk


def rgb_to_grayscale(img: sitk.Image) -> sitk.Image:
    c0 = sitk.VectorIndexSelectionCast(img, 0, sitk.sitkFloat32)
    c1 = sitk.VectorIndexSelectionCast(img, 1, sitk.sitkFloat32)
    c2 = sitk.VectorIndexSelectionCast(img, 2, sitk.sitkFloat32)
    return sitk.Cast((c0 + c1 + c2) / 3.0, sitk.sitkFloat32)


def load_image(path: str | Path, pixel_type: int = sitk.sitkFloat32) -> sitk.Image:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    image = sitk.ReadImage(str(path))

    if image.GetNumberOfComponentsPerPixel() > 1:
        image = rgb_to_grayscale(image)
    else:
        image = sitk.Cast(image, pixel_type)

    return image


def load_fixed_moving(ct_path: str | Path, mri_path: str | Path) -> Tuple[sitk.Image, sitk.Image]:
    """
    Load CT as fixed image and MRI as moving image.
    """
    fixed = load_image(ct_path)
    moving = load_image(mri_path)
    return fixed, moving


def print_image_info(name: str, image: sitk.Image) -> None:
    """
    Print useful debugging info about an image.
    """
    print(f"{name}:")
    print(f"  Size: {image.GetSize()}")
    print(f"  Spacing: {image.GetSpacing()}")
    print(f"  Origin: {image.GetOrigin()}")
    print(f"  Direction: {image.GetDirection()}")
    print(f"  PixelID: {image.GetPixelIDTypeAsString()}")