import SimpleITK as sitk
import argparse

def print_image_info(name, img):
    print(f"{name}:")
    print("  Size:", img.GetSize())
    print("  Spacing:", img.GetSpacing())
    print("  Origin:", img.GetOrigin())
    print("  Direction:", img.GetDirection())
    print("  PixelID:", img.GetPixelIDTypeAsString())
    print("  Components:", img.GetNumberOfComponentsPerPixel())
    print()

parser = argparse.ArgumentParser()
parser.add_argument("--ct", required=True)
parser.add_argument("--mri", required=True)
args = parser.parse_args()

ct = sitk.ReadImage(args.ct)
mri = sitk.ReadImage(args.mri)

print_image_info("CT", ct)
print_image_info("MRI", mri)