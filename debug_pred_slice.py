#!/usr/bin/env python3
"""\
Debug script: load a single prediction NIfTI, print its shape, and visualize one slice.

Target file (hard-coded for now):
  infer/SegTumor_DIY_PICAI_New_CNN_Encoder_infer/val_results/10012_1000012_pred.nii.gz

Usage:
  python debug_pred_slice.py
"""

from pathlib import Path

import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt


def main():
    # Hard-coded path as requested
    pred_path = Path("infer/SegTumor_DIY_PICAI_New_CNN_Encoder_infer/val_results/10012_1000012_pred.nii.gz")

    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")

    # Read with SimpleITK and convert to numpy
    img = sitk.ReadImage(str(pred_path))
    print(img.GetSize())
    arr = sitk.GetArrayFromImage(img)  # SimpleITK default: [D, H, W]
    # arr = np.transpose(arr, (2, 1, 0))
    print(f"Loaded prediction from: {pred_path}")
    print(f"Array shape: {arr.shape}")
    print(f"Array dtype: {arr.dtype}")
    print(f"Unique values: {np.unique(arr)}")

    # Visualize slice index 4 (0-based)
    slice_idx = 4
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {arr.shape}")

    depth = arr.shape[0]
    if not (0 <= slice_idx < depth):
        raise ValueError(f"slice_idx {slice_idx} out of range for depth {depth}")

    slice_img = arr[slice_idx]

    # Prepare output directory and file name
    out_dir = Path("debug_visuals")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{pred_path.stem}_slice_{slice_idx:02d}.png"

    plt.figure(figsize=(5, 5))
    plt.imshow(slice_img, cmap="gray")
    plt.title(f"Pred slice {slice_idx} (shape {arr.shape})")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.1, dpi=150)
    plt.close()

    print(f"Saved slice image to: {out_path}")


if __name__ == "__main__":
    main()
