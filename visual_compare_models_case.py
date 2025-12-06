#!/usr/bin/env python3
"""
Visualize multi-model predictions for a single case and single slice.

Configuration is provided via a JSON file, e.g.:

{
  "case_id": "10005_1000005",
  "slice_index": 6,

  "modalities": {
    "t2w": "./dataset/.../10005_1000005_0000.nii.gz",
    "adc": "./dataset/.../10005_1000005_0001.nii.gz",
    "dwi": "./dataset/.../10005_1000005_0002.nii.gz"
  },

  "gt_path": "./dataset/.../10005_1000005.nii.gz",

  "models": [
    {"name": "Ours",       "pred_path": "./infer/.../10005_1000005_pred.nii.gz"},
    {"name": "ALIEN_Net",  "pred_path": "./infer/.../10005_1000005_pred.nii.gz"},
    {"name": "UNet",       "pred_path": "./infer/.../10005_1000005_pred.nii.gz"}
  ],

  "output": {
    "save_dir": "./visual_compare",
    "filename": "case_10005_slice_06_compare.png"
  }
}

Usage:
  python visual_compare_models_case.py --config config_case_10005.json

Layout:
  - Column 1: ADC + red GT contours
  - Column 2: DWI + red GT contours
  - Column 3: T2W + red GT contours
  - Columns 4..(3+N): T2W + red GT + yellow Pred for each model
"""

import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from skimage import measure


def load_nii_as_numpy(path: Path) -> np.ndarray:
    """Load NIfTI file and return numpy array.

    By default, SimpleITK returns arrays in [D, H, W] order, but some
    external tools may save volumes as [H, W, D]. Here we only load and
    return the raw array; any axis correction that depends on an expected
    depth is handled in the caller.
    """
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)
    return arr


def draw_contours(ax, mask_slice: np.ndarray, color: str, label: str | None = None, linewidth: float = 2.0):
    """Draw binary mask contours on an axis using skimage.measure.find_contours.

    mask_slice: 2D array
    color: matplotlib color string, e.g. 'r', 'y'
    label: legend label (set only once per axis)
    """
    if mask_slice is None:
        return

    mask_bool = mask_slice.astype(bool)
    if mask_bool.max() == 0:
        # No positive voxels, skip
        return

    contours = measure.find_contours(mask_bool, 0.5)
    # Avoid duplicate legend labels
    existing_labels = ax.get_legend_handles_labels()[1]
    label_to_use = label if (label and label not in existing_labels) else ""

    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], color + '-', linewidth=linewidth, label=label_to_use)
        # Only label the first contour
        label_to_use = ""


def visualize_case_from_config(config_path: str) -> Path:
    config_path = Path(config_path)
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    case_id = cfg.get("case_id", "unknown_case")
    slice_index = int(cfg["slice_index"])  # required

    modalities = cfg["modalities"]
    t2w_path = Path(modalities["t2w"])
    adc_path = Path(modalities["adc"])
    dwi_path = Path(modalities["dwi"])

    gt_path = Path(cfg["gt_path"])

    models = cfg.get("models", [])
    if len(models) == 0:
        raise ValueError("No models provided in config['models']")

    output_cfg = cfg.get("output", {})
    save_dir = Path(output_cfg.get("save_dir", "./visual_compare"))
    filename = output_cfg.get("filename", f"{case_id}_slice_{slice_index:02d}_compare.png")
    # Whether to draw titles and legend on the figure
    show_titles = bool(output_cfg.get("show_titles", True))
    save_dir.mkdir(parents=True, exist_ok=True)
    output_path = save_dir / filename

    # Load volumes
    t2w = load_nii_as_numpy(t2w_path)  # expected [D, H, W]
    adc = load_nii_as_numpy(adc_path)
    dwi = load_nii_as_numpy(dwi_path)
    gt = load_nii_as_numpy(gt_path)

    # Safety checks on slice index and fix potential axis order issues
    num_slices = t2w.shape[0]
    if not (0 <= slice_index < num_slices):
        raise ValueError(f"slice_index {slice_index} out of range for T2W depth {num_slices}")

    # Ensure GT has same depth as modalities; if not, try to fix [H,W,D] -> [D,H,W]
    if gt.shape[0] != num_slices and gt.shape[-1] == num_slices:
        gt = np.transpose(gt, (2, 1, 0))
    if gt.shape[0] != num_slices:
        raise ValueError(f"GT depth {gt.shape[0]} does not match T2W depth {num_slices}")

    # Extract slices
    adc_slice = adc[slice_index]
    dwi_slice = dwi[slice_index]
    t2w_slice = t2w[slice_index]
    gt_slice = (gt[slice_index] > 0).astype(np.uint8)

    # Load all model predictions and extract same slice (using T2W as background)
    model_slices = []  # list of (model_name, pred_slice)
    for m in models:
        name = m["name"]
        pred_path = Path(m["pred_path"])
        pred_vol = load_nii_as_numpy(pred_path)

        # If depth is not on the first axis but matches on the last axis,
        # assume [H, W, D] and transpose to [D, H, W].
        if pred_vol.shape[0] != num_slices and pred_vol.shape[-1] == num_slices:
            pred_vol = np.transpose(pred_vol, (2, 1, 0))

        if pred_vol.shape[0] != num_slices:
            raise ValueError(
                f"Pred volume depth {pred_vol.shape[0]} for model '{name}' does not match T2W depth {num_slices}"
            )

        pred_slice = (pred_vol[slice_index] > 0).astype(np.uint8)
        model_slices.append((name, pred_slice))

    # Figure layout: 3 modalities + N models (all using T2W background on the right)
    num_models = len(model_slices)
    n_cols = 3 + num_models
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))

    if n_cols == 1:
        axes = [axes]

    # Column 1-3: ADC / DWI / T2W with red GT only
    modality_slices = [
        ("ADC", adc_slice),
        ("DWI", dwi_slice),
        ("T2W", t2w_slice),
    ]

    for ax, (mod_name, img_slice) in zip(axes[:3], modality_slices):
        ax.imshow(img_slice, cmap="gray")
        draw_contours(ax, gt_slice, color='r', label='GT' if show_titles else None)
        if show_titles:
            ax.set_title(f"{mod_name} (slice {slice_index})", fontsize=10)
        ax.axis("off")

    # Model columns: T2W background + red GT + yellow Pred
    for idx, (model_name, pred_slice) in enumerate(model_slices):
        ax = axes[3 + idx]
        ax.imshow(t2w_slice, cmap="gray")
        draw_contours(ax, gt_slice, color='r', label='GT' if show_titles else None)
        draw_contours(ax, pred_slice, color='y', label='Pred' if show_titles else None)
        if show_titles:
            ax.set_title(model_name, fontsize=10)
        ax.axis("off")

    # Add a global legend (use the first axis that has labels) if enabled
    if show_titles:
        handles, labels = axes[0].get_legend_handles_labels()
        if not labels:
            # If first axis has no labels, search others
            for ax in axes:
                handles, labels = ax.get_legend_handles_labels()
                if labels:
                    break
        if labels:
            fig.legend(handles, labels, loc="lower right", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.1, dpi=150)
    plt.close(fig)

    print(f"Saved comparison figure for case {case_id}, slice {slice_index} -> {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Visualize multi-model predictions for a single case and slice")
    parser.add_argument("--config", type=str, default="compare_sota_template_chengda.json", required=False, help="Path to JSON config file")
    args = parser.parse_args()

    visualize_case_from_config(args.config)


if __name__ == "__main__":
    main()
