#!/usr/bin/env python3

"""Visualize Grad-CAM heatmaps for multiple models on the same module.

Configuration is done directly in this file (no JSON):

- Specify case_id, slice_index, and paths for T2W/ADC/DWI/GT.
- Specify a list of models, each with:
  - name: label used in the figure
  - checkpoint: path to .pth file
  - model_type: string key to choose which builder to use
  - target_layer: name of the module to visualize (e.g. "decoder1")

The layout:
- Column 1-3: ADC / DWI / T2W with red GT contours only.
- Column 4..(3+N): Grad-CAM heatmap for each model (no GT / pred contours).
"""

from pathlib import Path

import numpy as np
import torch
import SimpleITK as sitk
import matplotlib.pyplot as plt
from skimage import measure
import cv2

from monai.transforms import ClipIntensityPercentiles
from pytorch_grad_cam import GradCAM

from Model.as_unetr import CDSA_Net
from Model.ablation import Backbone_Baseline, Backbone_SAEB, Backbone_ACF, Backbone_MRE, Backbone_MRE_ACF
import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='timm')


clip = ClipIntensityPercentiles(lower=0.5, upper=99.5, channel_wise=False)

def fix_cdsa_net_checkpoint(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if 'gre_dcgf_' in key:
            # Replace 'gre_dcgf_X' with 'BiCR_X'
            new_key = key.replace('gre_dcgf_', 'BiCR_')
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict


def load_nii(path: Path) -> np.ndarray:
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)
    return arr


def z_score_normalization(img: np.ndarray) -> np.ndarray:
    mask = img > 0
    if np.any(mask):
        mean_val = np.mean(img[mask])
        std_val = np.std(img[mask])
    else:
        mean_val = np.mean(img)
        std_val = np.std(img)
    return (img - mean_val) / (std_val + 1e-8)


def preprocess_modalities(t2w_path: Path, adc_path: Path, dwi_path: Path, gt_path: Path):
    t2w_raw = load_nii(t2w_path).astype(np.float32)
    adc_raw = load_nii(adc_path).astype(np.float32)
    dwi_raw = load_nii(dwi_path).astype(np.float32)
    gt_raw = load_nii(gt_path).astype(np.float32)

    gt_raw[gt_raw > 0] = 1.0

    t2w = z_score_normalization(clip(t2w_raw))
    adc = z_score_normalization(clip(adc_raw))
    dwi = z_score_normalization(clip(dwi_raw))

    t2w = t2w[np.newaxis, :]
    adc = adc[np.newaxis, :]
    dwi = dwi[np.newaxis, :]
    gt = gt_raw[np.newaxis, :]

    return adc, dwi, t2w, gt


class SemanticSegmentationTarget:
    def __init__(self, category, mask):
        self.category = category
        self.mask = torch.from_numpy(mask)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()
        
    def __call__(self, model_output):
        return (model_output[self.category, :, : ] * self.mask).sum()


def build_model(model_type: str, device: torch.device) -> torch.nn.Module:
    """Factory for different model architectures.

    You can extend this function with if/elif branches for other model types.
    """

    if model_type == "Backbone_Baseline":
        model = Backbone_Baseline(
            in_channels=2,  # ADC and DWI modalities
            out_channels=2,  # Background and lesion
            img_size=(16, 256, 256),
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=8,
            norm_name="instance",
        ).to(device)
        return model
    elif model_type == "Backbone_MRE":
        model = Backbone_MRE(
            in_channels=2,  # ADC and DWI modalities
            out_channels=2,  # Background and lesion
            img_size=(16, 256, 256),
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=8,
            norm_name="instance",
        ).to(device)
        return model
    elif model_type == "Backbone_ACF":
        model = Backbone_ACF(
            in_channels=2,  # ADC and DWI modalities
            out_channels=2,  # Background and lesion
            img_size=(16, 256, 256),
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=8,
            norm_name="instance",
        ).to(device)
        return model
    elif model_type == "Backbone_SAEB":
        model = Backbone_SAEB(
            in_channels=2,  # ADC and DWI modalities
            out_channels=2,  # Background and lesion
            img_size=(16, 256, 256),
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=8,
            norm_name="instance",
        ).to(device)
        return model
    elif model_type == "Backbone_MRE_ACF":
        model = Backbone_MRE_ACF(
            in_channels=2,  # ADC and DWI modalities
            out_channels=2,  # Background and lesion
            img_size=(16, 256, 256),
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=8,
            norm_name="instance",
        ).to(device)
        return model
    elif model_type == "CDSA_Net":
        model = CDSA_Net(
            in_channels=2,  # ADC and DWI modalities
            out_channels=2,  # Background and lesion
            img_size=(16, 256, 256),
            feature_size=16,
            hidden_size=768, 
            mlp_dim=3072,
            num_heads=8,
            norm_name='instance'
        ).to(device)
        return model
    raise ValueError(f"Unknown model_type: {model_type}")


# ============================
# User configuration (edit here)
# ============================

CASE_ID = "TianGuoHong"
SLICE_INDEX = 9

# Visualization crop size around GT center (in pixels). Set to None to disable cropping.
CROP_SIZE = 96

T2W_PATH = Path(f"dataset/ChengdaOnlyCSPca/nnUNet_val/imagesTs/{CASE_ID}_0000.nii.gz")
ADC_PATH = Path(f"dataset/ChengdaOnlyCSPca/nnUNet_val/imagesTs/{CASE_ID}_0001.nii.gz")
DWI_PATH = Path(f"dataset/ChengdaOnlyCSPca/nnUNet_val/imagesTs/{CASE_ID}_0002.nii.gz")
GT_PATH = Path(f"dataset/ChengdaOnlyCSPca/nnUNet_val/labelsTs/{CASE_ID}.nii.gz")

OUTPUT_DIR = Path("./gradcam_compare")
OUTPUT_FILENAME = f"Chengda-{CASE_ID}_slice{SLICE_INDEX:02d}_gradcam.png"
SHOW_TITLES = False

# Multiple models configuration: same module name can be reused across models.
# For different architectures, add appropriate model_type and extend build_model.
MODELS_CFG = [
    {
        "name": "Baseline",
        "checkpoint": "checkpoints/SegTumor_DIY_chengda_Backbone/model_epoch_300.pth",
        "model_type": "Backbone_Baseline",
        "target_layer": "decoder1",
    },
    {
        "name": "Backbone_MRE",
        "checkpoint": "checkpoints/SegTumor_DIY_chengda_Backbone_MRE/model_epoch_270.pth",
        "model_type": "Backbone_MRE",
        "target_layer": "decoder1",
    },
    {
        "name": "Backbone_ACF",
        "checkpoint": "checkpoints/SegTumor_DIY_chengda_Backbone_ACF/model_epoch_270.pth",
        "model_type": "Backbone_ACF",
        "target_layer": "decoder1",
    },
    {
        "name": "Backbone_SAEB",
        "checkpoint": "checkpoints/SegTumor_DIY_chengda_Backbone_SAEB/model_latest.pth",
        "model_type": "Backbone_SAEB",
        "target_layer": "decoder1",
    },
    {
        "name": "Backbone_MRE_ACF",
        "checkpoint": "checkpoints/SegTumor_DIY_chengda_Backbone_MRE_ACF/best_dice_model.pth",
        "model_type": "Backbone_MRE_ACF",
        "target_layer": "decoder1",
    },
    {
        "name": "ADSA-Net",
        "checkpoint": "checkpoints/SegTumor_DIY_New_CNN_Encoder/best_dice_model.pth",
        "model_type": "CDSA_Net",
        "target_layer": "decoder1",
    },
    
    # Baseline, Backbone_SAEB, Backbone_ACF, Backbone_MRE, Backbone_MRE_ACF, ADSA-Net

]  


def visualize_case() -> Path:
    case_id = CASE_ID
    slice_index = SLICE_INDEX

    t2w_path = T2W_PATH
    adc_path = ADC_PATH
    dwi_path = DWI_PATH

    gt_path = GT_PATH

    models_cfg = MODELS_CFG
    if len(models_cfg) == 0:
        raise ValueError("MODELS_CFG is empty; please configure at least one model.")

    # All models should share the same target_layer name for comparability.
    target_layer_name = models_cfg[0]["target_layer"]

    save_dir = OUTPUT_DIR
    filename = OUTPUT_FILENAME
    show_titles = SHOW_TITLES
    save_dir.mkdir(parents=True, exist_ok=True)
    output_path = save_dir / filename

    adc, dwi, t2w, gt = preprocess_modalities(t2w_path, adc_path, dwi_path, gt_path)

    num_slices = t2w.shape[1]
    if not (0 <= slice_index < num_slices):
        raise ValueError(f"slice_index {slice_index} out of range for depth {num_slices}")

    adc_slice = adc[0, slice_index]
    dwi_slice = dwi[0, slice_index]
    t2w_slice = t2w[0, slice_index]
    gt_slice = (gt[0, slice_index] > 0).astype(np.uint8)

    # Determine crop around GT center if enabled (center computed on full-size GT)
    def crop_around_center(arr: np.ndarray, cy: int, cx: int, size: int) -> np.ndarray:
        h, w = arr.shape
        half = size // 2
        y1 = max(0, cy - half)
        y2 = min(h, cy + half)
        x1 = max(0, cx - half)
        x2 = min(w, cx + half)
        return arr[y1:y2, x1:x2]

    use_crop = CROP_SIZE is not None and CROP_SIZE > 0
    crop_center = None
    if use_crop and gt_slice.max() > 0:
        ys, xs = np.where(gt_slice > 0)
        cy = int(ys.mean())
        cx = int(xs.mean())
        crop_center = (cy, cx)

        adc_slice = crop_around_center(adc_slice, cy, cx, CROP_SIZE)
        dwi_slice = crop_around_center(dwi_slice, cy, cx, CROP_SIZE)
        t2w_slice = crop_around_center(t2w_slice, cy, cx, CROP_SIZE)
        gt_slice = crop_around_center(gt_slice, cy, cx, CROP_SIZE)

    image_data = np.concatenate([adc, dwi, t2w], axis=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_tensor = torch.from_numpy(image_data).unsqueeze(0).to(device)

    mask_for_target = gt[0].astype(np.float32)
    targets_for_cam = [SemanticSegmentationTarget(1, gt.astype(np.float32))]

    cam_results = []

    for m in models_cfg:
        model_name = m["name"]
        ckpt_path = Path(m["checkpoint"])
        model_type = m["model_type"]
        target_layer_name = m["target_layer"]

        model = build_model(model_type, device)
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
        
        if model_type == "CDSA_Net":
            checkpoint = fix_cdsa_net_checkpoint(checkpoint["model_state_dict"])
            fixed_checkpoint = fix_cdsa_net_checkpoint(checkpoint)
            model.load_state_dict(fixed_checkpoint, strict=False)
        else:
            model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        with GradCAM(model=model, target_layers=[model.decoder1]) as cam:
            grayscale_cam = cam(input_tensor=image_tensor, targets=targets_for_cam)[0, :]

        if grayscale_cam.shape[0] != num_slices:
            raise ValueError(
                f"Grad-CAM depth {grayscale_cam.shape[0]} does not match image depth {num_slices}"
            )

        cam_slice = grayscale_cam[slice_index]

        # Apply same crop to CAM slice using the precomputed GT center
        if use_crop and crop_center is not None:
            cy, cx = crop_center
            cam_slice = crop_around_center(cam_slice, cy, cx, CROP_SIZE)
        cam_results.append((model_name, cam_slice))

    n_models = len(cam_results)
    n_cols = 3 + n_models
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))

    if n_cols == 1:
        axes = [axes]

    modality_slices = [
        ("ADC", adc_slice),
        ("DWI", dwi_slice),
        ("T2W", t2w_slice),
    ]

    for ax, (mod_name, img_slice) in zip(axes[:3], modality_slices):
        ax.imshow(img_slice, cmap="gray")
        if gt_slice.max() > 0:
            contours = measure.find_contours(gt_slice, 0.5)
            for contour in contours:
                ax.plot(contour[:, 1], contour[:, 0], "r-", linewidth=1.5)
        if show_titles:
            ax.set_title(f"{mod_name} (slice {slice_index})", fontsize=10)
        ax.axis("off")

    for idx, (model_name, cam_slice) in enumerate(cam_results):
        ax = axes[3 + idx]
        im = ax.imshow(cam_slice, cmap="jet")
        if show_titles:
            ax.set_title(model_name, fontsize=10)
        ax.axis("off")

    fig.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.1, dpi=150)
    plt.close(fig)

    print(
        f"Saved Grad-CAM comparison for case {case_id}, slice {slice_index}, "
        f"layer {target_layer_name} -> {output_path}"
    )
    return output_path


if __name__ == "__main__":
    visualize_case()
