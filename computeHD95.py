import os
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_erosion, generate_binary_structure
from scipy.spatial import cKDTree
from pathlib import Path

def get_surface_points(mask):
    """获取二值 mask 的表面点坐标"""
    struct = generate_binary_structure(3, 3)
    eroded = binary_erosion(mask, struct)
    surface = mask.astype(np.uint8) - eroded.astype(np.uint8)
    points = np.argwhere(surface > 0)
    return points

def compute_hd95(pred, gt, percentile=95):
    """计算 95% Hausdorff 距离"""
    pred = (pred > 0).astype(np.uint8)
    
    pred = np.transpose(pred, (2, 1, 0))  # PI-CAI时候必须取消注释
    
    gt = (gt > 0).astype(np.uint8)
    # print(f"Pred shape: {pred.shape}, GT shape: {gt.shape}")

    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        return float(np.sqrt(sum(np.square(pred.shape))))

    pred_surface = get_surface_points(pred)
    gt_surface = get_surface_points(gt)

    if len(pred_surface) == 0 or len(gt_surface) == 0:
        return 0.0

    tree_gt = cKDTree(gt_surface)
    tree_pred = cKDTree(pred_surface)

    dist_pred_to_gt, _ = tree_gt.query(pred_surface, k=1)
    dist_gt_to_pred, _ = tree_pred.query(gt_surface, k=1)

    all_distances = np.concatenate([dist_pred_to_gt, dist_gt_to_pred])
    hd95 = np.percentile(all_distances, percentile)
    return float(hd95)

def main():
    pred_dir = Path("infer/SegTumor_DIY_chengda_Backbone_infer/val_results")
    gt_dir = Path("dataset/ChengdaOnlyCSPca/nnUNet_val/labelsTs")
    
    # pred_dir = Path("infer/SegTumor_DIY_PICAI_New_CNN_Encoder_infer/val_results")
    # gt_dir = Path("dataset/PI-CAI/labelsTs")
    
    

    pred_files = sorted(pred_dir.glob("*_pred.nii.gz"))
    if len(pred_files) == 0:
        print(f"No prediction files found in {pred_dir}")
        return

    hd95_list = []

    print("Case-wise HD95 values:")
    for pred_file in pred_files:
        name = pred_file.stem 
        name = Path(name).stem.replace("_pred", "") 
        gt_file = gt_dir / f"{name}.nii.gz"

        if not gt_file.exists():
            print(f"Warning: GT file not found for {name}")
            continue

        pred_data = sitk.GetArrayFromImage(sitk.ReadImage(pred_file))
        gt_data = sitk.GetArrayFromImage(sitk.ReadImage(gt_file))

        hd95 = compute_hd95(pred_data, gt_data)
        print(f"{name}: HD95 = {hd95:.4f} mm")
        hd95_list.append(hd95)

    if hd95_list:
        mean_hd95 = np.mean(hd95_list)
        std_hd95 = np.std(hd95_list)
        print(f"\nAverage HD95: {mean_hd95:.4f} mm")
        print(f"Standard Deviation of HD95: {std_hd95:.4f} mm")
    else:
        print("No HD95 values computed.")

if __name__ == "__main__":
    main()