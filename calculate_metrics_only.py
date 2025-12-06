#!/usr/bin/env python3
import os
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
import argparse
import json
from itertools import permutations
from scipy.ndimage import binary_erosion, generate_binary_structure
from scipy.spatial import cKDTree
import SimpleITK as sitk


parser = argparse.ArgumentParser(description='Calculate metrics from prediction results')
parser.add_argument('--pred_dir', type=str, required=False, default="infer/SegTumor_DIY_PICAI_DWIAsLeading_infer/val_results",help='Directory containing prediction nii.gz files')

# parser.add_argument('--gt_dir', type=str, required=False, default="dataset/ChengdaOnlyCSPca/nnUNet_val/labelsTs", help='Directory containing ground truth nii.gz files')
parser.add_argument('--gt_dir', type=str, required=False, default="dataset/PI-CAI/labelsTs", help='Directory containing ground truth nii.gz files')

parser.add_argument('--output_file', type=str, default='metrics_results.json', help='Output JSON file for metrics')
parser.add_argument('--percentile', type=int, default=95, help='Percentile for Hausdorff distance (default: 95)')
parser.add_argument('--reference_json', type=str, default=None, help='Optional: Path to reference JSON file to compare results (e.g., A_Summary.json)')
args = parser.parse_args()


def calculate_dice(pred, target):
    """Calculate Dice coefficient"""
    smooth = 1e-6
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    intersection = (pred_flat * target_flat).sum()
    pred_sum = pred_flat.sum()
    target_sum = target_flat.sum()
    
    dice = (2. * intersection + smooth) / (pred_sum + target_sum + smooth)
    return dice


def calculate_miou(pred, target):
    """Calculate mean Intersection over Union"""
    smooth = 1e-6
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return iou


def get_surface_points(binary_mask):
    """
    Get surface points of a binary mask
    
    Args:
        binary_mask: Binary mask (numpy array)
    
    Returns:
        Array of surface point coordinates
    """
    # Create a structuring element for erosion (26-connectivity for 3D)
    struct = generate_binary_structure(3, 3)
    
    # Erode the mask
    eroded = binary_erosion(binary_mask, struct)
    
    # Surface = original mask - eroded mask
    surface = binary_mask.astype(np.uint8) - eroded.astype(np.uint8)
    
    # Get coordinates of surface points
    surface_points = np.argwhere(surface > 0)
    
    return surface_points


def calculate_hd95(pred, target, percentile=95):
    """
    Calculate 95th percentile Hausdorff Distance (HD95)
    
    HD95 is calculated as:
    1. Get surface points of both prediction and ground truth
    2. For each surface point in pred, find the minimum distance to target surface
    3. For each surface point in target, find the minimum distance to pred surface
    4. Take the 95th percentile of all these distances
    
    Args:
        pred: Binary prediction mask (numpy array)
        target: Binary ground truth mask (numpy array)
        percentile: Percentile for Hausdorff distance (default: 95)
    
    Returns:
        HD95 value (float)
    """
    # Convert to binary if not already
    pred = (pred > 0).astype(np.uint8)
    target = (target > 0).astype(np.uint8)
    
    # Check if both pred and target are empty
    if pred.sum() == 0 and target.sum() == 0:
        return 0.0
    
    # If one is empty and the other is not, return a large distance
    if pred.sum() == 0 or target.sum() == 0:
        # Return a large but finite value (e.g., image diagonal)
        max_distance = np.sqrt(sum(dim**2 for dim in pred.shape))
        return float(max_distance)
    
    try:
        # Get surface points
        pred_surface = get_surface_points(pred)
        target_surface = get_surface_points(target)
        
        if len(pred_surface) == 0 or len(target_surface) == 0:
            # If no surface points, return 0
            return 0.0
        
        # Use KDTree for efficient distance calculation
        # Calculate distances from pred surface to target surface
        target_tree = cKDTree(target_surface)
        distances_pred_to_target, _ = target_tree.query(pred_surface, k=1)
        
        # Calculate distances from target surface to pred surface
        pred_tree = cKDTree(pred_surface)
        distances_target_to_pred, _ = pred_tree.query(target_surface, k=1)
        
        # Combine all distances (cKDTree.query returns numpy array)
        all_distances = np.concatenate([distances_pred_to_target, distances_target_to_pred])
        
        # Calculate 95th percentile
        hd95 = np.percentile(all_distances, percentile)
        
        # Handle NaN or Inf values
        if np.isnan(hd95) or np.isinf(hd95):
            return 0.0
        
        return float(hd95)
        
    except Exception as e:
        print(f"Warning: Error calculating HD95: {e}")
        # Return a default value if calculation fails
        max_distance = np.sqrt(sum(dim**2 for dim in pred.shape))
        return float(max_distance)


def get_patient_name_from_pred_file(pred_file):
    """
    Extract patient name from prediction filename
    Same logic as dataset: split by '.' and take first part, then remove '_pred' suffix
    For 'ChenHuiFu_pred.nii.gz' -> 'ChenHuiFu'
    """
    # Get filename without path
    filename = pred_file.name  # e.g., 'ChenHuiFu_pred.nii.gz'
    
    # Split by '.' and take first part (same as dataset: str(img_path.name).split('.')[0])
    # This handles both .nii.gz and .nii cases
    base_name = filename.split('.')[0]  # e.g., 'ChenHuiFu_pred'
    
    # Remove common prediction suffixes
    if base_name.endswith('_pred'):
        return base_name[:-5]  # Remove '_pred'
    elif base_name.endswith('_prediction'):
        return base_name[:-11]  # Remove '_prediction'
    elif base_name.endswith('_pred_mask'):
        return base_name[:-10]  # Remove '_pred_mask'
    else:
        # If no recognized suffix, return as is (might be already correct)
        return base_name


def find_matching_gt_file(patient_name, gt_dir):
    """Find matching ground truth file for a patient name"""
    # Try different possible naming conventions (most common first)
    possible_names = [
        f"{patient_name}.nii.gz",           # Direct match
        f"{patient_name}.nii",              # Without .gz
        f"{patient_name}_0000.nii.gz",      # nnUNet format with channel suffix
        f"{patient_name}_label.nii.gz",     # With _label suffix
        f"{patient_name}_gt.nii.gz",        # With _gt suffix
        f"{patient_name}_seg.nii.gz",       # With _seg suffix
    ]
    
    for name in possible_names:
        gt_file = gt_dir / name
        if gt_file.exists():
            return gt_file
    
    # If no exact match, try to find files that start with the patient name
    # This handles cases where the patient name might have additional suffixes in GT
    for gt_file in sorted(gt_dir.glob(f"{patient_name}*.nii*")):
        return gt_file
    
    # Last resort: try to find files that contain the patient name anywhere
    for gt_file in sorted(gt_dir.glob(f"*{patient_name}*.nii*")):
        return gt_file
    
    return None


def main():
    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)
    output_file = Path(args.output_file)
    
    # Check if directories exist
    if not pred_dir.exists():
        raise ValueError(f"Prediction directory does not exist: {pred_dir}")
    if not gt_dir.exists():
        raise ValueError(f"Ground truth directory does not exist: {gt_dir}")
    
    # Find all prediction files
    pred_files = sorted(list(pred_dir.glob("*_pred.nii.gz")))
    if len(pred_files) == 0:
        # Try without _pred suffix
        pred_files = sorted(list(pred_dir.glob("*.nii.gz")))
        if len(pred_files) == 0:
            raise ValueError(f"No prediction files found in {pred_dir}")
        else:
            print(f"Warning: Found {len(pred_files)} files without '_pred' suffix. Make sure they are prediction files.")
    
    # Get all GT files for verification
    gt_files_list = sorted(list(gt_dir.glob("*.nii.gz")))
    print(f"Found {len(pred_files)} prediction files")
    print(f"Found {len(gt_files_list)} ground truth files")
    

    # Calculate metrics for each case
    all_dice_scores = []
    all_miou_scores = []
    all_hd95_scores = []
    case_metrics = []
    failed_cases = []
    
    for pred_file in tqdm(pred_files, desc='Calculating metrics'):
        try:
            # Get patient name (same logic as dataset: split by '.' and take first part)
            # For "ChenHuiFu_pred.nii.gz", we want "ChenHuiFu"
            patient_name = get_patient_name_from_pred_file(pred_file)
            
            # Verify patient name extraction (print first case)
            if len(case_metrics) == 0:
                print(f"\n=== File Matching Debug ===")
                print(f"Pred file: {pred_file.name}")
                print(f"Extracted patient name: {patient_name}")
            
            # Find matching ground truth file
            gt_file = find_matching_gt_file(patient_name, gt_dir)
            if gt_file is None:
                print(f"\nWarning: Could not find ground truth for patient '{patient_name}' (from file {pred_file.name})")
                print(f"  Searched in: {gt_dir}")
                print(f"  Available GT files (first 5): {sorted(list(gt_dir.glob('*.nii.gz')))[:5]}")
                failed_cases.append({
                    'patient_name': patient_name,
                    'pred_file': str(pred_file),
                    'reason': 'GT file not found'
                })
                continue
            
            # Verify file matching (print first case)
            if len(case_metrics) == 0:
                print(f"Matched GT file: {gt_file.name}")
                print(f"Patient name match: ✓\n")
            
            # Load prediction and ground truth
            # pred_img = nib.load(pred_file)
            pred_data = sitk.GetArrayFromImage(sitk.ReadImage(pred_file))
            gt_data = sitk.GetArrayFromImage(sitk.ReadImage(gt_file))
            
            
            # Debug: Print first case info
            if len(case_metrics) == 0:
                print(f"\n=== Debug Info for {patient_name} ===")
                print(f"Pred shape (original): {pred_data.shape}, dtype: {pred_data.dtype}")
                print(f"GT shape (original): {gt_data.shape}, dtype: {gt_data.dtype}")
            
            # Handle dimension mismatch: pred might be [D,H,W] while GT is [H,W,D] or [W,H,D]
            # Example: pred (16, 256, 256) [D,H,W] vs GT (256, 256, 16) [H,W,D]
            pred_shape = pred_data.shape
            gt_shape = gt_data.shape
            
            # Convert to float for processing
            pred_data = pred_data.astype(np.float32)
            gt_data = gt_data.astype(np.float32)
            
            # Check if shapes match (same dimensions, possibly in different order)
            if pred_shape != gt_shape:
                gt_data = np.transpose(gt_data, (2, 1, 0))

            
            # Now shapes should match (or we've done our best)
            if len(case_metrics) == 0:
                print(f"Final shapes - Pred: {pred_data.shape}, GT: {gt_data.shape}")
                print(f"Pred unique values: {np.unique(pred_data)}")
                print(f"GT unique values: {np.unique(gt_data)}")
                print(f"Pred non-zero voxels: {(pred_data > 0).sum()}")
                print(f"GT non-zero voxels: {(gt_data > 0).sum()}")
            
            # Process labels: convert to binary (same as dataset does: gt[gt > 0] = 1.0)
            # For predictions: values should be 0 or 1 (from argmax in test_prostate_val.py)
            #   In test_prostate_val.py: preds = torch.argmax(outputs_softmax, dim=1, keepdim=True)
            #   Then: pred_numpy = preds.squeeze().cpu().numpy().astype(np.uint8)
            #   So pred values are already 0 or 1
            # For ground truth: convert all > 0 to 1 (same as dataset processing: gt[gt > 0] = 1.0)
            
            # Ensure binary masks
            # For pred: threshold at 0.5 (should already be 0/1, but handle any floating point values)
            # This matches test_prostate_val.py where preds are from argmax (0 or 1)
            pred_binary = (pred_data > 0.5).astype(np.uint8)
            
            # For GT: convert all > 0 to 1 (exactly as dataset does: gt[gt > 0] = 1.0)
            # This ensures compatibility with labels that might have values > 1
            gt_binary = np.zeros_like(gt_data, dtype=np.uint8)
            gt_binary[gt_data > 0] = 1
            
            # Final shape verification
            if pred_binary.shape != gt_binary.shape:
                raise ValueError(f"Shape mismatch after processing for {patient_name}: "
                               f"pred {pred_binary.shape} vs GT {gt_binary.shape}")
            
            # Debug: Print first case metrics before calculation
            if len(case_metrics) == 0:
                intersection = (pred_binary * gt_binary).sum()
                pred_sum = pred_binary.sum()
                gt_sum = gt_binary.sum()
                print(f"After binarization - Pred sum: {pred_sum}, GT sum: {gt_sum}, Intersection: {intersection}")
            
            # Calculate metrics
            dice_score = calculate_dice(pred_binary, gt_binary)
            miou_score = calculate_miou(pred_binary, gt_binary)
            hd95_score = calculate_hd95(pred_binary, gt_binary, percentile=args.percentile)
            
            # Debug: Print first case results and compare with reference if available
            if len(case_metrics) == 0:
                print(f"Calculated - Dice: {dice_score:.4f}, mIoU: {miou_score:.4f}, HD95: {hd95_score:.4f}")
                
                # Compare with reference JSON if provided
                if args.reference_json and Path(args.reference_json).exists():
                    try:
                        import json as json_module
                        with open(args.reference_json, 'r') as f:
                            ref_data = json_module.load(f)
                        # Find matching case in reference
                        for ref_case in ref_data.get('cases', []):
                            if ref_case.get('patient_name') == patient_name:
                                ref_dice = ref_case.get('dice_score', 0)
                                ref_miou = ref_case.get('miou_score', 0)
                                print(f"Reference   - Dice: {ref_dice:.4f}, mIoU: {ref_miou:.4f}")
                                print(f"Difference  - Dice: {abs(dice_score - ref_dice):.4f}, mIoU: {abs(miou_score - ref_miou):.4f}")
                                if abs(dice_score - ref_dice) > 0.01:
                                    print(f"  ⚠ WARNING: Dice difference > 0.01! Possible mismatch.")
                                break
                    except Exception as e:
                        print(f"Could not compare with reference: {e}")
                print()
            
            # Store results
            all_dice_scores.append(dice_score)
            all_miou_scores.append(miou_score)
            all_hd95_scores.append(hd95_score)
            
            case_metrics.append({
                'patient_name': patient_name,
                'dice_score': round(dice_score, 4),
                'miou_score': round(miou_score, 4),
                'hd95_score': round(hd95_score, 4),
                'pred_file': str(pred_file),
                'gt_file': str(gt_file)
            })
            
            print(f"Patient: {patient_name}, Dice: {dice_score:.4f}, mIoU: {miou_score:.4f}, HD95: {hd95_score:.4f}")
            
        except Exception as e:
            print(f"Error processing {pred_file}: {e}")
            failed_cases.append({
                'patient_name': patient_name if 'patient_name' in locals() else 'unknown',
                'pred_file': str(pred_file),
                'reason': str(e)
            })
            continue
    
    # Calculate overall metrics
    if len(all_dice_scores) == 0:
        raise ValueError("No valid cases processed. Please check your prediction and ground truth directories.")
    
    avg_dice = round(np.mean(all_dice_scores), 4)
    avg_miou = round(np.mean(all_miou_scores), 4)
    avg_hd95 = round(np.mean(all_hd95_scores), 4)
    
    std_dice = round(np.std(all_dice_scores), 4)
    std_miou = round(np.std(all_miou_scores), 4)
    std_hd95 = round(np.std(all_hd95_scores), 4)
    
    median_dice = round(np.median(all_dice_scores), 4)
    median_miou = round(np.median(all_miou_scores), 4)
    median_hd95 = round(np.median(all_hd95_scores), 4)
    
    print(f"\n=== Overall Results ===")
    print(f"Number of cases: {len(case_metrics)}")
    print(f"Average Dice: {avg_dice:.4f} ± {std_dice:.4f} (median: {median_dice:.4f})")
    print(f"Average mIoU: {avg_miou:.4f} ± {std_miou:.4f} (median: {median_miou:.4f})")
    print(f"Average HD95: {avg_hd95:.4f} ± {std_hd95:.4f} (median: {median_hd95:.4f})")
    
    if failed_cases:
        print(f"\nWarning: {len(failed_cases)} cases failed to process:")
        for case in failed_cases:
            print(f"  - {case['patient_name']}: {case['reason']}")
    
    # Save results to JSON
    results = {
        'pred_dir': str(pred_dir),
        'gt_dir': str(gt_dir),
        'num_cases': len(case_metrics),
        'num_failed': len(failed_cases),
        'case_metrics': case_metrics,
        'dice_scores': [round(score, 4) for score in all_dice_scores],
        'miou_scores': [round(score, 4) for score in all_miou_scores],
        'hd95_scores': [round(score, 4) for score in all_hd95_scores],
        'overall_metrics': {
            'average_dice': avg_dice,
            'average_miou': avg_miou,
            'average_hd95': avg_hd95,
            'std_dice': std_dice,
            'std_miou': std_miou,
            'std_hd95': std_hd95,
            'median_dice': median_dice,
            'median_miou': median_miou,
            'median_hd95': median_hd95
        },
        'failed_cases': failed_cases if failed_cases else None
    }
    
    # Save to JSON file
    # output_file.parent.mkdir(parents=True, exist_ok=True)
    # with open(output_file, 'w', encoding='utf-8') as f:
    #     json.dump(results, f, indent=4, ensure_ascii=False)
    
    # print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()

