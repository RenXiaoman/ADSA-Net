#!/usr/bin/env python3

import os
import torch
import numpy as np
import nibabel as nib
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import json
import matplotlib.pyplot as plt
from skimage import measure
from Model.as_unetr import ADSA_Net
from Model.ALIEN import ALIEN

# Model
# from picai_baseline.unet.training_setup.neural_networks.unets import UNet
from monai.networks.nets import UNETR
from Model.ablation import Backbone_SAEB, Backbone_ACF, Backbone_MRE, Backbone_Baseline, Backbone_MRE_ACF

# Local imports
from dataset.dataset_nnunet import Lits_DataSet

parser = argparse.ArgumentParser(description='Test prostate validation set')
parser.add_argument('--model_path', type=str,default='checkpoints/SegTumor_DIY_New_CNN_Encoder/best_dice_model.pth', required=False, help='Path to trained model checkpoint')

# parser.add_argument('--data_path', type=str, default='dataset/PI-CAI', help='Path to dataset')
parser.add_argument('--data_path', type=str, default='dataset/ChengdaOnlyCSPca', help='Path to dataset')

parser.add_argument('--output_dir', type=str, default='val_results', help='Output directory for predictions')
parser.add_argument('--slices_dir', type=str, default='slice_contours', help='Directory for slice images with contours')
parser.add_argument('--gpu_ids', type=str, default='0', help='GPU IDs')
parser.add_argument('--keep_largest_cc', action='store_true',default=True, help='Keep only the largest connected component in predictions')
parser.add_argument('--mode', type=str, default='val', choices=['val', 'test'], help='Mode: val for validation, test for testing')

args = parser.parse_args()
    
    
def calculate_dice(preds, targets):
    """Calculate Dice coefficient"""
    smooth = 1e-6
    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)
    
    intersection = (preds_flat * targets_flat).sum()
    pred_sum = preds_flat.sum()
    target_sum = targets_flat.sum()
    
    dice = (2. * intersection + smooth) / (pred_sum + target_sum + smooth)
    return dice.item()

def calculate_miou(preds, targets):
    """Calculate mean Intersection over Union"""
    smooth = 1e-6
    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)
    
    intersection = (preds_flat * targets_flat).sum()
    union = preds_flat.sum() + targets_flat.sum() - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()

def calculate_dice_intervals(dice_scores):
    """Calculate distribution of Dice scores across intervals"""
    intervals = [
        (0.0, 0.1),   # 0~0.1
        (0.1, 0.2),   # 0.1~0.2
        (0.2, 0.3),   # 0.2~0.3
        (0.3, 0.4),   # 0.3~0.4
        (0.4, 0.5),   # 0.4~0.5
        (0.5, 1.0)    # 0.5~1.0
    ]
    
    interval_counts = {}
    for i, (lower, upper) in enumerate(intervals):
        if i == len(intervals) - 1:  # Last interval includes upper bound
            count = sum(lower <= score <= upper for score in dice_scores)
        else:
            count = sum(lower <= score < upper for score in dice_scores)
        interval_name = f"{lower:.1f}-{upper:.1f}"
        interval_counts[interval_name] = count
    
    return interval_counts

def save_slice_contours(mri_slice, gt_slice, pred_slice, output_path):
    """Save single slice with GT and prediction contours"""
    plt.figure(figsize=(8, 8))
    plt.imshow(mri_slice, cmap="gray")
    
    # Draw GT contours (red)
    contours = measure.find_contours(gt_slice, 0.5)
    for contour in contours:
        plt.plot(contour[:, 1], contour[:, 0], 'r-', linewidth=2, label='GT' if 'GT' not in plt.gca().get_legend_handles_labels()[1] else "")
    
    # Draw prediction contours (yellow)
    contours = measure.find_contours(pred_slice, 0.5)
    for contour in contours:
        plt.plot(contour[:, 1], contour[:, 0], 'y-', linewidth=2, label='Pred' if 'Pred' not in plt.gca().get_legend_handles_labels()[1] else "")
    
    plt.axis("off")
    plt.legend()
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0, dpi=100)
    plt.close()

def keep_largest_connected_component_3d(pred_data):
    """Keep only the largest connected component in 3D prediction volume"""
    import SimpleITK as sitk
    mask_img = sitk.GetImageFromArray(pred_data)
    cc = sitk.ConnectedComponent(mask_img, True)  
    cc_sorted = sitk.RelabelComponent(cc, sortByObjectSize=True)  # 按体积排序，最大区域编号=1
    largest_cc = sitk.Equal(cc_sorted, 1)                         # 保留最大连通域
    return sitk.GetArrayFromImage(largest_cc).astype(np.uint8)

def save_multimodal_slices_with_contours(adc_data, dwi_data, t2w_data, gt_data, pred_data, patient_name, output_dir):
    """Save 16 slices with 3 modalities side by side with GT and prediction contours"""
    patient_dir = output_dir / patient_name
    patient_dir.mkdir(parents=True, exist_ok=True)
    
    num_slices = t2w_data.shape[0]
    for z in range(num_slices):
        if z < 16:  # Only save first 16 slices
            # Get slices for all modalities
            adc_slice = adc_data[z]
            dwi_slice = dwi_data[z] 
            t2w_slice = t2w_data[z]
            gt_slice = gt_data[z].astype(bool)
            pred_slice = pred_data[z].astype(bool)
            
            # Create figure with 3 columns
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            modalities = [("ADC", adc_slice), ("DWI", dwi_slice), ("T2W", t2w_slice)]
            
            for ax, (modality_name, modality_slice) in zip(axes, modalities):
                ax.imshow(modality_slice, cmap="gray")
                
                # Draw GT contours (red)
                contours = measure.find_contours(gt_slice, 0.5)
                for contour in contours:
                    ax.plot(contour[:, 1], contour[:, 0], 'r-', linewidth=2, 
                           label='GT' if 'GT' not in ax.get_legend_handles_labels()[1] else "")
                
                # Draw prediction contours (yellow)
                contours = measure.find_contours(pred_slice, 0.5)
                for contour in contours:
                    ax.plot(contour[:, 1], contour[:, 0], 'y-', linewidth=2,
                           label='Pred' if 'Pred' not in ax.get_legend_handles_labels()[1] else "")
                
                ax.set_title(f"{modality_name} - Slice {z}")
                ax.axis("off")
            
            # Add legend only once
            handles, labels = axes[0].get_legend_handles_labels()
            if labels:
                fig.legend(handles, ["GT", "Pred"], loc="lower right", fontsize=12)
            
            plt.tight_layout()
            output_path = patient_dir / f"slice_{z:02d}.png"
            plt.savefig(output_path, bbox_inches="tight", pad_inches=0.1, dpi=100)
            plt.close()
    
    print(f"Saved 16 multimodal slices for {patient_name} to {patient_dir}")


def fix_adsa_net_checkpoint(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if 'gre_dcgf_' in key:
            # Replace 'gre_dcgf_X' with 'BiCR_X'
            new_key = key.replace('gre_dcgf_', 'BiCR_')
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict


def main():
    
    
    # Set device
    if args.gpu_ids != '-1':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
    # Load model
    model = ADSA_Net(
        in_channels=2,  # ADC and DWI modalities
        out_channels=2,  # Background and lesion
        img_size=(16, 256, 256),
        feature_size=16,
        hidden_size=768, 
        mlp_dim=3072,
        num_heads=8,
        norm_name='instance'
    ).to(device)

    # model = Backbone_SAEB(
    #     in_channels=2,  # ADC and DWI modalities
    #     out_channels=2,  # Background and lesion
    #     img_size=(16, 256, 256),
    #     feature_size=16,
    #     hidden_size=768,
    #     mlp_dim=3072,
    #     num_heads=8,
    #     norm_name='instance',
    # ).to(device)

    # model = ALIEN(n_classes=2, n_channels=3, trilinear=True).to(device)
    # model = Backbone_MRE(
    #     in_channels=2,  # ADC and DWI modalities
    #     out_channels=2,  # Background and lesion
    #     img_size=(16, 256, 256),
    #     feature_size=16,
    #     hidden_size=768,
    #     mlp_dim=3072,
    #     num_heads=8,
    #     norm_name='instance',
    #     # dropout_rate=0.2
    # ).to(device)
    
    # model = UNet(
    #     spatial_dims=3,
    #     in_channels=3,
    #     out_channels=2,
    #     strides=[(2, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2), (2, 2, 2)],
    #     channels=[32, 64, 128, 256, 512, 1024],
    #     ).to(device)
    
    # model = UNETR(in_channels=3, 
    #               out_channels=2, 
    #               img_size=(16, 256, 256), 
    #               spatial_dims=3).to(device)
    
    
    # Load checkpoint
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=True)
    
    checkpoint = checkpoint["model_state_dict"]
    
    checkpoint = fix_adsa_net_checkpoint(checkpoint)  # 注释即可
    
    model.load_state_dict(checkpoint)
    model.eval()
    print(f"Loaded model from: {args.model_path}")
    
    # Create parent directory based on model_path experiment name and mode
    model_path_parts = Path(args.model_path).parts
    if len(model_path_parts) >= 2:
        experiment_name = model_path_parts[-2]  # Get the experiment name from model path
    else:
        experiment_name = Path(args.model_path).stem  # Fallback to model file name
        
    experiment_name = '111111'

    suffix = "_infer" if args.mode == "val" else "_test"
    parent_dir = Path(f"infer/{experiment_name}{suffix}")
    parent_dir.mkdir(parents=True, exist_ok=True)

    # Create output directories as subdirectories of parent directory
    output_dir = parent_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    slices_dir = parent_dir / args.slices_dir
    slices_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset based on mode # nnUNet_val
    if args.mode == "val":
        if args.data_path.endswith('PI-CAI'):
            images_dir = 'imagesTs'
            labels_dir = 'labelsTs'
        elif args.data_path.endswith('ChengdaOnlyCSPca'):
            images_dir = 'nnUNet_val/imagesTs'
            labels_dir = 'nnUNet_val/labelsTs'
    else:  # test mode
        images_dir = 'imagesTs'
        labels_dir = 'labelsTs'
    
    val_dataset = Lits_DataSet(
        Path(args.data_path), 
        images_dir, 
        labels_dir,
        enable_augmentation=False
    )
    
    val_dataloader = DataLoader(
        dataset=val_dataset, 
        batch_size=1,  # Batch size 1 for individual patient inference
        num_workers=4, 
        shuffle=False
    )
    
    print(f"Validation dataset: {len(val_dataset)} samples")
    
    # Test loop
    all_dice_scores = []
    all_miou_scores = []
    case_metrics = []  # Store metrics for each individual case
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(val_dataloader, desc='Testing')):
            ADC, DWI, T2W, labels, patient_names = batch_data
            inputs = torch.cat([ADC, DWI, T2W], dim=1).to(device)  # 原始输入 T2W为主模态
            # inputs = torch.cat([ADC, T2W, DWI], dim=1).to(device)    # DWI为主模态
            # inputs = torch.cat([T2W, DWI, ADC], dim=1).to(device)  # ADC为主模态
            
            
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Apply softmax and get predictions
            outputs_softmax = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs_softmax, dim=1, keepdim=True)
            
            # Save prediction as NIfTI file
            patient_name = patient_names[0]
            pred_numpy = preds.squeeze().cpu().numpy().astype(np.uint8)
            
            # Conditionally keep only largest connected component
            if args.keep_largest_cc:
                pred_numpy_cleaned = keep_largest_connected_component_3d(pred_numpy)
                # Calculate metrics AFTER cleaning (using cleaned predictions)
                preds_cleaned = torch.from_numpy(pred_numpy_cleaned).unsqueeze(0).unsqueeze(0).to(device)
                dice_score = calculate_dice(preds_cleaned, labels)
                miou_score = calculate_miou(preds_cleaned, labels)
                # Use cleaned predictions for saving
                pred_numpy_to_save = pred_numpy_cleaned
            else:
                # Calculate metrics using original predictions
                dice_score = calculate_dice(preds, labels)
                miou_score = calculate_miou(preds, labels)
                # Use original predictions for saving
                pred_numpy_to_save = pred_numpy
            
            all_dice_scores.append(dice_score)
            all_miou_scores.append(miou_score)
            
            # Store individual case metrics (rounded to 4 decimal places)
            case_metrics.append({
                'patient_name': patient_name,
                'dice_score': round(dice_score, 4),
                'miou_score': round(miou_score, 4)
            })
            
            # Create NIfTI image (save predictions)
            pred_img = nib.Nifti1Image(pred_numpy_to_save, np.eye(4))
            pred_filename = output_dir / f"{patient_name}_pred.nii.gz"
            nib.save(pred_img, pred_filename)
            
            # Save multimodal slice images with contours
            mri_data = inputs.squeeze().cpu().numpy()  # (3, 16, 256, 256)
            gt_data = labels.squeeze().cpu().numpy().astype(bool)
            pred_data = pred_numpy.astype(bool)
            
            # Get all 3 modalities
            adc_data = mri_data[0]  # ADC channel
            dwi_data = mri_data[1]  # DWI channel  
            t2w_data = mri_data[2]  # T2W channel
            
            save_multimodal_slices_with_contours(adc_data, dwi_data, t2w_data, gt_data, pred_data, patient_name, slices_dir)
            
            print(f"Patient: {patient_name}, Dice: {dice_score:.4f}, mIoU: {miou_score:.4f}")
    
    # Calculate overall metrics (rounded to 4 decimal places)
    avg_dice = round(np.mean(all_dice_scores), 4)
    avg_miou = round(np.mean(all_miou_scores), 4)
    std_dice = round(np.std(all_dice_scores), 4)
    std_miou = round(np.std(all_miou_scores), 4)
    
    print(f"\n=== Overall Results ===")
    print(f"Connected component filtering: {'Enabled' if args.keep_largest_cc else 'Disabled'}")
    print(f"Average Dice: {avg_dice:.4f} ± {std_dice:.4f}")
    print(f"Average mIoU: {avg_miou:.4f} ± {std_miou:.4f}")
    print(f"All results saved to parent directory: {parent_dir}")
    print(f"- Predictions saved to: {output_dir}")
    print(f"- Slice images saved to: {slices_dir}")
    
    # Save results to JSON
    # Save results to JSON (using rounded values)
    results = {
        'model_path': args.model_path,
        'data_path': args.data_path,
        'keep_largest_cc': args.keep_largest_cc,
        'num_samples': len(val_dataset),
        'case_metrics': case_metrics,  # Individual case metrics (already rounded)
        'dice_scores': [round(score, 4) for score in all_dice_scores],
        'miou_scores': [round(score, 4) for score in all_miou_scores],
        'average_dice': avg_dice,
        'average_miou': avg_miou,
        'std_dice': std_dice,
        'std_miou': std_miou
    }
    
    results_file = output_dir / 'test_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Detailed results saved to: {results_file}")
    
    # Calculate Dice score intervals
    dice_interval_stats = calculate_dice_intervals([case['dice_score'] for case in case_metrics])
    
    # Save individual case metrics to separate A_Summary.json file
    summary_data = {
        'model_path': args.model_path,
        'data_path': args.data_path,
        'keep_largest_cc': args.keep_largest_cc,
        'num_cases': len(case_metrics),
        'cases': case_metrics,
        'overall_metrics': {
            'average_dice': avg_dice,
            'average_miou': avg_miou,
            'std_dice': std_dice,
            'std_miou': std_miou
        },
        'dice_interval_statistics': dice_interval_stats
    }
    
    summary_file = output_dir / 'A_Summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=4)
    
    print(f"Case summary saved to: {summary_file}")

if __name__ == "__main__":
    main()