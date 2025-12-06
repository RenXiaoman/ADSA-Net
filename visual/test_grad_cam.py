from pathlib import Path
from typing import Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import SimpleITK as sitk

import sys
import os

from monai.transforms import ClipIntensityPercentiles, NormalizeIntensity, ScaleIntensity


from Model.as_unetr import ADSA_Net

clip = ClipIntensityPercentiles(lower=0.5, upper=99.5, channel_wise=False)

def check_exists(*paths):
    """
    检查传入的所有路径是否存在，不存在就报错
    :param paths: 任意数量的 Path 或 str
    """
    for p in paths:
        p = Path(p)   # 保证类型安全
        if not p.exists():
            raise FileNotFoundError(f"❌ Path not found: {p}")
        
def load(file: Union[Path, str]) -> np.ndarray:
    itkimage = sitk.ReadImage(file)
    image = sitk.GetArrayFromImage(itkimage)
    return image


def z_score_normalization(img):
    # 只计算非零区域的均值和标准差
    non_zero_mask = img > 0
    if np.any(non_zero_mask):
        mean_val = np.mean(img[non_zero_mask])
        std_val = np.std(img[non_zero_mask])
    else:
        mean_val = np.mean(img)
        std_val = np.std(img)
        out = (img - mean_val) / (std_val + 1e-8)
        return out


def get_preprocessed_data(T2W_path, ADC_path, DWI_path, GT_path):
    # 验证上面3个path是否存在
    check_exists(T2W_path, ADC_path, DWI_path, GT_path)
    
    T2W_raw = load(T2W_path).astype(np.float32)  # [D,H,W]
    ADC_raw = load(ADC_path).astype(np.float32)
    DWI_raw = load(DWI_path).astype(np.float32)
    gt      = load(GT_path).astype(np.float32)
    
    # 清理标签值，将所有大于0的值转为1（二分类问题）
    gt[gt > 0] = 1.0

    # 基础预处理: clip + z-score归一化
    T2W = z_score_normalization(clip(T2W_raw))  # -> [D,H,W]
    ADC = z_score_normalization(clip(ADC_raw))  # -> [D,H,W]
    DWI = z_score_normalization(clip(DWI_raw))
    
    # 添加通道维度
    T2W = T2W[np.newaxis, :]  # [1, D, H, W]
    ADC = ADC[np.newaxis, :]  # [1, D, H, W]
    DWI = DWI[np.newaxis, :]
    gt = gt[np.newaxis, :]    # [1, D, H, W]
    
    # 合并多模态数据 [3, D, H, W]
    # image_data = np.concatenate([ADC, DWI, T2W], axis=0)
    
    return ADC, DWI, T2W, gt
        

        
        
        
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_path = 'checkpoints/SegTumor_DIY_PICAI_New_CNN_Encoder/best_dice_model.pth'
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

checkpoint = torch.load(model_path, map_location=device, weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])

print(f"Loaded model from: {model_path}")

input_t2w = Path('../dataset/PI-CAI/imagesTs/10005_1000005_0000.nii.gz')
input_adc = Path('../dataset/PI-CAI/imagesTs/10005_1000005_0001.nii.gz')
input_dwi = Path('../dataset/PI-CAI/imagesTs/10005_1000005_0002.nii.gz')
gt = Path('../dataset/PI-CAI/labelsTs/10005_1000005.nii.gz')


ADC, DWI, T2W, gt = get_preprocessed_data(input_adc, input_dwi, input_t2w, gt)


