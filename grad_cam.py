import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='timm')

from pathlib import Path
from typing import Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import SimpleITK as sitk
import cv2

import sys
import os

from monai.transforms import ClipIntensityPercentiles, NormalizeIntensity, ScaleIntensity
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

from Model.as_unetr import CDSA_Net
import matplotlib.pyplot as plt
from pylab import *
# from monai.visualize import GradCAM, CAM, OcclusionSensitivity



class CDSA_Net_GradCAM:
    def __init__(self, model, target_layer):
        """
        初始化Grad-CAM可视化
        
        参数:
            model: CDSA_Net模型
            target_layer: 目标层，如model.vit.blocks[-1].norm1
        """
        self.model = model
        self.target_layer = target_layer
        self.gradient = None
        self.activation = None
        
        # 注册钩子
        self.hook_handles = []
        self.register_hooks()
        
    def register_hooks(self):
        """注册前向和后向钩子"""
        def forward_hook(module, input, output):
            self.activation = output.detach()
            
        def backward_hook(module, grad_input, grad_output):
            self.gradient = grad_output[0].detach()
            
        # 注册前向和后向钩子
        self.hook_handles.append(
            self.target_layer.register_forward_hook(forward_hook)
        )
        self.hook_handles.append(
            self.target_layer.register_backward_hook(backward_hook)
        )
    
    def remove_hooks(self):
        """移除所有钩子"""
        for handle in self.hook_handles:
            handle.remove()
    
    def generate_cam(self, input_tensor, target_class=None):
        """
        生成Grad-CAM热力图
        
        参数:
            input_tensor: 输入张量 [B, C, D, H, W]
            target_class: 目标类别，如果为None则使用模型预测的类别
            
        返回:
            cam: Grad-CAM热力图
        """
        # 前向传播
        self.model.eval()
        output = self.model(input_tensor)
        
        if target_class is None:
            outputs_softmax = torch.softmax(output, dim=1)
            target_class = torch.argmax(outputs_softmax, dim=1, keepdim=True)
            # target_class = output.argmax(dim=1).item()
            
        # 反向传播
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # self.gradient shape is [1, 256, 768]
        
        # 计算权重 - 对最后一个维度取平均
        # self.gradient shape: [1, 256, 768]
        weights = torch.mean(self.gradient, dim=2, keepdim=True)  # [1, 256, 1]
        
        # 调整权重形状以匹配激活图
        # 假设self.activation是ViT的输出 [B, 256, 768] 或 [B, 256, H, W, D]
        if len(self.activation.shape) == 3:  # [B, 256, 768]
            # 如果是ViT的输出，需要reshape回空间维度
            # 这里假设原始空间维度是 [16, 16, 3] (16*16*3=768)
            # 需要根据实际模型结构调整这些值
            spatial_dims = (16, 16, 3)  # 需要根据实际情况调整
            weights = weights.reshape(1, 16, 16, 3)  # 调整为你需要的空间维度
            weights = weights.unsqueeze(0)  # 添加通道维度
            
            # 调整activation形状
            activation = self.activation.reshape(1, 256, *spatial_dims)
        else:
            activation = self.activation
            
        # 计算CAM
        cam = torch.sum(weights * activation, dim=1, keepdim=True)
        cam = F.relu(cam)  # 应用ReLU
        
        # 插值到输入大小
        cam = F.interpolate(cam, size=input_tensor.shape[2:], 
                           mode='trilinear', align_corners=False)
        
        # 归一化到[0,1]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.squeeze().cpu().numpy()

def visualize_3d_slice(cam, original_volume, slice_idx=None, axis=0):
    """
    可视化3D体积的特定切片
    
    参数:
        cam: 3D CAM热力图 [D, H, W]
        original_volume: 原始3D体积 [C, D, H, W]
        slice_idx: 切片索引，如果为None则显示中间切片
        axis: 切片的轴 (0, 1, 2)
    """
    if slice_idx is None:
        slice_idx = cam.shape[axis] // 2
    
    if axis == 0:
        cam_slice = cam[slice_idx, :, :]
        img_slice = original_volume[0, slice_idx, :, :]
    elif axis == 1:
        cam_slice = cam[:, slice_idx, :]
        img_slice = original_volume[0, :, slice_idx, :]
    else:  # axis == 2
        cam_slice = cam[:, :, slice_idx]
        img_slice = original_volume[0, :, :, slice_idx]
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(img_slice, cmap='gray')
    plt.axis('off')
    plt.title('Original Slice')
    
    plt.subplot(1, 2, 2)
    plt.imshow(img_slice, cmap='gray')
    plt.imshow(cam_slice, cmap='jet', alpha=0.5)
    plt.axis('off')
    plt.title('Grad-CAM')
    
    plt.tight_layout()
    plt.show()



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
        


class SemanticSegmentationTarget:
    def __init__(self, category, mask):
        self.category = category
        self.mask = torch.from_numpy(mask)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()
        
    def __call__(self, model_output):
        return (model_output[self.category, :, : ] * self.mask).sum()
    
    
        
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parallel = True
# model_path = 'checkpoints/SegTumor_DIY_New_CNN_Encoder/best_dice_model.pth'
model_name = 'SegTumor_DIY_PICAI_New_CNN_Encoder'
model_path = f'checkpoints/{model_name}/best_dice_model.pth'

name = '10005_1000005'
# 10005_1000005 10040_1000040
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

checkpoint = torch.load(model_path, map_location=device, weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Loaded model from: {model_path}")
# dataset/ChengdaOnlyCSPca/nnUNet_val/imagesTs labelsTs
# dataset/PI-CAI/imagesTs labelsTs
input_t2w = Path(f'dataset/PI-CAI/imagesTs/{name}_0000.nii.gz')
input_adc = Path(f'dataset/PI-CAI/imagesTs/{name}_0001.nii.gz')
input_dwi = Path(f'dataset/PI-CAI/imagesTs/{name}_0002.nii.gz')
gt = Path(f'dataset/PI-CAI/labelsTs/{name}.nii.gz')

ADC, DWI, T2W, gt = get_preprocessed_data(input_t2w, input_adc, input_dwi, gt)

image_data = np.concatenate([ADC, DWI, T2W], axis=0)
image_data = torch.from_numpy(image_data).to(device)
image_data = image_data.unsqueeze(0)

with torch.no_grad():
    outputs_for_pred = model(image_data)
    outputs_softmax = torch.softmax(outputs_for_pred, dim=1)
    preds = torch.argmax(outputs_softmax, dim=1).cpu().numpy()[0]  # (D,H,W)


# gt = torch.from_numpy(gt).to(device)

# for name, _ in model.named_modules(): print(name)


# 1.选择目标层进行可视化
# target_layer = model.vit.blocks[-1].norm1 
# encoder1.layer.conv1 : Transformer 辅助分支 | conv_head : CNN 主分支 
# | trans_1 : Fusion    | BiCR_1.dcgf.refine.conv GRE | 
# |  encoder1.layer.conv1  |  conv_head   |   BiCR_1.dcgf.refine.conv  |  decoder1 最后一层
target_layers = [model.encoder1.layer.conv1]       
layer_name = 'encoder1.layer.conv1'
targets = [SemanticSegmentationTarget(1, np.float32(gt))]
slice_idx = 9
# model.encoder1.layer.conv1
# model.conv_head
# model.BiCR_1.dcgf.refine.conv

def visualize_gradcam_3d(
    image_volume: np.ndarray,  # 输入3D图像 [D, H, W] 或 [C, D, H, W]
    cam_volume: np.ndarray,    # Grad-CAM 3D热图 [D, H, W]
    slice_idx: int = None,     # 指定要可视化的切片索引
    alpha: float = 0.5,        # 热图透明度
    use_rgb: bool = True,      # 是否使用RGB格式
    colormap: int = cv2.COLORMAP_JET,  # OpenCV色彩图
    save_path: Optional[str] = None,   # 保存路径
    dpi: int = 100,            # 图像DPI
    figsize: tuple = (15, 5)   # 图像大小
):
    """
    可视化3D Grad-CAM结果
    
    参数:
        image_volume: 3D图像数据 [D, H, W] 或 [C, D, H, W]
        cam_volume: Grad-CAM热图 [D, H, W]
        slice_idx: 要可视化的切片索引，如果为None则显示中间切片
        alpha: 热图透明度 (0-1)
        use_rgb: 是否使用RGB格式
        colormap: OpenCV色彩图
        save_path: 保存路径，如果为None则不保存
        dpi: 图像DPI
        figsize: 图像大小
    """
    # 确保输入是numpy数组
    image_volume = np.asarray(image_volume)
    cam_volume = np.asarray(cam_volume)
    
    # 如果输入是4D [C, D, H, W]，取第一个通道
    if image_volume.ndim == 4:
        image_volume = image_volume[0]  # 取第一个通道 [D, H, W]
    
    # 确保值范围在[0, 1]之间
    def normalize(x):
        x = x.astype(np.float32)
        x_min, x_max = x.min(), x.max()
        if x_max > x_min:
            x = (x - x_min) / (x_max - x_min)
        return x
    
    # 归一化图像和热图
    image_volume = normalize(image_volume)
    cam_volume = normalize(cam_volume)
    
    # 如果未指定切片索引，使用中间切片
    if slice_idx is None:
        slice_idx = image_volume.shape[0] // 2
    else:
        slice_idx = min(max(0, slice_idx), image_volume.shape[0] - 1)
    
    # 获取选中的切片
    img_slice = image_volume[slice_idx]  # [H, W]
    heatmap = cam_volume[slice_idx]      # [H, W]
    
    # 将热图转换为彩色
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), colormap)
    if use_rgb:
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    heatmap_colored = normalize(heatmap_colored)
    
    # 将单通道图像转换为3通道以进行叠加
    if len(img_slice.shape) == 2:
        img_slice = np.stack([img_slice] * 3, axis=-1)
    
    # 确保图像和热图形状匹配
    if img_slice.shape != heatmap_colored.shape:
        # 调整热图大小以匹配图像
        heatmap_colored = cv2.resize(heatmap_colored, (img_slice.shape[1], img_slice.shape[0]))
    
    # 创建叠加图像
    overlayed_img = (1 - alpha) * img_slice + alpha * heatmap_colored
    overlayed_img = np.clip(overlayed_img, 0, 1)
    
    # Create visualization
    fig, axes = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    
    # Heatmap
    axes.imshow(heatmap, cmap='jet')
    # axes.set_title('Grad-CAM Heatmap')
    # plt.colorbar(axes[0].imshow(heatmap, cmap='jet'), ax=axes[0], fraction=0.046, pad=0.04)
    axes.axis('off')
    
    plt.tight_layout()
    
    # Save the figure
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
        print(f"Figure saved to: {save_path}")
    
    plt.show()
    return overlayed_img

with GradCAM(model=model,
             target_layers=target_layers,
             ) as cam:  # 
    grayscale_cam = cam(input_tensor=image_data,
                        targets=targets)[0, :]  # (16, 256, 256)

    
    
    print(f'grayscale_cam shape {grayscale_cam.shape}')
    
    if parallel:
        t2w_slice = T2W[0, slice_idx]
        pred_slice = preds[slice_idx]
        cam_slice = grayscale_cam[slice_idx]

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        fig.subplots_adjust(wspace=0.05)  # 数字越小，间距越小

        axes[0].imshow(t2w_slice, cmap='gray')
        # 只勾勒预测的轮廓
        pred_bin = (pred_slice > 0).astype(np.uint8)
        axes[0].contour(pred_bin, levels=[0.5], colors='r', linewidths=1.5)
        axes[0].axis('off')

        im = axes[1].imshow(cam_slice, cmap='jet')
        axes[1].axis('off')
        # fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

        save_path = f'heatmap/{name}_{model_name}_{layer_name}_{slice_idx}_parallel.png'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=100)
        print(f"Figure saved to: {save_path}")
        plt.show()
    
    else:
        overlayed_img = visualize_gradcam_3d(
        image_volume=gt[0],  # 假设gt是张量且需要移动到CPU
        cam_volume=grayscale_cam,          # Grad-CAM热图
        slice_idx=slice_idx,               # 切片索引     alpha=0.5,                         # 热图透明度
        use_rgb=True,                      # 使用RGB格式
        save_path=f'heatmap/{name}_{model_name}_{layer_name}_{slice_idx}.png'  # 可选：保存路径
    )
 
