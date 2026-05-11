import SimpleITK as sitk
import numpy as np

def match_mean_std(fixed_image, moving_image):
    """
    将 moving_image 的灰度分布匹配到 fixed_image 的均值和标准差。
    
    Parameters:
        fixed_image (SimpleITK.Image): 参考图像
        moving_image (SimpleITK.Image): 待调整图像
    
    Returns:
        SimpleITK.Image: 均值方差匹配后的新图像
    """
    # 转为 numpy 数组
    fixed_np = sitk.GetArrayFromImage(fixed_image).astype(np.float32)
    moving_np = sitk.GetArrayFromImage(moving_image).astype(np.float32)
    
    # 计算均值和标准差
    mean_fixed = fixed_np.mean()
    std_fixed = fixed_np.std()
    
    mean_moving = moving_np.mean()
    std_moving = moving_np.std()
    
    # 均值方差匹配公式
    new_moving = (moving_np - mean_moving) / (std_moving + 1e-8) * std_fixed + mean_fixed
    
    # 转回 SimpleITK.Image
    new_image = sitk.GetImageFromArray(new_moving.astype(np.float32))
    new_image.CopyInformation(moving_image)  # 保留原始图像的空间信息
    return new_image

# -----------------------------
# 示例：读取两个 NIfTI 文件
# -----------------------------
# 假设 A.nii.gz 是参考图像，B.nii.gz 是需要增强的图像
fixed_path = "A.nii.gz"
moving_path = "B.nii.gz"

fixed_image = sitk.ReadImage(fixed_path)
moving_image = sitk.ReadImage(moving_path)

# 执行灰度分布匹配
enhanced_image = match_mean_std(fixed_image, moving_image)

# 保存增强后的图像
sitk.WriteImage(enhanced_image, "B_enhanced.nii.gz")
print("增强完成，结果保存为 B_enhanced.nii.gz")