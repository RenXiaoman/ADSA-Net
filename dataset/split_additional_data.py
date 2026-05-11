import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

def split_additional_dataset():
    """
    将额外的OnlycsPCA_copy数据划分到主数据集目录
    """
    # 源目录
    source_images_dir = Path("./dataset/Task205_picai_lesion/origin/images")
    source_labels_dir = Path("./dataset/Task205_picai_lesion/origin/labels")
    
    # 目标目录
    target_images_train_dir = Path("./dataset/Task205_picai_lesion/prostate158_imagesTr")
    target_labels_train_dir = Path("./dataset/Task205_picai_lesion/prostate158_labelsTr")
    target_images_val_dir = Path("./dataset/Task205_picai_lesion/prostate158_imagesTrVal")
    target_labels_val_dir = Path("./dataset/Task205_picai_lesion/prostate158_labelsTrVal")
    
    # 检查源目录是否存在
    if not source_images_dir.exists():
        raise FileNotFoundError(f"源图像目录不存在: {source_images_dir}")
    if not source_labels_dir.exists():
        raise FileNotFoundError(f"源标签目录不存在: {source_labels_dir}")
    
    # 创建目标目录
    target_images_train_dir.mkdir(exist_ok=True)
    target_labels_train_dir.mkdir(exist_ok=True)
    target_images_val_dir.mkdir(exist_ok=True)
    target_labels_val_dir.mkdir(exist_ok=True)
    
    # 获取所有标签文件
    label_files = sorted(list(source_labels_dir.glob("*.nii.gz")))
    print(f"找到 {len(label_files)} 个标签文件")
    
    if len(label_files) == 0:
        raise ValueError("源标签目录中没有找到.nii.gz文件")
    
    # 提取文件名（不含扩展名）
    file_names = [f.name.replace('.nii.gz', '') for f in label_files]
    
    # 划分训练集和验证集 (8:2比例)
    train_names, val_names = train_test_split(
        file_names, test_size=0.2, random_state=42, shuffle=True
    )
    
    print(f"训练集: {len(train_names)} 个文件")
    print(f"验证集: {len(val_names)} 个文件")
    
    # 移动训练集文件
    moved_train_count = 0
    for name in train_names:
        # 移动标签文件
        label_src = source_labels_dir / f"{name}.nii.gz"
        label_dst = target_labels_train_dir / f"{name}.nii.gz"
        
        # 移动图像文件 (可能有多个模态)
        image_patterns = [f"{name}_*.nii.gz", f"{name}.nii.gz"]
        for pattern in image_patterns:
            for image_src in source_images_dir.glob(pattern):
                image_dst = target_images_train_dir / image_src.name
                if image_src.exists():
                    shutil.move(str(image_src), str(image_dst))
                    moved_train_count += 1
        
        if label_src.exists():
            shutil.move(str(label_src), str(label_dst))
            moved_train_count += 1
    
    # 移动验证集文件
    moved_val_count = 0
    for name in val_names:
        # 移动标签文件
        label_src = source_labels_dir / f"{name}.nii.gz"
        label_dst = target_labels_val_dir / f"{name}.nii.gz"
        
        # 移动图像文件
        image_patterns = [f"{name}_*.nii.gz", f"{name}.nii.gz"]
        for pattern in image_patterns:
            for image_src in source_images_dir.glob(pattern):
                image_dst = target_images_val_dir / image_src.name
                if image_src.exists():
                    shutil.move(str(image_src), str(image_dst))
                    moved_val_count += 1
        
        if label_src.exists():
            shutil.move(str(label_src), str(label_dst))
            moved_val_count += 1
    
    print(f"成功移动 {moved_train_count} 个文件到训练集目录")
    print(f"成功移动 {moved_val_count} 个文件到验证集目录")
    
    # 验证移动结果
    verify_movement(target_images_train_dir, target_labels_train_dir, 
                   target_images_val_dir, target_labels_val_dir)
    
    return train_names, val_names

def verify_movement(images_train_dir, labels_train_dir, images_val_dir, labels_val_dir):
    """验证文件移动结果"""
    
    train_images = list(images_train_dir.glob("*.nii.gz"))
    train_labels = list(labels_train_dir.glob("*.nii.gz"))
    val_images = list(images_val_dir.glob("*.nii.gz"))
    val_labels = list(labels_val_dir.glob("*.nii.gz"))
    
    print(f"\n验证结果:")
    print(f"训练集图像: {len(train_images)} 个文件")
    print(f"训练集标签: {len(train_labels)} 个文件")
    print(f"验证集图像: {len(val_images)} 个文件")
    print(f"验证集标签: {len(val_labels)} 个文件")
    
    # 检查对应关系
    train_label_names = {f.name.replace('.nii.gz', '') for f in train_labels}
    val_label_names = {f.name.replace('.nii.gz', '') for f in val_labels}
    
    # 检查训练集图像文件是否都有对应的标签
    missing_train = []
    for image_file in train_images:
        base_name = image_file.stem
        if '_' in base_name:  # 处理带模态后缀的文件
            base_name = base_name.split('_')[0]
        if base_name not in train_label_names:
            missing_train.append(image_file.name)
    
    if missing_train:
        print(f"警告: {len(missing_train)} 个训练集图像文件没有对应的标签: {missing_train[:5]}")
    
    # 检查验证集图像文件是否都有对应的标签
    missing_val = []
    for image_file in val_images:
        base_name = image_file.stem
        if '_' in base_name:
            base_name = base_name.split('_')[0]
        if base_name not in val_label_names:
            missing_val.append(image_file.name)
    
    if missing_val:
        print(f"警告: {len(missing_val)} 个验证集图像文件没有对应的标签: {missing_val[:5]}")

if __name__ == "__main__":
    try:
        print("开始划分额外的OnlycsPCA_copy数据...")
        train_files, val_files = split_additional_dataset()
        print("\n数据划分完成！")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()