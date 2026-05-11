import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

def split_train_val_dataset(task_dir, images_dir_name, labels_dir_name, val_ratio=0.2, random_seed=42):
    """
    划分训练集和验证集
    
    Args:
        task_dir: Task目录路径
        images_dir_name: 图像目录名称
        labels_dir_name: 标签目录名称  
        val_ratio: 验证集比例
        random_seed: 随机种子
    """
    task_path = Path(task_dir)
    images_dir = task_path / images_dir_name
    labels_dir = task_path / labels_dir_name
    
    # 检查目录是否存在
    if not images_dir.exists():
        raise FileNotFoundError(f"图像目录不存在: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"标签目录不存在: {labels_dir}")
    
    # 获取所有标签文件
    label_files = sorted(list(labels_dir.glob("*.nii.gz")))
    print(f"找到 {len(label_files)} 个标签文件")
    
    if len(label_files) == 0:
        raise ValueError("标签目录中没有找到.nii.gz文件")
    
    # 提取文件名（不含任何扩展名）
    file_names = [f.name.replace('.nii.gz', '') for f in label_files]
    
    # 划分训练集和验证集
    train_names, val_names = train_test_split(
        file_names, test_size=val_ratio, random_state=random_seed, shuffle=True
    )
    
    print(f"训练集: {len(train_names)} 个文件")
    print(f"验证集: {len(val_names)} 个文件")
    
    # 创建验证集目录 (使用 Ts 后缀)
    images_val_dir = task_path / "picai_imagesTs"
    labels_val_dir = task_path / "picai_labelsTs"
    
    images_val_dir.mkdir(exist_ok=True)
    labels_val_dir.mkdir(exist_ok=True)
    
    # 移动验证集文件
    moved_count = 0
    for name in val_names:
        # 移动标签文件
        label_src = labels_dir / f"{name}.nii.gz"
        label_dst = labels_val_dir / f"{name}.nii.gz"
        
        # 移动图像文件 (0000, 0001, 0002)
        for modality in ["0000", "0001", "0002"]:
            image_src = images_dir / f"{name}_{modality}.nii.gz"
            image_dst = images_val_dir / f"{name}_{modality}.nii.gz"
            
            if image_src.exists():
                shutil.move(str(image_src), str(image_dst))
                moved_count += 1
        
        if label_src.exists():
            shutil.move(str(label_src), str(label_dst))
            moved_count += 1
    
    print(f"成功移动 {moved_count} 个文件到验证集目录")
    print(f"训练集目录: {images_dir}, {labels_dir}")
    print(f"验证集目录: {images_val_dir}, {labels_val_dir}")
    
    # 生成文件列表
    generate_file_lists(task_path, images_dir_name, labels_dir_name, train_names, val_names)
    
    return train_names, val_names

def generate_file_lists(task_path, images_dir_name, labels_dir_name, train_names, val_names):
    """生成训练集和验证集文件列表"""
    
    # 训练集文件列表
    with open(task_path / "train.txt", "w") as f:
        for name in train_names:
            f.write(f"{name}\n")
    
    # 验证集文件列表
    with open(task_path / "val.txt", "w") as f:
        for name in val_names:
            f.write(f"{name}\n")
    
    print(f"已生成 train.txt 和 val.txt 文件列表")

def verify_split(task_dir, images_dir_name, labels_dir_name):
    """验证划分结果"""
    task_path = Path(task_dir)
    
    images_dir = task_path / images_dir_name
    labels_dir = task_path / labels_dir_name
    images_val_dir = task_path / "picai_imagesTs"
    labels_val_dir = task_path / "picai_labelsTs"
    
    # 检查文件数量
    train_images = list(images_dir.glob("*.nii.gz"))
    train_labels = list(labels_dir.glob("*.nii.gz"))
    val_images = list(images_val_dir.glob("*.nii.gz"))
    val_labels = list(labels_val_dir.glob("*.nii.gz"))
    
    print(f"训练集图像: {len(train_images)} 个文件")
    print(f"训练集标签: {len(train_labels)} 个文件")
    print(f"验证集图像: {len(val_images)} 个文件")
    print(f"验证集标签: {len(val_labels)} 个文件")
    
    # 检查对应关系
    train_label_names = {f.name.replace('.nii.gz', '') for f in train_labels}
    val_label_names = {f.name.replace('.nii.gz', '') for f in val_labels}
    
    # 检查训练集图像文件是否都有对应的标签
    for image_file in train_images:
        base_name = image_file.name.split('_')[0]
        if base_name not in train_label_names:
            print(f"警告: 图像文件 {image_file.name} 没有对应的训练集标签")
    
    # 检查验证集图像文件是否都有对应的标签
    for image_file in val_images:
        base_name = image_file.name.split('_')[0]
        if base_name not in val_label_names:
            print(f"警告: 图像文件 {image_file.name} 没有对应的验证集标签")

if __name__ == "__main__":
    # 使用示例
    task_directory = "./dataset/Task205_picai_lesion/"
    images_dir = "picai_imagesTr"
    labels_dir = "picai_labelsTr"
    
    try:
        # 划分数据集
        train_files, val_files = split_train_val_dataset(
            task_directory, images_dir, labels_dir, val_ratio=0.2
        )
        
        # 验证划分结果
        print("\n验证划分结果:")
        verify_split(task_directory, images_dir, labels_dir)
        
        print("\n划分完成！")
        
    except Exception as e:
        print(f"错误: {e}")