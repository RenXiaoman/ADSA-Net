import os
from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
from os.path import join
import json
import time

# MONAI imports
from monai.losses import DiceFocalLoss
from monai.transforms import Compose
from monai.metrics import HausdorffDistanceMetric

#### Model
from Model.as_unetr import CDSA_Net
# from picai_baseline.unet.training_setup.neural_networks.unets import UNet

# Local imports
from Options.Options_x_PICAI import Options_x_PICAI_Improved # Options_x_PICAI_New_CNN_Encoder
from dataset.dataset_nnunet import Lits_DataSet
from utils.nnunet_augmentation import nnunet_style_augmentation


def poly_lr(epoch, max_epochs, initial_lr, exponent=0.9):
    """
    Polynomial learning rate decay from nnUNet
    lr = initial_lr * (1 - epoch/max_epochs)^exponent
    """
    return initial_lr * (1 - epoch / max_epochs)**exponent


# 备注：发现训练过程中从第30个epoch开始出现NaN损失值，这表明存在梯度爆炸或数值不稳定问题
# 解决方案：
# 1. 已添加梯度裁剪(torch.nn.utils.clip_grad_norm_)来防止梯度爆炸
# 2. 可考虑降低学习率或使用学习率预热(warmup)策略
# 3. AdamW优化器可能需要更小的学习率，比如1e-4而不是5e-4

# 如需启用学习率预热，可取消下面函数的注释并修改学习率更新部分
'''
def poly_lr_with_warmup(epoch, warmup_epochs=10, max_epochs=1000, initial_lr=1e-4, exponent=0.9):
    """
    Polynomial learning rate decay with warmup
    """
    if epoch < warmup_epochs:
        # Warmup阶段：线性增长
        return initial_lr * (epoch + 1) / warmup_epochs
    else:
        # PolyLR阶段：多项式衰减
        adjusted_epoch = epoch - warmup_epochs
        adjusted_max_epochs = max_epochs - warmup_epochs
        return initial_lr * (1 - adjusted_epoch / adjusted_max_epochs)**exponent
'''


# Metrics calculation functions
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
    

def calculate_loss(outputs, labels, loss_fn):
    """计算损失（单输出模式）"""
    return loss_fn(outputs, labels)
    
    

def calculate_miou(preds, targets):
    """Calculate mean Intersection over Union"""
    smooth = 1e-6
    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)

    intersection = (preds_flat * targets_flat).sum()
    union = preds_flat.sum() + targets_flat.sum() - intersection

    iou = (intersection + smooth) / (union + smooth)
    return iou.item()


def compute_95hd(pred, target):
    """计算95% hausdorff距离"""
    import numpy as np
    from scipy.spatial.distance import directed_hausdorff

    # 获取预测和目标的边界坐标
    pred_coords = np.where(pred)
    target_coords = np.where(target)

    if len(pred_coords[0]) == 0 or len(target_coords[0]) == 0:
        return 373.133  # 表示分割失败的大数值

    # 组合坐标 (x, y, z) 形式
    pred_points = np.column_stack(pred_coords)
    target_points = np.column_stack(target_coords)

    # 计算双向hausdorff距离
    dist_1 = directed_hausdorff(pred_points, target_points)[0]
    dist_2 = directed_hausdorff(target_points, pred_points)[0]

    # 返回hausdorff距离的最大值
    hd = max(dist_1, dist_2)
    return hd


def calculate_hd95(preds, targets):
    """Calculate 95th percentile of Hausdorff Distance"""
    # 确保输入是正确的形状 [B, H, W, D] 或其他合适的格式
    # 从独热编码转换为标签格式
    batch_size = preds.size(0)
    hd95_values = []

    for i in range(batch_size):
        single_pred = preds[i].squeeze().cpu().numpy()  # 移除批次维度和单通道维度
        single_target = targets[i].squeeze().cpu().numpy()

        # 转换为整数标签格式（背景0，前景1）
        single_pred_binary = (single_pred > 0).astype(int)
        single_target_binary = (single_target > 0).astype(int)

        # 计算hausdorff距离
        import numpy as np

        # 使用边界提取计算hausdorff距离
        if np.sum(single_pred_binary) == 0 and np.sum(single_target_binary) == 0:
            # 都是空集合，返回0
            hd95_values.append(0.0)
        elif np.sum(single_pred_binary) == 0 or np.sum(single_target_binary) == 0:
            # 一个是空集合，另一个不是，返回无穷大或一个大的值
            hd95_values.append(373.133)  # 用一个大的数值表示分割失败
        else:
            # 计算边界的hausdorff距离
            hd95_val = compute_95hd(single_pred_binary, single_target_binary)
            hd95_values.append(hd95_val)

    # 返回平均值
    if len(hd95_values) > 0:
        return np.mean(hd95_values)
    else:
        return float('inf')


def compute_95hd(pred, target):
    """计算95% hausdorff距离"""
    import numpy as np
    from scipy.spatial.distance import directed_hausdorff

    # 获取预测和目标的边界坐标
    pred_coords = np.where(pred)
    target_coords = np.where(target)

    if len(pred_coords[0]) == 0 or len(target_coords[0]) == 0:
        return 373.133  # 表示分割失败的大数值

    # 组合坐标 (x, y, z) 形式
    pred_points = np.column_stack(pred_coords)
    target_points = np.column_stack(target_coords)

    # 计算双向hausdorff距离
    dist_1 = directed_hausdorff(pred_points, target_points)[0]
    dist_2 = directed_hausdorff(target_points, pred_points)[0]

    # 返回hausdorff距离的最大值
    hd = max(dist_1, dist_2)
    return hd
    
    
def plot_result_mix(plot_data1, plot_data2, label1, label2, description, save_path, save_name, showCurrentBestLoss=None):
    plt.figure(figsize=(10, 6))
    plt.plot(plot_data1, label=label1)
    plt.plot(plot_data2, label=label2)
    
    if showCurrentBestLoss is not None:
        plt.title(str(description)+f' ({showCurrentBestLoss})')
    else:
        plt.title(description)
    plt.xlabel('Epoch')
    plt.ylabel(f'{save_name}')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(join(save_path, f'{save_name}.png'))
    plt.close()
        
        
def main():
    # Parse options
    opt_parser = Options_x_PICAI_Improved()
    opt = opt_parser.parse()
    
    def save_model_checkpoint(filename):
        """Save model checkpoint with all training metrics"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'train_dice': avg_train_dice,
            'val_dice': avg_val_dice,
            'train_miou': avg_train_miou,
            'val_miou': avg_val_miou,
            'train_hd95': avg_train_hd95,
            'val_hd95': avg_val_hd95,
            'best_val_loss': best_val_loss,
            'best_val_dice': best_val_dice,
            'best_val_miou': best_val_miou,
            'best_loss_epoch': best_loss_epoch,
            'best_dice_epoch': best_dice_epoch,
            'best_miou_epoch': best_miou_epoch,
            'current_lr': new_lr
        }, results_dir / filename)
        
        
    # Set device
    if opt.gpu_ids != '-1':
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
    # Initialize model
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
    
    
    # Resume training from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    best_val_dice = 0.0
    best_val_miou = 0.0
    best_val_hd95 = float('inf')  # 初始化为无穷大，因为HD95越小越好
    best_loss_epoch = 0
    best_dice_epoch = 0
    best_miou_epoch = 0
    best_hd95_epoch = 0
    
    # Define optimizer (using AdamW instead of SGD for potentially better convergence)
    # 注意：AdamW通常需要较小的学习率，如1e-4，以避免训练不稳定
    # optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.99, nesterov=True, weight_decay=opt.weight_decay)
    optimizer = optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay, betas=(0.9, 0.999), eps=1e-8)
    
    # Print model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params/1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params/1e6:.2f}M")

    # Define loss function - adjusted for small lesion segmentation
    dice_focal_loss = DiceFocalLoss(include_background=False,
                                    softmax=True, 
                                    to_onehot_y=True,
                                    gamma=2.0,           # Lower gamma for less extreme focal weighting
                                    weight=[0.2, 0.8],    # Higher weight for lesion class
                                    ).to(device)
    

    

    

    # Mixed precision training
    scaler = GradScaler('cuda')
    
    # Create results directory
    results_dir = Path(opt.checkpoints_dir) / opt.task_name
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize lists to store metrics
    train_losses = []
    val_losses = []
    train_dices = []
    val_dices = []
    train_mious = []
    val_mious = []
    train_hd95s = []
    val_hd95s = []
    best_val_loss = float('inf')
    best_val_dice = 0.0
    best_val_miou = 0.0
    best_loss_epoch = 0
    best_dice_epoch = 0
    best_miou_epoch = 0
    
    # Save training configuration
    config_file = results_dir / 'training_config.json'
    with open(config_file, 'w') as f:
        json.dump(vars(opt), f, indent=4)
    
    # Load dataset with augmentation for training, no augmentation for validation
    train_dataset = Lits_DataSet(Path(opt.datapath),
                                 'imagesTr', 
                                 'labelsTr', 
                                 enable_augmentation=True)
    val_dataset = Lits_DataSet(Path(opt.datapath), 
                               'imagesTs', 
                               'labelsTs',
                               enable_augmentation=False)
    
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, num_workers=opt.num_threads, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=opt.batch_size, num_workers=opt.num_threads, shuffle=False)
    
    print(f'Train dataset: {len(train_dataset)} samples, Val dataset: {len(val_dataset)} samples')

    # Training loop
    for epoch in range(start_epoch, opt.epoch):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0
        all_train_dice_scores = []  # 存储每个样本的Dice分数
        all_train_miou_scores = []  # 存储每个样本的mIoU分数
        all_train_hd95_scores = []  # 存储每个样本的HD95分数

        for batch_idx, batch_data in enumerate(tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{opt.epoch} [Train]')):
            ADC, DWI, T2W, labels, patient_names = batch_data
            inputs = torch.cat([ADC, DWI, T2W], dim=1).to(device)  # 合并ADC和T2W和DWI作为输入
            
            # inputs = torch.cat([ADC, DWI, T2W], dim=1).to(device)  # 原始输入 T2W为主模态
            # inputs = torch.cat([ADC, T2W, DWI], dim=1).to(device)    # DWI为主模态
            # inputs = torch.cat([T2W, DWI, ADC], dim=1).to(device)  # ADC为主模态
            
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            with autocast('cuda'):
                outputs = model(inputs)
                
                # 计算损失（单输出模式）
                loss = calculate_loss(outputs, labels, dice_focal_loss)
                
                # 计算预测结果
                outputs_softmax = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs_softmax, dim=1, keepdim=True)
            
            scaler.scale(loss).backward()

            # 添加梯度裁剪以防止梯度爆炸
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
            # 计算每个样本的Dice、mIoU和HD95（case级别）
            batch_size = preds.size(0)
            for i in range(batch_size):
                single_pred = preds[i:i+1]
                single_label = labels[i:i+1]
                dice_score = calculate_dice(single_pred, single_label)
                miou_score = calculate_miou(single_pred, single_label)
                hd95_score = calculate_hd95(single_pred, single_label)
                all_train_dice_scores.append(dice_score)
                all_train_miou_scores.append(miou_score)
                all_train_hd95_scores.append(hd95_score)
        
        avg_train_loss = train_loss / len(train_dataloader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_dice = 0
        val_miou = 0
        all_val_dice_scores = []  # 存储每个样本的Dice分数
        all_val_miou_scores = []  # 存储每个样本的mIoU分数
        all_val_hd95_scores = []  # 存储每个样本的HD95分数
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(tqdm(val_dataloader, desc=f'Epoch {epoch+1}/{opt.epoch} [Val]')):
                ADC, DWI, T2W, labels, patient_names = batch_data
                inputs = torch.cat([ADC, DWI, T2W], dim=1).to(device)  # 原始输入 T2W为主模态
                # inputs = torch.cat([ADC, T2W, DWI], dim=1).to(device)    # DWI为主模态
                # inputs = torch.cat([T2W, DWI, ADC], dim=1).to(device)  # ADC为主模态
                labels = labels.to(device)

                with autocast('cuda'):
                    outputs = model(inputs)
                    
                    # 计算损失（单输出模式）
                    loss = calculate_loss(outputs, labels, dice_focal_loss)
                    
                    # 计算预测结果
                    outputs_softmax = torch.softmax(outputs, dim=1)
                    preds = torch.argmax(outputs_softmax, dim=1, keepdim=True)

                val_loss += loss.item()
                
                # 计算每个样本的Dice、mIoU和HD95（case级别）
                batch_size = preds.size(0)
                for i in range(batch_size):
                    single_pred = preds[i:i+1]
                    single_label = labels[i:i+1]
                    dice_score = calculate_dice(single_pred, single_label)
                    miou_score = calculate_miou(single_pred, single_label)
                    hd95_score = calculate_hd95(single_pred, single_label)
                    all_val_dice_scores.append(dice_score)
                    all_val_miou_scores.append(miou_score)
                    all_val_hd95_scores.append(hd95_score)
        
        avg_train_loss = train_loss / len(train_dataloader)
        avg_train_dice = sum(all_train_dice_scores) / len(all_train_dice_scores)  # 样本级别平均
        avg_train_miou = sum(all_train_miou_scores) / len(all_train_miou_scores)  # 样本级别平均
        avg_train_hd95 = sum(all_train_hd95_scores) / len(all_train_hd95_scores)  # 样本级别平均
        avg_val_loss = val_loss / len(val_dataloader)
        avg_val_dice = sum(all_val_dice_scores) / len(all_val_dice_scores)  # 真正的样本级别平均
        avg_val_miou = sum(all_val_miou_scores) / len(all_val_miou_scores)  # 真正的样本级别平均
        avg_val_hd95 = sum(all_val_hd95_scores) / len(all_val_hd95_scores)  # 真正的样本级别平均
        
        # Store metrics
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_dices.append(avg_train_dice)
        val_dices.append(avg_val_dice)
        train_mious.append(avg_train_miou)
        val_mious.append(avg_val_miou)
        train_hd95s.append(avg_train_hd95)
        val_hd95s.append(avg_val_hd95)
        
        # Update learning rate using poly_lr (nnUNet style)
        new_lr = poly_lr(epoch + 1, opt.epoch, opt.lr, 0.9)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        
        epoch_time = time.time() - epoch_start_time
        print(f'Epoch [{epoch+1}/{opt.epoch}], Time: {epoch_time:.2f}s, LR: {new_lr:.6f}, Train Loss: {avg_train_loss:.4f}, Train Dice: {avg_train_dice:.4f}, Train mIoU: {avg_train_miou:.4f}, Val Loss: {avg_val_loss:.4f}, Val Dice: {avg_val_dice:.4f}, Val mIoU: {avg_val_miou:.4f}')
        
        # Save metrics to file
        metrics_file = results_dir / 'training_metrics.txt'
        with open(metrics_file, 'a') as f:
            f.write(f'Epoch {epoch+1}: Train Loss: {avg_train_loss:.6f}, Train Dice: {avg_train_dice:.6f}, Train mIoU: {avg_train_miou:.6f}, Val Loss: {avg_val_loss:.6f}, Val Dice: {avg_val_dice:.6f}, Val mIoU: {avg_val_miou:.6f}\n')
        
        # Update best validation metrics and save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_loss_epoch = epoch + 1
            save_model_checkpoint('best_loss_model.pth')
        
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            best_dice_epoch = epoch + 1
            save_model_checkpoint('best_dice_model.pth')

        if avg_val_miou > best_val_miou:
            best_val_miou = avg_val_miou
            best_miou_epoch = epoch + 1

        if avg_val_hd95 < best_val_hd95:  # HD95越小越好
            best_val_hd95 = avg_val_hd95
            best_hd95_epoch = epoch + 1
            save_model_checkpoint('best_hd95_model.pth')

        # Always save latest model for resume training
        save_model_checkpoint('model_latest.pth')
        
        # Save current model
        if (epoch + 1) % opt.model_save_fre == 0:
            save_model_checkpoint(f'model_epoch_{epoch+1}.pth')
        
        # Plot results every epoch
        plot_result_mix(train_losses, val_losses, 'Train_Loss', 'Val_Loss', 'Training and Validation Loss', str(results_dir), 'Loss_Curve', f'Best Loss: {best_val_loss:.6f} (Epoch {best_loss_epoch})')
        plot_result_mix(train_dices, val_dices, 'Train_Dice', 'Val_Dice', 'Training and Validation Dice', str(results_dir), 'Dice_Curve', f'Best Dice: {best_val_dice:.6f} (Epoch {best_dice_epoch})')
        plot_result_mix(train_mious, val_mious, 'Train_mIoU', 'Val_mIoU', 'Training and Validation mIoU', str(results_dir), 'mIoU_Curve', f'Best mIoU: {best_val_miou:.6f} (Epoch {best_miou_epoch})')
    
    print("Training completed!")


if __name__ == "__main__":
    main()