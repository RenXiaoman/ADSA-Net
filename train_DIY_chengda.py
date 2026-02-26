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

#### Model
from Model.as_unetr import CDSA_Net

# Local imports
from Options.Options_x import Options_x_chengda_optimized
from dataset.dataset_nnunet import Lits_DataSet


def poly_lr(epoch, max_epochs, initial_lr, exponent=0.9):
    """
    Polynomial learning rate decay from nnUNet
    lr = initial_lr * (1 - epoch/max_epochs)^exponent
    """
    return initial_lr * (1 - epoch / max_epochs)**exponent


def main():
    # Parse options
    opt_parser = Options_x_chengda_optimized()
    opt = opt_parser.parse()
    print(f'opt.lr is {opt.lr}')
    
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
        norm_name='instance',
        dropout_rate=0.2
    ).to(device)
    
    
    # Resume training from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    best_train_loss = float('inf')
    best_val_dice = 0.0
    best_val_miou = 0.0
    best_loss_epoch = 0
    best_train_loss_epoch = 0
    best_dice_epoch = 0
    best_miou_epoch = 0
    
    # Define optimizer (AdamW with poly LR decay)
    optimizer = optim.AdamW(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), weight_decay=opt.weight_decay)
    if opt.resume:
        if Path(opt.resume).exists():
            print(f"Resuming training from: {opt.resume}")
            checkpoint = torch.load(opt.resume)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint['best_val_loss']
            best_train_loss = checkpoint.get('best_train_loss', float('inf'))
            best_val_dice = checkpoint['best_val_dice']
            best_val_miou = checkpoint['best_val_miou']
            best_loss_epoch = checkpoint['best_loss_epoch']
            best_train_loss_epoch = checkpoint.get('best_train_loss_epoch', 0)
            best_dice_epoch = checkpoint['best_dice_epoch']
            best_miou_epoch = checkpoint['best_miou_epoch']
            
            # Restore learning rate from checkpoint
            if 'current_lr' in checkpoint:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = checkpoint['current_lr']
                print(f"Resumed learning rate: {checkpoint['current_lr']:.6f}")
            
            print(f"Resumed from epoch {start_epoch}, best val loss: {best_val_loss:.4f}")
        else:
            print(f"Warning: Checkpoint {opt.resume} not found, starting from scratch")
    
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
                                    weight=[0.3, 0.7],    # Higher weight for lesion class
                                    ).to(device)
    
    # nnUNet风格多尺度损失权重（从粗到细）
    deep_supervision_weights = [0.125, 0.25, 0.5, 1.0]  # 1/8, 1/4, 1/2, 1/1分辨率权重
    
    def downsample_gt(gt, scale_factor):
        """下采样GT标签到对应尺度"""
        if scale_factor == 1:
            return gt
        
        # 计算目标尺寸
        _, _, D, H, W = gt.shape
        target_size = (D // scale_factor, H // scale_factor, W // scale_factor)
        
        # 使用最近邻插值保持标签值
        gt_down = torch.nn.functional.interpolate(
            gt.float(), 
            size=target_size, 
            mode='nearest'
        ).long()
        return gt_down
    
    def calculate_multiscale_loss(outputs, labels, loss_fn, weights):
        """计算多尺度损失"""
        if not isinstance(outputs, list):
            # 单输出模式
            return loss_fn(outputs, labels)
        
        # 多尺度输出模式
        total_loss = 0
        for i, output in enumerate(outputs):
            scale_factor = 2 ** (len(outputs) - 1 - i)  # 计算下采样比例
            gt_down = downsample_gt(labels, scale_factor)
            loss = loss_fn(output, gt_down)
            total_loss += weights[i] * loss
        
        return total_loss
    
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
    
    def calculate_miou(preds, targets):
        """Calculate mean Intersection over Union"""
        smooth = 1e-6
        preds_flat = preds.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (preds_flat * targets_flat).sum()
        union = preds_flat.sum() + targets_flat.sum() - intersection
        
        iou = (intersection + smooth) / (union + smooth)
        return iou.item()
    
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
            'best_val_loss': best_val_loss,
            'best_train_loss': best_train_loss,
            'best_val_dice': best_val_dice,
            'best_val_miou': best_val_miou,
            'best_loss_epoch': best_loss_epoch,
            'best_train_loss_epoch': best_train_loss_epoch,
            'best_dice_epoch': best_dice_epoch,
            'best_miou_epoch': best_miou_epoch,
            'current_lr': new_lr
        }, results_dir / filename)
    
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
                                 'nnUNet_train/imagesTr', 
                                 'nnUNet_train/labelsTr', 
                                 enable_augmentation=True)
    val_dataset = Lits_DataSet(Path(opt.datapath), 
                               'nnUNet_val/imagesTs', 
                               'nnUNet_val/labelsTs',
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
        
        for batch_idx, batch_data in enumerate(tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{opt.epoch} [Train]')):
            ADC, DWI, T2W, labels, patient_names = batch_data
            inputs = torch.cat([ADC, DWI, T2W], dim=1).to(device)  # 合并ADC和T2W和DWI作为输入
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            with autocast('cuda'):
                outputs = model(inputs)
                
                # 使用多尺度损失计算
                loss = calculate_multiscale_loss(outputs, labels, dice_focal_loss, deep_supervision_weights)
                
                # 使用最终输出计算指标
                if isinstance(outputs, list):
                    # nnUNet风格多尺度输出
                    output_final = outputs[-1]  # 最终输出
                    outputs_softmax = torch.softmax(output_final, dim=1)
                    preds = torch.argmax(outputs_softmax, dim=1, keepdim=True)
                else:
                    # 单输出（向后兼容）
                    outputs_softmax = torch.softmax(outputs, dim=1)
                    preds = torch.argmax(outputs_softmax, dim=1, keepdim=True)
            
            scaler.scale(loss).backward()

            # Gradient clipping to prevent explosion
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
            # 计算每个样本的Dice和mIoU（case级别）
            batch_size = preds.size(0)
            for i in range(batch_size):
                single_pred = preds[i:i+1]
                single_label = labels[i:i+1]
                dice_score = calculate_dice(single_pred, single_label)
                miou_score = calculate_miou(single_pred, single_label)
                all_train_dice_scores.append(dice_score)
                all_train_miou_scores.append(miou_score)
        
        avg_train_loss = train_loss / len(train_dataloader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_dice = 0
        val_miou = 0
        all_val_dice_scores = []  # 存储每个样本的Dice分数
        all_val_miou_scores = []  # 存储每个样本的mIoU分数
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(tqdm(val_dataloader, desc=f'Epoch {epoch+1}/{opt.epoch} [Val]')):
                ADC, DWI, T2W, labels, patient_names = batch_data
                inputs = torch.cat([ADC, DWI, T2W], dim=1).to(device)  # 合并T2W和DWI作为输入
                labels = labels.to(device)

                with autocast('cuda'):
                    outputs = model(inputs)
                    
                    # 使用多尺度损失计算
                    loss = calculate_multiscale_loss(outputs, labels, dice_focal_loss, deep_supervision_weights)
                    
                    # 使用最终输出计算指标
                    if isinstance(outputs, list):
                        # nnUNet风格多尺度输出
                        output_final = outputs[-1]  # 最终输出
                        outputs_softmax = torch.softmax(output_final, dim=1)
                        preds = torch.argmax(outputs_softmax, dim=1, keepdim=True)
                    else:
                        # 单输出（向后兼容）
                        outputs_softmax = torch.softmax(outputs, dim=1)
                        preds = torch.argmax(outputs_softmax, dim=1, keepdim=True)

                val_loss += loss.item()
                
                # 计算每个样本的Dice和mIoU（case级别）
                batch_size = preds.size(0)
                for i in range(batch_size):
                    single_pred = preds[i:i+1]
                    single_label = labels[i:i+1]
                    dice_score = calculate_dice(single_pred, single_label)
                    miou_score = calculate_miou(single_pred, single_label)
                    all_val_dice_scores.append(dice_score)
                    all_val_miou_scores.append(miou_score)
        
        avg_train_loss = train_loss / len(train_dataloader)
        avg_train_dice = sum(all_train_dice_scores) / len(all_train_dice_scores)  # 样本级别平均
        avg_train_miou = sum(all_train_miou_scores) / len(all_train_miou_scores)  # 样本级别平均
        avg_val_loss = val_loss / len(val_dataloader)
        avg_val_dice = sum(all_val_dice_scores) / len(all_val_dice_scores)  # 真正的样本级别平均
        avg_val_miou = sum(all_val_miou_scores) / len(all_val_miou_scores)  # 真正的样本级别平均
        
        # Store metrics
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_dices.append(avg_train_dice)
        val_dices.append(avg_val_dice)
        train_mious.append(avg_train_miou)
        val_mious.append(avg_val_miou)
        
        # Update learning rate using poly_lr (nnUNet style)
        new_lr = poly_lr(epoch, opt.epoch, opt.lr, 0.9)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        
        epoch_time = time.time() - epoch_start_time
        print(f'Epoch [{epoch+1}/{opt.epoch}], Time: {epoch_time:.2f}s, LR: {new_lr:.8f}, Train Loss: {avg_train_loss:.4f}, Train Dice: {avg_train_dice:.4f}, Train mIoU: {avg_train_miou:.4f}, Val Loss: {avg_val_loss:.4f}, Val Dice: {avg_val_dice:.4f}, Val mIoU: {avg_val_miou:.4f}')
        
        # Save metrics to file
        metrics_file = results_dir / 'training_metrics.txt'
        with open(metrics_file, 'a') as f:
            f.write(f'Epoch {epoch+1}: Train Loss: {avg_train_loss:.6f}, Train Dice: {avg_train_dice:.6f}, Train mIoU: {avg_train_miou:.6f}, Val Loss: {avg_val_loss:.6f}, Val Dice: {avg_val_dice:.6f}, Val mIoU: {avg_val_miou:.6f}\n')
        
        # Update best validation metrics and save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_loss_epoch = epoch + 1
            save_model_checkpoint('best_loss_model.pth')

        # Update best training loss and save best training model
        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            best_train_loss_epoch = epoch + 1
            save_model_checkpoint('best_train_loss_model.pth')
        
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            best_dice_epoch = epoch + 1
            save_model_checkpoint('best_dice_model.pth')
        
        if avg_val_miou > best_val_miou:
            best_val_miou = avg_val_miou
            best_miou_epoch = epoch + 1
        
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