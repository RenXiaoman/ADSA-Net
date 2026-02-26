########################   Chengda   #######################
# Ours
python calculate_metrics_only.py --pred_dir infer/111111_infer/val_results --gt_dir dataset/ChengdaOnlyCSPca/nnUNet_val/labelsTs --output_file metrics/chengda/Ours.json

# ALIEN_Net
python calculate_metrics_only.py --pred_dir infer/SegTumor_ALIEN_chengda_infer/val_results --gt_dir dataset/ChengdaOnlyCSPca/nnUNet_val/labelsTs --output_file metrics/chengda/ALIEN_Net.json

# 3D UNet
python calculate_metrics_only.py --pred_dir infer/SegTumor_UNet_chengda_infer/val_results --gt_dir dataset/ChengdaOnlyCSPca/nnUNet_val/labelsTs --output_file metrics/chengda/UNet.json

# Attention_UNet
python calculate_metrics_only.py --pred_dir infer/SegTumor_Attention_UNet_chengda_infer/val_results --gt_dir dataset/ChengdaOnlyCSPca/nnUNet_val/labelsTs --output_file metrics/chengda/Attention_UNet.json

# UNETR
python calculate_metrics_only.py --pred_dir infer/SegTumor_UNETR_chengda_infer/val_results --gt_dir dataset/ChengdaOnlyCSPca/nnUNet_val/labelsTs --output_file metrics/chengda/UNETR.json

########################   PI-CAI   #######################
# Ours
python calculate_metrics_only.py --pred_dir infer/Cssssssssssssssshcsd_infer/val_results --gt_dir dataset/PI-CAI/labelsTs --output_file metrics/picai/Ours.json

# ALIEN_Net
python calculate_metrics_only.py --pred_dir infer/SegTumor_ALIEN_PICAI_infer/val_results --gt_dir dataset/PI-CAI/labelsTs --output_file metrics/picai/ALIEN_Net.json

# 3D UNet
python calculate_metrics_only.py --pred_dir infer/SegTumor_UNet_PICAI_infer/val_results --gt_dir dataset/PI-CAI/labelsTs --output_file metrics/picai/UNet.json

# Attention_UNet
python calculate_metrics_only.py --pred_dir infer/SegTumor_Attention_UNet_PICAI_infer/val_results --gt_dir dataset/PI-CAI/labelsTs --output_file metrics/picai/Attention_UNet.json

# UNETR
python calculate_metrics_only.py --pred_dir infer/SegTumor_UNETR_PICAI_infer/val_results --gt_dir dataset/PI-CAI/labelsTs --output_file metrics/picai/UNETR.json


