conda activate rxm
cd /home/ikun_server/clib/PycharmProjects/MoSID

python visualize_nii_slices.py \
 /home/ikun_server/clib/PycharmProjects/MoSID/dataset/ChengdaOnlyCSPca/nnUNet_val/imagesTs/HeLiZhi_0000.nii.gz \
 ./output \
 --mode single \
 --slice_index 6 \
 --label_path /home/ikun_server/clib/PycharmProjects/MoSID/dataset/ChengdaOnlyCSPca/nnUNet_val/labelsTs/HeLiZhi.nii.gz \


#######   PICAI   #######
python compare_models.py --inputs infer/SegTumor_DIY_PICAI_New_CNN_Encoder_infer/val_results/A_Summary.json --names "Ours"

python compare_models.py --inputs infer/SegTumor_ALIEN_PICAI_infer/val_results/A_Summary.json --names "ALIEN_Net"  --existing model_comparison.xlsx --output model_comparison.xlsx

python compare_models.py --inputs infer/SegTumor_UNet_PICAI_infer/val_results/A_Summary.json --names "3D UNet" 

python compare_models.py --inputs infer/SegTumor_Attention_UNet_PICAI_infer/val_results/A_Summary.json --names "Attention_UNet"  --existing model_comparison.xlsx --output model_comparison.xlsx

python compare_models.py --inputs infer/SegTumor_UNETR_PICAI_infer/val_results/A_Summary.json --names "UNETR"  --existing model_comparison.xlsx --output model_comparison.xlsx

python compare_models.py --inputs ../BMA-Net/DATASET/nnUNet_inference/Task005_PICAI/result/4/summary.json --names "BMA_Net"  --existing model_comparison.xlsx --output model_comparison.xlsx

python compare_models.py --inputs ../nnFormer/inferTs/Task005_PICAI/output_2/summary.json --names "nnFormer"  --existing model_comparison.xlsx --output model_comparison.xlsx

python compare_models.py --inputs ../picai_baseline/OUTPUT_2205/4/summary.json            --names "nnUNetv2"  --existing model_comparison.xlsx --output model_comparison.xlsx

python compare_models.py --inputs ../HmsU-Net/DATASET/nnUNet_inference/Task005_PICAI/result/0/summary.json --names "HmsU-Net"  --existing model_comparison.xlsx --output model_comparison.xlsx


#######   Chengda   #######
python compare_models.py --inputs infer/SegTumor_DIY_New_CNN_Encoder_infer/val_results/A_Summary.json --names "Ours"

python compare_models.py --inputs infer/SegTumor_ALIEN_chengda_infer/val_results/A_Summary.json --names "ALIEN_Net"  --existing model_comparison.xlsx --output model_comparison.xlsx

python compare_models.py --inputs infer/SegTumor_UNet_chengda_infer/val_results/A_Summary.json --names "3D UNet" 


python compare_models.py --inputs infer/SegTumor_Attention_UNet_chengda_infer/val_results/A_Summary.json --names "Attention_UNet"  --existing model_comparison.xlsx --output model_comparison.xlsx

python compare_models.py --inputs infer/SegTumor_UNETR_chengda_infer/val_results/A_Summary.json --names "UNETR"  --existing model_comparison.xlsx --output model_comparison.xlsx

python compare_models.py --inputs ../BMA-Net/DATASET/nnUNet_inference/Task004_prostate/result/2/summary.json --names "BMA_Net"  --existing model_comparison.xlsx --output model_comparison.xlsx

python compare_models.py --inputs ../nnFormer/inferTs/Task004_prostate/output_0/summary.json --names "nnFormer"  --existing model_comparison.xlsx --output model_comparison.xlsx

python compare_models.py --inputs ../picai_baseline/OUTPUT_2204/2/summary.json            --names "nnUNetv2"  --existing model_comparison.xlsx --output model_comparison.xlsx

python compare_models.py --inputs ../HmsU-Net/DATASET/nnUNet_inference/Task004_prostate/result/4/summary.json --names "HmsU-Net"  --existing model_comparison.xlsx --output model_comparison.xlsx


#######   Chengda Baseline   #######
python compare_models.py --inputs infer/SegTumor_DIY_chengda_Backbone_infer/val_results/A_Summary.json --names "Baseline" --output ablation.xlsx

python compare_models.py --inputs infer/SegTumor_DIY_chengda_Backbone_MRE_infer/val_results/A_Summary.json --names "Baseline+MRE"  --existing ablation.xlsx --output ablation.xlsx

python compare_models.py --inputs infer/SegTumor_DIY_chengda_Backbone_ACF_infer/val_results/A_Summary.json --names "Baseline+ACF"   --existing ablation.xlsx --output ablation.xlsx

python compare_models.py --inputs infer/SegTumor_DIY_chengda_Backbone_SAEB_infer/val_results/A_Summary.json --names "Baseline+SAEB" --existing ablation.xlsx --output ablation.xlsx

python compare_models.py --inputs infer/SegTumor_DIY_Chengda_NewFusion_infer/val_results/A_Summary.json --names "Baseline+MRE+ACF"  --existing ablation.xlsx --output ablation.xlsx

python compare_models.py --inputs infer/SegTumor_DIY_New_CNN_Encoder_infer/val_results/A_Summary.json --names "ADSA-Net"  --existing ablation.xlsx --output ablation.xlsx


#######   PI-CAI Baseline   #######
python compare_models.py --inputs infer/SegTumor_DIY_PICAI_Backbone_infer/val_results/A_Summary.json --names "Baseline" --output ablation_PICAI.xlsx

python compare_models.py --inputs infer/SegTumor_DIY_PICAI_Backbone_MRE_infer/val_results/A_Summary.json --names "Baseline+MRE"  --existing ablation_PICAI.xlsx --output ablation_PICAI.xlsx

python compare_models.py --inputs infer/SegTumor_DIY_PICAI_Backbone_ACF_infer/val_results/A_Summary.json --names "Baseline+ACF"   --existing ablation_PICAI.xlsx --output ablation_PICAI.xlsx

python compare_models.py --inputs infer/SegTumor_DIY_PICAI_Backbone_SAEB_infer/val_results/A_Summary.json --names "Baseline+SAEB" --existing ablation_PICAI.xlsx --output ablation_PICAI.xlsx

python compare_models.py --inputs infer/SegTumor_DIY_PICAI_Backbone_MRE_ACF_infer/val_results/A_Summary.json --names "Baseline+MRE+ACF"  --existing ablation_PICAI.xlsx --output ablation_PICAI.xlsx

python compare_models.py --inputs infer/SegTumor_DIY_PICAI_New_CNN_Encoder_infer/val_results/A_Summary.json --names "ADSA-Net"  --existing ablation_PICAI.xlsx --output ablation_PICAI.xlsx
