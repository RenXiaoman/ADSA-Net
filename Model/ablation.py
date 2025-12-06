"""
Ablation Study Models for ADSA-Net
包含5个消融实验模型:
1. Backbone_Baseline: 简单concat融合，不使用MRE/ACF/SAEB
2. Backbone_MRE: 仅使用MRE模块
3. Backbone_ACF: 仅使用ACF模块
4. Backbone_MRE_ACF: Baseline + MRE + ACF (不使用DCFB)
5. Backbone_SAEB: Baseline + SAEB结构感知增强模块
"""

from __future__ import annotations
from collections.abc import Sequence

import torch
import torch.nn as nn

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from monai.networks.blocks.convolutions import Convolution, ResidualUnit
from monai.networks.layers.factories import Norm
from monai.networks.nets.vit import ViT
from monai.utils import deprecated_arg, ensure_tuple_rep

from .multiscale import DCFB
from .as_unetr import (
    ConvBlock, MRE, ACF, GRE_DCGF3D,
    normalization, GeneralConv3dPreNorm
)


# ============================================================================
# Baseline融合模块: 简单Concat + 1x1 Conv
# ============================================================================
class SimpleConcat3D(nn.Module):
    """
    Baseline融合策略：简单拼接后1x1卷积降维
    输入: T2 [B, C, D, H, W], Aux [B, C, D, H, W]
    输出: [B, C, D, H, W]
    """
    def __init__(self, in_channel: int, norm='in'):
        super().__init__()
        self.fusion_conv = nn.Sequential(
            nn.Conv3d(in_channel * 2, in_channel, kernel_size=1, bias=False),
            normalization(in_channel, norm=norm),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, T2, Aux):
        concat = torch.cat([T2, Aux], dim=1)  # [B, 2C, D, H, W]
        out = self.fusion_conv(concat)        # [B, C, D, H, W]
        return out


# ============================================================================
# MRE-only融合模块
# ============================================================================
class MREOnly3D(nn.Module):
    """
    仅使用MRE进行粗粒度融合
    输入: T2 [B, C, D, H, W], Aux [B, C, D, H, W]
    输出: coarse [B, C, D, H, W]
    """
    def __init__(self, in_channel: int, norm='in'):
        super().__init__()
        self.mre = MRE(in_channel)
    
    def forward(self, T2, Aux):
        coarse, w_t2, w_aux = self.mre(T2, Aux)
        return coarse


# ============================================================================
# ACF-only融合模块
# ============================================================================
class ACFOnly3D(nn.Module):
    """
    仅使用ACF进行细粒度融合（不经过MRE权重调制）
    输入: T2 [B, C, D, H, W], Aux [B, C, D, H, W]
    输出: fine [B, C, D, H, W]
    """
    def __init__(self, in_channel: int, norm='in'):
        super().__init__()
        self.acf = ACF(in_channel, norm=norm)
    
    def forward(self, T2, Aux):
        fine = self.acf(T2, Aux)
        return fine


# ============================================================================
# 1. Backbone_Baseline: 简单Concat融合
# ============================================================================
class Backbone_Baseline(nn.Module):
    """
    消融实验1: Baseline
    使用简单concat融合，不使用MRE/ACF/SAEB
    """
    @deprecated_arg(
        name="pos_embed", since="1.2", removed="1.4", new_name="proj_type", msg_suffix="please use `proj_type` instead."
    )
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: Sequence[int] | int,
        feature_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        pos_embed: str = "conv",
        proj_type: str = "conv",
        norm_name: tuple | str = "instance",
        conv_block: bool = True,
        res_block: bool = True,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        qkv_bias: bool = False,
        save_attn: bool = False,
    ) -> None:
        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.num_layers = 12
        img_size = ensure_tuple_rep(img_size, spatial_dims)
        self.patch_size = ensure_tuple_rep(16, spatial_dims)
        self.feat_size = tuple(img_d // p_d for img_d, p_d in zip(img_size, self.patch_size))
        self.hidden_size = hidden_size
        self.classification = False
        
        # ViT Encoder
        self.vit = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            proj_type=proj_type,
            classification=self.classification,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
            qkv_bias=qkv_bias,
            save_attn=save_attn,
        )
        
        # Encoder blocks
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 2,
            num_layer=2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder4 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        
        # Decoder blocks
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        
        self.out = UnetOutBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=out_channels
        )
        
        self.proj_axes = (0, spatial_dims + 1) + tuple(d + 1 for d in range(spatial_dims))
        self.proj_view_shape = list(self.feat_size) + [self.hidden_size]
        
        # CNN Branch (T2W)
        self.conv_head = ConvBlock(
            spatial_dims=spatial_dims,
            in_channels=1,
            out_channels=16,
            dropout=dropout_rate,
            kernel_size=3,
        )
        
        self.trans_1 = Convolution(
            spatial_dims=spatial_dims,
            in_channels=16,
            out_channels=16,
            strides=1,
            kernel_size=3,
        )
        
        self.s_encoder1 = ResidualUnit(
            spatial_dims=spatial_dims,
            in_channels=16,
            out_channels=32,
            strides=2,
            kernel_size=3,
            adn_ordering="NDA",
            act=("leakyrelu", {"negative_slope": 0.2}),
            norm=Norm.BATCH,
            dropout=dropout_rate,
            dropout_dim=1
        )
        
        self.trans_2 = ConvBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * 2,
            out_channels=32,
            dropout=dropout_rate,
            kernel_size=3,
            strides=1,
        )
        
        self.s_encoder2 = ResidualUnit(
            spatial_dims=spatial_dims,
            in_channels=32,
            out_channels=64,
            strides=2,
            kernel_size=3,
            adn_ordering="NDA",
            act=("leakyrelu", {"negative_slope": 0.2}),
            norm=Norm.BATCH,
            dropout=dropout_rate,
            dropout_dim=1
        )
        
        self.trans_3 = ConvBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * 4,
            out_channels=64,
            dropout=dropout_rate,
            kernel_size=3,
            strides=1,
        )
        
        self.s_encoder3 = ResidualUnit(
            spatial_dims=spatial_dims,
            in_channels=64,
            out_channels=128,
            strides=2,
            kernel_size=3,
            adn_ordering="NDA",
            act=("leakyrelu", {"negative_slope": 0.2}),
            norm=Norm.BATCH,
            dropout=dropout_rate,
            dropout_dim=1
        )
        
        self.trans_4 = ConvBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * 8,
            out_channels=128,
            dropout=dropout_rate,
            kernel_size=3,
            strides=1,
        )
        
        self.s_encoder4 = ResidualUnit(
            spatial_dims=spatial_dims,
            in_channels=128,
            out_channels=256,
            strides=2,
            kernel_size=3,
            adn_ordering="NDA",
            act=("leakyrelu", {"negative_slope": 0.2}),
            norm=Norm.BATCH,
            dropout=dropout_rate,
            dropout_dim=1
        )
        
        self.conv1x1 = nn.Conv3d(
            in_channels=256,
            out_channels=768,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        
        self.trans_5 = ConvBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=hidden_size,
            dropout=dropout_rate,
            kernel_size=3,
            strides=1,
        )
        
        # 使用SimpleConcat融合模块替代GRE_DCGF3D
        self.BiCR_1 = SimpleConcat3D(in_channel=16, norm='in')
        self.BiCR_2 = SimpleConcat3D(in_channel=32, norm='in')
        self.BiCR_3 = SimpleConcat3D(in_channel=64, norm='in')
        self.BiCR_4 = SimpleConcat3D(in_channel=128, norm='in')
        self.BiCR_5 = SimpleConcat3D(in_channel=hidden_size, norm='in')

    def proj_feat(self, x):
        new_view = [x.size(0)] + self.proj_view_shape
        x = x.view(new_view)
        x = x.permute(self.proj_axes).contiguous()
        return x

    def forward(self, x_input):
        x_in, x_single = x_input[:, 0:2, :, :, :], x_input[:, 2:3, :, :, :]
        
        # ViT Encoder
        x, hidden_states_out = self.vit(x_in)
        
        enc1 = self.encoder1(x_in)
        
        x2 = hidden_states_out[3]
        enc2 = self.encoder2(self.proj_feat(x2))
        
        x3 = hidden_states_out[6]
        enc3 = self.encoder3(self.proj_feat(x3))
        
        x4 = hidden_states_out[9]
        enc4 = self.encoder4(self.proj_feat(x4))
        
        dec4 = self.proj_feat(x)
        
        # CNN Branch
        x_single_conv = self.conv_head(x_single)
        
        merge_x_1 = self.BiCR_1(x_single_conv, enc1)
        merge_x_1 = self.trans_1(merge_x_1)
        
        x2_single_in = self.s_encoder1(merge_x_1)
        merge_x_2 = self.BiCR_2(x2_single_in, enc2)
        merge_x_2 = self.trans_2(merge_x_2)
        
        x3_single_in = self.s_encoder2(merge_x_2)
        merge_x_3 = self.BiCR_3(x3_single_in, enc3)
        merge_x_3 = self.trans_3(merge_x_3)
        
        x_4_single_in = self.s_encoder3(merge_x_3)
        merge_x_4 = self.BiCR_4(x_4_single_in, enc4)
        merge_x_4 = self.trans_4(merge_x_4)
        
        x5_single_in = self.s_encoder4(merge_x_4)
        x5_single_in = self.conv1x1(x5_single_in)
        
        merge_x_5 = self.BiCR_5(x5_single_in, dec4)
        merge_x_5 = self.trans_5(merge_x_5)
        
        # Decoder
        dec3 = self.decoder4(merge_x_5, merge_x_4)
        dec2 = self.decoder3(dec3, merge_x_3)
        dec1 = self.decoder2(dec2, merge_x_2)
        out = self.decoder1(dec1, merge_x_1)
        
        output_final = self.out(out)
        return output_final


# ============================================================================
# 2. Backbone_MRE: 仅使用MRE模块
# ============================================================================
class Backbone_MRE(Backbone_Baseline):
    """
    消融实验2: Backbone + MRE
    仅使用MRE进行样本级全局可靠性评估
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 替换融合模块为MRE-only
        self.BiCR_1 = MREOnly3D(in_channel=16, norm='in')
        self.BiCR_2 = MREOnly3D(in_channel=32, norm='in')
        self.BiCR_3 = MREOnly3D(in_channel=64, norm='in')
        self.BiCR_4 = MREOnly3D(in_channel=128, norm='in')
        self.BiCR_5 = MREOnly3D(in_channel=self.hidden_size, norm='in')


# ============================================================================
# 3. Backbone_ACF: 仅使用ACF模块
# ============================================================================
class Backbone_ACF(Backbone_Baseline):
    """
    消融实验3: Backbone + ACF
    仅使用ACF进行体素级细粒度置信度融合
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 替换融合模块为ACF-only
        self.BiCR_1 = ACFOnly3D(in_channel=16, norm='in')
        self.BiCR_2 = ACFOnly3D(in_channel=32, norm='in')
        self.BiCR_3 = ACFOnly3D(in_channel=64, norm='in')
        self.BiCR_4 = ACFOnly3D(in_channel=128, norm='in')
        self.BiCR_5 = ACFOnly3D(in_channel=self.hidden_size, norm='in')


# ============================================================================
# 4. Backbone_MRE_ACF: Baseline + MRE + ACF (不使用DCFB)
# ============================================================================
class Backbone_MRE_ACF(Backbone_Baseline):
    """
    消融实验4: Backbone + MRE + ACF
    使用GRE_DCGF3D融合模块（包含MRE和ACF），但不使用DCFB模块
    这是ADSA_Net去掉DCFB后的版本
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 替换融合模块为GRE_DCGF3D（包含MRE和ACF）
        # 不使用SimpleConcat，而是使用MRE+ACF的融合方式
        self.BiCR_1 = GRE_DCGF3D(in_channel=16, norm='in')
        self.BiCR_2 = GRE_DCGF3D(in_channel=32, norm='in')
        self.BiCR_3 = GRE_DCGF3D(in_channel=64, norm='in')
        self.BiCR_4 = GRE_DCGF3D(in_channel=128, norm='in')
        self.BiCR_5 = GRE_DCGF3D(in_channel=self.hidden_size, norm='in')
    
    def forward(self, x_input):
        x_in, x_single = x_input[:, 0:2, :, :, :], x_input[:, 2:3, :, :, :]
        
        # ViT Encoder
        x, hidden_states_out = self.vit(x_in)
        
        enc1 = self.encoder1(x_in)
        
        x2 = hidden_states_out[3]
        enc2 = self.encoder2(self.proj_feat(x2))
        
        x3 = hidden_states_out[6]
        enc3 = self.encoder3(self.proj_feat(x3))
        
        x4 = hidden_states_out[9]
        enc4 = self.encoder4(self.proj_feat(x4))
        
        dec4 = self.proj_feat(x)
        
        # CNN Branch (不使用DCFB模块)
        x_single_conv = self.conv_head(x_single)
        
        merge_x_1 = self.BiCR_1(x_single_conv, enc1)
        merge_x_1 = self.trans_1(merge_x_1)
        
        x2_single_in = self.s_encoder1(merge_x_1)
        # 不使用dcfb1
        merge_x_2 = self.BiCR_2(x2_single_in, enc2)
        merge_x_2 = self.trans_2(merge_x_2)
        
        x3_single_in = self.s_encoder2(merge_x_2)
        # 不使用dcfb2
        merge_x_3 = self.BiCR_3(x3_single_in, enc3)
        merge_x_3 = self.trans_3(merge_x_3)
        
        x_4_single_in = self.s_encoder3(merge_x_3)
        # 不使用dcfb3
        merge_x_4 = self.BiCR_4(x_4_single_in, enc4)
        merge_x_4 = self.trans_4(merge_x_4)
        
        x5_single_in = self.s_encoder4(merge_x_4)
        # 不使用dcfb4
        x5_single_in = self.conv1x1(x5_single_in)
        # 不使用dcfb5
        merge_x_5 = self.BiCR_5(x5_single_in, dec4)
        merge_x_5 = self.trans_5(merge_x_5)
        
        # Decoder
        dec3 = self.decoder4(merge_x_5, merge_x_4)
        dec2 = self.decoder3(dec3, merge_x_3)
        dec1 = self.decoder2(dec2, merge_x_2)
        out = self.decoder1(dec1, merge_x_1)
        
        output_final = self.out(out)
        return output_final


# ============================================================================
# 5. Backbone_SAEB: Baseline + SAEB结构感知增强模块
# ============================================================================
class Backbone_SAEB(Backbone_Baseline):
    """
    消融实验4: Backbone + SAEB
    使用简单concat融合 + SAEB结构感知增强模块
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 添加DCFB (SAEB) 模块
        self.dcfb1 = DCFB(
            in_channels=32,
            out_channels=32,
            stride=1,
            kernel_sizes=[1, 3, 5],
            expansion_factor=2,
            dw_parallel=True,
            add=True,
            activation='relu6'
        )
        self.dcfb2 = DCFB(
            in_channels=64,
            out_channels=64,
            stride=1,
            kernel_sizes=[1, 3, 5],
            expansion_factor=2,
            dw_parallel=True,
            add=True,
            activation='relu6'
        )
        self.dcfb3 = DCFB(
            in_channels=128,
            out_channels=128,
            stride=1,
            kernel_sizes=[1, 3, 5],
            expansion_factor=2,
            dw_parallel=True,
            add=True,
            activation='relu6'
        )
        self.dcfb4 = DCFB(
            in_channels=256,
            out_channels=256,
            stride=1,
            kernel_sizes=[1, 3, 5],
            expansion_factor=2,
            dw_parallel=True,
            add=True,
            activation='relu6'
        )
        self.dcfb5 = DCFB(
            in_channels=768,
            out_channels=768,
            stride=1,
            kernel_sizes=[1, 3, 5],
            expansion_factor=2,
            dw_parallel=True,
            add=True,
            activation='relu6'
        )

    def forward(self, x_input):
        x_in, x_single = x_input[:, 0:2, :, :, :], x_input[:, 2:3, :, :, :]
        
        # ViT Encoder
        x, hidden_states_out = self.vit(x_in)
        
        enc1 = self.encoder1(x_in)
        
        x2 = hidden_states_out[3]
        enc2 = self.encoder2(self.proj_feat(x2))
        
        x3 = hidden_states_out[6]
        enc3 = self.encoder3(self.proj_feat(x3))
        
        x4 = hidden_states_out[9]
        enc4 = self.encoder4(self.proj_feat(x4))
        
        dec4 = self.proj_feat(x)
        
        # CNN Branch with SAEB
        x_single_conv = self.conv_head(x_single)
        
        merge_x_1 = self.BiCR_1(x_single_conv, enc1)
        merge_x_1 = self.trans_1(merge_x_1)
        
        x2_single_in = self.s_encoder1(merge_x_1)
        x2_single_in = self.dcfb1(x2_single_in)  # 添加SAEB
        merge_x_2 = self.BiCR_2(x2_single_in, enc2)
        merge_x_2 = self.trans_2(merge_x_2)
        
        x3_single_in = self.s_encoder2(merge_x_2)
        x3_single_in = self.dcfb2(x3_single_in)  # 添加SAEB
        merge_x_3 = self.BiCR_3(x3_single_in, enc3)
        merge_x_3 = self.trans_3(merge_x_3)
        
        x_4_single_in = self.s_encoder3(merge_x_3)
        x_4_single_in = self.dcfb3(x_4_single_in)  # 添加SAEB
        merge_x_4 = self.BiCR_4(x_4_single_in, enc4)
        merge_x_4 = self.trans_4(merge_x_4)
        
        x5_single_in = self.s_encoder4(merge_x_4)
        x5_single_in = self.dcfb4(x5_single_in)  # 添加SAEB
        x5_single_in = self.conv1x1(x5_single_in)
        x5_single_in = self.dcfb5(x5_single_in)  # 添加SAEB
        
        merge_x_5 = self.BiCR_5(x5_single_in, dec4)
        merge_x_5 = self.trans_5(merge_x_5)
        
        # Decoder
        dec3 = self.decoder4(merge_x_5, merge_x_4)
        dec2 = self.decoder3(dec3, merge_x_3)
        dec1 = self.decoder2(dec2, merge_x_2)
        out = self.decoder1(dec1, merge_x_1)
        
        output_final = self.out(out)
        return output_final


# ============================================================================
# 测试代码
# ============================================================================
if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    
    # 测试所有消融模型
    models = {
        "Baseline": Backbone_Baseline,
        "MRE": Backbone_MRE,
        "ACF": Backbone_ACF,
        "MRE_ACF": Backbone_MRE_ACF,
        "SAEB": Backbone_SAEB,
    }
    
    for name, ModelClass in models.items():
        print(f"\n{'='*50}")
        print(f"Testing {name}")
        print(f"{'='*50}")
        
        model = ModelClass(
            in_channels=2,
            out_channels=2,
            img_size=(16, 256, 256),
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=8,
            norm_name='instance'
        ).to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f'总参数量: {total_params / 1e6:.2f}M')
        print(f'可训练参数量: {trainable_params / 1e6:.2f}M')
        
        model.eval()
        x = torch.rand(2, 3, 16, 256, 256).to(device)
        
        with torch.no_grad():
            out = model(x)
        