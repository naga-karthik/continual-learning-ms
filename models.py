"""Implements the 3-layer 3D UNet and Attention 3D UNet models."""
import copy
import math

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F



def conv_norm_lrelu(feat_in, feat_out):
    """Conv3D + InstanceNorm3D + LeakyReLU block"""
    return nn.Sequential(
        nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
        nn.InstanceNorm3d(feat_out),
        nn.LeakyReLU()
    )


def norm_lrelu_conv(feat_in, feat_out):
    """InstanceNorm3D + LeakyReLU + Conv3D block"""
    return nn.Sequential(
        nn.InstanceNorm3d(feat_in),
        nn.LeakyReLU(),
        nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False)
    )


def lrelu_conv(feat_in, feat_out):
    """LeakyReLU + Conv3D block"""
    return nn.Sequential(
        nn.LeakyReLU(),
        nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False)
    )


def norm_lrelu_upscale_conv_norm_lrelu(feat_in, feat_out):
    """InstanceNorm3D + LeakyReLU + 2X Upsample + Conv3D + InstanceNorm3D + LeakyReLU block"""
    return nn.Sequential(
        nn.InstanceNorm3d(feat_in),
        nn.LeakyReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
        nn.InstanceNorm3d(feat_out),
        nn.LeakyReLU()
    )


def weights_init_kaiming(m):
    """Initialize weights according to method describe here:
    https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf
    """

    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


class GridAttentionBlockND(nn.Module):
    """Attention module to focus on important features passed through U-Net's decoder; Specific to Attention UNet
    .. seealso::
        Oktay, Ozan, et al. "Attention u-net: Learning where to look for the pancreas."
        arXiv preprint arXiv:1804.03999 (2018).
    Args:
        in_channels (int): Number of channels in the input image.
        gating_channels (int): Number of channels in the gating step.
        inter_channels (int): Number of channels in the intermediate gating step.
        dimension (int): Value of 2 or 3 to indicating whether it is used in a 2D or 3D model.
        sub_sample_factor (tuple or list): Convolution kernel size.
    Attributes:
        in_channels (int): Number of channels in the input image.
        gating_channels (int): Number of channels in the gating step.
        inter_channels (int): Number of channels in the intermediate gating step.
        dimension (int): Value of 2 or 3 to indicating whether it is used in a 2D or 3D model.
        sub_sample_factor (tuple or list): Convolution kernel size.
        upsample_mode (str): 'bilinear' or 'trilinear' related to the use of 2D or 3D models.
        W (Sequential): Sequence of convolution and batch normalization layers.
        theta (Conv2d or Conv3d): Convolution layer for gating operation.
        phi (Conv2d or Conv3d): Convolution layer for gating operation.
        psi (Conv2d or Conv3d): Convolution layer for gating operation.
    """
    def __init__(self, in_channels, gating_channels, inter_channels=None, dimension=3, sub_sample_factor=(2, 2, 2)):
        super(GridAttentionBlockND, self).__init__()

        assert dimension == 3

        # Downsampling rate for the input featuremap
        if isinstance(sub_sample_factor, tuple):
            self.sub_sample_factor = sub_sample_factor
        elif isinstance(sub_sample_factor, list):
            self.sub_sample_factor = tuple(sub_sample_factor)
        else:
            self.sub_sample_factor = tuple([sub_sample_factor]) * dimension

        # Default parameter set
        self.dimension = dimension
        self.sub_sample_kernel_size = self.sub_sample_factor

        # Number of channels (pixel dimensions)
        self.in_channels = in_channels
        self.gating_channels = gating_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            ino = nn.InstanceNorm3d     # replaced batch norm to instance norm
            self.upsample_mode = 'trilinear'
        else:
            raise NotImplementedError

        # Output transform
        self.W = nn.Sequential(
            conv_nd(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0),
            ino(self.in_channels))

        # Theta^T * x_ij + Phi^T * gating_signal + bias
        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=self.sub_sample_kernel_size, stride=self.sub_sample_factor, padding=0,
                             bias=False)
        self.phi = conv_nd(in_channels=self.gating_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0, bias=True)
        self.psi = conv_nd(in_channels=self.inter_channels, out_channels=1, kernel_size=1, stride=1, padding=0,
                           bias=True)

        # Initialise weights
        for m in self.children():
            m.apply(weights_init_kaiming)

        # Define the operation
        self.operation_function = self._concatenation

    def forward(self, x, g):
        output = self.operation_function(x, g)
        return output

    def _concatenation(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        # theta => (b, c, t, h, w) -> (b, i_c, t, h, w) -> (b, i_c, thw)
        # phi   => (b, g_d) -> (b, i_c)
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()   # same as theta_x.shape

        # g (b, c, t', h', w') -> phi_g (b, i_c, t', h', w')
        #  Relu(theta_x + phi_g + bias) -> f = (b, i_c, thw) -> (b, i_c, t/s1, h/s2, w/s3)
        phi_g = F.interpolate(self.phi(g), size=theta_x_size[2:], mode=self.upsample_mode, align_corners=True)
        f = F.relu(theta_x + phi_g, inplace=True)

        #  psi^T * f -> (b, psi_i_c, t/s1, h/s2, w/s3)
        sigm_psi_f = torch.sigmoid(self.psi(f))

        # upsample the attentions and multiply
        sigm_psi_f = F.interpolate(sigm_psi_f, size=input_size[2:], mode=self.upsample_mode, align_corners=True)
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)

        return W_y, sigm_psi_f


class UnetGridGatingSignal3(nn.Module):
    """Operation to extract important features for a specific task using 1x1x1 convolution (Gating) which is used in the
    attention blocks.
    Args:
        in_size (int): Number of channels in the input image.
        out_size (int): Number of channels in the output image.
        kernel_size (tuple): Convolution kernel size.
        is_instancenorm (bool): Boolean indicating whether to apply instance normalization or not.
    Attributes:
        conv1 (Sequential): 3D convolution, batch normalization and ReLU activation.
    """
    def __init__(self, in_size, out_size, kernel_size=(1, 1, 1), is_instancenorm=True):
        super(UnetGridGatingSignal3, self).__init__()
        if is_instancenorm:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, (1, 1, 1), (0, 0, 0)),
                                       nn.InstanceNorm3d(out_size),
                                       nn.ReLU(inplace=True))
        else:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, (1, 1, 1), (0, 0, 0)),
                                       nn.ReLU(inplace=True))

        # initialise the blocks
        for m in self.children():
            weights_init_kaiming(m)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        return outputs


# ---------------------------- ModifiedUNet3D Encoder Implementation -----------------------------
class ModifiedUNet3DEncoder(nn.Module):
    """Encoder for ModifiedUNet3D. Adapted from ivadomed.models"""
    def __init__(self, in_channels=1, base_n_filter=8, attention=False):
        super(ModifiedUNet3DEncoder, self).__init__()

        self.attention = attention

        # Initialize common operations
        self.lrelu = nn.LeakyReLU()
        self.dropout3d = nn.Dropout3d(p=0.5)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # Level 1 context pathway
        self.conv3d_c1_1 = nn.Conv3d(in_channels, base_n_filter, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3d_c1_2 = nn.Conv3d(base_n_filter, base_n_filter, kernel_size=3, stride=1, padding=1, bias=False)
        self.lrelu_conv_c1 = lrelu_conv(base_n_filter, base_n_filter)
        self.inorm3d_c1 = nn.InstanceNorm3d(base_n_filter)

        # Level 2 context pathway
        self.conv3d_c2 = nn.Conv3d(base_n_filter, base_n_filter * 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm_lrelu_conv_c2 = norm_lrelu_conv(base_n_filter * 2, base_n_filter * 2)
        self.inorm3d_c2 = nn.InstanceNorm3d(base_n_filter * 2)

        # Level 3 context pathway
        self.conv3d_c3 = nn.Conv3d(base_n_filter * 2, base_n_filter * 4, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm_lrelu_conv_c3 = norm_lrelu_conv(base_n_filter * 4, base_n_filter * 4)
        self.inorm3d_c3 = nn.InstanceNorm3d(base_n_filter * 4)

        # Level 4 context pathway
        self.conv3d_c4 = nn.Conv3d(base_n_filter * 4, base_n_filter * 8, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm_lrelu_conv_c4 = norm_lrelu_conv(base_n_filter * 8, base_n_filter * 8)
        self.inorm3d_c4 = nn.InstanceNorm3d(base_n_filter * 8)
        # level 0 localization pathway --> COMMENT/UNCOMMENT FOR 3-LEVEL UNET
        self.norm_lrelu_upscale_conv_norm_lrelu_l0 = norm_lrelu_upscale_conv_norm_lrelu(base_n_filter * 8, base_n_filter * 4)

        # # Level 5 context pathway, level 0 localization pathway
        # self.conv3d_c5 = nn.Conv3d(base_n_filter * 8, base_n_filter * 16, kernel_size=3, stride=2, padding=1, bias=False)
        # self.norm_lrelu_conv_c5 = norm_lrelu_conv(base_n_filter * 16, base_n_filter * 16)
        # self.norm_lrelu_upscale_conv_norm_lrelu_l0 = norm_lrelu_upscale_conv_norm_lrelu(base_n_filter * 16, base_n_filter * 8)

        # adding the "attention gates" part
        if self.attention:
            print("Training U-Net with Attention Gates! ")
            self.gating = UnetGridGatingSignal3(base_n_filter * 16, base_n_filter * 8, kernel_size=(1,1,1), is_instancenorm=True)

            # attention blocks
            self.attentionblock2 = GridAttentionBlockND(in_channels=base_n_filter * 2,
                                                        gating_channels=base_n_filter * 8,
                                                        inter_channels=base_n_filter * 2,
                                                        sub_sample_factor=(2, 2, 2))
            self.attentionblock3 = GridAttentionBlockND(in_channels=base_n_filter * 4,
                                                        gating_channels=base_n_filter * 8,
                                                        inter_channels=base_n_filter * 4,
                                                        sub_sample_factor=(2, 2, 2))
            self.attentionblock4 = GridAttentionBlockND(in_channels=base_n_filter * 8,
                                                        gating_channels=base_n_filter * 8,
                                                        inter_channels=base_n_filter * 8,
                                                        sub_sample_factor=(2, 2, 2))
            self.inorm3d_l0 = nn.InstanceNorm3d(base_n_filter * 16)

    def forward(self, x):
        #  Level 1 context pathway
        out = self.conv3d_c1_1(x)
        residual_1 = out
        out = self.lrelu(out)
        out = self.conv3d_c1_2(out)
        out = self.dropout3d(out)
        out = self.lrelu_conv_c1(out)

        # Element Wise Summation
        out += residual_1
        context_1 = self.lrelu(out)
        out = self.inorm3d_c1(out)
        out = self.lrelu(out)

        # Level 2 context pathway
        out = self.conv3d_c2(out)
        residual_2 = out
        out = self.norm_lrelu_conv_c2(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c2(out)
        out += residual_2
        out = self.inorm3d_c2(out)
        out = self.lrelu(out)
        context_2 = out

        # Level 3 context pathway
        out = self.conv3d_c3(out)
        residual_3 = out
        out = self.norm_lrelu_conv_c3(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c3(out)
        out += residual_3
        out = self.inorm3d_c3(out)
        out = self.lrelu(out)
        context_3 = out

        # Level 4 context pathway
        out = self.conv3d_c4(out)
        residual_4 = out
        out = self.norm_lrelu_conv_c4(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c4(out)
        out += residual_4

        # # UNCOMMENT THE LINES BELOW FOR A 4-LEVEL UNET     
        # out = self.inorm3d_c4(out)
        # out = self.lrelu(out)
        # context_4 = out

        # # Level 5
        # out = self.conv3d_c5(out)
        # residual_5 = out
        # out = self.norm_lrelu_conv_c5(out)
        # out = self.dropout3d(out)
        # out = self.norm_lrelu_conv_c5(out)
        # out += residual_5

        if self.attention:
            out = self.inorm3d_l0(out)
            out = self.lrelu(out)

            gating = self.gating(out)
            context_4, attention4 = self.attentionblock4(context_4, gating)
            context_3, attention3 = self.attentionblock3(context_3, gating)
            context_2, attention2 = self.attentionblock2(context_2, gating)

        out = self.norm_lrelu_upscale_conv_norm_lrelu_l0(out)

        context_features = [context_1, context_2, context_3]

        return out, context_features


# ---------------------------- ModifiedUNet3D Decoder Implementation -----------------------------
class ModifiedUNet3DDecoder(nn.Module):
    """Decoder for ModifiedUNet3D. Adapted from ivadomed.models"""
    def __init__(self, n_classes=1, base_n_filter=8):
        super(ModifiedUNet3DDecoder, self).__init__()

        # Initialize common operations
        self.lrelu = nn.LeakyReLU()
        self.dropout3d = nn.Dropout3d(p=0.5)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # FOR A 3-LEVEL UNET USE THIS
        self.conv3d_l0 = nn.Conv3d(base_n_filter * 4, base_n_filter * 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.inorm3d_l0 = nn.InstanceNorm3d(base_n_filter * 4)

        # Level 1 localization pathway
        self.conv_norm_lrelu_l1 = conv_norm_lrelu(base_n_filter * 8, base_n_filter * 8)
        self.conv3d_l1 = nn.Conv3d(base_n_filter * 8, base_n_filter * 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm_lrelu_upscale_conv_norm_lrelu_l1 = norm_lrelu_upscale_conv_norm_lrelu(base_n_filter * 4, base_n_filter * 2)

        # Level 2 localization pathway
        self.conv_norm_lrelu_l2 = conv_norm_lrelu(base_n_filter * 4, base_n_filter * 4)
        self.conv3d_l2 = nn.Conv3d(base_n_filter * 4, base_n_filter * 2, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm_lrelu_upscale_conv_norm_lrelu_l2 = norm_lrelu_upscale_conv_norm_lrelu(base_n_filter * 2, base_n_filter)

        # # Level 3 localization pathway
        # self.conv_norm_lrelu_l3 = conv_norm_lrelu(base_n_filter * 4, base_n_filter * 4)
        # self.conv3d_l3 = nn.Conv3d(base_n_filter * 4, base_n_filter * 2, kernel_size=1, stride=1, padding=0, bias=False)
        # self.norm_lrelu_upscale_conv_norm_lrelu_l3 = norm_lrelu_upscale_conv_norm_lrelu(base_n_filter * 2, base_n_filter)

        # Level 3 localization pathway
        self.conv_norm_lrelu_l3 = conv_norm_lrelu(base_n_filter * 2, base_n_filter * 2)
        self.conv3d_l3 = nn.Conv3d(base_n_filter * 2, n_classes, kernel_size=1, stride=1, padding=0, bias=False)

        self.ds2_1x1_conv3d = nn.Conv3d(base_n_filter * 4, n_classes, kernel_size=1, stride=1, padding=0, bias=False)
        self.ds3_1x1_conv3d = nn.Conv3d(base_n_filter * 2, n_classes, kernel_size=1, stride=1, padding=0, bias=False)

        # ----------------------------------------------------------------------------------------
        # FOR A 4-LEVEL UNET USE UNCOMMENT AND USE THIS
        # self.conv3d_l0 = nn.Conv3d(base_n_filter * 8, base_n_filter * 8, kernel_size=1, stride=1, padding=0, bias=False)
        # self.inorm3d_l0 = nn.InstanceNorm3d(base_n_filter * 8)

        # # Level 1 localization pathway
        # self.conv_norm_lrelu_l1 = conv_norm_lrelu(base_n_filter * 16, base_n_filter * 16)
        # self.conv3d_l1 = nn.Conv3d(base_n_filter * 16, base_n_filter * 8, kernel_size=1, stride=1, padding=0, bias=False)
        # self.norm_lrelu_upscale_conv_norm_lrelu_l1 = norm_lrelu_upscale_conv_norm_lrelu(base_n_filter * 8, base_n_filter * 4)

        # # Level 2 localization pathway
        # self.conv_norm_lrelu_l2 = conv_norm_lrelu(base_n_filter * 8, base_n_filter * 8)
        # self.conv3d_l2 = nn.Conv3d(base_n_filter * 8, base_n_filter * 4, kernel_size=1, stride=1, padding=0, bias=False)
        # self.norm_lrelu_upscale_conv_norm_lrelu_l2 = norm_lrelu_upscale_conv_norm_lrelu(base_n_filter * 4, base_n_filter * 2)

        # # Level 3 localization pathway
        # self.conv_norm_lrelu_l3 = conv_norm_lrelu(base_n_filter * 4, base_n_filter * 4)
        # self.conv3d_l3 = nn.Conv3d(base_n_filter * 4, base_n_filter * 2, kernel_size=1, stride=1, padding=0, bias=False)
        # self.norm_lrelu_upscale_conv_norm_lrelu_l3 = norm_lrelu_upscale_conv_norm_lrelu(base_n_filter * 2, base_n_filter)

        # # Level 4 localization pathway
        # self.conv_norm_lrelu_l4 = conv_norm_lrelu(base_n_filter * 2, base_n_filter * 2)
        # self.conv3d_l4 = nn.Conv3d(base_n_filter * 2, n_classes, kernel_size=1, stride=1, padding=0, bias=False)

        # self.ds2_1x1_conv3d = nn.Conv3d(base_n_filter * 8, n_classes, kernel_size=1, stride=1, padding=0, bias=False)
        # self.ds3_1x1_conv3d = nn.Conv3d(base_n_filter * 4, n_classes, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x, context_features):
        # Get context features from the encoder
        # context_1, context_2, context_3, context_4 = context_features
        context_1, context_2, context_3 = context_features

        out = self.conv3d_l0(x)
        out = self.inorm3d_l0(out)
        out = self.lrelu(out)
        # print(f" dec l0 shape: {out.shape}")

        # Level 1 localization pathway
        out = torch.cat([out, context_3], dim=1)
        # out = torch.cat([out, context_4], dim=1)
        out = self.conv_norm_lrelu_l1(out)
        out = self.conv3d_l1(out)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l1(out)
        # print(f" dec l1 shape: {out.shape}")

        # Level 2 localization pathway
        out = torch.cat([out, context_2], dim=1)
        out = self.conv_norm_lrelu_l2(out)
        ds2 = out
        out = self.conv3d_l2(out)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l2(out)
        # print(f" dec l2 shape: {out.shape}")

        # Level 3 localization pathway
        out = torch.cat([out, context_1], dim=1)
        # print(f"context 1 shape: {context_1.shape}")
        out = self.conv_norm_lrelu_l3(out)
        # ds3 = out
        out_pred = self.conv3d_l3(out)
        # out = self.norm_lrelu_upscale_conv_norm_lrelu_l3(out)

        # # Level 4 localization pathway
        # out = torch.cat([out, context_1], dim=1)
        # out = self.conv_norm_lrelu_l4(out)
        # out_pred = self.conv3d_l4(out)

        ds2_1x1_conv = self.ds2_1x1_conv3d(ds2)
        ds1_ds2_sum_upscale = self.upsample(ds2_1x1_conv)
        # ds3_1x1_conv = self.ds3_1x1_conv3d(ds3)
        # ds1_ds2_sum_upscale_ds3_sum = ds1_ds2_sum_upscale + ds3_1x1_conv
        # ds1_ds2_sum_upscale_ds3_sum_upscale = self.upsample(ds1_ds2_sum_upscale_ds3_sum)

        out = out_pred + ds1_ds2_sum_upscale

        # Final Activation Layer
        out = F.relu(out) / F.relu(out).max() if bool(F.relu(out).max()) else F.relu(out)

        return out


# ---------------------------- ModifiedUNet3D Implementation -----------------------------
class ModifiedUNet3D(nn.Module):
    """ModifiedUNet3D with Encoder + Decoder. Adapted from ivadomed.models"""
    def __init__(self, cfg):
        super(ModifiedUNet3D, self).__init__()
        self.cfg = cfg
        self.unet_encoder = ModifiedUNet3DEncoder(cfg, in_channels=1 if cfg.task == '1' else 2,
                                                  base_n_filter=cfg.base_n_filter, attention=cfg.attention_gates)
        self.unet_decoder = ModifiedUNet3DDecoder(cfg, n_classes=1, base_n_filter=cfg.base_n_filter)

    def forward(self, x):
        # x: (B, 1, SV, SV, SV)

        x, context_features = self.unet_encoder(x)

        # x: (B, 4 * F, SV // 4, SV // 4, SV // 4)
        # context_features: [3]
        #   0 -> (B, F, SV, SV, SV)
        #   1 -> (B, 2 * F, SV / 2, SV / 2, SV / 2)
        #   2 -> (B, 4 * F, SV / 4, SV / 4, SV / 4)

        seg_preds = self.unet_decoder(x, context_features)

        return seg_preds



if __name__ == "__main__":

    x = torch.randn((4, 1, 64, 64, 64))
    encoder = ModifiedUNet3DEncoder(in_channels=1, base_n_filter=32)
    decoder = ModifiedUNet3DDecoder(n_classes=1, base_n_filter=32)
    x, context_feats = encoder(x)
    print(x.shape)
    preds = decoder(x, context_feats)

    print(preds.shape)