"""FocalMatter-style decoder for the six StarMatting feature scales."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from maggie.network.encoder.pvt_star import PVTEncoderLayer
from maggie.network.module.transformer import nchw_to_nlc, nlc_to_nchw


class ContextAggregation(nn.Module):
    """Parallel depthwise atrous context aggregation at the coarsest scale."""

    def __init__(self, in_channels, out_channels, rates=(1, 2, 4, 8)):
        super().__init__()
        if out_channels % len(rates) != 0:
            raise ValueError('out_channels must be divisible by the number of rates')
        branch_channels = out_channels // len(rates)
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=rate,
                          dilation=rate, groups=in_channels, bias=False),
                nn.Conv2d(in_channels, branch_channels, 1, bias=False),
                nn.BatchNorm2d(branch_channels),
                nn.GELU(),
            )
            for rate in rates
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.fuse(torch.cat([branch(x) for branch in self.branches], dim=1))


class SkipAlignGate(nn.Module):
    """Project a skip feature and learn a channel-wise relevance gate."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        hidden_channels = max(out_channels // 4, 8)
        self.align = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, hidden_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, skip, output_size):
        skip = self.align(skip)
        if skip.shape[-2:] != output_size:
            skip = F.interpolate(
                skip, size=output_size, mode='bilinear', align_corners=False)
        return skip * self.gate(skip)


class DepthwiseSeparableConv(nn.Module):
    """Depthwise spatial filtering followed by pointwise channel mixing."""

    def __init__(self, in_channels, out_channels, activation='gelu'):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        if activation == 'prelu':
            self.activation = nn.PReLU(out_channels)
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation is None:
            self.activation = nn.Identity()
        else:
            raise ValueError(f'unsupported activation: {activation}')

    def forward(self, x):
        return self.activation(self.norm(self.pointwise(self.depthwise(x))))


class StarRefine(nn.Module):
    """StarMatting EMLLA+SCM layers applied at one decoder resolution."""

    def __init__(self, channels, depth=2, num_heads=4, mlp_ratio=4,
                 dr_ratio=1, drop_path=0.0, stage_index=0):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError('channels must be divisible by num_heads')
        self.blocks = nn.ModuleList([
            PVTEncoderLayer(
                embed_dims=channels,
                num_heads=num_heads,
                feedforward_channels=channels * mlp_ratio,
                drop_path_rate=drop_path,
                use_conv_ffn=True,
                dr_ratio=dr_ratio,
                idx=stage_index,
                layer_idx=layer_index,
                num_layer=depth,
                structure='TE')
            for layer_index in range(depth)
        ])

    def forward(self, x):
        hw_shape = x.shape[-2:]
        x = nchw_to_nlc(x)
        for block in self.blocks:
            x = block(x, hw_shape)
        return nlc_to_nchw(x, hw_shape)


class FocalDecoder(nn.Module):
    """Decode ``(x1, x2, x3, x4, x5, x6)`` StarMatting features.

    The feature order is shallow-to-deep:

    - ``x1``: RGB input at full resolution.
    - ``x2``: StarMatting stem output at 1/2 resolution.
    - ``x3``--``x6``: stage outputs at 1/4, 1/8, 1/16 and 1/32.

    Args:
        in_channels: Channels of x1--x6 in shallow-to-deep order.
        out_channels: Decoder channels for x6--x3 in deep-to-shallow order.
        refine_depths: Number of EMLLA+SCM layers at x6--x3.
    """

    def __init__(self, in_channels=(3, 20, 20, 40, 100, 160),
                 out_channels=(128, 96, 64, 32),
                 refine_depths=(2, 2, 1, 1), num_heads=(4, 4, 4, 4),
                 dr_ratios=(4, 2, 2, 1),
                 mlp_ratios=(4, 4, 4, 4),
                 neck_channels=32, rgb_channels=8,
                 ms_supervision=True, eval_output='alpha', **kwargs):
        super().__init__()
        if len(in_channels) != 6:
            raise ValueError('in_channels must describe x1 through x6')
        if len(out_channels) != 4:
            raise ValueError('out_channels must describe x6 through x3')
        if (len(refine_depths) != 4 or len(num_heads) != 4 or len(dr_ratios) != 4 or len(mlp_ratios) != 4):
            raise ValueError('refine_depths, num_heads, dr_ratios and mlp_ratios must have four values')

        c1, c2, c3, c4, c5, c6 = in_channels
        d6, d5, d4, d3 = out_channels
        self.input_channels = tuple(in_channels)
        self.ms_supervision = ms_supervision
        if eval_output not in ('alpha', 'neck'):
            raise ValueError("eval_output must be 'alpha' or 'neck'")
        # Evaluation-only selector. Training always returns all six outputs when multi-scale supervision is enabled.
        self.eval_output = eval_output

        self.top_reduce = nn.Conv2d(c6, d6, 1, bias=False)
        self.top_context = ContextAggregation(c6, d6)
        self.top_fuse = nn.Sequential(
            nn.Conv2d(2 * d6, d6, 1, bias=False),
            nn.BatchNorm2d(d6),
            nn.GELU(),
        )
        self.refine6 = StarRefine(d6, refine_depths[0], num_heads[0], mlp_ratio=mlp_ratios[0], dr_ratio=dr_ratios[0], stage_index=3)

        self.up6_to5 = nn.ConvTranspose2d(d6, d6, 2, stride=2, bias=False)
        self.skip5 = SkipAlignGate(c5, d5)
        self.fuse5 = self._fusion(d6 + d5, d5)
        self.refine5 = StarRefine(d5, refine_depths[1], num_heads[1],
                                  mlp_ratio=mlp_ratios[1],
                                  dr_ratio=dr_ratios[1], stage_index=2)

        self.up5_to4 = nn.ConvTranspose2d(d5, d5, 2, stride=2, bias=False)
        self.skip4 = SkipAlignGate(c4, d4)
        self.fuse4 = self._fusion(d5 + d4, d4)
        self.refine4 = StarRefine(d4, refine_depths[2], num_heads[2], mlp_ratio=mlp_ratios[2], dr_ratio=dr_ratios[2], stage_index=1)

        self.skip3 = SkipAlignGate(c3, d3)
        self.fuse3 = self._fusion(d4 + d3, d3)
        self.refine3 = StarRefine(d3, refine_depths[3], num_heads[3], mlp_ratio=mlp_ratios[3], dr_ratio=dr_ratios[3], stage_index=0)

        # FocalMatter neck: bring the decoded x3 feature to 1/2 resolution
        # and fuse it with the encoder stem feature x2.
        self.stem_align = SkipAlignGate(c2, d3)
        self.neck = nn.Sequential(
            DepthwiseSeparableConv(2 * d3, neck_channels),
            DepthwiseSeparableConv(neck_channels, neck_channels),
        )

        # Full-resolution head.
        self.rgb_stem = nn.Sequential(
            nn.Conv2d(c1, rgb_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(rgb_channels),
            nn.PReLU(rgb_channels),
        )
        self.alpha_head = nn.Conv2d(neck_channels + rgb_channels, 1, kernel_size=1)

        self.pred6 = nn.Conv2d(d6, 1, 1)
        self.pred5 = nn.Conv2d(d5, 1, 1)
        self.pred4 = nn.Conv2d(d4, 1, 1)
        self.pred3 = nn.Conv2d(d3, 1, 1)
        self.pred_neck = nn.Conv2d(neck_channels, 1, 1)

    @staticmethod
    def _fusion(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    @staticmethod
    def _resize(x, reference):
        if x.shape[-2:] != reference.shape[-2:]:
            x = F.interpolate(x, size=reference.shape[-2:], mode='bilinear', align_corners=False)
        return x

    def forward(self, features):
        if len(features) != 6:
            raise ValueError('FocalMatterDecoder expects (x1, x2, x3, x4, x5, x6)')
        
        x1, x2, x3, x4, x5, x6 = features
        actual_channels = tuple(feature.shape[1] for feature in features)
        if actual_channels != self.input_channels:
            raise ValueError(
                f'feature channels {actual_channels} do not match '
                f'configured channels {self.input_channels}')

        p6 = self.refine6(self.top_fuse(torch.cat([self.top_reduce(x6), self.top_context(x6)], dim=1)))

        up5 = self._resize(self.up6_to5(p6), x5)
        p5 = self.refine5(self.fuse5(torch.cat([up5, self.skip5(x5, x5.shape[-2:])], dim=1)))

        up4 = self._resize(self.up5_to4(p5), x4)
        p4 = self.refine4(self.fuse4(torch.cat([up4, self.skip4(x4, x4.shape[-2:])], dim=1)))

        up3 = F.interpolate(p4, size=x3.shape[-2:], mode='bilinear', align_corners=False)
        p3 = self.refine3(self.fuse3(torch.cat([up3, self.skip3(x3, x3.shape[-2:])], dim=1)))

        neck_feature = F.interpolate(p3, size=x2.shape[-2:], mode='bilinear', align_corners=False)
        neck_feature = self.neck(torch.cat([neck_feature, self.stem_align(x2, x2.shape[-2:])], dim=1))

        neck_pred = self.pred_neck(neck_feature)
        head_feature = F.interpolate(neck_feature, size=x1.shape[-2:], mode='bilinear', align_corners=False)
        alpha = self.alpha_head(torch.cat([head_feature, self.rgb_stem(x1)], dim=1))

        if self.training and self.ms_supervision:
            return [self.pred6(p6), self.pred5(p5), self.pred4(p4), self.pred3(p3), neck_pred, alpha]
        if self.eval_output == 'neck':
            # Evaluation utilities undo dataset padding/resizing assuming the
            # prediction is at input resolution. Upsample the x2 neck head
            # here so its IoU is measured after the same interpolation that a
            # deployment pipeline would apply, without changing training.
            neck_pred = F.interpolate(neck_pred, size=x1.shape[-2:], mode='bilinear', align_corners=False)
            return [neck_pred]
        return [alpha]


def focal_decoder(**kwargs):
    return FocalDecoder(**kwargs)
