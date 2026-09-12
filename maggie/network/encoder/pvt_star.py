# Copyright (c) OpenMMLab. All rights reserved.
import math
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from mmcv.cnn import Conv2d, build_activation_layer, build_norm_layer
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.bricks.transformer import MultiheadAttention
from mmengine.model import BaseModule, ModuleList, Sequential
from mmengine.model.weight_init import trunc_normal_
from mmengine.runner import load_state_dict
from mmengine.utils import to_2tuple
from maggie.utils.logger import get_root_logger
from maggie.network.module.transformer import PatchEmbed, nchw_to_nlc, nlc_to_nchw
from timm.models.layers import DropPath

def pvt_convert(ckpt):
    new_ckpt = OrderedDict()
    # Process the concat between q linear weights and kv linear weights
    use_abs_pos_embed = False
    use_conv_ffn = False
    for k in ckpt.keys():
        if k.startswith('pos_embed'):
            use_abs_pos_embed = True
        if k.find('dwconv') >= 0:
            use_conv_ffn = True
    for k, v in ckpt.items():
        if k.startswith('head'):
            continue
        if k.startswith('norm.'):
            continue
        if k.startswith('cls_token'):
            continue
        if k.startswith('pos_embed'):
            stage_i = int(k.replace('pos_embed', ''))
            new_k = k.replace(f'pos_embed{stage_i}',
                              f'layers.{stage_i - 1}.1.0.pos_embed')
            if stage_i == 4 and v.size(1) == 50:  # 1 (cls token) + 7 * 7
                new_v = v[:, 1:, :]  # remove cls token
            else:
                new_v = v
        elif k.startswith('patch_embed'):
            stage_i = int(k.split('.')[0].replace('patch_embed', ''))
            new_k = k.replace(f'patch_embed{stage_i}',
                              f'layers.{stage_i - 1}.0')
            new_v = v
            if 'proj.' in new_k:
                new_k = new_k.replace('proj.', 'projection.')
        elif k.startswith('block'):
            stage_i = int(k.split('.')[0].replace('block', ''))
            layer_i = int(k.split('.')[1])
            new_layer_i = layer_i + use_abs_pos_embed
            new_k = k.replace(f'block{stage_i}.{layer_i}',
                              f'layers.{stage_i - 1}.1.{new_layer_i}')
            new_v = v
            if 'attn.q.' in new_k:
                sub_item_k = k.replace('q.', 'kv.')
                new_k = new_k.replace('q.', 'attn.in_proj_')
                new_v = torch.cat([v, ckpt[sub_item_k]], dim=0)
            elif 'attn.kv.' in new_k:
                continue
            elif 'attn.proj.' in new_k:
                new_k = new_k.replace('proj.', 'attn.out_proj.')
            elif 'attn.sr.' in new_k:
                new_k = new_k.replace('sr.', 'sr.')
            elif 'mlp.' in new_k:
                string = f'{new_k}-'
                new_k = new_k.replace('mlp.', 'ffn.layers.')
                if 'fc1.weight' in new_k or 'fc2.weight' in new_k:
                    new_v = v.reshape((*v.shape, 1, 1))
                new_k = new_k.replace('fc1.', '0.')
                new_k = new_k.replace('dwconv.dwconv.', '1.')
                if use_conv_ffn:
                    new_k = new_k.replace('fc2.', '4.')
                else:
                    new_k = new_k.replace('fc2.', '3.')
                string += f'{new_k} {v.shape}-{new_v.shape}'
        elif k.startswith('norm'):
            stage_i = int(k[4])
            new_k = k.replace(f'norm{stage_i}', f'layers.{stage_i - 1}.2')
            new_v = v
        else:
            new_k = k
            new_v = v
        new_ckpt[new_k] = new_v

    return new_ckpt


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class StemNet(nn.Module):
    def __init__(self, in_channels, embed_dim):
        super(StemNet, self).__init__()

        self.conv1 = ConvBNReLU(in_channels, embed_dim, kernel_size=3, stride=2, padding=1)
        self.conv2 = ConvBNReLU(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1, groups=embed_dim)
        self.conv3 = ConvBNReLU(embed_dim, embed_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class PRMv1(BaseModule):
    def __init__(self, kernel_size=3, downsample_ratio=2, dilations=[1, 2, 3, 4, 5, 6], in_chans=3, embed_dim=16,
                 op='cat', init_cfg=dict(type='Kaiming', layer='Conv2d')):

        super().__init__(init_cfg)

        self.dilations = dilations
        self.embed_dim = embed_dim
        self.downsample_ratio = downsample_ratio
        self.op = op
        self.kernel_size = kernel_size
        self.stride = downsample_ratio

        self.convs = nn.ModuleList()

        for i, dilation in enumerate(self.dilations):
            dilated_kernel_size = (self.kernel_size - 1) * dilation + 1
            padding = math.ceil((dilated_kernel_size - self.stride) / 2)
            self.convs.append(nn.Sequential(
                  *[nn.Conv2d(in_channels=in_chans, out_channels=embed_dim, kernel_size=self.kernel_size,
                              stride=self.stride, padding=padding, dilation=dilation),
                  nn.GELU(), ]))

    def forward(self, x):
        B, C, W, H = x.shape
        y = self.convs[0](x)
        for i in range(1, len(self.dilations)):
            _y = self.convs[i](x)
            y = torch.cat((y, _y), dim=1)

        return y


class SAM(BaseModule):
    def __init__(self, kernel_size=3, downsample_ratio=1, dilations=[1, 2, 3], in_chans=32, embed_dim=32,
                 op='sum', init_cfg=dict(type='Kaiming', layer='Conv2d'), norm_cfg=dict(type='LN')):

        super().__init__(init_cfg)

        self.dilations = dilations
        self.embed_dim = embed_dim
        self.downsample_ratio = downsample_ratio
        self.op = op
        self.kernel_size = kernel_size
        self.stride = downsample_ratio

        self.convs = nn.ModuleList()

        self.norm = build_norm_layer(norm_cfg, in_chans)[1]
        self.res_proj= nn.Sequential(
            nn.Linear(in_chans,embed_dim),
            nn.SiLU()
        )
        self.in_proj =  nn.Linear(in_chans,embed_dim)

        self.out_proj =nn.Sequential(
            nn.Conv2d(
                in_channels=embed_dim,
                out_channels=in_chans,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(in_chans))

        for i, dilation in enumerate(self.dilations):
            dilated_kernel_size = (self.kernel_size - 1) * dilation + 1
            padding = math.ceil((dilated_kernel_size - self.stride) / 2)
            self.convs.append(nn.Sequential(
                *[nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=dilated_kernel_size,
                            stride=self.stride, padding=padding, groups=embed_dim),
                  nn.GELU(), ]))

    def forward(self, x,hw_shape):

        shortcut =x
        x = self.norm(x)

        x = self.in_proj(x)*self.res_proj(x)
        x = nlc_to_nchw(x,hw_shape)

        y = self.convs[0](x)
        for i in range(1, len(self.dilations)):
            _y = self.convs[i](x)
            y += _y

        y = self.out_proj(y)
        y = y + nlc_to_nchw(shortcut, hw_shape)

        return y


class SCM(BaseModule):
    """Modified from MixFFN of PVT.

    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`.
        feedforward_channels (int): The hidden dimension of FFNs.
        act_cfg (dict, optional): The activation config for FFNs.
            Default: dict(type='GELU').
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut.
            Default: None.
        use_conv (bool): If True, add 3x3 DWConv between two Linear layers.
            Defaults: False.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 act_cfg=dict(type='GELU'),
                 ffn_drop=0.,
                 dropout_layer=None,
                 use_conv=False,
                 init_cfg=None):
        super(SCM, self).__init__(init_cfg=init_cfg)

        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.act_cfg = act_cfg
        activate = build_activation_layer(act_cfg)

        in_channels = embed_dims
        fc1 = Conv2d(
            in_channels=in_channels,
            out_channels=feedforward_channels,
            kernel_size=1,
            stride=1,
            bias=True)
        if use_conv:
            # 3x3 depth wise conv to provide positional encode information
            dw_conv = Conv2d(
                in_channels=feedforward_channels,
                out_channels=feedforward_channels,
                kernel_size=3,
                stride=1,
                padding=(3 - 1) // 2,
                bias=True,
                groups=feedforward_channels)
        fc2 = Conv2d(
            in_channels=feedforward_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            bias=True)
        drop = nn.Dropout(ffn_drop)
        norm = nn.BatchNorm2d(embed_dims)
        layers1 = [fc1, activate, drop]
        layers2 = [fc2, norm, drop]
        if use_conv:
            layers1.insert(1, dw_conv)
        self.layers1 = Sequential(*layers1)
        self.layers2 = Sequential(*layers2)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else torch.nn.Identity()

    def forward(self, x, hw_shape, identity=None):
        out = nlc_to_nchw(x, hw_shape)
        out = self.layers1(out)
        out = out * out
        out = self.layers2(out)
        out = nchw_to_nlc(out)
        if identity is None:
            identity = x
        out = self.dropout_layer(out)

        return identity + out


class SpatialReductionAttention(MultiheadAttention):
    """An implementation of Spatial Reduction Attention of PVT.

    This module is modified from MultiheadAttention which is a module from
    mmcv.cnn.bricks.transformer.

    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        attn_drop (float): A Dropout layer on attn_output_weights.
            Default: 0.0.
        proj_drop (float): A Dropout layer after `nn.MultiheadAttention`.
            Default: 0.0.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut. Default: None.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default: False.
        qkv_bias (bool): enable bias for qkv if True. Default: True.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        sr_ratio (int): The ratio of spatial reduction of Spatial Reduction
            Attention of PVT. Default: 1.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=None,
                 batch_first=True,
                 qkv_bias=True,
                 norm_cfg=dict(type='LN'),
                 sr_ratio=1,
                 init_cfg=None):
        super().__init__(
            embed_dims,
            num_heads,
            attn_drop,
            proj_drop,
            batch_first=batch_first,
            dropout_layer=dropout_layer,
            bias=qkv_bias,
            init_cfg=init_cfg)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = Conv2d(
                in_channels=embed_dims,
                out_channels=embed_dims,
                kernel_size=sr_ratio,
                stride=sr_ratio)
            # The ret[0] of build_norm_layer is norm name.
            self.norm = build_norm_layer(norm_cfg, embed_dims)[1]

        # handle the BC-breaking from https://github.com/open-mmlab/mmcv/pull/1418 # noqa
        from mmpose import digit_version, mmcv_version
        if mmcv_version < digit_version('1.3.17'):
            warnings.warn('The legacy version of forward function in'
                          'SpatialReductionAttention is deprecated in'
                          'mmcv>=1.3.17 and will no longer support in the'
                          'future. Please upgrade your mmcv.')
            self.forward = self.legacy_forward

    def forward(self, x, hw_shape, identity=None):
        x_q = x

        if self.sr_ratio > 1:
            x_kv = nlc_to_nchw(x, hw_shape)
            x_kv = self.sr(x_kv)
            x_kv = nchw_to_nlc(x_kv)
            x_kv = self.norm(x_kv)
        else:
            x_kv = x

        if identity is None:
            identity = x_q

        # Because the dataflow('key', 'query', 'value') of
        # ``torch.nn.MultiheadAttention`` is (num_query, batch,
        # embed_dims), We should adjust the shape of dataflow from
        # batch_first (batch, num_query, embed_dims) to num_query_first
        # (num_query ,batch, embed_dims), and recover ``attn_output``
        # from num_query_first to batch_first.
        if self.batch_first:
            x_q = x_q.transpose(0, 1)
            x_kv = x_kv.transpose(0, 1)

        out = self.attn(query=x_q, key=x_kv, value=x_kv)[0]

        if self.batch_first:
            out = out.transpose(0, 1)

        return identity + self.dropout_layer(self.proj_drop(out))

    def legacy_forward(self, x, hw_shape, identity=None):
        """multi head attention forward in mmcv version < 1.3.17."""
        x_q = x
        if self.sr_ratio > 1:
            x_kv = nlc_to_nchw(x, hw_shape)
            x_kv = self.sr(x_kv)
            x_kv = nchw_to_nlc(x_kv)
            x_kv = self.norm(x_kv)
        else:
            x_kv = x

        if identity is None:
            identity = x_q

        out = self.attn(query=x_q, key=x_kv, value=x_kv)[0]

        return identity + self.dropout_layer(self.proj_drop(out))


class LinearAttention(nn.Module):
    r""" Linear Attention with LePE.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
    """

    def __init__(self, dim, num_heads, qkv_bias=True,dr_ratio=1,idx=None, **kwargs):

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.dr_ratio = dr_ratio
        if (dim // dr_ratio) % num_heads != 0:
            raise ValueError(f"dim // dr_ratio ({dim // dr_ratio}) must be divisible by num_heads ({num_heads})")

        self.q = nn.Linear(dim, dim//dr_ratio, bias=qkv_bias)
        self.k = nn.Linear(dim, dim//dr_ratio, bias=qkv_bias)

        self.idx=idx

        self.elu = nn.ELU()
        self.lepe = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)

    def forward(self, x, hw_shape):#H, W
        """
        Args:
            x: input features with shape of (B, N, C)
        """
        b, n, c = x.shape

        num_heads = self.num_heads

        q=self.q(x)
        k=self.k(x)

        head_dim = c // num_heads
        shortcut=v=x
        # q, k, v: b, n, c

        q = self.elu(q) + 1.0
        k = self.elu(k) + 1.0

        q = q.reshape(b, n, num_heads, -1).permute(0, 2, 1, 3)
        k = k.reshape(b, n, num_heads, -1).permute(0, 2, 1, 3)
        v = v.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)

        z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k.transpose(-2, -1) * (n ** -0.5)) @ (v * (n ** -0.5))
        x = q @ kv * z

        x = x.transpose(1, 2).reshape(b, n, c)

        shortcut = nlc_to_nchw(shortcut, hw_shape)
        shortcut = self.lepe(shortcut)
        shortcut = nchw_to_nlc(shortcut)

        return x + shortcut


class EMLLABlock(nn.Module):
    r""" EMLLA Block.

    Modified from MLLA in the paper "Demystify Mamba in Vision: A Linear Attention Perspective."
    <https://arxiv.org/abs/2405.16605.pdf>`_.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, qkv_bias=True, drop_path=0.,
                 norm_layer=nn.LayerNorm, sr_ratio=1, dr_ratio=1, idx=None, layer_idx=None, num_layer=None,
                 structure=None, **kwargs):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        if structure not in ['TE', 'CP']:
            raise ValueError(f"Invalid structure: {structure}. Must be 'TE' or 'CP'.")
        self.act = nn.SiLU()
        self.in_proj = nn.Linear(dim, dim)
        self.act_proj = nn.Linear(dim, dim)
        self.idx = idx
        self.layer_idx = layer_idx
        self.num_layer = num_layer

        self.is_middle_layer = (structure == 'TE' and layer_idx != num_layer - 1) or \
                          (structure == 'CP' and layer_idx != 0 and layer_idx != num_layer - 1)
        if self.is_middle_layer:
            self.dwc = nn.Sequential(
                nn.Conv2d(dim, dim, 7, padding=9, groups=dim, dilation=3),
                nn.Conv2d(dim, dim, 5, padding=2, groups=dim),
                nn.GELU()
            )
            self.out_proj = nn.Sequential(nn.Conv2d(dim, dim, 1), nn.BatchNorm2d(dim))
        else:
            self.dwc = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
            self.attn = LinearAttention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                        dr_ratio=dr_ratio, idx=idx)
            self.out_proj = nn.Linear(dim, dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()


    def forward(self, x, hw_shape, identity=None):
        
        if self.is_middle_layer:
            act_res = self.act(self.act_proj(x))
            x = self.in_proj(x)
            x = nlc_to_nchw(x * act_res, hw_shape)
            x = self.dwc(x)
            x = self.out_proj(x)
            x = nchw_to_nlc(x)
            # 末层/首层分支
        else:
            act_res = self.act(self.act_proj(x))
            x = self.in_proj(x)
            x = nlc_to_nchw(x, hw_shape)
            x = self.act(self.dwc(x))
            x = nchw_to_nlc(x)
            x = self.attn(x, hw_shape)
            x = self.out_proj(x * act_res)

        return identity + self.drop_path(x)


class PVTEncoderLayer(BaseModule):
    """Implements one encoder layer in PVT.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        drop_rate (float): Probability of an element to be zeroed.
            after the feed forward layer. Default: 0.0.
        attn_drop_rate (float): The drop out rate for attention layer.
            Default: 0.0.
        drop_path_rate (float): stochastic depth rate. Default: 0.0.
        qkv_bias (bool): enable bias for qkv if True.
            Default: True.
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        sr_ratio (int): The ratio of spatial reduction of Spatial Reduction
            Attention of PVT. Default: 1.
        use_conv_ffn (bool): If True, use Convolutional FFN to replace FFN.
            Default: False.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 qkv_bias=True,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 sr_ratio=1,
                 use_conv_ffn=False,
                 init_cfg=None,
                 dr_ratio=1,
                 idx=None,
                 layer_idx=None,
                 num_layer=None,
                 structure=None):
        super(PVTEncoderLayer, self).__init__(init_cfg=init_cfg)

        # The ret[0] of build_norm_layer is norm name.
        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]

        # self.attn = SpatialReductionAttention(
        #     embed_dims=embed_dims,
        #     num_heads=num_heads,
        #     attn_drop=attn_drop_rate,
        #     proj_drop=drop_rate,
        #     dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
        #     qkv_bias=qkv_bias,
        #     norm_cfg=norm_cfg,
        #     sr_ratio=sr_ratio)

        self.attn = EMLLABlock(dim=embed_dims,
                               num_heads=num_heads,
                               drop_path=drop_path_rate,
                               sr_ratio=sr_ratio,
                               dr_ratio=dr_ratio,
                               idx=idx,
                               layer_idx=layer_idx,
                               num_layer=num_layer,
                               structure=structure)

        # The ret[0] of build_norm_layer is norm name.
        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]

        self.ffn = SCM(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            use_conv=use_conv_ffn,
            act_cfg=act_cfg)

    def forward(self, x, hw_shape):
        x = self.attn(self.norm1(x), hw_shape, identity=x)
        x = self.ffn(self.norm2(x), hw_shape, identity=x)

        return x


class AbsolutePositionEmbedding(BaseModule):
    """An implementation of the absolute position embedding in PVT.

    Args:
        pos_shape (int): The shape of the absolute position embedding.
        pos_dim (int): The dimension of the absolute position embedding.
        drop_rate (float): Probability of an element to be zeroed.
            Default: 0.0.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self, pos_shape, pos_dim, drop_rate=0., init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        if isinstance(pos_shape, int):
            pos_shape = to_2tuple(pos_shape)
        elif isinstance(pos_shape, tuple):
            if len(pos_shape) == 1:
                pos_shape = to_2tuple(pos_shape[0])
            assert len(pos_shape) == 2, \
                f'The size of image should have length 1 or 2, ' \
                f'but got {len(pos_shape)}'
        self.pos_shape = pos_shape
        self.pos_dim = pos_dim

        self.pos_embed = nn.Parameter(
            torch.zeros(1, pos_shape[0] * pos_shape[1], pos_dim))
        self.drop = nn.Dropout(p=drop_rate)

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)

    def resize_pos_embed(self, pos_embed, input_shape, mode='bilinear'):
        """Resize pos_embed weights.

        Resize pos_embed using bilinear interpolate method.

        Args:
            pos_embed (torch.Tensor): Position embedding weights.
            input_shape (tuple): Tuple for (downsampled input image height,
                downsampled input image width).
            mode (str): Algorithm used for upsampling:
                ``'nearest'`` | ``'linear'`` | ``'bilinear'`` | ``'bicubic'`` |
                ``'trilinear'``. Default: ``'bilinear'``.

        Return:
            torch.Tensor: The resized pos_embed of shape [B, L_new, C].
        """
        assert pos_embed.ndim == 3, 'shape of pos_embed must be [B, L, C]'
        pos_h, pos_w = self.pos_shape
        pos_embed_weight = pos_embed[:, (-1 * pos_h * pos_w):]
        pos_embed_weight = pos_embed_weight.reshape(
            1, pos_h, pos_w, self.pos_dim).permute(0, 3, 1, 2).contiguous()
        pos_embed_weight = F.interpolate(
            pos_embed_weight, size=input_shape, mode=mode)
        pos_embed_weight = torch.flatten(pos_embed_weight,
                                         2).transpose(1, 2).contiguous()
        pos_embed = pos_embed_weight

        return pos_embed

    def forward(self, x, hw_shape, mode='bilinear'):
        pos_embed = self.resize_pos_embed(self.pos_embed, hw_shape, mode)
        return self.drop(x + pos_embed)


class PyramidVisionTransformer(BaseModule):
    """Pyramid Vision Transformer (PVT)

    Implementation of `Pyramid Vision Transformer: A Versatile Backbone for
    Dense Prediction without Convolutions
    <https://arxiv.org/pdf/2102.12122.pdf>`_.

    Args:
        pretrain_img_size (int | tuple[int]): The size of input image when
            pretrain. Defaults: 224.
        in_channels (int): Number of input channels. Default: 3.
        embed_dims (int): Embedding dimension. Default: 64.
        num_stags (int): The num of stages. Default: 4.
        num_layers (Sequence[int]): The layer number of each transformer encode
            layer. Default: [3, 4, 6, 3].
        num_heads (Sequence[int]): The attention heads of each transformer
            encode layer. Default: [1, 2, 5, 8].
        patch_sizes (Sequence[int]): The patch_size of each patch embedding.
            Default: [4, 2, 2, 2].
        strides (Sequence[int]): The stride of each patch embedding.
            Default: [4, 2, 2, 2].
        paddings (Sequence[int]): The padding of each patch embedding.
            Default: [0, 0, 0, 0].
        sr_ratios (Sequence[int]): The spatial reduction rate of each
            transformer encode layer. Default: [8, 4, 2, 1].
        out_indices (Sequence[int] | int): Output from which stages.
            Default: (0, 1, 2, 3).
        mlp_ratios (Sequence[int]): The ratio of the mlp hidden dim to the
            embedding dim of each transformer encode layer.
            Default: [8, 8, 4, 4].
        qkv_bias (bool): Enable bias for qkv if True. Default: True.
        drop_rate (float): Probability of an element to be zeroed.
            Default 0.0.
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0.
        drop_path_rate (float): stochastic depth rate. Default 0.1.
        use_abs_pos_embed (bool): If True, add absolute position embedding to
            the patch embedding. Defaults: True.
        use_conv_ffn (bool): If True, use Convolutional FFN to replace FFN.
            Default: False.
        mul_scl_ipt (str | bool): Optional multi-scale input fusion mode.
            Set to ``'cat'`` or ``'add'`` to fuse features from a
            half-resolution input. ``False`` disables the branch.
        cxt (Sequence, optional): Select the number of shallow feature maps
            to concatenate with the deepest feature map. ``False`` disables
            context fusion.
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        pretrained (str, optional): model pretrained path. Default: None.
        convert_weights (bool): The flag indicates whether the
            pre-trained model is from the original repo. We may need
            to convert some keys to make it compatible.
            Default: True.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default:
            ``[
                dict(type='TruncNormal', std=.02, layer=['Linear']),
                dict(type='Constant', val=1, layer=['LayerNorm']),
                dict(type='Normal', std=0.01, layer=['Conv2d'])
            ]``
    """

    def __init__(self,
                 pretrain_img_size=224,
                 in_channels=3,
                 embed_dims=64,
                 num_stages=4,
                 num_layers=[3, 4, 6, 3],
                 num_heads=[1, 2, 5, 8],
                 patch_sizes=[4, 2, 2, 2],
                 strides=[4, 2, 2, 2],
                 paddings=[0, 0, 0, 0],
                 sr_ratios=[8, 4, 2, 1],
                 out_indices=(0, 1, 2, 3),
                 mlp_ratios=[8, 8, 4, 4],
                 prm_ratio=[1,2,2,2],
                 dr_ratio=[1,2,2,4],
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 use_abs_pos_embed=True,
                 norm_after_stage=False,
                 use_conv_ffn=False,
                 structure=None,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN', eps=1e-6),
                 convert_weights=True,
                 init_cfg=[
                     dict(type='TruncNormal', std=.02, layer=['Linear']),
                     dict(type='Constant', val=1, layer=['LayerNorm']),
                     dict(type='Kaiming', layer=['Conv2d'])
                 ],
                 cxt=False,
                 mul_scl_ipt=False,
                 include_stem_out=False):
        super().__init__(init_cfg=init_cfg)

        if embed_dims <= 32:
            self.stem = StemNet(in_channels=in_channels, embed_dim=embed_dims)
        else:
            self.prmv1 = PRMv1()

        self.convert_weights = convert_weights
        # Optional multi-scale input branch.  ``False`` keeps the original
        # single-scale forward path and checkpoint behaviour unchanged;
        # ``'cat'`` and ``'add'`` fuse features from a half-resolution input.
        self.cxt = cxt
        self.mul_scl_ipt = mul_scl_ipt
        self.include_stem_out = include_stem_out
        if isinstance(pretrain_img_size, int):
            pretrain_img_size = to_2tuple(pretrain_img_size)
        elif isinstance(pretrain_img_size, tuple):
            if len(pretrain_img_size) == 1:
                pretrain_img_size = to_2tuple(pretrain_img_size[0])
            assert len(pretrain_img_size) == 2, \
                f'The size of image should have length 1 or 2, ' \
                f'but got {len(pretrain_img_size)}'

        self.embed_dims = embed_dims

        self.num_stages = num_stages
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.patch_sizes = patch_sizes
        self.strides = strides
        self.sr_ratios = sr_ratios
        assert num_stages == len(num_layers) == len(num_heads) \
               == len(patch_sizes) == len(strides) == len(sr_ratios)

        self.out_indices = out_indices
        assert max(out_indices) < self.num_stages

        # transformer encoder
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(num_layers))
        ]  # stochastic num_layer decay rule

        cur = 0
        self.layers = ModuleList()
        for i, num_layer in enumerate(num_layers):
            embed_dims_i = embed_dims * num_heads[i]
            patch_embed = PatchEmbed(
                in_channels=in_channels,
                embed_dims=embed_dims_i,
                kernel_size=patch_sizes[i],
                stride=strides[i],
                padding=paddings[i],
                bias=True,
                norm_cfg=norm_cfg,
                idx_stage=i, )

            layers = ModuleList()
            if use_abs_pos_embed:
                pos_shape = pretrain_img_size // np.prod(patch_sizes[:i + 1])
                pos_embed = AbsolutePositionEmbedding(
                    pos_shape=pos_shape,
                    pos_dim=embed_dims_i,
                    drop_rate=drop_rate)
                layers.append(pos_embed)
            layers.extend([
                PVTEncoderLayer(
                    embed_dims=embed_dims_i,
                    num_heads=num_heads[i],
                    feedforward_channels=mlp_ratios[i] * embed_dims_i,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=dpr[cur + idx],
                    qkv_bias=qkv_bias,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    sr_ratio=sr_ratios[i],
                    use_conv_ffn=use_conv_ffn,
                    dr_ratio=dr_ratio[i],
                    idx=i,
                    layer_idx=idx,
                    num_layer=num_layer,
                    structure=structure) for idx in range(num_layer)
            ])
            in_channels = embed_dims_i
            # The ret[0] of build_norm_layer is norm name.
            if norm_after_stage:
                norm = build_norm_layer(norm_cfg, embed_dims_i)[1]
            else:
                norm = nn.Identity()

            sam = SAM(in_chans=embed_dims_i, embed_dim=embed_dims_i* prm_ratio[i],norm_cfg=norm_cfg)

            self.layers.append(ModuleList([patch_embed, layers, norm, sam]))

            cur += num_layer

    def init_weights(self):
        """Initialize the weights in backbone."""

        if (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            logger = get_root_logger()
            state_dict = get_state_dict(
                self.init_cfg['checkpoint'], map_location='cpu')
            logger.warn(f'Load pre-trained model for '
                        f'{self.__class__.__name__} from original repo')

            if self.convert_weights:
                # Because pvt backbones are not supported by mmcls,
                # so we need to convert pre-trained weights to match this
                # implementation.
                state_dict = pvt_convert(state_dict)
            load_state_dict(self, state_dict, strict=False, logger=logger)

        else:
            super(PyramidVisionTransformer, self).init_weights()

    def _forward_features(self, x):
        """Run the PVT stages and return one feature map per stage."""
        if self.embed_dims <= 32:
            x=self.stem(x)
        else:
            x = self.prmv1(x)
        
        stem_out = x

        stage_outs = []
        for i, layer in enumerate(self.layers):
            x, hw_shape = layer[0](x)

            for j, block in enumerate(layer[1]):
                x = block(x, hw_shape)

            x = layer[2](x)

            x = layer[3](x,hw_shape)
            stage_outs.append(x)

        return stage_outs, stem_out

    def forward(self, x):
        outs = [x]
        stage_outs, stem_out = self._forward_features(x)

        if self.mul_scl_ipt:
            _, _, height, width = x.shape
            x_pyramid = F.interpolate(x, size=(height // 2, width // 2), mode='bilinear', align_corners=False)
            pyramid_outs, pyramid_stem = self._forward_features(x_pyramid)

            # Treat boolean True as the natural concatenation mode. 
            fusion_mode = 'cat' if self.mul_scl_ipt is True else self.mul_scl_ipt
            if fusion_mode == 'cat':
                if self.include_stem_out:
                    stem_out = torch.cat((stem_out, F.interpolate(pyramid_stem, size=stem_out.shape[2:], mode='bilinear', align_corners=False)), dim=1)
                    
                stage_outs = [
                    torch.cat(
                        (feature, F.interpolate(pyramid_feature, size=feature.shape[2:], mode='bilinear', align_corners=False)), dim=1)
                    for feature, pyramid_feature in zip(stage_outs, pyramid_outs)
                ]
            elif fusion_mode == 'add':
                if self.include_stem_out:
                    stem_out = stem_out + F.interpolate(pyramid_stem, size=stem_out.shape[2:], mode='bilinear', align_corners=False)
                    
                stage_outs = [
                    feature + F.interpolate(pyramid_feature, size=feature.shape[2:], mode='bilinear', align_corners=False)
                    for feature, pyramid_feature in zip(stage_outs, pyramid_outs)
                ]

        if self.cxt:
            context_count = 1 if self.cxt is True else len(self.cxt)
            context_features = [
                F.interpolate(feature, size=stage_outs[-1].shape[2:], mode='bilinear', align_corners=False)
                for feature in stage_outs[:-1][-context_count:]
            ]
            stage_outs[-1] = torch.cat(context_features + [stage_outs[-1]], dim=1)
        
        if self.include_stem_out:
            outs.append(stem_out)

        for i, feature in enumerate(stage_outs):
            if i in self.out_indices:
                outs.append(feature)

        return outs


class PyramidVisionTransformerV2(PyramidVisionTransformer):
    """Implementation of `PVTv2: Improved Baselines with Pyramid Vision
    Transformer <https://arxiv.org/pdf/2106.13797.pdf>`_."""

    def __init__(self, **kwargs):
        super(PyramidVisionTransformerV2, self).__init__(
            patch_sizes=[3, 3, 3, 3],
            paddings=[1, 1, 1, 1],
            strides=[2, 2, 2, 2],
            use_abs_pos_embed=False,
            norm_after_stage=True,
            use_conv_ffn=True,
            **kwargs)


class StarMatting(PyramidVisionTransformer):
    def __init__(self, **kwargs):
        super(StarMatting, self).__init__(
            patch_sizes=[3, 3, 3, 3],
            paddings=[1, 1, 1, 1],
            strides=[2, 2, 2, 2],
            use_abs_pos_embed=False,
            norm_after_stage=False,
            use_conv_ffn=True,
            **kwargs)


def star_matting(**kwargs):
    """Constructs a resnet_encoder_25 model.
    """
    model = StarMatting(**kwargs)

    return model
