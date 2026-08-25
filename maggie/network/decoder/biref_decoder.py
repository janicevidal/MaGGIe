import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from kornia.filters import laplacian
from mmcv.cnn import ConvModule, build_norm_layer
from maggie.network.module.transformer import nchw_to_nlc, nlc_to_nchw

def image2patches(image, grid_h=2, grid_w=2, patch_ref=None, transformation='b c (hg h) (wg w) -> (b hg wg) c h w'):
    if patch_ref is not None:
        grid_h, grid_w = image.shape[-2] // patch_ref.shape[-2], image.shape[-1] // patch_ref.shape[-1]
    patches = rearrange(image, transformation, hg=grid_h, wg=grid_w)
    return patches

def patches2image(patches, grid_h=2, grid_w=2, patch_ref=None, transformation='(b hg wg) c h w -> b c (hg h) (wg w)'):
    if patch_ref is not None:
        grid_h, grid_w = patch_ref.shape[-2] // patches[0].shape[-2], patch_ref.shape[-1] // patches[0].shape[-1]
    image = rearrange(patches, transformation, hg=grid_h, wg=grid_w)
    return image


class IPTBlock(nn.Module):
    def __init__( self, in_channels, out_channels, norm_cfg, act_cfg):
        super().__init__()
        
        self.conv1 = ConvModule(in_channels, out_channels, kernel_size=1, norm_cfg=norm_cfg, act_cfg=None)
        self.conv2 = ConvModule(out_channels, out_channels, kernel_size=3, groups=out_channels, stride=1, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv3 = ConvModule(out_channels, out_channels, kernel_size=1, norm_cfg=norm_cfg, act_cfg=None)
    def forward(self, x):
        x = self.conv1(x)
        out = self.conv3(self.conv2(x))
        
        return out
    
    
class GDTBlock(nn.Module):
    def __init__( self, in_channels, out_channels, norm_cfg, act_cfg):
        super().__init__()
        
        self.conv1 = ConvModule(in_channels, in_channels, kernel_size=3, groups=in_channels, stride=1, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv2 = ConvModule(in_channels, out_channels, kernel_size=1, norm_cfg=norm_cfg, act_cfg=None)
    def forward(self, x):
        x = self.conv1(x)
        out = self.conv2(x)
        
        return out


class ReconBlock(nn.Module):
    def __init__(self, kernel_size=3, dilations=[1, 2, 3], in_chans=32, out_chans=32):

        super().__init__()

        self.dilations = dilations
        self.out_chans = out_chans
        self.kernel_size = kernel_size

        self.convs = nn.ModuleList()

        self.norm = build_norm_layer(dict(type='LN'), in_chans)[1]
        self.res_proj = nn.Sequential(
            nn.Linear(in_chans, in_chans),
            nn.SiLU()
        )
        self.in_proj = nn.Linear(in_chans, in_chans)

        self.out_proj =nn.Sequential(
            nn.Conv2d(
                in_channels=in_chans,
                out_channels=out_chans,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(out_chans))

        for i, dilation in enumerate(self.dilations):
            dilated_kernel_size = (self.kernel_size - 1) * dilation + 1
            padding = math.ceil((dilated_kernel_size - 1) / 2)
            self.convs.append(nn.Sequential(
                *[nn.Conv2d(in_channels=in_chans, out_channels=in_chans, kernel_size=dilated_kernel_size,
                            stride=1, padding=padding, groups=in_chans),
                  nn.GELU(), ]))

    def forward(self, x):
        _, _, H, W = x.size()
        x = nchw_to_nlc(x)
        x = self.norm(x)

        x = self.in_proj(x) * self.res_proj(x)
        x = nlc_to_nchw(x, (H, W))

        y = self.convs[0](x)
        for i in range(1, len(self.dilations)):
            _y = self.convs[i](x)
            y += _y

        y = self.out_proj(y)

        return y


class BiRefDecoder(nn.Module):
    def __init__(self, in_channels, ipt_channels, out_channels,
                 ms_supervision=True, split=True, dec_ipt=True, out_ref=True,
                 share_p3_patches=False):
        super(BiRefDecoder, self).__init__()
        
        self.ms_supervision = ms_supervision
        self.dec_ipt = dec_ipt
        self.out_ref = out_ref
        self.share_p3_patches = share_p3_patches
        
        norm_cfg=dict(type='BN')
        act_cfg=dict(type='ReLU')
        
        self.up_kwargs = {'mode': 'bilinear', 'align_corners': False}
        
        if self.dec_ipt:
            self.split = split

            ipt_blk_in_channels = [2**i*3 for i in (10, 8, 6, 4, 0)] if self.split else [3] * 5

            ipt_blk5_in_channels = (ipt_blk_in_channels[1] if self.share_p3_patches else ipt_blk_in_channels[0])
            self.ipt_blk5 = IPTBlock(ipt_blk5_in_channels, ipt_channels[0], norm_cfg, act_cfg)
            self.ipt_blk4 = IPTBlock(ipt_blk_in_channels[1], ipt_channels[1], norm_cfg, act_cfg)
            self.ipt_blk3 = IPTBlock(ipt_blk_in_channels[2], ipt_channels[2], norm_cfg, act_cfg)
            self.ipt_blk2 = IPTBlock(ipt_blk_in_channels[3], ipt_channels[3], norm_cfg, act_cfg)
            self.ipt_blk1 = IPTBlock(ipt_blk_in_channels[4], ipt_channels[4], norm_cfg, act_cfg)
        else:
            self.split = None
        
        dec_blk_in_channels = out_channels.copy()
        
        if self.dec_ipt:
            dec_blk_in_channels[0] = in_channels[0] + ipt_channels[0]
            dec_blk_in_channels[1] = out_channels[0] + ipt_channels[1]
            dec_blk_in_channels[2] = out_channels[1] + ipt_channels[2]
            dec_blk_in_channels[3] = out_channels[2] + ipt_channels[3]
            
            # dec_blk_in_channels = [out_channels[i] +  ipt_channels[i] for i in range(len(in_channels))]

        self.decoder_block4 = ReconBlock(in_chans=dec_blk_in_channels[0], out_chans=out_channels[0])
        self.decoder_block3 = ReconBlock(in_chans=dec_blk_in_channels[1], out_chans=out_channels[1])
        self.decoder_block2 = ReconBlock(in_chans=dec_blk_in_channels[2], out_chans=out_channels[2])
        self.decoder_block1 = ReconBlock(in_chans=dec_blk_in_channels[3], out_chans=out_channels[3])
        
        self.conv_out1 = ConvModule(in_channels=(out_channels[3] + (ipt_channels[4] if self.dec_ipt else 0)), out_channels=1, kernel_size=1, norm_cfg=None, act_cfg=None)

        # Backbone+PyramidNeck --> lateral block --> DecoderBlock
        self.lateral_block3 = ConvModule(in_channels=in_channels[1], out_channels=out_channels[0], kernel_size=1, bias=False, norm_cfg=norm_cfg, act_cfg=None)
        self.lateral_block2 = ConvModule(in_channels=in_channels[2], out_channels=out_channels[1], kernel_size=1, bias=False, norm_cfg=norm_cfg, act_cfg=None)
        self.lateral_block1 = ConvModule(in_channels=in_channels[3], out_channels=out_channels[2], kernel_size=1, bias=False, norm_cfg=norm_cfg, act_cfg=None)

        if self.ms_supervision:
            self.conv_ms_spvn_4 = ConvModule(in_channels=out_channels[0], out_channels=1, kernel_size=1, norm_cfg=None, act_cfg=None)
            self.conv_ms_spvn_3 = ConvModule(in_channels=out_channels[1], out_channels=1, kernel_size=1, norm_cfg=None, act_cfg=None)
            self.conv_ms_spvn_2 = ConvModule(in_channels=out_channels[2], out_channels=1, kernel_size=1, norm_cfg=None, act_cfg=None)

            if self.out_ref:
                _N = 16
                self.gdt_convs_4 = GDTBlock(out_channels[0], _N, norm_cfg, act_cfg)
                self.gdt_convs_3 = GDTBlock(out_channels[1], _N, norm_cfg, act_cfg)
                self.gdt_convs_2 = GDTBlock(out_channels[2], _N, norm_cfg, act_cfg)

                self.gdt_convs_pred_4 = ConvModule(in_channels=_N, out_channels=1, kernel_size=1, norm_cfg=None, act_cfg=None)
                self.gdt_convs_pred_3 = ConvModule(in_channels=_N, out_channels=1, kernel_size=1, norm_cfg=None, act_cfg=None)
                self.gdt_convs_pred_2 = ConvModule(in_channels=_N, out_channels=1, kernel_size=1, norm_cfg=None, act_cfg=None)
                
                self.gdt_convs_attn_4 = ConvModule(in_channels=_N, out_channels=1, kernel_size=1, norm_cfg=None, act_cfg=None)
                self.gdt_convs_attn_3 = ConvModule(in_channels=_N, out_channels=1, kernel_size=1, norm_cfg=None, act_cfg=None)
                self.gdt_convs_attn_2 = ConvModule(in_channels=_N, out_channels=1, kernel_size=1, norm_cfg=None, act_cfg=None)

    def forward(self, features):
        if self.training and self.out_ref:
            input = features[0]
            features.append(laplacian(torch.mean(input, dim=1).unsqueeze(1), kernel_size=5))
            
            outs_gdt_pred = []
            outs_gdt_label = []
            x, x1, x2, x3, x4, gdt_gt = features
        else:
            x, x1, x2, x3, x4 = features
        
        outs = []

        if self.dec_ipt:
            if self.share_p3_patches:
                if self.split:
                    # x3 is available before decoding and has the same spatial
                    # size as _p3, avoiding a circular dependency on _p3.
                    patches_p3 = image2patches(x, patch_ref=x3, transformation=('b c (hg h) (wg w) -> b (c hg wg) h w'))
                else:
                    patches_p3 = F.interpolate(x, size=x3.shape[2:], **self.up_kwargs)
                patches_p4 = F.adaptive_avg_pool2d(patches_p3, output_size=x4.shape[2:])
            else:
                patches_p4 = image2patches(x, patch_ref=x4, transformation=('b c (hg h) (wg w) -> b (c hg wg) h w')) if self.split else x
            x4 = torch.cat((x4, self.ipt_blk5(patches_p4)), 1)
            
        p4 = self.decoder_block4(x4)
        m4 = self.conv_ms_spvn_4(p4) if self.ms_supervision and self.training else None
        
        if self.out_ref:
            p4_gdt = self.gdt_convs_4(p4)
            if self.training:
                # >> GT:
                m4_dia = m4
                gdt_label_main_4 = gdt_gt * F.interpolate(m4_dia, size=gdt_gt.shape[2:], **self.up_kwargs)
                outs_gdt_label.append(gdt_label_main_4)
                # >> Pred:
                gdt_pred_4 = self.gdt_convs_pred_4(p4_gdt)
                outs_gdt_pred.append(gdt_pred_4)
            gdt_attn_4 = self.gdt_convs_attn_4(p4_gdt).sigmoid()
            # >> Finally:
            p4 = p4 * gdt_attn_4
            
        _p4 = F.interpolate(p4, size=x3.shape[2:], **self.up_kwargs)
        _p3 = _p4 + self.lateral_block3(x3)

        if self.dec_ipt:
            if not self.share_p3_patches:
                patches_p3 = image2patches(x, patch_ref=_p3, transformation=('b c (hg h) (wg w) -> b (c hg wg) h w')) if self.split else x
            _p3 = torch.cat((_p3, self.ipt_blk4(patches_p3)), 1)
            
        p3 = self.decoder_block3(_p3)
        m3 = self.conv_ms_spvn_3(p3) if self.ms_supervision and self.training else None
        if self.out_ref:
            p3_gdt = self.gdt_convs_3(p3)
            if self.training:
                # >> GT:
                # m3 --dilation--> m3_dia
                # G_3^gt * m3_dia --> G_3^m, which is the label of gradient
                m3_dia = m3
                gdt_label_main_3 = gdt_gt * F.interpolate(m3_dia, size=gdt_gt.shape[2:], **self.up_kwargs)
                outs_gdt_label.append(gdt_label_main_3)
                # >> Pred:
                # p3 --conv--BN--> F_3^G, where F_3^G predicts the \hat{G_3} with xx
                # F_3^G --sigmoid--> A_3^G
                gdt_pred_3 = self.gdt_convs_pred_3(p3_gdt)
                outs_gdt_pred.append(gdt_pred_3)
            gdt_attn_3 = self.gdt_convs_attn_3(p3_gdt).sigmoid()
            # >> Finally:
            # p3 = p3 * A_3^G
            p3 = p3 * gdt_attn_3
            
        _p3 = F.interpolate(p3, size=x2.shape[2:], **self.up_kwargs)
        _p2 = _p3 + self.lateral_block2(x2)

        if self.dec_ipt:
            patches_batch = image2patches(x, patch_ref=_p2, transformation='b c (hg h) (wg w) -> b (c hg wg) h w') if self.split else x
            _p2 = torch.cat((_p2, self.ipt_blk3(patches_batch)), 1)
            
        p2 = self.decoder_block2(_p2)
        m2 = self.conv_ms_spvn_2(p2) if self.ms_supervision and self.training else None
        
        if self.out_ref:
            p2_gdt = self.gdt_convs_2(p2)
            if self.training:
                # >> GT:
                m2_dia = m2
                gdt_label_main_2 = gdt_gt * F.interpolate(m2_dia, size=gdt_gt.shape[2:], **self.up_kwargs)
                outs_gdt_label.append(gdt_label_main_2)
                # >> Pred:
                gdt_pred_2 = self.gdt_convs_pred_2(p2_gdt)
                outs_gdt_pred.append(gdt_pred_2)
            gdt_attn_2 = self.gdt_convs_attn_2(p2_gdt).sigmoid()
            # >> Finally:
            p2 = p2 * gdt_attn_2
            
        _p2 = F.interpolate(p2, size=x1.shape[2:], **self.up_kwargs)
        _p1 = _p2 + self.lateral_block1(x1)

        if self.dec_ipt:
            patches_batch = image2patches(x, patch_ref=_p1, transformation='b c (hg h) (wg w) -> b (c hg wg) h w') if self.split else x
            _p1 = torch.cat((_p1, self.ipt_blk2(patches_batch)), 1)
        _p1 = self.decoder_block1(_p1)
        _p1 = F.interpolate(_p1, size=x.shape[2:], **self.up_kwargs)

        if self.dec_ipt:
            patches_batch =  x
            _p1 = torch.cat((_p1, self.ipt_blk1(patches_batch)), 1)
        p1_out = self.conv_out1(_p1)

        if self.ms_supervision and self.training:
            outs.append(m4)
            outs.append(m3)
            outs.append(m2)
            
        outs.append(p1_out)
        
        return outs if not (self.out_ref and self.training) else ([outs_gdt_pred, outs_gdt_label], outs)
    
def biref_decoder(**kwargs):
    model = BiRefDecoder(**kwargs)
    return model
