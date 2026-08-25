import torch
import torch.nn as nn
from torch.nn import functional as F
from huggingface_hub import PyTorchModelHubMixin
from yacs.config import CfgNode

from ..decoder import *
from ..encoder import *
from ..loss import hybrid_e_loss


class BiRefNetBinary(nn.Module, PyTorchModelHubMixin):
    """Single-branch BiRefNet for binary person segmentation.

    The encoder and decoder follow BiRefNetPro, but this model has no matting
    losses, GDT refinement target, or auxiliary semantic head. During training,
    all decoder outputs are supervised by binary person masks. Evaluation keeps
    the existing ``alpha_pred`` output key for compatibility with eval_image.
    """

    def __init__(self, cfg):
        super().__init__()
        if isinstance(cfg, dict):
            cfg = CfgNode(init_dict=cfg)
        self.cfg = cfg

        self.encoder = eval(cfg.encoder)(**cfg.encoder_args)

        decoder_args = cfg.decoder_args.clone()
        self.loss_cfg = decoder_args.pop('binary_loss', None)
        if self.loss_cfg is None:
            raise ValueError(
                "BiRefNetBinary requires decoder_args.binary_loss")
        if decoder_args.out_ref:
            raise ValueError(
                "BiRefNetBinary requires decoder_args.out_ref=false")
        self.decoder = eval(cfg.decoder)(**decoder_args)

        self.decoder.apply(self._init_weights)
        if hasattr(self.encoder, 'init_weights'):
            self.encoder.init_weights()

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(
                module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(
                module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (
                nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                nn.SyncBatchNorm, nn.GroupNorm, nn.LayerNorm,
                nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
            if module.weight is not None:
                nn.init.constant_(module.weight, 1)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    @staticmethod
    def _dice_loss(logits, target):
        probability = logits.sigmoid()
        intersection = (probability * target).sum(dim=(1, 2, 3))
        denominator = probability.sum(dim=(1, 2, 3)) + target.sum(
            dim=(1, 2, 3))
        return 1 - ((2 * intersection + 1) /
                    (denominator + 1)).mean()

    @staticmethod
    def _hard_negative_loss(logits, target, ratio):
        loss_map = F.binary_cross_entropy_with_logits(
            logits, target, reduction='none')
        sample_losses = []
        for pixel_loss, sample_target in zip(loss_map, target):
            background_loss = pixel_loss[sample_target < 0.5]
            if background_loss.numel() == 0:
                continue
            num_hard = max(1, int(background_loss.numel() * ratio))
            sample_losses.append(
                torch.topk(background_loss, num_hard).values.mean())
        if sample_losses:
            return torch.stack(sample_losses).mean()
        return logits.sum() * 0

    def _prepare_target(self, target):
        if target.ndim == 3:
            target = target.unsqueeze(1)
        if target.ndim != 4 or target.shape[1] != 1:
            raise ValueError(
                "Binary target must have shape (B, 1, H, W)")
        target = target.float()
        if target.detach().max() > 1:
            target = target / 255.0
        return (target > self.loss_cfg.target_threshold).float()

    def compute_loss(self, scaled_preds, target):
        target = self._prepare_target(target)
        coarse_preds = scaled_preds[:-1]
        scale_weights = list(self.loss_cfg.scale_weights)
        if len(scale_weights) != len(coarse_preds):
            raise ValueError(
                "binary_loss.scale_weights must contain one weight for each "
                f"coarse output, got {len(scale_weights)} weights for "
                f"{len(coarse_preds)} outputs")

        zero = scaled_preds[-1].sum() * 0
        coarse_bce = zero
        coarse_dice = zero
        for logits, scale_weight in zip(coarse_preds, scale_weights):
            target_lvl = F.interpolate(
                target, size=logits.shape[-2:], mode='nearest')
            coarse_bce = coarse_bce + scale_weight * (
                F.binary_cross_entropy_with_logits(logits, target_lvl))
            coarse_dice = coarse_dice + scale_weight * (
                self._dice_loss(logits, target_lvl))

        final_logits = scaled_preds[-1]
        if final_logits.shape[-2:] != target.shape[-2:]:
            final_logits = F.interpolate(
                final_logits, size=target.shape[-2:], mode='bilinear',
                align_corners=False)
        hybrid = hybrid_e_loss(
            final_logits, target,
            kernel_size=self.loss_cfg.hybrid_e_kernel_size,
            boundary_factor=self.loss_cfg.hybrid_e_boundary_factor)
        hard_background = self._hard_negative_loss(
            final_logits, target, self.loss_cfg.hard_negative_ratio)

        total = self.loss_cfg.loss_weight * (
            self.loss_cfg.coarse_bce_weight * coarse_bce +
            self.loss_cfg.coarse_dice_weight * coarse_dice +
            self.loss_cfg.hybrid_e_weight * hybrid +
            self.loss_cfg.hard_negative_weight * hard_background)
        return {
            'binary_coarse_bce': coarse_bce,
            'binary_coarse_dice': coarse_dice,
            'binary_hybrid_e': hybrid,
            'binary_hard_bg': hard_background,
            'total': total,
        }

    def forward(self, batch, **kwargs):
        image = batch['image']
        embeddings = self.encoder(image)
        scaled_preds = self.decoder(embeddings)
        output = {'alpha_pred': scaled_preds[-1].sigmoid()}

        if not self.training:
            return output

        target = next(
            (batch[key] for key in ('alpha', 'mask', 'masks')
             if key in batch), None)
        if target is None:
            raise KeyError(
                "BiRefNetBinary training requires alpha, mask, or masks")
        return output, self.compute_loss(scaled_preds, target)
