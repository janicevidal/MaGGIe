"""Binary segmentation architecture for the FocalDecoder.

The decoder exposes six predictions during training: x32/x16/x8 coarse
outputs, x4/x2/x1 refined outputs.  Coarse BCE+Dice and fine Hybrid-E losses
are applied to their respective groups.
"""

import torch.nn.functional as F

from maggie.network.loss import hybrid_e_loss
from .birefnet_binary import BiRefNetBinary


class FocalNetBinary(BiRefNetBinary):
    """BiRefNetBinary variant paired with ``focal_decoder``."""

    expected_coarse_outputs = 3
    expected_total_outputs = 6

    def __init__(self, cfg):
        super().__init__(cfg)
        decoder_name = str(cfg.decoder)
        if decoder_name not in ('focal_decoder', 'FocalDecoder'):
            raise ValueError('FocalBiRefNetBinary requires decoder: focal_decoder')
        configured_weights = list(self.loss_cfg.scale_weights)
        if len(configured_weights) != 3:
            raise ValueError(
                'FocalDecoder coarse supervision uses x32/x16/x8; set binary_loss.scale_weights to three values, got '
                f'{len(configured_weights)}')

    def compute_loss(self, scaled_preds, target):
        if len(scaled_preds) != self.expected_total_outputs:
            raise ValueError(
                'FocalDecoder must return three coarse and three refined '
                f'predictions (six total), got {len(scaled_preds)} outputs')
        target = self._prepare_target(target)
        coarse_preds = scaled_preds[:3]  # x32, x16, x8
        fine_preds = scaled_preds[3:]    # x4, x2, x1
        scale_weights = list(self.loss_cfg.scale_weights)

        zero = scaled_preds[-1].sum() * 0
        coarse_bce = zero
        coarse_dice = zero
        for logits, weight in zip(coarse_preds, scale_weights):
            target_lvl = F.interpolate(target, size=logits.shape[-2:], mode='nearest')
            coarse_bce = coarse_bce + weight * F.binary_cross_entropy_with_logits(logits, target_lvl)
            coarse_dice = coarse_dice + weight * self._dice_loss(logits, target_lvl)

        fine_weights = list(self.loss_cfg.get('fine_scale_weights', [1., 1., 1.]))
        if len(fine_weights) != 3:
            raise ValueError('fine_scale_weights must contain x4/x2/x1 weights')
        hybrid_levels = []
        hybrid = zero
        for logits, weight in zip(fine_preds, fine_weights):
            # Hybrid-E is defined at the target (full) resolution for all
            # refined levels. This keeps its boundary-aware pooling kernel in
            # a consistent pixel scale for x4, x2 and x1 predictions.
            logits_full = logits
            if logits_full.shape[-2:] != target.shape[-2:]:
                logits_full = F.interpolate(logits_full, size=target.shape[-2:], mode='bilinear', align_corners=False)
            value = hybrid_e_loss(logits_full, target,
                                  kernel_size=self.loss_cfg.hybrid_e_kernel_size,
                                  boundary_factor=self.loss_cfg.hybrid_e_boundary_factor)
            hybrid = hybrid + weight * value
            hybrid_levels.append(value)

        final_logits = fine_preds[-1]
        if final_logits.shape[-2:] != target.shape[-2:]:
            final_logits = F.interpolate(final_logits, size=target.shape[-2:], mode='bilinear', align_corners=False)
        final_ssim = self.ssim_loss(final_logits.sigmoid(), target)
        if self.loss_cfg.get('focal_enabled', False) and self.loss_cfg.get('focal_weight', 0.) > 0:
            focal = self._focal_loss(final_logits, target,
                                     gamma=self.loss_cfg.get('focal_gamma', 1.5),
                                     alpha=self.loss_cfg.get('focal_alpha', 0.5))
        else:
            focal = zero
        if self.loss_cfg.get('hard_negative_enabled', True) and self.loss_cfg.hard_negative_weight > 0:
            hard_background = self._hard_negative_loss(final_logits, target, self.loss_cfg.hard_negative_ratio)
        else:
            hard_background = zero

        total = self.loss_cfg.loss_weight * (
            self.loss_cfg.coarse_bce_weight * coarse_bce +
            self.loss_cfg.coarse_dice_weight * coarse_dice +
            self.loss_cfg.hybrid_e_weight * hybrid +
            self.loss_cfg.final_ssim_weight * final_ssim +
            self.loss_cfg.hard_negative_weight * hard_background +
            self.loss_cfg.get('focal_weight', 0.) * focal)
        return {
            'coarse_bce': coarse_bce,
            'coarse_dice': coarse_dice,
            'hybrid_e': hybrid,
            'hybrid_e_x4': hybrid_levels[0],
            'hybrid_e_x2': hybrid_levels[1],
            'hybrid_e_x1': hybrid_levels[2],
            'ssim': final_ssim,
            'focal': focal,
            'hard_bg': hard_background,
            'total': total,
        }
