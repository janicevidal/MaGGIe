"""Matting architecture paired with the six-output FocalDecoder."""

import torch.nn.functional as F

from .birefnet_pro import BiRefNetPro
from ...utils.utils import compute_unknown


class FocalNetMatting(BiRefNetPro):
    """BiRefNetPro loss wrapper with explicit FocalDecoder supervision.

    FocalDecoder returns predictions in the order
    ``x32, x16, x8, x4, x2, x1``.  The first two outputs receive the
    MODNet-style coarse semantic objective at their native resolution.  The
    last four outputs are resized to the alpha target and receive the
    configured pixel criteria.  Gradient/Laplacian unknown-region losses are
    applied to x8, x4 and x2, while the global versions are applied to x1.
    """

    expected_total_outputs = 6
    coarse_output_count = 2
    coarse_scale_names = ('x32', 'x16')
    fine_scale_names = ('x8', 'x4', 'x2', 'x1')
    unknown_scale_names = ('x8', 'x4', 'x2')

    def __init__(self, cfg):
        super().__init__(cfg)
        decoder_name = str(cfg.decoder)
        if decoder_name not in ('focal_decoder', 'FocalDecoder'):
            raise ValueError('FocalNetMatting requires decoder: focal_decoder')
        if cfg.decoder_args.out_ref:
            raise ValueError('FocalNetMatting requires decoder_args.out_ref=false')
        if not cfg.decoder_args.ms_supervision:
            raise ValueError('FocalNetMatting training requires ms_supervision=true')

        loss_cfg = cfg.decoder_args.get('matting_loss', {})
        self.coarse_scale_weights = tuple(loss_cfg.get('coarse_scale_weights', (1.0, 1.0)))
        self.fine_scale_weights = tuple(loss_cfg.get('fine_scale_weights', (0.5, 0.7, 0.9, 1.0)))
        self.unknown_scale_weights = tuple(loss_cfg.get('unknown_scale_weights', (0.25, 0.5, 0.75)))
        unknown_kernel_sizes = tuple(loss_cfg.get('unknown_kernel_sizes', (19, 15, 11, 7)))
        unknown_gt_support_kernel_sizes = tuple(loss_cfg.get('unknown_gt_support_kernel_sizes', (31, 25, 19, 15)))

        if len(unknown_kernel_sizes) == len(self.unknown_scale_names):
            unknown_kernel_sizes = unknown_kernel_sizes + (7,)
        if len(unknown_gt_support_kernel_sizes) == len(self.unknown_scale_names):
            unknown_gt_support_kernel_sizes = (unknown_gt_support_kernel_sizes + (15,))
        self.unknown_kernel_sizes = unknown_kernel_sizes
        self.unknown_gt_support_kernel_sizes = (unknown_gt_support_kernel_sizes)
        self.mae_unknown_factors = tuple(loss_cfg.get('mae_unknown_factors', (0.0, 0.0, 0.0, 0.0)))
        self.soft_unknown_thresholds = tuple(loss_cfg.get('soft_unknown_thresholds', (5.0, 250.0)))
        self.hard_boundary_threshold = float(loss_cfg.get('hard_boundary_threshold', 0.5))
        self.include_prediction_error = bool(loss_cfg.get('include_prediction_error', False))
        self.prediction_error_thresholds = tuple(loss_cfg.get('prediction_error_thresholds', (0.35, 0.30, 0.25, 0.20)))
        self.unknown_source = str(loss_cfg.get('unknown_source', 'gt_pred_limited')).lower()
        self._validate_scale_config()

    def _validate_scale_config(self):
        expected_lengths = (
            ('coarse_scale_weights', self.coarse_scale_weights, len(self.coarse_scale_names)),
            ('fine_scale_weights', self.fine_scale_weights, len(self.fine_scale_names)),
            ('unknown_scale_weights', self.unknown_scale_weights, len(self.unknown_scale_names)),
            ('unknown_kernel_sizes', self.unknown_kernel_sizes, len(self.fine_scale_names)),
            ('unknown_gt_support_kernel_sizes', self.unknown_gt_support_kernel_sizes, len(self.fine_scale_names)),
            ('mae_unknown_factors', self.mae_unknown_factors, len(self.fine_scale_names)),
            ('prediction_error_thresholds', self.prediction_error_thresholds, len(self.fine_scale_names)),
        )
        for name, values, expected in expected_lengths:
            if len(values) != expected:
                raise ValueError(f'matting_loss.{name} must contain {expected} values, got {len(values)}')
        if any(weight < 0 for weight in (self.coarse_scale_weights + self.fine_scale_weights + self.unknown_scale_weights + self.mae_unknown_factors)):
            raise ValueError('matting loss scale weights must be non-negative')
        if any(kernel < 2 or kernel > 30 for kernel in self.unknown_kernel_sizes):
            raise ValueError('unknown_kernel_sizes must be between 2 and 30 because compute_unknown only provides kernels up to size 29')
        if any(kernel < 1 for kernel in self.unknown_gt_support_kernel_sizes):
            raise ValueError('unknown_gt_support_kernel_sizes must be positive')
        if any(support < core for support, core in zip(self.unknown_gt_support_kernel_sizes, self.unknown_kernel_sizes)):
            raise ValueError('unknown_gt_support_kernel_sizes must not be smaller than unknown_kernel_sizes')
        if (len(self.soft_unknown_thresholds) != 2 or not 0 <= self.soft_unknown_thresholds[0] <  self.soft_unknown_thresholds[1] <= 255):
            raise ValueError('matting_loss.soft_unknown_thresholds must contain two increasing values in [0, 255]')
        if not 0 < self.hard_boundary_threshold < 1:
            raise ValueError('matting_loss.hard_boundary_threshold must be in (0, 1)')
        if any(not 0 < threshold < 1 for threshold in self.prediction_error_thresholds):
            raise ValueError('matting_loss.prediction_error_thresholds values must be in (0, 1)')
        
        valid_unknown_sources = {'pred', 'gt', 'union', 'gt_pred_limited'}
        if self.unknown_source not in valid_unknown_sources:
            raise ValueError('matting_loss.unknown_source must be one of '
                f'{sorted(valid_unknown_sources)}, got {self.unknown_source}')

    @staticmethod
    def _dilate_mask(mask, kernel_size):
        """Dilate an NCHW bool mask while preserving its spatial shape."""
        if kernel_size == 1:
            return mask.bool()
        pad_before = kernel_size // 2
        pad_after = kernel_size - 1 - pad_before
        mask = F.pad(mask.float(), (pad_before, pad_after, pad_before, pad_after), mode='constant', value=0)
        return F.max_pool2d(mask, kernel_size=kernel_size, stride=1).bool()

    def _build_gt_unknown_base(self, alphas):
        """Find GT transition pixels for both soft and binary alpha mattes."""
        alpha = alphas.detach().clamp(0, 1)
        soft_lower, soft_upper = (threshold / 255.0 for threshold in self.soft_unknown_thresholds)
        soft_unknown = ((alpha > soft_lower) & (alpha < soft_upper))

        # Detect the hard foreground/background contour after binarization.
        # Computing the local range on continuous alpha is too sensitive to
        # near-white/near-black quantization noise inside an object.
        binary_fg = (alpha >= self.hard_boundary_threshold).to(alpha.dtype)
        local_max = F.max_pool2d(binary_fg, kernel_size=3, stride=1, padding=1)
        local_min = -F.max_pool2d(-binary_fg, kernel_size=3, stride=1, padding=1)
        hard_boundary = local_max != local_min
        return soft_unknown | hard_boundary

    def _build_unknown_weight(self, pred_alpha, alphas, index, gt_unknown_base=None):
        """Combine prediction uncertainty with a GT-derived transition band.

        ``gt_pred_limited`` guarantees supervision on the GT transition and
        only retains prediction uncertainty close to that transition.  When
        enabled, confident prediction errors are then added without the GT
        support restriction so false positives/negatives also get refined.
        """
        kernel_size = self.unknown_kernel_sizes[index]
        pred_unknown = compute_unknown(
            pred_alpha.detach().clamp(1e-7, 1 - 1e-7),
            k_size=kernel_size,
            is_train=self.training,
            lower_thres=1.0 / 255.0,
            upper_thres=0.98).bool()
        if self.unknown_source == 'pred':
            unknown = pred_unknown
        else:
            if gt_unknown_base is None:
                gt_unknown_base = self._build_gt_unknown_base(alphas)
            gt_unknown = self._dilate_mask(gt_unknown_base, kernel_size)
            if self.unknown_source == 'gt':
                unknown = gt_unknown
            elif self.unknown_source == 'union':
                # This is the direct counterpart of MaGGIe's ``unknown_gt | unknown_pred_os8`` reweighting.
                unknown = gt_unknown | pred_unknown
            else:
                gt_support = self._dilate_mask(gt_unknown_base, self.unknown_gt_support_kernel_sizes[index])
                unknown = gt_unknown | (pred_unknown & gt_support)

        if self.include_prediction_error:
            pred_error = ((pred_alpha.detach() - alphas.detach()).abs() > self.prediction_error_thresholds[index])
            unknown = unknown | pred_error
        return unknown.to(device=pred_alpha.device, dtype=pred_alpha.dtype)

    def compute_loss(self, pred, alphas, images_dist=None, phas_dist=None):
        if alphas is None:
            raise KeyError('FocalNetMatting training requires batch[\'alpha\']')
        if len(pred) != self.expected_total_outputs:
            raise ValueError('FocalDecoder must return x32/x16/x8/x4/x2/x1 (six outputs), '
                f'got {len(pred)}')

        scaled_preds = pred
        total_loss = alphas.new_zeros(())
        loss_dict = {}
        alpha_lap_pyramid = self.lap_loss.build_pyramid(alphas)
        gt_unknown_base = (None if self.unknown_source == 'pred' else self._build_gt_unknown_base(alphas))

        # Coarse semantic supervision: x32 and x16 only. These branches are deliberately not upsampled for pixel-level matting losses.
        coarse_semantic = alphas.new_zeros(())
        for scale_name, pred_lvl, scale_weight in zip(self.coarse_scale_names, scaled_preds[:self.coarse_output_count], self.coarse_scale_weights):
            target = self.build_coarse_semantic_target(alphas, pred_lvl.shape[-2:])
            value = (F.mse_loss(pred_lvl.sigmoid(), target) * self.coarse_semantic_loss_weight * scale_weight)
            coarse_semantic = coarse_semantic + value
            loss_dict[f'coarse_semantic_{scale_name}'] = value
        total_loss = total_loss + coarse_semantic
        loss_dict['coarse_semantic'] = coarse_semantic

        # Pixel criteria on x8/x4/x2/x1 at full target resolution.
        fine_preds = scaled_preds[self.coarse_output_count:]
        fine_full = []
        for index, (scale_name, pred_lvl, scale_weight) in enumerate(zip(self.fine_scale_names, fine_preds, self.fine_scale_weights)):
            pred_alpha = pred_lvl.sigmoid()
            if pred_alpha.shape[-2:] != alphas.shape[-2:]:
                pred_alpha = F.interpolate(pred_alpha, size=alphas.shape[-2:], mode='bilinear', align_corners=False)
            fine_full.append(pred_alpha)

            # Build the mask once and share it between MAE reweighting and
            # the x8/x4/x2 detail losses.  x1 only needs a mask when its MAE unknown factor is enabled.
            needs_detail_unknown = index < len(self.unknown_scale_names)
            needs_mae_unknown = ('mae' in self.criterions_last and self.mae_unknown_factors[index] > 0)
            unknown = None
            if needs_detail_unknown or needs_mae_unknown:
                unknown = self._build_unknown_weight(pred_alpha, alphas, index, gt_unknown_base)

            # Keep the historical criterion interface (criteria consume sigmoid probabilities, including BCELoss and SSIMLoss).
            for criterion_name, criterion in self.criterions_last.items():
                if criterion_name == 'ssim' and index != len(fine_preds) - 1:
                    continue
                if (criterion_name == 'mae' and self.mae_unknown_factors[index] > 0):
                    # Base weight 1 keeps all foreground/background pixels in
                    # the objective; unknown pixels receive an additive
                    # configurable weight, matching MaGGIe-style reweighting.
                    pixel_weight = (1.0 + self.mae_unknown_factors[index] * unknown)
                    value = (F.l1_loss(pred_alpha, alphas, reduction='none') * pixel_weight).sum() / pixel_weight.sum().clamp_min(1e-6)
                else:
                    value = criterion(pred_alpha, alphas)
                value = (value * self.lambdas_pix_last[criterion_name] * self.pix_loss_weight * scale_weight)
                total_loss = total_loss + value
                loss_dict[criterion_name] = (loss_dict.get(criterion_name, alphas.new_zeros(())) + value)
                loss_dict[f'{criterion_name}_{scale_name}'] = value

            # x8, x4 and x2 receive progressively narrower unknown-region
            # detail losses. Avoid strong global high-frequency supervision on predictions that were bilinearly upsampled.
            if needs_detail_unknown:
                unknown_weight = self.unknown_scale_weights[index]
                grad_un = self.grad_loss(pred_alpha, alphas, mask=unknown)
                lap_un = self.lap_loss(pred_alpha, alphas, unknown, target_pyramid=alpha_lap_pyramid)
                grad_un = grad_un * self.loss_alpha_gradun_w * unknown_weight
                lap_un = lap_un * self.loss_alpha_lapun_w * unknown_weight
                total_loss = total_loss + grad_un + lap_un
                loss_dict['grad_un'] = loss_dict.get('grad_un', alphas.new_zeros(())) + grad_un
                loss_dict['lap_un'] = loss_dict.get('lap_un', alphas.new_zeros(())) + lap_un
                loss_dict[f'grad_un_{scale_name}'] = grad_un
                loss_dict[f'lap_un_{scale_name}'] = lap_un

        # Global detail losses are defined on the final x1 prediction only.
        final_alpha = fine_full[-1]
        grad = self.grad_loss(final_alpha, alphas) * self.loss_alpha_grad_w
        lap = self.lap_loss(final_alpha, alphas, target_pyramid=alpha_lap_pyramid) * self.loss_alpha_lap_w
        total_loss = total_loss + grad + lap
        loss_dict['grad'] = grad
        loss_dict['lap'] = lap

        if images_dist is not None and phas_dist is not None:
            ddc = F.l1_loss(images_dist, phas_dist) * self.loss_alpha_ddc_w
            total_loss = total_loss + ddc
            loss_dict['ddc'] = ddc

        loss_dict['total'] = total_loss
        return loss_dict

    def forward(self, batch, **kwargs):
        image = batch['image']
        alphas = batch.get('alpha', None)
        embedding = self.encoder(image)
        pred = self.decoder(embedding)
        output = {'alpha_pred': pred[-1].sigmoid()}

        if not self.training:
            return output

        if self.cfg.use_ddc_loss:
            images_dist, phas_dist = self.consistency(image, output['alpha_pred'])
        else:
            images_dist, phas_dist = None, None
        return output, self.compute_loss(pred, alphas, images_dist, phas_dist)


def focalnet_matting(**kwargs):
    """Compatibility constructor for config-based model lookup."""
    return FocalNetMatting(**kwargs)
