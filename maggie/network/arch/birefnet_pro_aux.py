import torch
import torch.nn as nn
from torch.nn import functional as F
from huggingface_hub import PyTorchModelHubMixin
from yacs.config import CfgNode

from ..encoder import *
from ..decoder import *
from ..loss import GradientLoss, hybrid_e_loss, LapLoss, SSIMLoss
from ...utils.utils import compute_unknown


class SemanticAuxHead(nn.Module):
    """Training-only FPN head with gated quarter-resolution detail."""

    def __init__(self, in_channels, hidden_channels=64):
        super().__init__()
        if len(in_channels) != 4:
            raise ValueError(
                "SemanticAuxHead expects channels for x4, x3, x2, and x1")

        self.context_projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, hidden_channels, kernel_size=1,
                          bias=False),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU(inplace=True))
            for channels in in_channels[:-1]
        ])
        self.context_fuse = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True))
        self.detail_projection = nn.Sequential(
            nn.Conv2d(in_channels[-1], hidden_channels, kernel_size=1,
                      bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True))
        # The gate is generated only from deep semantic context, preventing
        # shallow grass or texture responses from entering unconditionally.
        self.detail_gate = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.Sigmoid())
        self.output_fuse = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1))

    def forward(self, features):
        if len(features) != 4:
            raise ValueError("SemanticAuxHead expects features [x4, x3, x2, x1]")

        context_size = features[-2].shape[-2:]
        context = None
        for feature, projection in zip(features[:-1], self.context_projections):
            feature = projection(feature)
            if feature.shape[-2:] != context_size:
                feature = F.interpolate(feature, size=context_size, mode='bilinear', align_corners=False)
            context = feature if context is None else context + feature

        context = self.context_fuse(context)
        output_size = features[-1].shape[-2:]
        context = F.interpolate(context, size=output_size, mode='bilinear', align_corners=False)
        detail = self.detail_projection(features[-1])
        fused = context + self.detail_gate(context) * detail
        return self.output_fuse(fused)


class BiRefNetProAux(nn.Module, PyTorchModelHubMixin):
    def __init__(self, cfg):
        super(BiRefNetProAux, self).__init__()
        if isinstance(cfg, dict):
            cfg = CfgNode(init_dict=cfg)
        self.cfg = cfg

        self.encoder = eval(cfg.encoder)(**cfg.encoder_args)
        self.decoder = eval(cfg.decoder)(**cfg.decoder_args)

        semantic_cfg = cfg.get('semantic_aux', None)
        self.use_semantic_aux = (
            semantic_cfg is not None and semantic_cfg.enabled)
        if self.use_semantic_aux:
            self.semantic_cfg = semantic_cfg
            # Encoder outputs are [image, x1, x2, x3, x4]. Deep x4/x3/x2
            # context gates x1 detail, producing quarter-resolution logits.
            semantic_in_channels = list(cfg.decoder_args.in_channels)
            self.semantic_head = SemanticAuxHead(
                semantic_in_channels,
                hidden_channels=semantic_cfg.hidden_channels)

        # Some weights for loss
        self.lap_loss = LapLoss(channels=1)
        self.grad_loss = GradientLoss()
        
        self.criterions_last = {}
        if 'bce' in cfg.lambdas_pix_last and cfg.lambdas_pix_last['bce']:
            self.criterions_last['bce'] = nn.BCELoss()
            # self.criterions_last['bce'] = nn.BCEWithLogitsLoss()
        if 'ssim' in cfg.lambdas_pix_last and cfg.lambdas_pix_last['ssim']:
            self.criterions_last['ssim'] = SSIMLoss()
        if 'mae' in cfg.lambdas_pix_last and cfg.lambdas_pix_last['mae']:
            self.criterions_last['mae'] = nn.L1Loss()
        
        self.lambdas_pix_last = self.cfg.lambdas_pix_last
        self.pix_loss_weight = self.cfg.pix_loss_weight
        self.coarse_semantic_loss_weight = (
            self.cfg.coarse_semantic_loss_weight)
        self.loss_alpha_lap_w = self.cfg.loss_alpha_lap_w
        self.loss_alpha_grad_w = self.cfg.loss_alpha_grad_w
        self.loss_alpha_lapun_w = self.cfg.loss_alpha_lapun_w
        self.loss_alpha_gradun_w = self.cfg.loss_alpha_gradun_w
        self.loss_alpha_ddc_w = self.cfg.loss_alpha_ddc_w  
        
        if self.cfg.decoder_args.out_ref:
            self.criterion_gdt = nn.BCELoss()
            # self.criterion_gdt = nn.BCEWithLogitsLoss()
            self.gdt_loss_weight = self.cfg.gdt_loss_weight

        # Init weights
        self.decoder.apply(self._init_weights)
        if self.use_semantic_aux:
            self.semantic_head.apply(self._init_weights)
        
        if hasattr(self.encoder, 'init_weights'):
            self.encoder.init_weights()
    
    def _init_weights(self, module):
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
    
    def consistency(self, image, alpha, kernel_size=11):
        b, c, h, w = image.shape
        mean = image.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = image.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        image = image * std + mean
        unfold_image = F.unfold(image, kernel_size=kernel_size, padding=kernel_size // 2).view(b, c, kernel_size ** 2, h, w)
        image_dist = torch.norm(image.view(b, c, 1, h, w) - unfold_image, 2, dim=1)
        image_dist, indices = torch.topk(image_dist, k=kernel_size, dim=1, largest=False)
        unfold_alpha = F.unfold(alpha, kernel_size=kernel_size, padding=kernel_size // 2).view(b, kernel_size ** 2, h, w)
        alpha_dist = torch.gather(alpha - unfold_alpha, dim=1, index=indices)

        return image_dist, alpha_dist

    def forward(self, batch, **kwargs):
        '''
        batch:
            image: b, 3, h, w 
                image tensors
            alpha: b, 1, h, w
                GT alpha matte
        '''

        # Forward encoder
        x = batch['image']
        alphas = batch.get('alpha', None)

        # Forward through encoder
        embedding = self.encoder(x)

        if not self.training:
            pred = self.decoder(embedding)
            output = {'alpha_pred': pred[-1].sigmoid()}
            return output

        if alphas is None:
            raise KeyError("Training BiRefNetPro requires an 'alpha' target")

        is_matting = batch.get('is_matting')
        if is_matting is None:
            is_matting = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
        else:
            is_matting = is_matting.reshape(-1).bool()
        if is_matting.numel() != x.shape[0]:
            raise ValueError("is_matting must contain one flag per sample")
        if not self.use_semantic_aux and not torch.all(is_matting):
            raise ValueError(
                "Binary-only samples require model.semantic_aux.enabled=true")

        output = {}
        loss_dict = {}
        total_loss = x.sum() * 0

        if self.use_semantic_aux:
            quarter_semantic_logits = self.semantic_head(
                [embedding[4], embedding[3], embedding[2], embedding[1]])
            semantic_target = (
                alphas > self.semantic_cfg.target_threshold).float()
            # Supervise the prediction at the original target resolution.
            # Interpolate logits (rather than probabilities) so losses using
            # BCEWithLogitsLoss retain the correct numerical semantics.
            semantic_logits = F.interpolate(
                quarter_semantic_logits,
                size=semantic_target.shape[-2:],
                mode='bilinear', align_corners=False)
            semantic_loss = self.compute_semantic_loss(
                semantic_logits, semantic_target)
            loss_dict.update(semantic_loss)
            total_loss = total_loss + semantic_loss['semantic']
            output['semantic_pred'] = semantic_logits.sigmoid()

        if torch.any(is_matting):
            matting_embedding = [
                feature[is_matting] for feature in embedding
            ]
            pred = self.decoder(matting_embedding)
            scaled_preds = pred[1] if self.cfg.decoder_args.out_ref else pred
            alpha_pred = scaled_preds[-1].sigmoid()

            if torch.all(is_matting):
                output['alpha_pred'] = alpha_pred
            else:
                full_alpha_pred = alpha_pred.new_zeros(
                    (x.shape[0], *alpha_pred.shape[1:]))
                full_alpha_pred[is_matting] = alpha_pred
                output['alpha_pred'] = full_alpha_pred

            if self.cfg.use_ddc_loss:
                images_dist, phas_dist = self.consistency(
                    x[is_matting], alpha_pred)
            else:
                images_dist, phas_dist = None, None

            matting_loss = self.compute_loss(
                pred, alphas[is_matting], images_dist, phas_dist)
            total_loss = total_loss + matting_loss.pop('total')
            loss_dict.update(matting_loss)
        else:
            output['alpha_pred'] = x.new_zeros(
                (x.shape[0], 1, *x.shape[-2:]))

        loss_dict['total'] = total_loss
        return output, loss_dict

    def compute_semantic_loss(self, logits, target):
        if logits.shape != target.shape:
            raise ValueError(
                "Full-resolution semantic logits and target must have "
                f"identical shapes, got {logits.shape} and {target.shape}")
        zero = logits.sum() * 0
        semantic_sum = zero
        loss_dict = {}

        bce_weight = self.semantic_cfg.get('bce_weight', 0.0)
        if bce_weight:
            bce = F.binary_cross_entropy_with_logits(logits, target)
            semantic_sum = semantic_sum + bce_weight * bce
            loss_dict['semantic_bce'] = bce

        dice_weight = self.semantic_cfg.get('dice_weight', 0.0)
        if dice_weight:
            probability = logits.sigmoid()
            intersection = (probability * target).sum(dim=(1, 2, 3))
            denominator = probability.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
            dice = 1 - ((2 * intersection + 1) / (denominator + 1)).mean()
            semantic_sum = semantic_sum + dice_weight * dice
            loss_dict['semantic_dice'] = dice

        hybrid_weight = self.semantic_cfg.get('hybrid_e_weight', 0.0)
        if hybrid_weight:
            hybrid = hybrid_e_loss(
                logits, target,
                kernel_size=self.semantic_cfg.get('hybrid_e_kernel_size', 5),
                boundary_factor=self.semantic_cfg.get('hybrid_e_boundary_factor', 5.0))
            semantic_sum = semantic_sum + hybrid_weight * hybrid
            loss_dict['semantic_hybrid_e'] = hybrid

        hard_negative_weight = self.semantic_cfg.get('hard_negative_weight', 0.0)
        if hard_negative_weight:
            loss_map = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
            hard_negative_losses = []
            for sample_loss, sample_target in zip(loss_map, target):
                negative_loss = sample_loss[sample_target < 0.5]
                if negative_loss.numel() == 0:
                    continue
                num_hard = max(1, int(negative_loss.numel() * self.semantic_cfg.hard_negative_ratio))
                hard_negative_losses.append( torch.topk(negative_loss, num_hard).values.mean())
            if hard_negative_losses:
                hard_negative = torch.stack(hard_negative_losses).mean()
            else:
                hard_negative = zero
            semantic_sum = (
                semantic_sum + hard_negative_weight * hard_negative)
            loss_dict['semantic_hard_bg'] = hard_negative

        if not loss_dict:
            raise ValueError(
                "At least one semantic auxiliary loss weight must be nonzero")

        loss_dict['semantic'] = (
            self.semantic_cfg.loss_weight * semantic_sum)
        return loss_dict

    @staticmethod
    def build_coarse_semantic_target(alpha, output_size):
        """Create MODNet's blurred low-resolution matte target."""
        target = F.interpolate(
            alpha, size=output_size, mode='bilinear', align_corners=False)

        # Exact output of scipy.ndimage.gaussian_filter applied to a centered
        # impulse in a 3x3 array with sigma=0.8 (the referenced MODNet code).
        kernel = target.new_tensor([
            [0.062610564, 0.124999902, 0.062610564],
            [0.124999902, 0.249558134, 0.124999902],
            [0.062610564, 0.124999902, 0.062610564],
        ]).view(1, 1, 3, 3)
        kernel = kernel.expand(target.shape[1], 1, 3, 3)

        target = F.pad(target, (1, 1, 1, 1), mode='reflect')
        return F.conv2d(target, kernel, groups=target.shape[1])

    def compute_loss(self, pred, alphas, images_dist, phas_dist):
        total_loss = 0
        loss_dict = {}
        
        scaled_preds = pred
        
        if self.cfg.decoder_args.out_ref:
            (outs_gdt_pred, outs_gdt_label), scaled_preds = scaled_preds
            for _idx, (_gdt_pred, _gdt_label) in enumerate(zip(outs_gdt_pred, outs_gdt_label)):
                _gdt_pred = F.interpolate(_gdt_pred, size=_gdt_label.shape[2:], mode='bilinear').sigmoid().clamp(1e-7, 1 - 1e-7)
                # _gdt_label = _gdt_label.sigmoid()
                _gdt_label = _gdt_label.sigmoid().clamp(1e-7, 1 - 1e-7)
                loss_gdt = self.criterion_gdt(_gdt_pred, _gdt_label) if _idx == 0 else self.criterion_gdt(_gdt_pred, _gdt_label) + loss_gdt
            
            loss_dict['gdt'] = loss_gdt * self.gdt_loss_weight
            total_loss += loss_gdt * self.gdt_loss_weight

        fine_level_start = max(len(scaled_preds) - 2, 0)
        for idx, pred_lvl in enumerate(scaled_preds):
            is_coarse_level = idx < fine_level_start

            if is_coarse_level:
                semantic_target = self.build_coarse_semantic_target(
                    alphas, pred_lvl.shape[-2:])
                coarse_semantic_loss = F.mse_loss(
                    pred_lvl.sigmoid(), semantic_target)
                coarse_semantic_loss *= self.coarse_semantic_loss_weight
                total_loss += coarse_semantic_loss
                loss_dict['coarse_semantic'] = (
                    loss_dict.get('coarse_semantic', 0.) +
                    coarse_semantic_loss)
                # The coarse branches use only this semantic objective. Do not
                # upsample them for the pixel-level L1/SSIM objectives below.
                continue

            if pred_lvl.shape != alphas.shape:
                pred_lvl = F.interpolate(
                    pred_lvl, size=alphas.shape[2:], mode='bilinear',
                    align_corners=False)
                scaled_weight = 1.0
            else:
                scaled_weight = len(scaled_preds)
            
            # pred_sigmoid = pred_lvl.sigmoid()
            
            for criterion_name, criterion in self.criterions_last.items():
                # Only x8 and x1 reach this block; x32/x16 are supervised by
                # the native-resolution semantic objective above. SSIM is
                # reserved for the final full-resolution x1 prediction.
                if (criterion_name == 'ssim' and
                        idx != len(scaled_preds) - 1):
                    continue
                _loss = criterion(pred_lvl.sigmoid(), alphas) * self.lambdas_pix_last[criterion_name] * self.pix_loss_weight * scaled_weight
                total_loss += _loss
                loss_dict[criterion_name] = loss_dict.get(criterion_name, 0.) + _loss
                # if criterion_name == 'bce':
                #     _loss = criterion(pred_lvl, alphas) 
                # else:
                #     _loss = criterion(pred_sigmoid, alphas)
                
                # _loss = _loss * self.lambdas_pix_last[criterion_name] * self.pix_loss_weight * scaled_weight
                # total_loss += _loss
                # loss_dict[criterion_name] = loss_dict.get(criterion_name, 0.) + _loss
            
            if idx in [2, 3]:
                pred_sigmoid = pred_lvl.sigmoid().clamp(1e-7, 1 - 1e-7)
                
                k_size = 15 if idx == 2 else 11
                    
                weight_mask = compute_unknown(
                    pred_sigmoid,
                    k_size=k_size,
                    is_train=self.training,
                    lower_thres=1.0/255.0,
                    upper_thres=0.98
                ).to(pred_sigmoid.device)
                
                loss_grad = self.grad_loss(pred_sigmoid, alphas, mask=weight_mask)
                loss_grad *= self.loss_alpha_gradun_w
                total_loss += loss_grad
                
                loss_lap = self.lap_loss(pred_sigmoid, alphas, weight_mask)
                loss_lap *= self.loss_alpha_lapun_w
                total_loss += loss_lap
                
                loss_dict['grad_unknown'] = loss_dict.get('grad_unknown', 0.) + loss_grad
                loss_dict['lap_unknown'] = loss_dict.get('lap_unknown', 0.) + loss_lap
                
        final_pred = scaled_preds[-1]
        final_sigmoid = final_pred.sigmoid()
        
        loss_grad = self.grad_loss(final_sigmoid, alphas) * self.loss_alpha_grad_w
        loss_lap = self.lap_loss(final_sigmoid, alphas) * self.loss_alpha_lap_w
        
        total_loss += loss_grad + loss_lap
        loss_dict['grad'] = loss_grad
        loss_dict['lap'] = loss_lap
        
        if images_dist is not None and phas_dist is not None:
            loss_ddc = F.l1_loss(images_dist, phas_dist) * self.loss_alpha_ddc_w
            total_loss += loss_ddc
            loss_dict['ddc'] = loss_ddc
        
        loss_dict['total'] = total_loss

        return loss_dict
