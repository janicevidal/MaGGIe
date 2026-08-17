import torch
import torch.nn as nn
from torch.nn import functional as F
from huggingface_hub import PyTorchModelHubMixin
from yacs.config import CfgNode

from ..encoder import *
from ..decoder import *
from ..loss import LapLoss, GradientLoss, SSIMLoss
from ...utils.utils import compute_unknown


class BiRefNetPro(nn.Module, PyTorchModelHubMixin):
    def __init__(self, cfg):
        super(BiRefNetPro, self).__init__()
        if isinstance(cfg, dict):
            cfg = CfgNode(init_dict=cfg)
        self.cfg = cfg

        self.encoder = eval(cfg.encoder)(**cfg.encoder_args)
        self.decoder = eval(cfg.decoder)(**cfg.decoder_args)

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
        
        # Forward through decoder
        pred = self.decoder(embedding)
        
        output = {}
        output['alpha_pred'] = pred[-1][-1].sigmoid()
        
        # Compute loss during training
        if self.training:
            if self.cfg.use_ddc_loss:
                images_dist, phas_dist = self.consistency(x, output['alpha_pred'])
            else:
                images_dist, phas_dist = None, None
            
            loss_dict = self.compute_loss(pred, alphas, images_dist, phas_dist)
            
            return output, loss_dict
        
        return output

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

        for idx, pred_lvl in enumerate(scaled_preds):
            if pred_lvl.shape != alphas.shape:
                pred_lvl = F.interpolate(pred_lvl, size=alphas.shape[2:], mode='bilinear')
                scaled_weight = 1.0
            else:
                scaled_weight = len(scaled_preds)
            
            # pred_sigmoid = pred_lvl.sigmoid()
            
            for criterion_name, criterion in self.criterions_last.items():
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