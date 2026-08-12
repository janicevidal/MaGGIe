import torch.nn as nn
from torch.nn import functional as F
from huggingface_hub import PyTorchModelHubMixin
from yacs.config import CfgNode

from ..encoder import *
from ..decoder import *
from ..loss import LapLoss, GradientLoss, SSIMLoss


class BiRefNet(nn.Module, PyTorchModelHubMixin):
    def __init__(self, cfg):
        super(BiRefNet, self).__init__()
        if isinstance(cfg, dict):
            cfg = CfgNode(init_dict=cfg)
        self.cfg = cfg

        self.encoder = eval(cfg.encoder)(**cfg.encoder_args)
        self.decoder = eval(cfg.decoder)(**cfg.decoder_args)

        # Some weights for loss
        self.lap_loss = LapLoss()
        self.grad_loss = GradientLoss()
        
        self.criterions_last = {}
        if 'bce' in cfg.lambdas_pix_last and cfg.lambdas_pix_last['bce']:
            self.criterions_last['bce'] = nn.BCELoss()
        if 'ssim' in cfg.lambdas_pix_last and cfg.lambdas_pix_last['ssim']:
            self.criterions_last['ssim'] = SSIMLoss()
        if 'mae' in cfg.lambdas_pix_last and cfg.lambdas_pix_last['mae']:
            self.criterions_last['mae'] = nn.L1Loss()
        
        self.lambdas_pix_last = self.cfg.lambdas_pix_last
        self.pix_loss_weight = self.cfg.pix_loss_weight
        self.loss_alpha_lap_w = self.cfg.loss_alpha_lap_w
        self.loss_alpha_grad_w = self.cfg.loss_alpha_grad_w
        
        if self.cfg.decoder_args.out_ref:
            self.criterion_gdt = nn.BCELoss()
            self.gdt_loss_weight = self.cfg.gdt_loss_weight

        # Init weights
        self._init_weights(self.decoder)
        
        if hasattr(self.encoder, 'init_weights'):
            self.encoder.init_weights()
    
    def _init_weights(self, module):
        for name, p in module.named_parameters():
            if p.dim() > 1:
                if 'conv' in name or isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(p, mode='fan_out', nonlinearity='relu')
                elif 'linear' in name or isinstance(module, nn.Linear):
                    nn.init.kaiming_normal_(p, mode='fan_in', nonlinearity='relu')
                else:
                    nn.init.kaiming_normal_(p, mode='fan_out', nonlinearity='relu')
            elif 'bias' in name:
                nn.init.constant_(p, 0)

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
            loss_dict = self.compute_loss(pred, alphas)
            
            return output, loss_dict
        
        return output

    def compute_loss(self, pred, alphas):
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

        for _, pred_lvl in enumerate(scaled_preds):
            if pred_lvl.shape != alphas.shape:
                pred_lvl = F.interpolate(pred_lvl, size=alphas.shape[2:], mode='bilinear')
            for criterion_name, criterion in self.criterions_last.items():
                _loss = criterion(pred_lvl.sigmoid(), alphas) * self.lambdas_pix_last[criterion_name] * self.pix_loss_weight
                total_loss += _loss
                loss_dict[criterion_name] = loss_dict.get(criterion_name, 0.) + _loss / len(scaled_preds)
                
        final_pred = scaled_preds[-1]
        if final_pred.shape != alphas.shape:
            final_pred = F.interpolate(final_pred, size=alphas.shape[2:], mode='bilinear', align_corners=True)
        final_sigmoid = final_pred.sigmoid()
        
        loss_grad = self.grad_loss(final_sigmoid, alphas) * self.loss_alpha_grad_w
        loss_lap = self.lap_loss(final_sigmoid, alphas) * self.loss_alpha_lap_w
        
        total_loss += loss_grad + loss_lap
        loss_dict['grad'] = loss_grad
        loss_dict['lap'] = loss_lap
        
        loss_dict['total'] = total_loss

        return loss_dict