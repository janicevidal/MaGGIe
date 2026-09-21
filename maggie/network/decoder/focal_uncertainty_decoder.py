"""FocalMatter decoder with plug-in NIG uncertainty heads.

The class reproduces :class:`FocalDecoder` module for module and only appends
three 1x1 heads on the very same full-resolution head input.  They predict the
parameters of the Normal-Inverse-Gamma (NIG) distribution used by *dugMatting*
(Wu et al., ICML 2023): ``omega`` (evidence precision of the mean),
``evidence_alpha`` and ``beta`` (Inverse-Gamma shape and scale).

Only the matte is needed to run the model; the three extra parameters are
consumed by the training objective, so they are skipped outside training unless
``eval_uncertainty`` asks for them (for visualisation or a test-time
uncertainty-guided refinement).

Backward compatibility is deliberate: ``alpha_head`` keeps its name *and* its
``(neck_channels + rgb_channels) -> 1`` shape, so a checkpoint trained with
:class:`FocalDecoder` loads with no missing, unexpected or shape-mismatched
key.  The new heads start zero-gated (``reset_uncertainty_heads``), which puts
``omega``/``evidence_alpha``/``beta`` at their floors everywhere and turns the
NIG objective into a plain per-pixel weighted regression until training moves
them.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .focal_decoder import FocalDecoder


class FocalUncertaintyDecoder(FocalDecoder):
    """FocalDecoder plus the decomposed-uncertainty (NIG) heads.

    Args:
        nig_heads: Build the three extra heads.  When false the decoder is
            exactly :class:`FocalDecoder` and still returns a plain prediction
            list, so it can be paired with any existing architecture.
        eval_uncertainty: Also compute the NIG parameters at inference.  Leave
            it off for deployment; it is only needed to look at the aleatoric
            map or to drive a test-time refinement.
        omega_floor, evidence_alpha_floor, beta_floor: Lower bounds added to the
            softplus activations.  ``evidence_alpha_floor`` must exceed 2 because
            ``Var[sigma^2] = beta**2 / ((alpha - 1)**2 * (alpha - 2))`` is only
            finite for ``alpha > 2``; the defaults follow the reference
            implementation (softplus + 0.1 / + 2.1 / + 0.1).

    Forward contract: returns the usual list of six predictions, or a
    ``(predictions, uncertainty)`` tuple while the uncertainty parameters are
    requested, where ``uncertainty`` is a dict of ``omega``,
    ``evidence_alpha`` and ``beta`` tensors at the full-resolution head.
    """

    def __init__(self, *args, nig_heads=False, eval_uncertainty=False,
                 omega_floor=0.1, evidence_alpha_floor=2.1, beta_floor=0.1,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.nig_heads = bool(nig_heads)
        self.eval_uncertainty = bool(eval_uncertainty)
        self.omega_floor = float(omega_floor)
        self.evidence_alpha_floor = float(evidence_alpha_floor)
        self.beta_floor = float(beta_floor)

        if self.omega_floor <= 0.0:
            raise ValueError('omega_floor must be positive')
        if self.evidence_alpha_floor <= 2.0:
            raise ValueError(
                'evidence_alpha_floor must exceed 2 so that '
                'Var[sigma^2] = beta^2 / ((alpha - 1)^2 (alpha - 2)) stays finite')
        if self.beta_floor <= 0.0:
            raise ValueError('beta_floor must be positive')

        self.uncertainty_head_names = (
            'omega_head', 'evidence_alpha_head', 'beta_head')
        if self.nig_heads:
            # Reuse the exact head input of ``alpha_head`` (FocalMatter's last
            # fusion of the decoded feature and the RGB stem), matching
            # ``conv_lamda/conv_alpha/conv_beta`` in the reference code.
            head_in_channels = self.alpha_head.in_channels
            self.omega_head = nn.Conv2d(head_in_channels, 1, kernel_size=1)
            self.evidence_alpha_head = nn.Conv2d(head_in_channels, 1, kernel_size=1)
            self.beta_head = nn.Conv2d(head_in_channels, 1, kernel_size=1)
            self.reset_uncertainty_heads()

    def reset_uncertainty_heads(self):
        """Zero-gate the uncertainty heads so they start at their floors.

        Random initialisation would start the NIG objective from an arbitrary
        per-pixel noise level, which is a hidden source of instability when
        finetuning a pretrained alpha branch.  Note that ``BiRefNetPro``
        re-initialises ``self.decoder`` *after* constructing it, so the
        architecture calls this again from its own ``__init__``.
        """
        if not self.nig_heads:
            return 0
        reset = 0
        for name in self.uncertainty_head_names:
            head = getattr(self, name)
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
            reset += head.weight.numel() + head.bias.numel()
        return reset

    def predict_uncertainty(self, head_input):
        """Return the NIG parameters of the full-resolution matte.

        The activations follow the reference implementation: the matte mean
        uses a sigmoid, and the three positive quantities use softplus with a
        floor.  ``beta`` grows monotonically with ``E[sigma^2] = beta/(alpha-1)``
        and ``alpha`` controls how heavy the distribution tail is.
        """
        if not self.nig_heads:
            raise RuntimeError('FocalUncertaintyDecoder was built with nig_heads=False')
        return {
            'omega': F.softplus(self.omega_head(head_input)) + self.omega_floor,
            'evidence_alpha': F.softplus(self.evidence_alpha_head(head_input)) + self.evidence_alpha_floor,
            'beta': F.softplus(self.beta_head(head_input)) + self.beta_floor,
        }

    def _wants_uncertainty(self):
        return self.nig_heads and (self.training or self.eval_uncertainty)

    def forward(self, features):
        p6, p5, p4, p3, neck_feature = self.decode(features)
        neck_pred = self.pred_neck(neck_feature)
        head_feature = self.full_resolution_feature(neck_feature, features[0].shape[-2:])
        
        head_input = torch.cat([head_feature, self.rgb_stem(features[0])], dim=1)
        alpha = self.alpha_head(head_input)

        predictions = [self.pred6(p6), self.pred5(p5), self.pred4(p4), self.pred3(p3), neck_pred, alpha]
        uncertainty = self.predict_uncertainty(head_input) if self._wants_uncertainty() else None

        if self.training and self.ms_supervision:
            selected = predictions
        elif self.eval_output == 'neck':
            # Evaluation utilities undo dataset padding/resizing assuming the
            # prediction is at input resolution. Upsample the x2 neck head
            # here so its IoU is measured after the same interpolation that a
            # deployment pipeline would apply, without changing training.
            selected = [F.interpolate(neck_pred, size=features[0].shape[-2:], mode='bilinear', align_corners=False)]
        else:
            selected = [alpha]

        return selected if uncertainty is None else (selected, uncertainty)


def focal_uncertainty_decoder(**kwargs):
    return FocalUncertaintyDecoder(**kwargs)
