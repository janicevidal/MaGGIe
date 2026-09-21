"""FocalMatter matting architecture with the dugMatting uncertainty objective.

``FocalNetMattingNIG`` is :class:`FocalNetMatting` plus one optional loss term.
Everything that already trains today is untouched: the six-scale supervision,
the coarse semantic objective on x32/x16, the unknown-region gradient and
Laplacian losses, the MAE unknown weighting and the DDC consistency loss all
keep running exactly as before.  The NIG term is added on top of the final
full-resolution prediction only, mirroring the reference implementation, which
attaches its four heads to a single last layer.

Two ways to use it, both backward compatible:

* ``nig_weight: 0`` (default) -- the decoder may carry the uncertainty heads and
  the result is numerically identical to :class:`FocalNetMatting`, because the
  heads are not even evaluated.
* ``nig_weight > 0`` -- ``decoder: focal_uncertainty_decoder`` with
  ``decoder_args.nig_heads: true`` is required; the objective then becomes
  ``total + nig_weight * criterion_nig(...)``.

Existing checkpoints load unchanged: ``decoder.alpha_head`` keeps its name and
shape, and the new heads are reported as missing keys and start zero-gated at
their floors (see :class:`FocalUncertaintyDecoder`).
"""

from .focalnet_matting import FocalNetMatting
from ..loss import criterion_nig


class FocalNetMattingNIG(FocalNetMatting):
    """FocalNetMatting with an evidential (Normal-Inverse-Gamma) objective.

    The NIG parameters are interpreted as in the paper:

    * ``E[sigma^2] = beta / (alpha - 1)`` is the *aleatoric* (data noise) map;
    * ``Var[gamma] = beta / (omega * (alpha - 1))`` is the *epistemic* (model)
      map;
    * ``Var[sigma^2] = beta^2 / ((alpha - 1)^2 * (alpha - 2))`` indicates where
      the aleatoric estimate itself is unreliable.

    All three are exposed on the output dict (``aleatoric``, ``epistemic``,
    ``aleatoric_variance``) whenever the decoder was asked for them, which makes
    them directly usable for visualisation, for gating a refinement module or
    for reweighting the existing unknown-region losses.

    Args (config):
        decoder_args.nig_heads: build the three uncertainty heads.
        decoder_args.eval_uncertainty: also produce them at inference.
        decoder_args.matting_loss.nig_weight: weight of the NIG term (0 disables
            it and keeps the previous training behaviour bit for bit).
        decoder_args.matting_loss.nig_lambda: evidence regulariser of the NIG
            term, 0.01 in the reference implementation.
    """

    supported_decoders = FocalNetMatting.supported_decoders + (
        'focal_uncertainty_decoder', 'FocalUncertaintyDecoder')

    def __init__(self, cfg):
        super().__init__(cfg)
        loss_cfg = cfg.decoder_args.get('matting_loss', {})
        self.nig_weight = float(loss_cfg.get('nig_weight', 0.0))
        self.nig_lambda = float(loss_cfg.get('nig_lambda', 0.01))
        if self.nig_weight < 0.0:
            raise ValueError('matting_loss.nig_weight must be non-negative')
        if self.nig_lambda < 0.0:
            raise ValueError('matting_loss.nig_lambda must be non-negative')
        if self.nig_weight > 0.0 and not getattr(self.decoder, 'nig_heads', False):
            raise ValueError(
                'matting_loss.nig_weight > 0 needs the uncertainty heads: use '
                'decoder: focal_uncertainty_decoder with '
                'decoder_args.nig_heads=true')

        # BiRefNetPro re-initialises the decoder after building it, so the
        # zero-gating of the new heads has to be re-applied here.
        reset = getattr(self.decoder, 'reset_uncertainty_heads', None)
        if self.nig_weight > 0.0 and callable(reset):
            reset()

    @staticmethod
    def split_prediction(pred):
        """Split ``decoder(...)`` into ``(predictions, uncertainty_or_None)``.

        Accepts both the plain prediction list returned by every existing
        decoder and the ``(predictions, uncertainty)`` pair returned by
        :class:`FocalUncertaintyDecoder` when the NIG parameters are active.
        """
        if isinstance(pred, tuple):
            if len(pred) != 2:
                raise ValueError(
                    'expected a (predictions, uncertainty) pair, got a tuple of '
                    'length {}'.format(len(pred)))
            return pred[0], pred[1]
        return pred, None

    @staticmethod
    def uncertainty_maps(uncertainty):
        """Derive the aleatoric/epistemic maps from the NIG parameters."""
        omega = uncertainty['omega']
        evidence_alpha = uncertainty['evidence_alpha'].clamp_min(1.0 + 1e-6)
        beta = uncertainty['beta']
        safe_shape = (evidence_alpha - 1.0).clamp_min(1e-6)
        return {
            'aleatoric': beta / safe_shape,
            'epistemic': beta / (omega.clamp_min(1e-6) * safe_shape),
            'aleatoric_variance': (
                beta ** 2 / (safe_shape ** 2 * (evidence_alpha - 2.0).clamp_min(1e-6))),
        }

    def compute_nig_loss(self, uncertainty, alpha_pred, alphas):
        """Negative log-likelihood of the full-resolution matte."""
        if alphas is None:
            raise KeyError("FocalNetMattingNIG training requires batch['alpha']")
        return criterion_nig(
            alpha_pred,
            uncertainty['omega'],
            uncertainty['evidence_alpha'],
            uncertainty['beta'],
            alphas.clamp(0.0, 1.0),
            lam=self.nig_lambda,
            reduction='mean',
        )

    def forward(self, batch, **kwargs):
        image = batch['image']
        alphas = batch.get('alpha', None)
        embedding = self.encoder(image)
        pred, uncertainty = self.split_prediction(self.decoder(embedding))

        output = {'alpha_pred': pred[-1].sigmoid()}
        if uncertainty is not None:
            output.update(self.uncertainty_maps(uncertainty))

        if not self.training:
            return output

        if self.cfg.use_ddc_loss:
            images_dist, phas_dist = self.consistency(image, output['alpha_pred'])
        else:
            images_dist, phas_dist = None, None

        loss_dict = self.compute_loss(pred, alphas, images_dist, phas_dist)

        if self.nig_weight > 0.0:
            if uncertainty is None:
                raise RuntimeError('the NIG objective needs the uncertainty heads, but the decoder returned plain predictions')
            nig_loss = (self.compute_nig_loss(uncertainty, output['alpha_pred'], alphas) * self.nig_weight)
            loss_dict['nig'] = nig_loss
            loss_dict['total'] = loss_dict['total'] + nig_loss

        return output, loss_dict


def focalnet_matting_nig(**kwargs):
    """Compatibility constructor for config-based model lookup."""
    return FocalNetMattingNIG(**kwargs)
