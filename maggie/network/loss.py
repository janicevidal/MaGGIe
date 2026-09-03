import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
# from pudb.remote import set_trace


def hybrid_e_loss(pred, mask, kernel_size=31, boundary_factor=5.0,
                  eps=1e-8):
    """Boundary-aware BCE, enhanced-alignment, and weighted IoU loss.

    Args:
        pred: Binary foreground logits in ``(B, C, H, W)`` format.
        mask: Binary targets with the same shape as ``pred``.
        kernel_size: Odd averaging kernel used to locate boundary regions.
        boundary_factor: Additional weight assigned around target boundaries.
        eps: Numerical stability term.

    The calculation is promoted to float32 under mixed precision because the
    enhanced-alignment denominator can otherwise underflow for constant masks.
    """
    if pred.ndim != 4 or mask.ndim != 4:
        raise ValueError("hybrid_e_loss expects pred and mask in BCHW format")
    if pred.shape != mask.shape:
        raise ValueError("pred and mask must have identical shapes")
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd number")

    if pred.dtype in (torch.float16, torch.bfloat16):
        pred = pred.float()
    mask = mask.to(device=pred.device, dtype=pred.dtype)

    padding = kernel_size // 2
    local_average = F.avg_pool2d(
        mask, kernel_size=kernel_size, stride=1, padding=padding)
    weight = 1 + boundary_factor * torch.abs(local_average - mask)

    pixel_bce = F.binary_cross_entropy_with_logits(
        pred, mask, reduction='none')
    weighted_bce = (
        (weight * pixel_bce).sum(dim=(2, 3)) + eps
    ) / (weight.sum(dim=(2, 3)) + eps)

    probability = pred.sigmoid()
    pred_centered = probability - probability.mean(
        dim=(2, 3), keepdim=True)
    mask_centered = mask - mask.mean(dim=(2, 3), keepdim=True)
    alignment = (
        2.0 * pred_centered * mask_centered + eps
    ) / (
        pred_centered.square() + mask_centered.square() + eps
    )
    enhanced_alignment = (1 + alignment).square() / 4.0
    e_loss = 1.0 - enhanced_alignment.mean(dim=(2, 3))

    intersection = (probability * mask * weight).sum(dim=(2, 3))
    union = ((probability + mask) * weight).sum(dim=(2, 3))
    weighted_iou = 1.0 - (
        intersection + 1.0 + eps
    ) / (
        union - intersection + 1.0 + eps
    )

    return (weighted_bce + e_loss + weighted_iou).mean()


def _loss_dtSSD(pred, gt, mask):
    b, n_f, _, h, w = pred.shape
    dadt = pred[:, 1:] - pred[:, :-1]
    dgdt = gt[:, 1:] - gt[:, :-1]
    # import pdb; pdb.set_trace()
    diff = (dadt - dgdt) ** 2
    diff = diff * mask[:, 1:]
    # import pdb; pdb.set_trace()
    diff = torch.sum(diff) / torch.sum(mask[:, 1:] + 1e-6)
    return diff

def _loss_dtSSD_ohem_smooth_l1(pred, gt, mask, ratio=0.5, threshold=5e-5):
    '''
    ratio: maximum ratio of hard examples
    threshold: minimum loss of hard examples
    '''
    dadt = torch.abs(pred[:, 1:] - pred[:, :-1])
    dgdt = torch.abs(gt[:, 1:] - gt[:, :-1])
    diff = F.smooth_l1_loss(dadt, dgdt, reduction='none', beta=0.5)
    diff = diff * mask[:, 1:]
    if mask[:, 1:].sum() == 0:
        return diff.mean()
    diff = diff[mask[:, 1:] > 0]
    
    # Ohem
    diff, _ = torch.sort(diff, descending=True)
    
    min_value = diff[int(math.floor(diff.numel() * ratio))]
    min_value = max(min_value, threshold)
    min_value = min(min_value, diff.max())

    hard_loss = diff[diff >= min_value].mean()
    return hard_loss

def loss_dtSSD(pred, gt, mask):
    loss = _loss_dtSSD(pred, gt, mask) #+ _loss_dtSSD(torch.flip(pred, dims=(1,)), torch.flip(gt, dims=(1,)), torch.flip(mask, dims=(1,)))
    # loss = _loss_dtSSD_ohem_smooth_l1(pred, gt, mask)
    return loss
    # b, n_f, _, h, w = pred.shape
    # import pdb; pdb.set_trace()
    # dadt = pred[:, 1:] - pred[:, :-1]
    # dgdt = gt[:, 1:] - gt[:, :-1]
    # diff = (dadt - dgdt) ** 2
    # diff = diff * mask[:, 1:]
    # # import pdb; pdb.set_trace()
    # diff = torch.sum(diff) / torch.sum(mask[:, 1:])
    # return diff
    # metric = torch.sqrt(torch.sum((dadt - dgdt) ** 2, dim=(2, 3, 4)))
    # metric = torch.sum(metric)
    # count = ((n_f - 1) * b)
    # if torch.isnan(metric).any():
    #     # set_trace()
    # return metric/ (count + 1e-4)

def loss_comp(alpha_pred, alpha_gt, fg, bg, mask):
    comp_pred = alpha_pred * fg + (1 - alpha_pred) * bg
    comp_gt = alpha_gt * fg + (1 - alpha_gt) * bg
    loss = torch.sum(F.l1_loss(comp_pred, comp_gt, reduction='none') * mask) / (mask.sum() + 1e-6)
    return loss

class GradientLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.kernel_x, self.kernel_y = self.sobel_kernel()
        self.eps = eps

    def forward(self, logit, label, mask=None):
        if len(label.shape) == 3:
            label = label.unsqueeze(1)
        if mask is not None:
            if len(mask.shape) == 3:
                mask = mask.unsqueeze(1)
            logit = logit * mask
            label = label * mask
            # import pdb; pdb.set_trace()
            loss = torch.sum(
                F.l1_loss(self.sobel(logit), self.sobel(label), reduction='none')) / (
                    mask.sum() + self.eps)
        else:
            loss = F.l1_loss(self.sobel(logit), self.sobel(label), 'mean')

        return loss

    def sobel(self, input):
        """Using Sobel to compute gradient. Return the magnitude."""
        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, we expect NCHW, but it is ",
                             input.shape)

        n, c, h, w = input.shape
        
        self.kernel_x = self.kernel_x.to(input.device)
        self.kernel_y = self.kernel_y.to(input.device)

        input_pad = input.reshape(n * c, 1, h, w)
        input_pad = F.pad(input_pad, pad=[1, 1, 1, 1], mode='replicate')
        grad_x = F.conv2d(input_pad, self.kernel_x, padding=0)
        grad_y = F.conv2d(input_pad, self.kernel_y, padding=0)

        mag = torch.sqrt(grad_x * grad_x + grad_y * grad_y + self.eps)
        mag = mag.reshape(n, c, h, w)

        return mag

    def sobel_kernel(self):
        kernel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0],
                                     [-1.0, 0.0, 1.0]]).float()
        kernel_x = kernel_x / kernel_x.abs().sum()
        kernel_y = kernel_x.permute(1, 0)
        kernel_x = kernel_x.unsqueeze(0).unsqueeze(0)
        kernel_y = kernel_y.unsqueeze(0).unsqueeze(0)
        return kernel_x, kernel_y

def gauss_kernel(size=5, device=torch.device('cpu'), channels=3):
    kernel = torch.tensor([[1., 4., 6., 4., 1],
                           [4., 16., 24., 16., 4.],
                           [6., 24., 36., 24., 6.],
                           [4., 16., 24., 16., 4.],
                           [1., 4., 6., 4., 1.]])
    kernel /= 256.
    kernel = kernel.repeat(channels, 1, 1, 1)
    kernel = kernel.to(device)
    return kernel

def downsample(x):
    return x[:, :, ::2, ::2]

def upsample(x):
    cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[2]*2, x.shape[3])
    cc = cc.permute(0,1,3,2)
    cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3]*2, device=x.device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[2]*2, x.shape[3]*2)
    x_up = cc.permute(0,1,3,2)
    return conv_gauss(x_up, 4*gauss_kernel(channels=x.shape[1], device=x.device))

def conv_gauss(img, kernel):
    img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
    out = torch.nn.functional.conv2d(img, kernel.to(img.device), groups=img.shape[1])
    return out

def laplacian_pyramid(img, kernel, max_levels=3):
    current = img
    pyr = []
    for level in range(max_levels):
        filtered = conv_gauss(current, kernel)
        down = downsample(filtered)
        up = upsample(down)
        diff = current-up
        pyr.append(diff)
        current = down
    return pyr

def weight_pyramid(x, max_levels=3):
    current = x
    pyr = []
    for level in range(max_levels):
        down = downsample(current)
        pyr.append(current)
        current = down
    return pyr

class LapLoss(torch.nn.Module):
    def __init__(self, max_levels=3, channels=3):
        super(LapLoss, self).__init__()
        self.max_levels = max_levels
        self.gauss_kernel = gauss_kernel(channels=channels)

    def build_pyramid(self, image):
        """Build a Laplacian pyramid for reuse across loss evaluations."""
        return laplacian_pyramid(
            img=image,
            kernel=self.gauss_kernel,
            max_levels=self.max_levels)

    def l1_loss(self, input, target, weight=None):
        if weight is None:
            return F.l1_loss(input, target)
        else:
            return (F.l1_loss(input, target, reduction='none') * weight).sum() / (weight.sum() + 1e-6)
        
    def forward(self, input, target, weight=None, target_pyramid=None):
        pyr_input = self.build_pyramid(input)
        pyr_target = (
            self.build_pyramid(target)
            if target_pyramid is None else target_pyramid)
        if len(pyr_target) != len(pyr_input):
            raise ValueError(
                "target_pyramid must contain the same number of levels as "
                f"the input pyramid ({len(pyr_input)}), got "
                f"{len(pyr_target)}")
        weights = [None] * len(pyr_input)
        if weight is not None:
            weights = weight_pyramid(weight, max_levels=self.max_levels)
        
        total_loss = 0
        for i in range(len(pyr_input)):
            total_loss += self.l1_loss(pyr_input[i], pyr_target[i], weights[i])
        return total_loss

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self, yhat, y ):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss


class SSIMLoss(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            self.window = window
            self.channel = channel
        return 1 - (1 + _ssim(img1, img2, window, self.window_size, channel, self.size_average)) / 2


def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def SSIM(x, y):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = nn.AvgPool2d(3, 1, 1)(x)
    mu_y = nn.AvgPool2d(3, 1, 1)(y)
    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)

    sigma_x = nn.AvgPool2d(3, 1, 1)(x * x) - mu_x_sq
    sigma_y = nn.AvgPool2d(3, 1, 1)(y * y) - mu_y_sq
    sigma_xy = nn.AvgPool2d(3, 1, 1)(x * y) - mu_x_mu_y

    SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
    SSIM = SSIM_n / SSIM_d

    return torch.clamp((1 - SSIM) / 2, 0, 1)
