import gc
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import skimage.measure
from scipy.ndimage import convolve, distance_transform_edt
from skimage.morphology import skeletonize
from .dist import synchronize, gather
from multiprocessing import Pool
from joblib import Parallel, delayed

def reshape2D(x):
    return x.reshape(-1, *x.shape[-2:])

class Metric(object):
    higher_is_better = False

    def __init__(self):
        self.reset()
    
    def reset(self):
        self.score = 0
        self.count = 0

    # def reshape(self, pred, gt):
    #     gt_shape = gt.shape
    #     if len(pred.shape) > 4:
            
    #     if sum(pred.shape) != sum(gt.shape):
    #         pred = cv2.resize(pred, (gt.shape[-2], gt.shape[-1]), interpolation=cv2.INTER_LINEAR)
    #     return pred

    def compute_metric(self, pred, gt, **kargs):
        raise NotImplementedError
    
    def gather_metric(self, rank=0):
        synchronize()
        gather_score = gather(self.score, dst=rank)
        gather_score = sum(gather_score)
        gather_count = gather(self.count, dst=rank)
        gather_count = sum(gather_count)
        self.score = gather_score
        self.count = gather_count


    def update(self, pred, gt, trimap=None, **kargs):
        
        mask = None
        if trimap is not None:
            mask = (trimap > 0).astype('float32')
        else:
            mask = np.ones_like(gt).astype('float32')

        pred = reshape2D(pred)
        gt = reshape2D(gt)
        mask = reshape2D(mask)

        # pred, gt = self.reshape(pred, gt)
        score, count = self.compute_metric(pred, gt, mask, **kargs)
        # import pdb; pdb.set_trace()
        # self.count += count
        # self.score += score
        self.count += count
        self.score += score
        return score * 1.0 / count

    def average(self):
        return self.score / (self.count + 1e-6)


_BINARY_EPS = np.spacing(1)


def _prepare_binary_data(pred, gt):
    """Convert a prediction/target pair to [0, 1] and bool respectively."""
    pred = pred.astype(np.float64)
    gt = gt.astype(np.float64)
    if pred.size and pred.max() > 1:
        pred = pred / 255.0
    pred = np.clip(pred, 0, 1)
    threshold = 0.5 if not gt.size or gt.max() <= 1 else 128
    return pred, gt > threshold


def _adaptive_threshold(pred):
    return min(2 * pred.mean(), 1)


class BinaryMetric(Metric):
    """Base class for per-image binary segmentation metrics."""

    def compute_binary(self, pred, gt, valid):
        raise NotImplementedError

    def compute_metric(self, pred, gt, mask, **kargs):
        scores = []
        for pred_i, gt_i, mask_i in zip(pred, gt, mask):
            pred_i, gt_i = _prepare_binary_data(pred_i, gt_i)
            valid = mask_i > 0
            score = (self.compute_binary(pred_i, gt_i, valid)
                     if np.any(valid) else 0.0)
            scores.append(score)
        return float(np.sum(scores)), len(scores)


def _binary_confusion(pred, gt, valid, threshold=0.5):
    """Return TP, TN, FP, and FN counts inside the valid region."""
    pred = pred >= threshold
    gt = gt.astype(bool)
    tp = np.count_nonzero(pred & gt & valid)
    tn = np.count_nonzero(~pred & ~gt & valid)
    fp = np.count_nonzero(pred & ~gt & valid)
    fn = np.count_nonzero(~pred & gt & valid)
    return tp, tn, fp, fn


class IoU(BinaryMetric):
    higher_is_better = True

    def compute_binary(self, pred, gt, valid):
        pred = pred >= 0.5
        intersection = np.count_nonzero(pred & gt & valid)
        union = np.count_nonzero((pred | gt) & valid)
        return intersection / union if union else 1.0


class Dice(BinaryMetric):
    higher_is_better = True

    def compute_binary(self, pred, gt, valid):
        pred = pred >= 0.5
        intersection = np.count_nonzero(pred & gt & valid)
        denominator = np.count_nonzero(pred & valid) + np.count_nonzero(gt & valid)
        return 2 * intersection / denominator if denominator else 1.0


class Precision(BinaryMetric):
    higher_is_better = True

    def compute_binary(self, pred, gt, valid):
        pred = pred >= 0.5
        predicted = np.count_nonzero(pred & valid)
        true_positive = np.count_nonzero(pred & gt & valid)
        if predicted:
            return true_positive / predicted
        return 1.0 if not np.any(gt & valid) else 0.0


class Recall(BinaryMetric):
    higher_is_better = True

    def compute_binary(self, pred, gt, valid):
        target = np.count_nonzero(gt & valid)
        true_positive = np.count_nonzero((pred >= 0.5) & gt & valid)
        return true_positive / target if target else 1.0


class F1Score(Dice):
    """Foreground F1 at threshold 0.5; mathematically equal to Dice."""


class PixelAccuracy(BinaryMetric):
    """Fraction of correctly classified foreground and background pixels."""

    higher_is_better = True

    def compute_binary(self, pred, gt, valid):
        tp, tn, fp, fn = _binary_confusion(pred, gt, valid)
        total = tp + tn + fp + fn
        return (tp + tn) / total if total else 1.0


class Specificity(BinaryMetric):
    """True-negative rate, measuring rejection of background pixels."""

    higher_is_better = True

    def compute_binary(self, pred, gt, valid):
        _, tn, fp, _ = _binary_confusion(pred, gt, valid)
        negatives = tn + fp
        return tn / negatives if negatives else 1.0


class BalancedAccuracy(BinaryMetric):
    """Mean of foreground recall and background specificity."""

    higher_is_better = True

    def compute_binary(self, pred, gt, valid):
        tp, tn, fp, fn = _binary_confusion(pred, gt, valid)
        positives = tp + fn
        negatives = tn + fp
        recall = tp / positives if positives else 1.0
        specificity = tn / negatives if negatives else 1.0
        return 0.5 * (recall + specificity)


class BalancedErrorRate(BalancedAccuracy):
    """Balanced error rate: 1 - BalancedAccuracy; lower is better."""

    higher_is_better = False

    def compute_binary(self, pred, gt, valid):
        return 1.0 - super().compute_binary(pred, gt, valid)


class MeanIoU(BinaryMetric):
    """Mean IoU of foreground and background classes at threshold 0.5."""

    higher_is_better = True

    def compute_binary(self, pred, gt, valid):
        tp, tn, fp, fn = _binary_confusion(pred, gt, valid)
        foreground_union = tp + fp + fn
        background_union = tn + fp + fn
        foreground_iou = tp / foreground_union if foreground_union else 1.0
        background_iou = tn / background_union if background_union else 1.0
        return 0.5 * (foreground_iou + background_iou)


# Common names used by segmentation evaluation tools and config files.
Accuracy = PixelAccuracy
F1 = F1Score
BER = BalancedErrorRate
mIoU = MeanIoU


class BinaryMAE(BinaryMetric):
    def compute_binary(self, pred, gt, valid):
        if not np.any(valid):
            return 0.0
        return np.mean(np.abs(pred[valid] - gt[valid]))


class BinaryMSE(BinaryMetric):
    def compute_binary(self, pred, gt, valid):
        if not np.any(valid):
            return 0.0
        return np.mean((pred[valid] - gt[valid]) ** 2)


# Names used by common binary-segmentation evaluation scripts.
MAE = BinaryMAE
SegMSE = BinaryMSE


def _f_measure_curve(pred, gt, beta=0.3):
    pred = (pred * 255).astype(np.uint8)
    bins = np.linspace(0, 256, 257)
    fg_hist, _ = np.histogram(pred[gt], bins=bins)
    bg_hist, _ = np.histogram(pred[~gt], bins=bins)
    true_positives = np.cumsum(np.flip(fg_hist))
    positives = true_positives + np.cumsum(np.flip(bg_hist))
    precision = true_positives / np.maximum(positives, 1)
    recall = true_positives / max(np.count_nonzero(gt), 1)
    numerator = (1 + beta) * precision * recall
    denominator = beta * precision + recall
    return np.divide(
        numerator, denominator, out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator != 0)


def _adaptive_f_measure(pred, gt, beta=0.3):
    binary_pred = pred >= _adaptive_threshold(pred)
    intersection = np.count_nonzero(binary_pred & gt)
    if intersection == 0:
        return 0.0
    precision = intersection / max(np.count_nonzero(binary_pred), 1)
    recall = intersection / max(np.count_nonzero(gt), 1)
    return (1 + beta) * precision * recall / (beta * precision + recall)


class FMeasure(BinaryMetric):
    """Maximum F-measure over 256 thresholds (beta=0.3)."""

    higher_is_better = True

    def compute_binary(self, pred, gt, valid):
        return _f_measure_curve(pred[valid], gt[valid]).max()


class MeanFMeasure(FMeasure):
    """Mean F-measure over 256 thresholds."""

    def compute_binary(self, pred, gt, valid):
        return _f_measure_curve(pred[valid], gt[valid]).mean()


class AdaptiveFMeasure(FMeasure):
    def compute_binary(self, pred, gt, valid):
        return _adaptive_f_measure(pred[valid], gt[valid])


class SMeasure(BinaryMetric):
    """Structure measure from the referenced binary-mask evaluator."""

    higher_is_better = True

    def __init__(self, alpha=0.5):
        self.alpha = alpha
        super().__init__()

    @staticmethod
    def _s_object(pred, gt):
        values = pred[gt]
        if values.size == 0:
            return 0.0
        mean = values.mean()
        std = values.std(ddof=1) if values.size > 1 else 0.0
        return 2 * mean / (mean ** 2 + 1 + std + _BINARY_EPS)

    def _object(self, pred, gt):
        foreground = pred * gt
        background = (1 - pred) * (~gt)
        weight = gt.mean()
        return (weight * self._s_object(foreground, gt) +
                (1 - weight) * self._s_object(background, ~gt))

    @staticmethod
    def _ssim(pred, gt):
        if pred.size == 0:
            return 0.0
        pred_mean = pred.mean()
        gt_mean = gt.mean()
        if pred.size == 1:
            return float(np.isclose(pred_mean, gt_mean))
        pred_var = np.sum((pred - pred_mean) ** 2) / (pred.size - 1)
        gt_var = np.sum((gt - gt_mean) ** 2) / (gt.size - 1)
        covariance = np.sum(
            (pred - pred_mean) * (gt - gt_mean)) / (gt.size - 1)
        alpha = 4 * pred_mean * gt_mean * covariance
        beta = ((pred_mean ** 2 + gt_mean ** 2) *
                (pred_var + gt_var))
        if alpha != 0:
            return alpha / (beta + _BINARY_EPS)
        return 1.0 if beta == 0 else 0.0

    def _region(self, pred, gt):
        h, w = gt.shape
        if np.any(gt):
            y, x = np.argwhere(gt).mean(axis=0).round().astype(int)
        else:
            x, y = round(w / 2), round(h / 2)
        x = min(max(x + 1, 1), w)
        y = min(max(y + 1, 1), h)
        regions = (
            (slice(0, y), slice(0, x)),
            (slice(0, y), slice(x, w)),
            (slice(y, h), slice(0, x)),
            (slice(y, h), slice(x, w)))
        weights = (
            x * y / (h * w), y * (w - x) / (h * w),
            (h - y) * x / (h * w), (h - y) * (w - x) / (h * w))
        return sum(
            weight * self._ssim(pred[region], gt[region])
            for region, weight in zip(regions, weights))

    def compute_binary(self, pred, gt, valid):
        pred = np.where(valid, pred, 0)
        gt = gt & valid
        foreground_ratio = gt.mean()
        if foreground_ratio == 0:
            return 1 - pred.mean()
        if foreground_ratio == 1:
            return pred.mean()
        score = (self.alpha * self._object(pred, gt) +
                 (1 - self.alpha) * self._region(pred, gt))
        return max(0.0, score)


def _e_measure_parts(fg_fg, fg_bg, pred_fg, pred_bg, gt_fg, size):
    bg_fg = gt_fg - fg_fg
    bg_bg = pred_bg - bg_fg
    parts = (fg_fg, fg_bg, bg_fg, bg_bg)
    mean_pred = pred_fg / size
    mean_gt = gt_fg / size
    combinations = (
        (1 - mean_pred, 1 - mean_gt),
        (1 - mean_pred, -mean_gt),
        (-mean_pred, 1 - mean_gt),
        (-mean_pred, -mean_gt))
    result = 0
    for part, (pred_value, gt_value) in zip(parts, combinations):
        alignment = (2 * pred_value * gt_value /
                     (pred_value ** 2 + gt_value ** 2 + _BINARY_EPS))
        result = result + ((alignment + 1) ** 2 / 4) * part
    return result


def _e_measure_curve(pred, gt):
    pred = (pred * 255).astype(np.uint8)
    bins = np.linspace(0, 256, 257)
    fg_hist, _ = np.histogram(pred[gt], bins=bins)
    bg_hist, _ = np.histogram(pred[~gt], bins=bins)
    fg_fg = np.cumsum(np.flip(fg_hist))
    fg_bg = np.cumsum(np.flip(bg_hist))
    pred_fg = fg_fg + fg_bg
    size = gt.size
    gt_fg = np.count_nonzero(gt)
    pred_bg = size - pred_fg
    if gt_fg == 0:
        enhanced_sum = pred_bg
    elif gt_fg == size:
        enhanced_sum = pred_fg
    else:
        enhanced_sum = _e_measure_parts(
            fg_fg, fg_bg, pred_fg, pred_bg, gt_fg, size)
    return np.clip(enhanced_sum / max(size - 1, _BINARY_EPS), 0, 1)


def _adaptive_e_measure(pred, gt):
    binary_pred = pred >= _adaptive_threshold(pred)
    fg_fg = np.count_nonzero(binary_pred & gt)
    fg_bg = np.count_nonzero(binary_pred & ~gt)
    pred_fg = fg_fg + fg_bg
    size = gt.size
    gt_fg = np.count_nonzero(gt)
    pred_bg = size - pred_fg
    if gt_fg == 0:
        enhanced_sum = pred_bg
    elif gt_fg == size:
        enhanced_sum = pred_fg
    else:
        enhanced_sum = _e_measure_parts(
            fg_fg, fg_bg, pred_fg, pred_bg, gt_fg, size)
    return float(np.clip(
        enhanced_sum / max(size - 1, _BINARY_EPS), 0, 1))


class EMeasure(BinaryMetric):
    """Mean enhanced-alignment measure over 256 thresholds."""

    higher_is_better = True

    def compute_binary(self, pred, gt, valid):
        return _e_measure_curve(pred[valid], gt[valid]).mean()


class MaxEMeasure(EMeasure):
    def compute_binary(self, pred, gt, valid):
        return _e_measure_curve(pred[valid], gt[valid]).max()


class AdaptiveEMeasure(EMeasure):
    def compute_binary(self, pred, gt, valid):
        return _adaptive_e_measure(pred[valid], gt[valid])


class WeightedFMeasure(BinaryMetric):
    higher_is_better = True

    @staticmethod
    def _gaussian_kernel(shape=(7, 7), sigma=5):
        m, n = [(size - 1) / 2 for size in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        kernel = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        kernel[kernel < np.finfo(kernel.dtype).eps * kernel.max()] = 0
        if kernel.sum() != 0:
            kernel /= kernel.sum()
        return kernel

    def compute_binary(self, pred, gt, valid):
        pred = np.where(valid, pred, 0)
        gt = gt & valid
        if not np.any(gt):
            return 0.0
        distance, indices = distance_transform_edt(
            ~gt, return_indices=True)
        error = np.abs(pred - gt)
        propagated_error = error.copy()
        propagated_error[~gt] = error[
            indices[0][~gt], indices[1][~gt]]
        smoothed_error = convolve(
            propagated_error, self._gaussian_kernel(),
            mode='constant', cval=0)
        minimum_error = np.where(
            gt & (smoothed_error < error), smoothed_error, error)
        importance = np.where(
            ~gt, 2 - np.exp(np.log(0.5) / 5 * distance), 1)
        weighted_error = minimum_error * importance
        true_positive = gt.sum() - weighted_error[gt].sum()
        false_positive = weighted_error[~gt].sum()
        recall = 1 - weighted_error[gt].mean()
        precision = true_positive / (
            true_positive + false_positive + _BINARY_EPS)
        return (2 * recall * precision /
                (recall + precision + _BINARY_EPS))


def _mask_to_boundary(mask, dilation_ratio=0.02):
    h, w = mask.shape
    dilation = max(1, int(round(dilation_ratio * np.sqrt(h ** 2 + w ** 2))))
    padded = cv2.copyMakeBorder(
        mask.astype(np.uint8), 1, 1, 1, 1,
        cv2.BORDER_CONSTANT, value=0)
    eroded = cv2.erode(
        padded, np.ones((3, 3), dtype=np.uint8), iterations=dilation)
    return mask.astype(np.uint8) - eroded[1:h + 1, 1:w + 1]


def _boundary_iou_curve(pred, gt):
    gt_boundary = _mask_to_boundary(gt) > 0
    pred_boundary = _mask_to_boundary((pred * 255).astype(np.uint8))
    bins = np.linspace(0, 256, 257)
    fg_hist, _ = np.histogram(pred_boundary[gt_boundary], bins=bins)
    bg_hist, _ = np.histogram(pred_boundary[~gt_boundary], bins=bins)
    true_positive = np.cumsum(np.flip(fg_hist))
    false_positive = np.cumsum(np.flip(bg_hist))
    target = np.count_nonzero(gt_boundary)
    denominator = target + false_positive
    return np.divide(
        true_positive, denominator,
        out=np.ones_like(true_positive, dtype=np.float64),
        where=denominator != 0)


class BIoU(BinaryMetric):
    """Maximum boundary IoU over 256 thresholds."""

    higher_is_better = True

    def compute_binary(self, pred, gt, valid):
        pred = np.where(valid, pred, 0)
        gt = gt & valid
        return _boundary_iou_curve(pred, gt).max()


class MeanBIoU(BIoU):
    def compute_binary(self, pred, gt, valid):
        pred = np.where(valid, pred, 0)
        gt = gt & valid
        return _boundary_iou_curve(pred, gt).mean()


class MBA(BinaryMetric):
    """Mean boundary accuracy over five boundary widths."""

    higher_is_better = True

    def compute_binary(self, pred, gt, valid):
        pred = (pred > 128.0 / 255.0) & valid
        gt = gt & valid
        h, w = gt.shape
        accuracies = []
        max_radius = (w + h) / 300
        for index in range(5):
            radius = 1 + int((max_radius - 1) / 5 * index)
            radius = max(radius, 1)
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
            boundary = cv2.morphologyEx(
                gt.astype(np.uint8), cv2.MORPH_GRADIENT, kernel) > 0
            boundary &= valid
            if not np.any(boundary):
                accuracies.append(float(np.array_equal(pred, gt)))
            else:
                accuracies.append(np.mean(pred[boundary] == gt[boundary]))
        return np.mean(accuracies)


class HCE(BinaryMetric):
    """Human correction effort; lower is better."""

    @staticmethod
    def _filter_boundary_condition(boundaries, mask, condition):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        condition = cv2.dilate(condition.astype(np.uint8), kernel)
        labels = skimage.measure.label(mask)
        independent = np.ones(len(np.unique(labels)))
        independent[0] = 0
        index_map = np.zeros_like(condition)
        selected_boundaries = []
        for boundary in boundaries:
            pieces = []
            piece = []
            for point in boundary[:, 0]:
                column, row = point
                if condition[row, column] == 0 or index_map[row, column] != 0:
                    if piece:
                        pieces.append(piece)
                        piece = []
                    continue
                piece.append([column, row])
                index_map[row, column] += 1
                independent[labels[row, column]] = 0
            if piece:
                pieces.append(piece)
            if len(pieces) > 1:
                first_x, first_y = pieces[0][0]
                last_x, last_y = pieces[-1][-1]
                if abs(first_x - last_x) <= 1 and abs(first_y - last_y) <= 1:
                    pieces[-1].extend(pieces[0][::-1])
                    del pieces[0]
            selected_boundaries.extend(
                np.asarray(piece)[:, None, :] for piece in pieces if piece)
        return selected_boundaries, independent.sum()

    @staticmethod
    def _polygon_points(boundaries, epsilon=2.0):
        return sum(
            len(cv2.approxPolyDP(boundary, epsilon, False))
            for boundary in boundaries)

    def compute_binary(self, pred, gt, valid):
        pred = (pred > 128.0 / 255.0) & valid
        gt = gt & valid
        gt_skeleton = skeletonize(gt)
        union = pred | gt
        true_positive = pred & gt
        false_positive = pred & ~gt
        false_negative = gt & ~pred
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        relaxed_union = cv2.erode(
            union.astype(np.uint8), kernel, iterations=5) > 0

        relaxed_fp = false_positive & relaxed_union
        for _ in range(5):
            relaxed_fp = cv2.dilate(relaxed_fp.astype(np.uint8), kernel) > 0
            relaxed_fp &= ~(true_positive | false_negative)
        relaxed_fp &= false_positive

        relaxed_fn = false_negative & relaxed_union
        for _ in range(5):
            relaxed_fn = cv2.dilate(relaxed_fn.astype(np.uint8), kernel) > 0
            relaxed_fn &= ~(true_positive | false_positive)
        relaxed_fn &= false_negative
        relaxed_fn |= gt_skeleton & ~true_positive

        fp_contours, _ = cv2.findContours(
            relaxed_fp.astype(np.uint8), cv2.RETR_TREE,
            cv2.CHAIN_APPROX_NONE)
        fn_contours, _ = cv2.findContours(
            relaxed_fn.astype(np.uint8), cv2.RETR_TREE,
            cv2.CHAIN_APPROX_NONE)
        fp_boundaries, fp_independent = self._filter_boundary_condition(
            fp_contours, relaxed_fp, true_positive | relaxed_fn)
        fn_boundaries, fn_independent = self._filter_boundary_condition(
            fn_contours, relaxed_fn,
            ~(true_positive | relaxed_fp | relaxed_fn))
        return (self._polygon_points(fp_boundaries) + fp_independent +
                self._polygon_points(fn_boundaries) + fn_independent)


# Short names used by the referenced evaluator.
S = SMeasure
F = FMeasure
E = EMeasure
WF = WeightedFMeasure

class SAD(Metric):
    
    def compute_metric(self, pred, gt, mask, **kargs):
        '''
        pred, gt: numpy array
        (N, *, H, W)
        '''
        # return np.sum(np.abs(pred - gt) * mask) * 0.001, mask.shape[0]
        diff = np.abs(pred - gt) * mask
        sad = np.sum(diff, axis=(1, 2))
        return sad.sum() * 1e-3, mask.shape[0]

class MSE(Metric):
    
    def compute_metric(self, pred, gt, mask, **kargs):
        '''
        pred, gt: numpy array
        (N, *, H, W)
        '''
        # return np.sum(((pred - gt) ** 2) * mask) * 1000, mask.sum()
        diff = ((pred - gt) ** 2) * mask
        mse = np.mean(diff, axis=(1, 2)) / (mask.sum(axis=(1, 2)) + 1e-6)
        return mse.sum() * 1e10, mask.shape[0]

class MAD(Metric):

    def compute_metric(self, pred, gt, mask, **kargs):
        # return np.sum(np.abs(pred - gt) * mask) * 1000, mask.sum()
        diff = np.abs(pred - gt) * mask
        mad = np.mean(diff, axis=(1, 2)) / (mask.sum(axis=(1, 2)) + 1e-6)
        return mad.sum() * 1e10, mask.shape[0]

# class MAD_fg(Metric):
#     def compute_metric(self, pred, gt, mask, **kargs):
#         mask = (mask == 2).float()
#         return np.sum(np.abs(pred - gt) * mask) * 1000, mask.sum()

# class MAD_bg(Metric):
#     def compute_metric(self, pred, gt, mask, **kargs):
#         mask = (mask == 0).float()
#         return np.sum(np.abs(pred - gt) * mask) * 1000, mask.sum()

# class MAD_unk(Metric):
#     def compute_metric(self, pred, gt, mask, **kargs):
#         mask = (mask == 1).float()
#         return np.sum(np.abs(pred - gt) * mask) * 1000, mask.sum()

# class Conn(Metric):
    
#     def compute_metric(self, pred, gt, mask, **kargs):
#         conn_err = 0
#         B = pred.shape[0]
#         # mask = np.ones_like(mask)
#         pool = Pool(B)
#         for err in pool.imap(self.compute_conn, zip(pred, gt, mask)):
#             conn_err += err
#         # for i in range(pred.shape[0]):
#         #     conn_err += self.compute_conn((pred[i], gt[i], mask[i]))
#         pool.close()
#         # import pdb; pdb.set_trace()
#         return conn_err, B

#     def compute_conn(self, args):
#         """
#         update metric.
#         Args:
#             pred (np.ndarray): The value range is [0., 1.].
#             gt (np.ndarray): The value range is [0, 1].
#             step (float, optional): Step of threshold when computing intersection between
#             `gt` and `pred`. Default: 0.1.
#         """
#         pred, gt, roi_mask = args
#         step=0.1
#         thresh_steps = np.arange(0, 1 + step, step)
#         round_down_map = -np.ones_like(gt)
#         for i in range(1, len(thresh_steps)):
#             gt_thresh = gt >= thresh_steps[i]
#             pred_thresh = pred >= thresh_steps[i]
#             intersection = (gt_thresh & pred_thresh).astype(np.uint8)

#             # connected components
#             _, output, stats, _ = cv2.connectedComponentsWithStats(intersection, connectivity=4)
#             # start from 1 in dim 0 to exclude background
#             size = stats[1:, -1]

#             # largest connected component of the intersection
#             omega = np.zeros_like(gt)
#             if len(size) != 0:
#                 max_id = np.argmax(size)
#                 # plus one to include background
#                 omega[output == max_id + 1] = 1

#             mask = (round_down_map == -1) & (omega == 0)
#             round_down_map[mask] = thresh_steps[i - 1]
#         round_down_map[round_down_map == -1] = 1

#         gt_diff = gt - round_down_map
#         pred_diff = pred - round_down_map
#         # only calculate difference larger than or equal to 0.15
#         gt_phi = 1 - gt_diff * (gt_diff >= 0.15)
#         pred_phi = 1 - pred_diff * (pred_diff >= 0.15)
#         conn_diff = np.sum(np.abs(gt_phi - pred_phi) * roi_mask)
#         return conn_diff

# class Conn(Metric):

#     def compute_metric(self, pred, gt, mask, **kargs):
#         conn_err = 0
#         B = pred.shape[0]
#         # mask = np.ones_like(mask)
#         pool = Pool(B)
#         for err in pool.imap(self.compute_conn, zip(pred, gt, mask)):
#             conn_err += err * 0.001
#         # for i in range(pred.shape[0]):
#         #     conn_err += self.compute_conn((pred[i], gt[i], mask[i]))
#         pool.close()
#         # import pdb; pdb.set_trace()
#         return conn_err, B

#     def compute_conn(self, args):
#         """
#         update metric.
#         Args:
#             pred (np.ndarray): The value range is [0., 1.].
#             gt (np.ndarray): The value range is [0, 1].
#             step (float, optional): Step of threshold when computing intersection between
#             `gt` and `pred`. Default: 0.1.
#         """
#         pred, gt, roi_mask = args
#         step=0.1
#         thresh_steps = np.arange(0, 1 + step, step)
#         round_down_map = -np.ones_like(gt)
#         for i in range(1, len(thresh_steps)):
#             gt_thresh = gt >= thresh_steps[i]
#             pred_thresh = pred >= thresh_steps[i]
#             intersection = (gt_thresh & pred_thresh).astype(np.uint8)

#             cc, num = skimage.measure.label(intersection, connectivity=1, return_num=True)
#             omega = np.zeros_like(intersection)
#             if num > 0:
#                 # find the largest connected region
#                 max_id = np.argmax(np.bincount(cc.flatten())[1:]) + 1
#                 omega[cc == max_id] = 1

#             mask = (round_down_map == -1) & (omega == 0)
#             round_down_map[mask] = thresh_steps[i - 1]
#         round_down_map[round_down_map == -1] = 1

#         gt_diff = gt - round_down_map
#         pred_diff = pred - round_down_map
#         # only calculate difference larger than or equal to 0.15
#         gt_phi = 1 - gt_diff * (gt_diff >= 0.15)
#         pred_phi = 1 - pred_diff * (pred_diff >= 0.15)
#         conn_diff = np.sum(np.abs(gt_phi - pred_phi) * roi_mask)
#         return conn_diff

class Conn(Metric):

    def compute_metric(self, pred, gt, mask, **kargs):
        conn_err = self.compute_conn(pred, gt, mask) * 0.001
        B = pred.shape[0]
        return conn_err, B

    @staticmethod
    def compute_largest_connected_component(intersection):
        cc, num = skimage.measure.label(intersection, connectivity=1, return_num=True)
        omega = np.zeros_like(intersection)
        if num > 0:
            max_id = np.argmax(np.bincount(cc.flatten())[1:]) + 1
            omega[cc == max_id] = 1
        return omega

    def compute_conn(self, pred, gt, roi_mask):
        """
        update metric.
        Args:
            pred (np.ndarray): The value range is [0., 1.].
            gt (np.ndarray): The value range is [0, 1].
            step (float, optional): Step of threshold when computing intersection between
            `gt` and `pred`. Default: 0.1.
        """
        step=0.1
        B = pred.shape[0]
        thresh_steps = np.arange(0, 1 + step, step)
        round_down_map = -np.ones_like(gt)
        all_intersections = []
        for b in range(B):
            for i in range(1, len(thresh_steps)):
                gt_thresh = gt[b] >= thresh_steps[i]
                pred_thresh = pred[b] >= thresh_steps[i]
                intersection = (gt_thresh & pred_thresh).astype(np.uint8)
                all_intersections.append(intersection)

        
        # with Pool(4) as p:
        #     all_omegas = p.map(self.compute_largest_connected_component, all_intersections)
        all_omegas = Parallel(n_jobs=min(10, len(all_intersections)))(delayed(self.compute_largest_connected_component)(intersection) for intersection in all_intersections)

        j = 0
        for b in range(B):
            for i in range(1, len(thresh_steps)):
                omega = all_omegas[j]
                j += 1
                mask = (round_down_map[b] == -1) & (omega == 0)
                round_down_map[b][mask] = thresh_steps[i - 1]
        
        round_down_map[round_down_map == -1] = 1

        gt_diff = gt - round_down_map
        pred_diff = pred - round_down_map
        # only calculate difference larger than or equal to 0.15
        gt_phi = 1 - gt_diff * (gt_diff >= 0.15)
        pred_phi = 1 - pred_diff * (pred_diff >= 0.15)
        # import pdb; pdb.set_trace()
        conn_diff = np.sum(np.abs(gt_phi - pred_phi) * roi_mask)
        # if conn_diff > 30000:
        #     cv2.imwrite("mask.png", gt_phi[0] * 255)
        #     cv2.imwrite("pred.png", pred_phi[0] * 255)
        #     import pdb; pdb.set_trace()
        del all_omegas, all_intersections, round_down_map, gt_diff, pred_diff, gt_phi, pred_phi
        gc.collect()
        return conn_diff

# class Grad(Metric):
    
#     def gaussian(self, x, sigma):
#         return np.exp(-x**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))

#     def dgaussian(self, x, sigma):
#         return -x * self.gaussian(x, sigma) / sigma**2

#     def gauss_filter(self, sigma, epsilon=1e-2):
#         half_size = np.ceil(
#             sigma * np.sqrt(-2 * np.log(np.sqrt(2 * np.pi) * sigma * epsilon)))
#         size = int(2 * half_size + 1)

#         # create filter in x axis
#         filter_x = np.zeros((size, size))
#         for i in range(size):
#             for j in range(size):
#                 filter_x[i, j] = self.gaussian(
#                     i - half_size, sigma) * self.dgaussian(j - half_size, sigma)

#         # normalize filter
#         norm = np.sqrt((filter_x**2).sum())
#         filter_x = filter_x / norm
#         filter_y = np.transpose(filter_x)

#         return filter_x, filter_y

#     def gauss_gradient(self, img, sigma):
#         filter_x, filter_y = self.gauss_filter(sigma)
#         img_filtered_x = cv2.filter2D(
#             img, -1, filter_x, borderType=cv2.BORDER_REPLICATE)
#         img_filtered_y = cv2.filter2D(
#             img, -1, filter_y, borderType=cv2.BORDER_REPLICATE)
#         return np.sqrt(img_filtered_x**2 + img_filtered_y**2)
    
#     def compute_grad(self, args):
#         pred, gt, mask = args
#         sigma=1.4
#         gt = gt.astype(np.float64)
#         pred = pred.astype(np.float64)
#         gt_normed = np.zeros_like(gt)
#         pred_normed = np.zeros_like(pred)
#         cv2.normalize(gt, gt_normed, 1., 0., cv2.NORM_MINMAX)
#         cv2.normalize(pred, pred_normed, 1., 0., cv2.NORM_MINMAX)

#         gt_grad = self.gauss_gradient(gt_normed, sigma).astype(np.float32)
#         pred_grad = self.gauss_gradient(pred_normed, sigma).astype(np.float32)

#         grad_diff = (((gt_grad - pred_grad)**2) * mask).sum()

#         return grad_diff
    
#     def compute_metric(self, pred, gt, mask, **kargs):
#         grad_err = 0
#         B = pred.shape[0]
#         pool = Pool(B)
#         for err in pool.imap(self.compute_grad, zip(pred, gt, mask)):
#             grad_err += err * 0.001
#         pool.close()
#         return grad_err, B

class Grad(Metric):
    def __init__(self):
        super().__init__()
        sigma = 1.4
        self.filter_x, self.filter_y = self.gauss_filter(sigma)
        
        # Convert filters to PyTorch tensors and move to GPU
        self.filter_x = torch.tensor(self.filter_x, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.filter_y = torch.tensor(self.filter_y, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    def gaussian(self, x, sigma):
        return np.exp(-x**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))

    def dgaussian(self, x, sigma):
        return -x * self.gaussian(x, sigma) / sigma**2

    def gauss_filter(self, sigma, epsilon=1e-2):
        half_size = np.ceil(
            sigma * np.sqrt(-2 * np.log(np.sqrt(2 * np.pi) * sigma * epsilon)))
        size = int(2 * half_size + 1)

        # create filter in x axis
        filter_x = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                filter_x[i, j] = self.gaussian(
                    i - half_size, sigma) * self.dgaussian(j - half_size, sigma)

        # normalize filter
        norm = np.sqrt((filter_x**2).sum())
        filter_x = filter_x / norm
        filter_y = np.transpose(filter_x)

        return filter_x, filter_y

    def gauss_gradient(self, img):
        img_filtered_x = F.conv2d(img, self.filter_x, padding=self.filter_x.shape[-1]//2)
        img_filtered_y = F.conv2d(img, self.filter_y, padding=self.filter_y.shape[-1]//2)
        return torch.sqrt(img_filtered_x**2 + img_filtered_y**2)
    
    def compute_grad(self, pred, gt, mask):
        gt = gt.float().unsqueeze(1) # B x 1 x H x W
        pred = pred.float().unsqueeze(1)#.cuda()
        mask = mask.float().unsqueeze(1)#.cuda()

        gt_normed = (gt - gt.min()) / (gt.max() - gt.min() + 1e-6)
        pred_normed = (pred - pred.min()) / (pred.max() - pred.min() + 1e-6)

        gt_grad = self.gauss_gradient(gt_normed)
        pred_grad = self.gauss_gradient(pred_normed)

        grad_diff = (((gt_grad - pred_grad)**2) * mask).sum().item()
        del gt_grad, pred_grad, gt_normed, pred_normed
        torch.cuda.empty_cache()
        gc.collect()
        return grad_diff
    
    def compute_metric(self, pred, gt, mask, device='cuda', **kargs):
        
        pred = torch.from_numpy(pred).to(device)
        gt = torch.from_numpy(gt).to(device)
        mask = torch.from_numpy(mask).to(device)

        self.filter_x = self.filter_x.to(device)
        self.filter_y = self.filter_y.to(device)

        grad_err = self.compute_grad(pred, gt, mask) * 0.001
        B = pred.shape[0]
        return grad_err, B

class dtSSD(Metric):

    def update(self, pred, gt, trimap=None, **kargs):
        mask = None
        if trimap is not None:
            mask = (trimap == 1).astype('float32')
        else:
            mask = np.ones_like(gt).astype('float32')

        if pred.ndim == 4:
            pred = pred[None]
            gt = gt[None]
            mask = mask[None]
        dadt = pred[:, 1:] - pred[:, :-1]
        dgdt = gt[:, 1:] - gt[:, :-1]
        mask_0 = mask[:, :-1]
        err_m = (dadt - dgdt) ** 2
        err_m = err_m * mask_0
        err = np.sqrt(np.sum(err_m, axis=(0, 1, 3, 4)))
        err = np.sum(err) * 0.1
        num = mask_0.shape[2] #mask_0.sum()

        # dtSSD for each instance in each video

        self.score += err
        self.count += num
        return err / (num + 1e-10)
    
class MESSDdt(Metric):
    def calcOpticalFlow(self, frames):
        prev, curr = frames
        flow = cv2.calcOpticalFlowFarneback(prev.astype(np.uint8), curr.astype(np.uint8), None,  
                                        0.5, 5, 10, 2, 7, 1.5, 
                                        cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        return flow
    
    def compute_single_video(self, pred, gt, mask):
        pred = reshape2D(pred)
        gt = reshape2D(gt)
        
        B, h, w = gt.shape
        pool = Pool(B)
        flows = []
        items = [t for t in (gt * 255)]
        for flow in pool.imap(self.calcOpticalFlow, zip(items[:-1], items[1:])):
            flows.append(flow)
        flow = torch.from_numpy(np.rint(np.array(flows)).astype(np.int64))
        pool.close()

        pred = torch.from_numpy(pred)
        gt = torch.from_numpy(gt)
        mask = torch.from_numpy(mask)
        pred_0 = pred[:-1, ...]
        pred_1 = pred[1:, ...]
        target_0 = gt[:-1, ...]
        target_1 = gt[1:, ...]
        mask_0 = mask[:-1, ...]
        mask_1 = mask[1:, ...]
        
        B, h, w = target_0.shape
        x = torch.arange(0, w)
        y = torch.arange(0, h)
        xx, yy = torch.meshgrid([y, x])
        coords = torch.stack([yy, xx], dim=2).unsqueeze(0).repeat((B, 1, 1, 1))
        coords_n = (coords + flow)
        coords_y = coords_n[..., 0].clamp(0, h-1)
        coords_x = coords_n[..., 1].clamp(0, w-1)
        indices = coords_y * w + coords_x
        pred_1 = torch.take(pred_1, indices)
        target_1 = torch.take(target_1, indices)
        mask_1 = torch.take(mask_1, indices)

        error_map = (pred_0-target_0).pow(2) * mask_0 - (pred_1-target_1).pow(2) * mask_1

        error = error_map.abs().view(mask_0.shape[0], -1).sum(dim=1) # (N_f - 1) x HW
        num = mask_0.view(mask_0.shape[0], -1).sum(dim=1) + 1. # (N_f - 1) x HW
        
        error = error.cpu().numpy().sum() / num.cpu().numpy().sum()
        # num = num.cpu().numpy().sum()
        return error
    
    def update(self, pred, gt, trimap=None, **kargs):
        if pred.ndim == 5:
            pred = pred.squeeze(0)
            gt = gt.squeeze(0)
        mask = None
        if trimap is not None:
            mask = (trimap == 1).astype('float32')
        else:
            mask = np.ones_like(gt).astype('float32')

        error = 0
        count = 0

        # N_F x N_I x H x W


        for i in range(pred.shape[1]):
            try:
                e = self.compute_single_video(pred[:, i], gt[:, i], mask[:, i])
            except Exception as exception:
                print(exception)
                continue
            error += e * 10000
            count += 1
        # all_omegas = Parallel(n_jobs=len(pred))(delayed(self.compute_single_video)(pred[i], gt[i], mask[i]) for intersection in all_intersections)
            
        self.score += error # Sum of error for each instance
        self.count += count # Add number of instances
        return error / (count + 1e-8)
        

def build_metric(metrics):
    '''
    metrics: list of str
    returns:
    dict of metric name and metric class
    '''
    metric_dict = {}
    for metric in metrics:
        # try:
        metric_dict[metric] = eval(metric)()
        # except:
        #     raise NotImplementedError(f'metric {metric} is not implemented')
    return metric_dict
