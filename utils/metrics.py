import numpy as np
from scipy.ndimage import distance_transform_edt, convolve


def _to_numpy(pred, gt):
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    return pred, gt


def mae_score(pred, gt):
    pred, gt = _to_numpy(pred, gt)
    return float(np.mean(np.abs(pred - gt)))


def dice_score(pred, gt, threshold=0.5, eps=1e-8):
    pred, gt = _to_numpy(pred, gt)
    pred_bin = (pred >= threshold).astype(np.float64)
    gt_bin = (gt >= threshold).astype(np.float64)
    intersection = (pred_bin * gt_bin).sum()
    union = pred_bin.sum() + gt_bin.sum()
    return float((2.0 * intersection + eps) / (union + eps))


def iou_score(pred, gt, threshold=0.5, eps=1e-8):
    pred, gt = _to_numpy(pred, gt)
    pred_bin = (pred >= threshold).astype(np.float64)
    gt_bin = (gt >= threshold).astype(np.float64)
    intersection = (pred_bin * gt_bin).sum()
    union = pred_bin.sum() + gt_bin.sum() - intersection
    return float((intersection + eps) / (union + eps))


def weighted_f_measure(pred, gt, beta2=1.0, eps=1e-8):
    pred, gt = _to_numpy(pred, gt)
    gt_bin = gt > 0.5

    if gt_bin.sum() == 0:
        return 0.0

    dist, idx = distance_transform_edt(~gt_bin, return_indices=True)
    error = np.abs(pred - gt)

    error_t = error.copy()
    outside = ~gt_bin
    error_t[outside] = error[idx[0][outside], idx[1][outside]]

    kernel = np.ones((7, 7)) / 49.0
    ea = convolve(error_t, kernel, mode="constant")

    min_e_ea = error.copy()
    replace_mask = gt_bin & (ea < error)
    min_e_ea[replace_mask] = ea[replace_mask]

    b = np.ones_like(gt, dtype=np.float64)
    b[outside] = 2.0 - np.exp(np.log(0.5) / 5.0 * dist[outside])

    ew = min_e_ea * b

    tp_w = gt_bin.sum() - ew[gt_bin].sum()
    fp_w = ew[outside].sum()

    r = 1.0 - ew[gt_bin].mean() if gt_bin.sum() > 0 else 0.0
    p = tp_w / (tp_w + fp_w + eps)
    q = (1 + beta2) * r * p / (r + beta2 * p + eps)
    return float(max(q, 0.0))


def _object_score(pred_region, gt_region):
    mask = gt_region.astype(bool)
    if mask.sum() == 0:
        return 0.0
    x = pred_region[mask].mean()
    sigma = pred_region[mask].std()
    return float(2.0 * x / (x ** 2 + 1.0 + sigma + 1e-8))


def _s_object(pred, gt):
    fg = pred.copy()
    fg[gt == 0] = 0
    bg = 1.0 - pred
    bg[gt == 1] = 0
    u = gt.mean()
    return u * _object_score(fg, gt) + (1 - u) * _object_score(bg, 1 - gt)


def _centroid(gt):
    if gt.sum() == 0:
        h, w = gt.shape
        return h // 2, w // 2
    ys, xs = np.where(gt > 0.5)
    return int(round(ys.mean())), int(round(xs.mean()))


def _divide_regions(mat, cy, cx):
    h, w = mat.shape
    lt = mat[:cy, :cx]
    rt = mat[:cy, cx:]
    lb = mat[cy:, :cx]
    rb = mat[cy:, cx:]
    return lt, rt, lb, rb


def _ssim_region(pred, gt):
    h, w = pred.shape
    n = h * w
    if n == 0:
        return 0.0
    x = pred.mean()
    y = gt.mean()
    sigma_x2 = ((pred - x) ** 2).sum() / (n - 1 + 1e-8)
    sigma_y2 = ((gt - y) ** 2).sum() / (n - 1 + 1e-8)
    sigma_xy = ((pred - x) * (gt - y)).sum() / (n - 1 + 1e-8)

    alpha = 4 * x * y * sigma_xy
    beta = (x ** 2 + y ** 2) * (sigma_x2 + sigma_y2)

    if alpha != 0:
        return float(alpha / (beta + 1e-8))
    elif alpha == 0 and beta == 0:
        return 1.0
    return 0.0


def _s_region(pred, gt):
    cy, cx = _centroid(gt)
    h, w = gt.shape
    cy = max(1, min(cy, h - 1))
    cx = max(1, min(cx, w - 1))

    gt_lt, gt_rt, gt_lb, gt_rb = _divide_regions(gt, cy, cx)
    pred_lt, pred_rt, pred_lb, pred_rb = _divide_regions(pred, cy, cx)

    total = h * w
    w_lt = (gt_lt.shape[0] * gt_lt.shape[1]) / total
    w_rt = (gt_rt.shape[0] * gt_rt.shape[1]) / total
    w_lb = (gt_lb.shape[0] * gt_lb.shape[1]) / total
    w_rb = (gt_rb.shape[0] * gt_rb.shape[1]) / total

    q1 = _ssim_region(pred_lt, gt_lt)
    q2 = _ssim_region(pred_rt, gt_rt)
    q3 = _ssim_region(pred_lb, gt_lb)
    q4 = _ssim_region(pred_rb, gt_rb)

    return w_lt * q1 + w_rt * q2 + w_lb * q3 + w_rb * q4


def s_measure(pred, gt, alpha=0.5):
    pred, gt = _to_numpy(pred, gt)
    gt_bin = (gt > 0.5).astype(np.float64)
    y = gt_bin.mean()

    if y == 0:
        return float(1.0 - pred.mean())
    if y == 1:
        return float(pred.mean())

    score = alpha * _s_object(pred, gt_bin) + (1 - alpha) * _s_region(pred, gt_bin)
    return float(max(score, 0.0))


def _e_measure_at_threshold(pred_bin, gt_bin, eps=1e-8):
    fm_mean = pred_bin.mean()
    gt_mean = gt_bin.mean()
    align_fm = pred_bin - fm_mean
    align_gt = gt_bin - gt_mean
    align_matrix = 2 * align_gt * align_fm / (align_gt ** 2 + align_fm ** 2 + eps)
    enhanced = ((align_matrix + 1) ** 2) / 4.0
    return enhanced.sum() / (gt_bin.size - 1 + eps)


def e_measure_curve(pred, gt, num_thresholds=255):
    pred, gt = _to_numpy(pred, gt)
    gt_bin = (gt > 0.5).astype(np.float64)

    if gt_bin.sum() == 0:
        fm_bin = (pred > 0.5).astype(np.float64)
        return np.array([1.0 - fm_bin.mean()] * num_thresholds)
    if gt_bin.sum() == gt_bin.size:
        fm_bin = (pred > 0.5).astype(np.float64)
        return np.array([fm_bin.mean()] * num_thresholds)

    thresholds = np.linspace(0, 1, num_thresholds)
    scores = np.zeros(num_thresholds)
    for i, th in enumerate(thresholds):
        pred_bin = (pred >= th).astype(np.float64)
        scores[i] = _e_measure_at_threshold(pred_bin, gt_bin)
    return scores


def mean_e_measure(pred, gt, num_thresholds=255):
    curve = e_measure_curve(pred, gt, num_thresholds)
    return float(curve.mean())


def max_e_measure(pred, gt, num_thresholds=255):
    curve = e_measure_curve(pred, gt, num_thresholds)
    return float(curve.max())


class MetricMeter:
    def __init__(self, num_e_thresholds=255):
        self.num_e_thresholds = num_e_thresholds
        self.reset()

    def reset(self):
        self.dice_list = []
        self.iou_list = []
        self.wfm_list = []
        self.sm_list = []
        self.mae_list = []
        self.e_curves = []

    def update(self, pred, gt):
        pred = np.clip(pred, 0.0, 1.0)
        gt = np.clip(gt, 0.0, 1.0)
        self.dice_list.append(dice_score(pred, gt))
        self.iou_list.append(iou_score(pred, gt))
        self.wfm_list.append(weighted_f_measure(pred, gt))
        self.sm_list.append(s_measure(pred, gt))
        self.mae_list.append(mae_score(pred, gt))
        self.e_curves.append(e_measure_curve(pred, gt, self.num_e_thresholds))

    def summary(self):
        e_curves = np.stack(self.e_curves, axis=0) if self.e_curves else np.zeros((1, self.num_e_thresholds))
        mean_curve = e_curves.mean(axis=0)
        return {
            "mDice": float(np.mean(self.dice_list)) if self.dice_list else 0.0,
            "mIoU": float(np.mean(self.iou_list)) if self.iou_list else 0.0,
            "wFmeasure": float(np.mean(self.wfm_list)) if self.wfm_list else 0.0,
            "Smeasure": float(np.mean(self.sm_list)) if self.sm_list else 0.0,
            "meanEmeasure": float(mean_curve.mean()),
            "maxEmeasure": float(mean_curve.max()),
            "MAE": float(np.mean(self.mae_list)) if self.mae_list else 0.0,
        }
