import numpy as np

def compute_metrics(pred, target):
    pred = (pred > 0.5).astype(np.float32)
    target = (target > 0.5).astype(np.float32)
    inter = np.sum(pred * target)
    union = np.sum(pred) + np.sum(target) - inter
    
    return {
        "dice": float((2.0 * inter + 1e-5) / (np.sum(pred) + np.sum(target) + 1e-5)),
        "iou": float((inter + 1e-5) / (union + 1e-5)),
        "precision": float((inter + 1e-5) / (np.sum(pred) + 1e-5)),
        "recall": float((inter + 1e-5) / (np.sum(target) + 1e-5)),
        "mae": float(np.mean(np.abs(pred - target)))
    }
