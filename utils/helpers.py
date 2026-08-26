import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def save_checkpoint(state, is_best, checkpoint_dir, model_name):
    os.makedirs(os.path.join(checkpoint_dir, model_name), exist_ok=True)
    torch.save(state, os.path.join(checkpoint_dir, model_name, "last.pth"))
    if is_best:
        torch.save(state, os.path.join(checkpoint_dir, model_name, "best.pth"))

def load_checkpoint(checkpoint_path, model):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])

def save_visualization(image, gt, pred, save_path):
    image = image.permute(1, 2, 0).cpu().numpy()
    gt = gt.squeeze().cpu().numpy()
    pred = (pred.squeeze().cpu().numpy() > 0.5).astype(np.float32)
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image); axes[0].set_title("Image"); axes[0].axis("off")
    axes[1].imshow(gt, cmap="gray"); axes[1].set_title("Ground Truth"); axes[1].axis("off")
    axes[2].imshow(pred, cmap="gray"); axes[2].set_title("Prediction"); axes[2].axis("off")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout(); plt.savefig(save_path); plt.close()
