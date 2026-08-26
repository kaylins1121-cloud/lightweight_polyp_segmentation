import argparse
import os

import cv2
import numpy as np
import torch

from models import build_model
from utils.dataset import build_multi_test_datasets, IMAGENET_MEAN, IMAGENET_STD
from utils.metrics import MetricMeter
from utils.gradcam import GradCAM, overlay_cam_on_image, denormalize_image
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--test_path", type=str, default="dataset/TestDataset")
    parser.add_argument("--input_size", type=int, default=352)
    parser.add_argument("--out_dir", type=str, default="inference_outputs")
    parser.add_argument("--gradcam", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = build_model(args.model, in_channels=3, num_classes=1).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_datasets = build_multi_test_datasets(args.test_path, image_size=args.input_size)

    gradcam = None
    if args.gradcam:
        target_layer = getattr(model, "gradcam_target_layer", None)
        if target_layer is not None:
            gradcam = GradCAM(model, target_layer)

    os.makedirs(args.out_dir, exist_ok=True)

    for name, dataset in test_datasets.items():
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        meter = MetricMeter()
        sub_dir = os.path.join(args.out_dir, name)
        os.makedirs(sub_dir, exist_ok=True)

        for images, masks, filenames in loader:
            images = images.to(device)

            if gradcam is not None:
                images.requires_grad_(True)
                cam, probs = gradcam(images)
                image_rgb = denormalize_image(images[0].detach(), IMAGENET_MEAN, IMAGENET_STD)
                overlay = overlay_cam_on_image(image_rgb, cam)
                cv2.imwrite(os.path.join(sub_dir, f"{filenames[0]}_cam.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            else:
                with torch.no_grad():
                    logits = model(images)
                    probs = torch.sigmoid(logits)

            pred_np = probs[0, 0].detach().cpu().numpy()
            gt_np = masks.numpy()[0, 0]
            meter.update(pred_np, gt_np)

            pred_mask = (pred_np > 0.5).astype(np.uint8) * 255
            cv2.imwrite(os.path.join(sub_dir, f"{filenames[0]}_pred.png"), pred_mask)

        summary = meter.summary()
        print(f"[{name}] " + " ".join(f"{k}={v:.4f}" for k, v in summary.items()))


if __name__ == "__main__":
    main()
