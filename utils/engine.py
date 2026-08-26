import os
import random

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.dataset import PolypTrainDataset, build_multi_test_datasets, rescale_batch, IMAGENET_MEAN, IMAGENET_STD
from utils.metrics import MetricMeter
from utils.seed import seed_worker, get_generator
from utils.gradcam import GradCAM, overlay_cam_on_image, denormalize_image
from utils.logger import AverageMeter, CSVLogger

import numpy as np
import cv2


def build_dataloaders(args, seed):
    train_set = PolypTrainDataset(args.train_path, image_size=args.input_size, augment=True)
    generator = get_generator(seed)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=True,
    )

    test_datasets = build_multi_test_datasets(args.test_path, image_size=args.input_size)
    test_loaders = {
        name: DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers)
        for name, ds in test_datasets.items()
    }
    return train_loader, test_loaders


def build_optimizer(args, model):
    if args.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)


def train_one_epoch(model, loader, criterion, optimizer, device, args, epoch):
    model.train()
    loss_meter = AverageMeter()
    progress = tqdm(loader, desc=f"Epoch {epoch}", leave=False)

    for images, masks in progress:
        images, masks = images.to(device), masks.to(device)
        rate = random.choice(args.multiscale_rates)
        size = int(round(args.input_size * rate / 32) * 32)
        images_s, masks_s = rescale_batch(images, masks, size)

        optimizer.zero_grad()
        logits = model(images_s)
        loss = criterion(logits, masks_s)
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        loss_meter.update(loss.item(), images.size(0))
        progress.set_postfix(loss=loss_meter.avg)

    return loss_meter.avg


@torch.no_grad()
def evaluate(model, test_loaders, device, args):
    model.eval()
    all_summaries = {}
    for name, loader in test_loaders.items():
        meter = MetricMeter(num_e_thresholds=args.num_e_thresholds)
        for images, masks, _ in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()[0, 0]
            gt = masks.numpy()[0, 0]
            meter.update(probs, gt)
        all_summaries[name] = meter.summary()

    overall = {}
    if all_summaries:
        keys = list(next(iter(all_summaries.values())).keys())
        for k in keys:
            overall[k] = float(np.mean([s[k] for s in all_summaries.values()]))
    return overall, all_summaries


def generate_gradcam(model, test_loaders, device, args, seed, run_tag):
    if not args.gradcam:
        return

    target_layer = getattr(model, "gradcam_target_layer", None)
    if target_layer is None:
        print("Model không khai báo gradcam_target_layer, bỏ qua GradCAM")
        return

    out_dir = os.path.join(args.gradcam_path, run_tag)
    os.makedirs(out_dir, exist_ok=True)

    gradcam = GradCAM(model, target_layer)
    count = 0

    for name, loader in test_loaders.items():
        for images, masks, filenames in loader:
            if count >= args.gradcam_samples:
                return
            images = images.to(device)
            images.requires_grad_(True)

            cam, probs = gradcam(images)
            image_rgb = denormalize_image(images[0].detach(), IMAGENET_MEAN, IMAGENET_STD)
            overlay = overlay_cam_on_image(image_rgb, cam)

            pred_mask = (probs[0, 0].cpu().numpy() > 0.5).astype(np.uint8) * 255

            out_name = f"{name}_{filenames[0]}"
            cv2.imwrite(os.path.join(out_dir, f"{out_name}_image.png"), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(out_dir, f"{out_name}_cam.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(out_dir, f"{out_name}_pred.png"), pred_mask)

            count += 1
