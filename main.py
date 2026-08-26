import os
import json

import numpy as np
import torch

from config import parse_args
from models import build_model, list_models
from losses import build_loss, list_losses
from utils.seed import set_seed
from utils.engine import build_dataloaders, build_optimizer, train_one_epoch, evaluate, generate_gradcam
from utils.logger import CSVLogger


def run_single(args, seed):
    set_seed(seed, deterministic=args.deterministic)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    run_tag = f"{args.model}_{args.loss}_seed{seed}"
    print(f"\n===== Bắt đầu huấn luyện: {run_tag} =====")

    train_loader, test_loaders = build_dataloaders(args, seed)

    model = build_model(args.model, in_channels=3, num_classes=1).to(device)
    criterion = build_loss(args.loss)
    optimizer = build_optimizer(args, model)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_epoch, gamma=args.decay_rate)

    log_path = os.path.join(args.log_path, f"{run_tag}.csv")
    metric_keys = ["mDice", "mIoU", "wFmeasure", "Smeasure", "meanEmeasure", "maxEmeasure", "MAE"]
    logger = CSVLogger(log_path, fieldnames=["epoch", "train_loss", "lr"] + metric_keys)

    best_dice = -1.0
    best_metrics = None
    os.makedirs(args.save_path, exist_ok=True)
    ckpt_path = os.path.join(args.save_path, f"{run_tag}_best.pth")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, args, epoch)
        current_lr = optimizer.param_groups[0]["lr"]

        row = {"epoch": epoch, "train_loss": train_loss, "lr": current_lr}

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            overall, _ = evaluate(model, test_loaders, device, args)
            row.update(overall)
            print(f"[{run_tag}] Epoch {epoch}/{args.epochs} loss={train_loss:.4f} mDice={overall.get('mDice', 0):.4f} mIoU={overall.get('mIoU', 0):.4f}")

            if overall.get("mDice", 0) > best_dice:
                best_dice = overall["mDice"]
                best_metrics = overall
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "metrics": overall,
                    "args": vars(args),
                    "seed": seed,
                }, ckpt_path)
        else:
            for k in metric_keys:
                row[k] = ""

        logger.log(row)
        scheduler.step()

    if best_metrics is None:
        best_metrics, _ = evaluate(model, test_loaders, device, args)

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    generate_gradcam(model, test_loaders, device, args, seed, run_tag)

    print(f"===== Kết thúc {run_tag}: best mDice={best_metrics.get('mDice', 0):.4f} =====")
    return best_metrics


def main():
    args = parse_args()

    if args.model not in list_models():
        raise ValueError(f"Model '{args.model}' không tồn tại. Các model khả dụng: {list_models()}")
    if args.loss not in list_losses():
        raise ValueError(f"Loss '{args.loss}' không tồn tại. Các loss khả dụng: {list_losses()}")

    os.makedirs(args.log_path, exist_ok=True)

    if args.multiseed:
        seeds = args.seeds[:5] if len(args.seeds) >= 5 else args.seeds
        all_results = []
        for seed in seeds:
            metrics = run_single(args, seed)
            metrics["seed"] = seed
            all_results.append(metrics)

        summary = {"model": args.model, "loss": args.loss, "seeds": seeds, "runs": all_results}
        metric_keys = ["mDice", "mIoU", "wFmeasure", "Smeasure", "meanEmeasure", "maxEmeasure", "MAE"]
        aggregate = {}
        for k in metric_keys:
            values = [r[k] for r in all_results if k in r]
            aggregate[k] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
        summary["aggregate"] = aggregate

        summary_path = os.path.join(args.log_path, f"{args.model}_{args.loss}_multiseed_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print("\n===== Kết quả tổng hợp multiseed (mean ± std) =====")
        for k, v in aggregate.items():
            print(f"{k}: {v['mean']:.4f} ± {v['std']:.4f}")
        print(f"Đã lưu tại: {summary_path}")
    else:
        run_single(args, args.seed)


if __name__ == "__main__":
    main()
