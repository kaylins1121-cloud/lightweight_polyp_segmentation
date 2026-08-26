import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_csv", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="plots")
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.log_csv)
    df = df.dropna(subset=["mDice"])

    os.makedirs(args.out_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.log_csv))[0]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(df["epoch"], df["train_loss"])
    axes[0, 0].set_title("Train Loss")
    axes[0, 0].set_xlabel("Epoch")

    axes[0, 1].plot(df["epoch"], df["mDice"], label="mDice")
    axes[0, 1].plot(df["epoch"], df["mIoU"], label="mIoU")
    axes[0, 1].set_title("mDice / mIoU")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].legend()

    axes[1, 0].plot(df["epoch"], df["Smeasure"], label="S-measure")
    axes[1, 0].plot(df["epoch"], df["wFmeasure"], label="weighted F-measure")
    axes[1, 0].set_title("S-measure / weighted F-measure")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].legend()

    axes[1, 1].plot(df["epoch"], df["MAE"], label="MAE")
    axes[1, 1].set_title("MAE")
    axes[1, 1].set_xlabel("Epoch")

    plt.tight_layout()
    out_path = os.path.join(args.out_dir, f"{base_name}.png")
    plt.savefig(out_path, dpi=150)
    print(f"Đã lưu biểu đồ tại: {out_path}")


if __name__ == "__main__":
    main()
