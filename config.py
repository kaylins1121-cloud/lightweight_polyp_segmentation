import argparse


def get_parser():
    parser = argparse.ArgumentParser(description="Lightweight Polyp Segmentation Training Framework")

    parser.add_argument("--model", type=str, required=True, help="Tên model đã đăng ký trong models/")
    parser.add_argument("--loss", type=str, required=True, help="Tên loss đã đăng ký trong losses/")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--multiseed", action="store_true", help="Huấn luyện 5 lần với 5 seed khác nhau")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 2024])

    parser.add_argument("--input_size", type=int, default=352)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "adam", "sgd"])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--decay_rate", type=float, default=0.1)
    parser.add_argument("--decay_epoch", type=int, default=50)
    parser.add_argument("--clip", type=float, default=0.5)
    parser.add_argument("--multiscale_rates", type=float, nargs="+", default=[0.75, 1.0, 1.25])

    parser.add_argument("--train_path", type=str, default="dataset/TrainDataset")
    parser.add_argument("--test_path", type=str, default="dataset/TestDataset")
    parser.add_argument("--save_path", type=str, default="checkpoints")
    parser.add_argument("--log_path", type=str, default="logs")

    parser.add_argument("--gradcam", action="store_true", default=True)
    parser.add_argument("--no_gradcam", dest="gradcam", action="store_false")
    parser.add_argument("--gradcam_samples", type=int, default=8)
    parser.add_argument("--gradcam_path", type=str, default="gradcam_outputs")

    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--num_e_thresholds", type=int, default=255)

    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--no_deterministic", dest="deterministic", action="store_false")

    parser.add_argument("--device", type=str, default="cuda")

    return parser


def parse_args():
    return get_parser().parse_args()
