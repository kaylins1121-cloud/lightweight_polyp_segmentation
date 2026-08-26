import argparse
import os
import zipfile

from huggingface_hub import hf_hub_download

REPO_ID = "Zyna1121/lightpranet-polyp-data"


def download_and_extract(filename, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    local_path = hf_hub_download(repo_id=REPO_ID, filename=filename, repo_type="dataset")
    with zipfile.ZipFile(local_path, "r") as zf:
        zf.extractall(out_dir)
    print(f"Đã giải nén {filename} vào {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="dataset")
    parser.add_argument("--train_only", action="store_true")
    parser.add_argument("--test_only", action="store_true")
    args = parser.parse_args()

    if not args.test_only:
        download_and_extract("TrainDataset.zip", args.out_dir)
    if not args.train_only:
        download_and_extract("TestDataset.zip", args.out_dir)


if __name__ == "__main__":
    main()
