import os
import argparse
import yaml
import torch
import torch.optim as optim
import pandas as pd
from tqdm import tqdm

from models import get_model
from losses.loss import StructureLoss
from datasets.polyp_dataset import download_and_extract_dataset, get_train_loaders, get_test_loader
from utils.helpers import set_seed, save_checkpoint, load_checkpoint, save_visualization
from metrics.segmentation import compute_metrics

def train(config):
    set_seed(config["validation"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    download_and_extract_dataset(config["data"]["root"])
    train_loader, val_loader = get_train_loaders(
        config["data"]["root"], config["training"]["batch_size"],
        config["data"]["image_size"], config["validation"]["split"],
        config["validation"]["seed"], config["hardware"]["num_workers"]
    )
    
    model = get_model(config["model"]["name"]).to(device)
    criterion = StructureLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config["training"]["lr"])
    
    best_dice = 0.0
    for epoch in range(config["training"]["epochs"]):
        model.train()
        train_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}"):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        model.eval()
        dice_scores = []
        with torch.no_grad():
            for images, masks in val_loader:
                preds = torch.sigmoid(model(images.to(device))).cpu().numpy()
                for p, m in zip(preds, masks.numpy()):
                    dice_scores.append(compute_metrics(p, m)["dice"])
        mean_dice = sum(dice_scores) / len(dice_scores)
        print(f"Epoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.4f} | Val Dice: {mean_dice:.4f}")
        
        is_best = mean_dice > best_dice
        if is_best: best_dice = mean_dice
        save_checkpoint({"state_dict": model.state_dict()}, is_best, config["checkpoint"]["dir"], config["model"]["name"])

def test(config, save_maps=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(config["model"]["name"]).to(device)
    load_checkpoint(os.path.join(config["checkpoint"]["dir"], config["model"]["name"], "best.pth"), model)
    model.eval()
    
    datasets = ["Kvasir", "CVC-ClinicDB", "CVC-ColonDB", "CVC-300", "ETIS-LaribPolypDB"]
    results = []
    
    for ds in datasets:
        try:
            loader = get_test_loader(config["data"]["root"], ds, config["data"]["image_size"])
        except Exception:
            continue
        metrics_list = []
        for idx, (images, masks) in enumerate(tqdm(loader, desc=f"Testing {ds}")):
            images, masks = images.to(device), masks.to(device)
            with torch.no_grad():
                preds = torch.sigmoid(model(images))
            m = compute_metrics(preds.cpu().numpy()[0], masks.cpu().numpy()[0])
            metrics_list.append(m)
            if save_maps:
                save_visualization(images[0], masks[0], preds[0], os.path.join("results", config["model"]["name"], ds, f"{idx:04d}.png"))
        if metrics_list:
            mean_m = {k: sum(x[k] for x in metrics_list)/len(metrics_list) for k in metrics_list[0]}
            mean_m["Dataset"] = ds
            results.append(mean_m)
            
    df = pd.DataFrame(results)
    print("\n" + df.to_string(index=False))
    os.makedirs(os.path.join("results", config["model"]["name"]), exist_ok=True)
    df.to_csv(os.path.join("results", config["model"]["name"], "results.csv"), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="unet")
    parser.add_argument("--mode", type=str, default="train_test", choices=["train", "test", "train_test"])
    parser.add_argument("--save_maps", action="store_true")
    args = parser.parse_args()
    
    with open("configs/train.yaml", "r") as f:
        config = yaml.safe_load(f)
    config["model"]["name"] = args.model
    
    if args.mode in ["train", "train_test"]: train(config)
    if args.mode in ["test", "train_test"]: test(config, save_maps=args.save_maps)
