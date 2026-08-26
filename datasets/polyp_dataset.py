import os
import zipfile
from glob import glob
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from huggingface_hub import hf_hub_download

class PolypDataset(Dataset):
    def __init__(self, image_paths, mask_paths, image_size=352):
        self.image_paths = sorted(image_paths)
        self.mask_paths = sorted(mask_paths)
        self.image_size = image_size
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")

        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)

        img_tensor = torch.tensor(np.array(image), dtype=torch.float32).permute(2, 0, 1) / 255.0
        mask_tensor = torch.tensor(np.array(mask), dtype=torch.float32).unsqueeze(0) / 255.0
        return img_tensor, (mask_tensor > 0.5).float()

def download_and_extract_dataset(data_root="./dataset"):
    os.makedirs(data_root, exist_ok=True)
    for zip_name in ["TrainDataset.zip", "TestDataset.zip"]:
        zip_path = os.path.join(data_root, zip_name)
        if not os.path.exists(zip_path):
            print(f"Downloading {zip_name} from Hugging Face...")
            hf_hub_download(repo_id="Zyna1121/lightpranet-polyp-data", repo_type="dataset", filename=zip_name, local_dir=data_root)
        
        extract_folder = os.path.join(data_root, zip_name.replace(".zip", ""))
        if not os.path.exists(extract_folder):
            print(f"Extracting {zip_name}...")
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(data_root)
    print("Dataset ready!")

def get_train_loaders(data_root, batch_size=16, image_size=352, val_split=0.2, seed=42, num_workers=2):
    img_dir = os.path.join(data_root, "TrainDataset", "images")
    mask_dir = os.path.join(data_root, "TrainDataset", "masks")
    
    img_paths = sorted(glob(os.path.join(img_dir, "*.*")))
    mask_paths = [os.path.join(mask_dir, os.path.basename(p)) for p in img_paths]
    
    indices = np.arange(len(img_paths))
    np.random.seed(seed); np.random.shuffle(indices)
    split = int(len(indices) * (1 - val_split))
    
    train_ds = PolypDataset([img_paths[i] for i in indices[:split]], [mask_paths[i] for i in indices[:split]], image_size)
    val_ds = PolypDataset([img_paths[i] for i in indices[split:]], [mask_paths[i] for i in indices[split:]], image_size)
    
    return DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True), \
           DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

def get_test_loader(data_root, dataset_name, image_size=352, num_workers=2):
    sub_path = os.path.join(data_root, "TestDataset", dataset_name)
    img_paths = sorted(glob(os.path.join(sub_path, "images", "*.*")))
    mask_paths = [os.path.join(sub_path, "masks", os.path.basename(p)) for p in img_paths]
    return DataLoader(PolypDataset(img_paths, mask_paths, image_size), batch_size=1, shuffle=False, num_workers=num_workers)
