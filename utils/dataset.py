import os
import random
from PIL import Image, ImageEnhance

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _list_images(folder):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    return sorted([f for f in os.listdir(folder) if f.lower().endswith(exts)])


class PolypTrainDataset(Dataset):
    def __init__(self, root, image_size=352, augment=True):
        self.image_dir = os.path.join(root, "image")
        self.mask_dir = os.path.join(root, "masks")
        if not os.path.isdir(self.mask_dir):
            self.mask_dir = os.path.join(root, "masks")
        self.image_files = _list_images(self.image_dir)
        self.mask_files = _list_images(self.mask_dir)
        assert len(self.image_files) == len(self.mask_files), "Số lượng ảnh và mask không khớp"
        self.image_size = image_size
        self.augment = augment

        self.img_transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        self.mask_transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_files)

    def _augment(self, image, mask):
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        if random.random() < 0.5:
            angle = random.choice([90, 180, 270])
            image = image.rotate(angle)
            mask = mask.rotate(angle)
        if random.random() < 0.3:
            factor = random.uniform(0.7, 1.3)
            image = ImageEnhance.Brightness(image).enhance(factor)
        if random.random() < 0.3:
            factor = random.uniform(0.7, 1.3)
            image = ImageEnhance.Contrast(image).enhance(factor)
        return image, mask

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_dir, self.image_files[idx])).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, self.mask_files[idx])).convert("L")

        if self.augment:
            image, mask = self._augment(image, mask)

        image = self.img_transform(image)
        mask = self.mask_transform(mask)
        mask = (mask > 0.5).float()
        return image, mask


class PolypTestDataset(Dataset):
    def __init__(self, root, image_size=352, dataset_name=None):
        self.image_dir = os.path.join(root, "images")
        self.mask_dir = os.path.join(root, "masks")
        self.image_files = _list_images(self.image_dir)
        self.mask_files = _list_images(self.mask_dir)
        assert len(self.image_files) == len(self.mask_files), "Số lượng ảnh và mask không khớp"
        self.image_size = image_size
        self.dataset_name = dataset_name or os.path.basename(os.path.normpath(root))

        self.img_transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        self.mask_transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        name = self.image_files[idx]
        image = Image.open(os.path.join(self.image_dir, name)).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, self.mask_files[idx])).convert("L")

        image = self.img_transform(image)
        mask = self.mask_transform(mask)
        mask = (mask > 0.5).float()
        return image, mask, name


def build_multi_test_datasets(test_root, image_size=352):
    datasets = {}
    if os.path.isdir(os.path.join(test_root, "images")):
        datasets[os.path.basename(os.path.normpath(test_root))] = PolypTestDataset(test_root, image_size)
        return datasets

    for name in sorted(os.listdir(test_root)):
        sub_root = os.path.join(test_root, name)
        if os.path.isdir(sub_root) and os.path.isdir(os.path.join(sub_root, "images")):
            datasets[name] = PolypTestDataset(sub_root, image_size, dataset_name=name)
    return datasets


def rescale_batch(images, masks, size):
    if images.shape[-1] == size:
        return images, masks
    images = torch.nn.functional.interpolate(images, size=(size, size), mode="bilinear", align_corners=True)
    masks = torch.nn.functional.interpolate(masks, size=(size, size), mode="nearest")
    return images, masks
