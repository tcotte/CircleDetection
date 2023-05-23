import os

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

IMG_SIZE = 128


class CustomImageDataset(Dataset):
    def __init__(self, img_dir: str, label_dir: str, label_needed: bool = True, transform=None, target_transform=None):
        self.label_needed = label_needed
        self.img_labels = os.listdir(img_dir)
        self.label_dir = label_dir
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        filename = os.listdir(self.img_dir)[idx]
        img_path = os.path.join(self.img_dir, filename)
        image = read_image(img_path)
        txtfile = filename[:-4] + ".txt"
        txtfile = open(os.path.join(self.label_dir, txtfile), "r")
        bbox = txtfile.readline().split(" ")
        bbox = [float(x) / IMG_SIZE for x in bbox]
        txtfile.close()
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            bbox = self.target_transform(bbox)

        if self.label_needed:
            return image.type(torch.FloatTensor), 0, torch.as_tensor(bbox), filename
        else:
            return image.type(torch.FloatTensor), bbox
