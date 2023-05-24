import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

IMG_SIZE = 128


class CustomImageDataset(Dataset):
    def __init__(self, img_dir: str, label_dir: str, label_needed: bool = True, transform=None):
        self.label_needed = label_needed
        self.img_labels = os.listdir(img_dir)
        self.label_dir = label_dir
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        filename = os.listdir(self.img_dir)[idx]
        img_path = os.path.join(self.img_dir, filename)
        image = cv2.imread(img_path)
        # By default OpenCV uses BGR color space for color images,
        # so we need to convert the image to RGB color space.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        txtfile = filename[:-4] + ".txt"
        txtfile = open(os.path.join(self.label_dir, txtfile), "r")
        bboxes = txtfile.readline().split(" ")
        txtfile.close()

        bboxes = np.array([[float(x) / IMG_SIZE for x in bboxes]])
        category_ids = np.zeros(len(bboxes), dtype=int)

        if self.transform:
            augmented = self.transform(image=image, bboxes=bboxes, category_ids=category_ids)
            image = augmented['image']
            bboxes = augmented['bboxes']

        if self.label_needed:
            return image, 0, torch.as_tensor(bboxes), filename
        else:
            return image.type(torch.FloatTensor), bboxes
