import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


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
        txtfile = os.path.join(self.label_dir, txtfile)

        if os.stat(txtfile).st_size > 0:
            opened_file = open(txtfile, "r")
            bboxes = opened_file.readline().split(" ")
            opened_file.close()

            # bboxes = np.array([[float(x) / IMG_SIZE for x in bboxes]])
            bboxes = np.array(bboxes, dtype=float)
            if bboxes.ndim == 1:
                bboxes = np.expand_dims(bboxes, axis=0)

            img_h, img_w = image.shape[:2]
            bboxes[:, 0] = bboxes[:, 0] / img_w
            bboxes[:, 1] = bboxes[:, 1] / img_h
            bboxes[:, 2] = bboxes[:, 2] / img_w
            bboxes[:, 3] = bboxes[:, 3] / img_h
            category_ids = np.ones(len(bboxes), dtype=int)
            probabilities = np.ones(len(bboxes), dtype=float)

        else:
            print("no ann")
            bboxes = np.array([[0, 0, 1, 1]], dtype=int)
            category_ids = np.zeros(len(bboxes), dtype=int)
            probabilities = np.zeros(len(bboxes), dtype=float)

        if self.transform:
            augmented = self.transform(image=image, bboxes=bboxes, category_ids=category_ids)
            image = augmented['image']
            bboxes = augmented['bboxes']

        if self.label_needed:
            return image, category_ids, torch.as_tensor(bboxes), probabilities, filename
        else:
            return image.type(torch.FloatTensor), bboxes
