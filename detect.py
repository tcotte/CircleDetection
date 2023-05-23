import os
import time

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt

from metrics import get_iou
from torch_datasets import CustomImageDataset
from train import transforms

val_dataset = CustomImageDataset(
    img_dir=r"datasets/dataset_circle/val/img",
    label_dir=r"datasets/dataset_circle/val/labels")


validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False)

if __name__ == "__main__":
    # for i in range(10):
    #     img, label = dataset[i]
    #     plt.imshow(img.permute(1, 2, 0))
    #     print(label)
    #     plt.show()
    #
    train_dataset = CustomImageDataset(
        img_dir=r"datasets/dataset_circle/train/img",
        label_dir=r"datasets/dataset_circle/train/labels", transform=transforms)

    index = 18

    start = time.time()

    model = torch.load(r"C:\Users\tristan_cotte\PycharmProjects\CircleDetection\trained_models\500_epochs.pt")
    model.eval()


    with torch.no_grad():
        # print(train_dataset[0][0])
        image = val_dataset[index][0].type(torch.FloatTensor)

        image = torch.unsqueeze(image, 0)

        output = model(image)[0]
        coords = 128*output[0]


        print(get_iou(np.array(128*train_dataset[index][2]), coords.numpy()))

        coords = np.round(coords.numpy())
        coords = [int(x) for x in coords]


        image = cv2.imread(os.path.join(r"datasets/dataset_circle/val/img", val_dataset[index][-1]))
        cv2.rectangle(image, (coords[0], coords[1]), (coords[2], coords[3]), (0, 255, 255), 1)
        plt.imshow(image)
        plt.show()

    print(time.time()-start)