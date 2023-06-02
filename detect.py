import os
import time

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt

from logger import WeightandBiaises
from torch_datasets import CustomImageDataset
from train import test_transform

BATCH_SIZE = 8

val_dataset = CustomImageDataset(
    img_dir=r"datasets/dataset_circle/val/img",
    label_dir=r"datasets/dataset_circle/val/labels", transform=test_transform)

validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

if __name__ == "__main__":



    index = 21

    start = time.time()

    model = torch.load(r"C:\Users\tristan_cotte\PycharmProjects\CircleDetection\trained_models\500_epochs.pt", map_location=torch.device('cpu'))
    model.eval()


    # for (images, labels, bboxes, filenames) in validation_loader:
    #     # send the input to the device
    #     labels = torch.Tensor(labels)
    #     # bboxes = torch.stack(bboxes, dim=1)
    #
    #     bboxes = bboxes.to(torch.float32)
    #     bboxes = torch.squeeze(bboxes, 1)
    #
    #     predictions = model(images)
    #
    #     print(predictions)
    #
    #     logger = WeightandBiaises(interval_display=1)
    #     logger.plot_one_batch(pred_batch=predictions[0], x_batch=images, y_batch=[None]*BATCH_SIZE, e=1)
    #     break


    with torch.no_grad():
        # print(train_dataset[0][0])
        image = val_dataset[index][0].type(torch.FloatTensor)

        image = torch.unsqueeze(image, 0)

        output = model(image)[0]
        coords = 128*output[0]


        # print(get_iou(np.array(128*train_dataset[index][2]), coords.numpy()))

        coords = np.round(coords.numpy())
        coords = [int(x) for x in coords]


        image = cv2.imread(os.path.join(r"datasets/dataset_circle/val/img", val_dataset[index][-1]))
        cv2.rectangle(image, (coords[0], coords[1]), (coords[2], coords[3]), (0, 255, 255), 1)
        plt.imshow(image)
        plt.show()

    print(time.time()-start)