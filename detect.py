import os
import time
from imutils import paths
import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from matplotlib import pyplot as plt

from torch_datasets import CustomImageDataset
from train import test_transform

BATCH_SIZE = 1

val_dataset = CustomImageDataset(
    img_dir=r"datasets/dataset_circle/val/img",
    label_dir=r"datasets/dataset_circle/val/labels", transform=test_transform)

img_test_transform = A.Compose([
    A.augmentations.geometric.resize.Resize(683, 1024, interpolation=1, always_apply=False, p=1),
    A.Normalize(always_apply=True),
    ToTensorV2()]
)

validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

if __name__ == "__main__":
    index = 32

    model = torch.load(
        r"C:\Users\tristan_cotte\PycharmProjects\CircleDetection\trained_models\legio_lr_10-4_finetune.pt",
        map_location=torch.device('cpu'))
    model.eval()

    images = paths.list_images(r"C:\Users\tristan_cotte\Downloads\detect_circle_legio\detect_circle_legio\output")

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

        # image = val_dataset[index][0].type(torch.FloatTensor)

        start = time.time()

        # im0 = cv2.imread(list(images)[index])
        im0 = cv2.imread(
            os.path.join(r"C:\Users\tristan_cotte\Pictures\Dataset_legio\PHOTOS IA\LEGIOS\LECTURE LEGIO DU 250322",
                         "EV22-07023.001 L3 C-25-03-2022_RAW.png"))
        img_w, img_h = im0.shape[1], im0.shape[0]
        augmented = img_test_transform(image=im0)
        image = augmented['image']

        image = torch.unsqueeze(image, 0)

        output = model(image)
        bboxes = output[0]
        bboxes[:, 0] = bboxes[:, 0] * img_w
        bboxes[:, 1] = bboxes[:, 1] * img_h
        bboxes[:, 2] = bboxes[:, 2] * img_w
        bboxes[:, 3] = bboxes[:, 3] * img_h
        prob = output[1][0][0]

        print("probability {}".format(torch.softmax(prob, dim=-1).item() * 100))

        # print(get_iou(np.array(128*train_dataset[index][2]), coords.numpy()))

        coords = np.round(bboxes.numpy())[0]
        coords = [int(x) for x in coords]

        print(time.time() - start)

        # image = cv2.imread(os.path.join(r"datasets/dataset_circle/val/img", val_dataset[index][-1]))
        cv2.rectangle(im0, (coords[0], coords[1]), (coords[2], coords[3]), (0, 255, 255), 1)
        plt.imshow(im0)
        plt.show()
