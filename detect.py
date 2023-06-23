import argparse
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

# C:\Users\tristan_cotte\Pictures\Dataset_legio\YOLO\X\501.002CHDL3-14-03-2022.png

parser = argparse.ArgumentParser(description='Train custom model which enables to detect circles')
parser.add_argument('--batch-size', type=int, default=16, help='total batch size')
parser.add_argument('--name', type=str, default=None, help='save to project/name')
parser.add_argument('--model-path', type=str, help='Model path which has to detect circle')
parser.add_argument('--imgsz', nargs='+', default=[683, 1024],  type=int, help="Image size used for the training")
parser.add_argument('--src', type=str, required=True, help='Picture(s) where the AI has to detect circle')

args = parser.parse_args()

def timing_function(some_function):
    def wrapper(*args,**kwargs):
        t1 = time.time()
        ret = some_function(*args, **kwargs)
        t2 = time.time()
        print("Time it took to {fn}: {time:.2f} seconds".format(fn=some_function.__name__, time=t2-t1))
        if ret is not None:
            return ret
    return wrapper

@timing_function
def load_model(model_path: str):
    model = torch.load(
            model_path,
            map_location=torch.device('cpu'))
    model.eval()
    return model

@timing_function
def predict(model, image):
    return model(image)



BATCH_SIZE = args.batch_size
IMGSZ = args.imgsz


img_test_transform = A.Compose([
        A.augmentations.geometric.resize.Resize(*IMGSZ, interpolation=1, always_apply=False, p=1),
        A.Normalize(always_apply=True),
        ToTensorV2()]
)

model = load_model(args.model_path)


if os.path.isdir(args.src):


    val_dataset = CustomImageDataset(
        img_dir=r"datasets/dataset_circle/val/img",
        label_dir=r"datasets/dataset_circle/val/labels", transform=img_test_transform)

    validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

else:
    im0 = cv2.imread(
        r"C:\Users\tristan_cotte\Documents\Evry\Visual_AI\01_Datas\EV22-00501.005 CH L1-09-03-2022_RAW.png")
    im0 = cv2.resize(im0, IMGSZ, interpolation=cv2.INTER_AREA)




    with torch.no_grad():
        # print(train_dataset[0][0])

        # image = val_dataset[index][0].type(torch.FloatTensor)


        # im0 = cv2.imread(list(images)[index])
        im0 = cv2.imread(args.src)
        img_w, img_h = im0.shape[1], im0.shape[0]
        augmented = img_test_transform(image=im0)
        image = augmented['image']

        image = torch.unsqueeze(image, 0)

        output = predict(model, image)

        bboxes = output[0]
        bboxes[:, 0] = bboxes[:, 0] * img_w
        bboxes[:, 1] = bboxes[:, 1] * img_h
        bboxes[:, 2] = bboxes[:, 2] * img_w
        bboxes[:, 3] = bboxes[:, 3] * img_h

        print("probability {:.2f}%".format(torch.squeeze(output[2]).item() * 100))

        # print(get_iou(np.array(128*train_dataset[index][2]), coords.numpy()))

        coords = np.round(bboxes.numpy())[0]
        coords = [int(x) for x in coords]


        # image = cv2.imread(os.path.join(r"datasets/dataset_circle/val/img", val_dataset[index][-1]))
        cv2.rectangle(im0, (coords[0], coords[1]), (coords[2], coords[3]), (0, 255, 255), 1)
        plt.imshow(im0)
        plt.show()
