import os

import cv2
import numpy as np
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--img_w', type=int, default=128, help='Image width')
parser.add_argument('--img_h', type=int, default=128, help='Image height')
parser.add_argument('--path', type=str, required=True, help='Dataset path')
parser.add_argument('--nb_samples', type=int, required=True, help='Number of samples in the created dataset')

args = parser.parse_args()

img_height = args.img_h
img_width = args.img_w
nb_samples = args.nb_samples
path_dataset = args.path

if __name__ == "__main__":
    print("[INFO] Directories creation")
    os.makedirs(os.path.join(path_dataset, "img"))
    os.makedirs(os.path.join(path_dataset, "labels"))

    print("[INFO] Dataset creation")

    for i in tqdm(range(1, nb_samples)):
        image = np.zeros((img_height, img_width, 3), np.uint8)

        radius = np.random.randint(low=5, high=round(img_height / 3))
        x_center_pos = np.random.randint(low=radius, high=img_width - 2 * radius)
        y_center_pos = np.random.randint(low=radius, high=img_height - 2 * radius)

        color = tuple(np.random.randint(low=0, high=200, size=3, dtype=int))
        # convert to native Python types to be supported by OpenCV
        color = [int(x) for x in color]

        cv2.circle(image, (x_center_pos, y_center_pos), radius, color, -1)

        cv2.imwrite(os.path.join(path_dataset, "img", str(i) + '.png'), image)

        # xyxy
        rect = [x_center_pos - radius, y_center_pos - radius, x_center_pos + radius, y_center_pos + radius]
        rect = [str(x) for x in rect]
        with open(os.path.join(path_dataset, "labels", str(i) + '.txt'), 'w') as f:
            f.write(" ".join(rect))
