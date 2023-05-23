import os

import cv2
import numpy as np
from tqdm import tqdm

img_height = 128
img_width = 128
nb_samples = 500

path_dataset = r"C:\Users\tristan_cotte\PycharmProjects\CircleDetection\datasets\dataset_circle\val"

if __name__ == "__main__":
    print("[INFO] Directories creation")
    os.makedirs(os.path.join(path_dataset, "img"))
    os.makedirs(os.path.join(path_dataset, "labels"))

    print("[INFO] Dataset creation")

    for i in tqdm(range(1, nb_samples)):
        image = np.zeros((img_height, img_width, 3), np.uint8)

        radius = np.random.randint(low=5, high=round(img_height / 3))
        x_center_pos = np.random.randint(low=radius, high=img_width-2*radius)
        y_center_pos = np.random.randint(low=radius, high=img_height-2*radius)

        color = tuple(np.random.randint(low=0, high=200, size=3, dtype=int))
        # convert to native Python types to be supported by OpenCV
        color = [int(x) for x in color]

        cv2.circle(image, (x_center_pos, y_center_pos), radius, color, -1)

        cv2.imwrite(os.path.join(path_dataset, "img", str(i) + '.png'), image)


        # xyxy
        rect = [x_center_pos-radius, y_center_pos-radius, x_center_pos+radius, y_center_pos + radius]
        rect = [str(x) for x in rect]
        with open(os.path.join(path_dataset, "labels", str(i) + '.txt'), 'w') as f:
            f.write(" ".join(rect))


