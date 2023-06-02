import json

import cv2
import os
from natsort import natsorted
from tqdm import tqdm
from matplotlib import pyplot as plt

OUTPUT_FOLDER = r"C:\Users\tristan_cotte\Pictures\dataset_idea\IDEA"


def write_in_lableme(box, img_path, filename, img_size):
    labelme_shape = [{
        'label': 'circle',
        'points': [[box[0], box[1]], [box[2], box[3]]],
        'group_id': None,
        'shape_type': 'rectangle',
        'flags': {}
    }]

    annotation = {
        "version": "4.2.5",
        "flags": {},
        "shapes": labelme_shape,
        "imagePath": img_path,
        "imageData": None,
        "imageHeight": img_size[0],
        "imageWidth": img_size[1]
    }

    with open(os.path.join(OUTPUT_FOLDER, filename + '.json'), 'w') as outfile:
        json.dump(annotation, outfile)


if __name__ == "__main__":
    path = r"C:\Users\tristan_cotte\Pictures\dataset_idea\IDEA"
    img_ext = ".jpg"

    for i in tqdm(natsorted(os.listdir(path))):
        if i.endswith(img_ext):
            filename = i[:-4]
            img_path = os.path.join(path, i)
            img = cv2.imread(img_path)

            aspect_ratio = img.shape[0] / img.shape[1]
            img_w = 500
            resized_image = cv2.resize(img, (img_w, round(500 * aspect_ratio)))
            gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)  # Convert the image to gray scale

            thres = 255
            gray = cv2.blur(gray, (1, 1))
            edges = cv2.Canny(image=gray, threshold1=170, threshold2=180,
                              apertureSize=3, L2gradient=True)


            circles = cv2.HoughCircles(image=edges, method=cv2.HOUGH_GRADIENT, dp=0.6,
                                       minDist=120, param1=thres, param2=30, minRadius=10,
                                       maxRadius=138)
            if circles is not None:
                for circle in circles[0, :]:
                    a, b = int(circle[0]), int(circle[1])
                    # radius = int(circle[2])
                    # print(radius)
                    # print(a,b)
                    # cv2.circle(img=resized_image, center=(a, b), radius=radius, color=(255, 0, 0),
                    #            thickness=1)

                    ratio = img.shape[1] / img_w

                    x_center = round(circle[0] * ratio)
                    y_center = round(circle[1] * ratio)
                    radius = round(circle[2] * ratio)
                    coords = [x_center - radius, y_center - radius, x_center + radius, y_center + radius]


                    write_in_lableme(coords, img_path, filename, img.shape[:2])
            else:
                print("[WARN] No circle was detected in the file {}".format(filename))
