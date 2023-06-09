import os
import typing
import numpy as np


def create_directory(path):
    if not os.path.isdir(path):
        # if the directory is not present then create it.
        os.makedirs(path)


def fix_bboxes(bboxes):
    for box in bboxes:
        if box[0] > box[2]:
            box[2] = box[0]
        if box[1] > box[3]:
            box[3] = box[1]
    return bboxes


def remap_coords(bboxes: np.array, img_size: typing.Tuple[int, int]):
    bboxes[:, 0] = bboxes[:, 0] * img_size[1]
    bboxes[:, 1] = bboxes[:, 1] * img_size[0]
    bboxes[:, 2] = bboxes[:, 2] * img_size[1]
    bboxes[:, 3] = bboxes[:, 3] * img_size[0]
    return np.round(bboxes).astype(int)
