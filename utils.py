import os


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
