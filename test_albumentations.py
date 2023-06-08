import random

import albumentations as A
import cv2
import torch
from albumentations.pytorch import ToTensorV2
from matplotlib import pyplot as plt

from torch_datasets import CustomImageDataset

BOX_COLOR = (255, 0, 0)  # Red
TEXT_COLOR = (255, 255, 255)  # White


def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):
    """Visualizes a single bounding box on the image"""
    x_min, y_min, x_max, y_max = [round(x * img.shape[0]) for x in bbox]
    #     x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35,
        color=TEXT_COLOR,
        lineType=cv2.LINE_AA,
    )
    return img


def visualize(image, bboxes, category_ids, category_id_to_name):
    if isinstance(image, torch.Tensor):
        img = image.permute(1, 2, 0).numpy()
    else:
        img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name)
    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.imshow(img)


bbox_format = 'albumentations'
train_transform = A.Compose(
    [
        A.augmentations.geometric.transforms.Affine (scale=(0.5, 1), translate_percent=(0.15, 0.5), keep_ratio=True, p=0.5),
        A.Equalize(mode='cv', by_channels=True, mask=None, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        # A.Rotate(limit=90, p=0.5, border_mode=0, rotate_method="ellipse"),
        A.CLAHE(p=0.5),
        A.OneOf([
            A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            A.GridDistortion(p=0.5),
        ], p=0.0),
        A.Normalize(always_apply=True),
        A.augmentations.geometric.resize.Resize (683, 1024, interpolation=1, always_apply=False, p=1),
        ToTensorV2()],
    bbox_params=A.BboxParams(format=bbox_format, label_fields=['category_ids']),
)

train_dataset = CustomImageDataset(
    img_dir=r"datasets/dataset_circle/train/img",
    label_dir=r"datasets/dataset_circle/train/labels",
    transform=train_transform)

# We will use the mapping from category_id to the class name
# to visualize the class label for the bounding box on the image
category_id_to_name = {0: 'circle'}

if __name__ == "__main__":
    random.seed(9)
    sample = train_dataset[10]
    print(sample[-1])
    # transformed = transform(image=sample[0], bboxes=sample[2])
    visualize(
        sample[0],
        sample[2],
        [0],
        category_id_to_name,
    )
    plt.show()
