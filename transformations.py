import albumentations as A
from albumentations.pytorch import ToTensorV2



bbox_format = 'albumentations'
train_transform = A.Compose(
    [
        # A.augmentations.geometric.transforms.Affine (scale=(0.5, 1), translate_percent=(0.15, 0.5), keep_ratio=True, p=0.5),
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

        A.augmentations.geometric.resize.Resize(*IMGSZ, interpolation=1, always_apply=False, p=1),
        ToTensorV2()],
    bbox_params=A.BboxParams(format=bbox_format, label_fields=['category_ids']),
)

test_transform = A.Compose([
    # A.augmentations.geometric.resize.Resize(*IMGSZ, interpolation=1, always_apply=False, p=1),
    A.Normalize(always_apply=True),
    ToTensorV2()],
    bbox_params=A.BboxParams(format=bbox_format, label_fields=['category_ids'])
)