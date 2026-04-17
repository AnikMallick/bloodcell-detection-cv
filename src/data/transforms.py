import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(img_size=512):
    return A.Compose([
        A.LongestMaxSize(max_size=img_size),
        
        A.PadIfNeeded(min_height=img_size, min_width=img_size,
                      border_mode=0, fill=(123, 117, 104)),  # ImageNet mean color # padding is required as fix size is expected by model

        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2), # if we flip vertically it wont effect blood images

        # Color augmentations — only affect pixel values, not box coords
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, p=0.3),
        A.GaussianBlur(blur_limit=3, p=0.2),

        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()

    ], bbox_params=A.BboxParams(
        format='pascal_voc',      # input format is [x1, y1, x2, y2] in pixels
        label_fields=['labels'],
        clip=True, # clip boxes to image boundary after augmentation
        min_area=16, # drop boxes that become smaller than 16px² (noise)
        min_visibility=0.3 # drop boxes where <30% of area is still visible
    ))


def get_val_transforms(img_size=512):
    return A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(min_height=img_size, min_width=img_size,
                      border_mode=0, fill=(123, 117, 104)),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['labels'],
        clip=True,
        min_area=16,
        min_visibility=0.3
    ))