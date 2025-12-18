# augmentations.py
import albumentations as A
import cv2

def get_augmentations(out_size=512):
    aug = A.Compose([
        A.OneOf([
            A.Affine(rotate=(-20,20), translate_percent=(0.0,0.05), shear=(-8,8), mode=cv2.BORDER_CONSTANT, p=0.6),
            A.PiecewiseAffine(scale=(0.02,0.05), nb_rows=4, nb_cols=4, p=0.4),
        ], p=0.9),
        A.RandomBrightnessContrast(p=0.7),
        A.GaussNoise(var_limit=(5.0, 30.0), p=0.4),
        A.HorizontalFlip(p=0.2),
        A.Resize(out_size, out_size),
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    return aug
