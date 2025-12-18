# dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2

from utils import flat_to_kp_array
from preprocess import apply_mask_and_resize, make_heatmaps

class FaceKeypointDataset(Dataset):
    def __init__(self, index_list, images_dir, seg_wrapper, transform=None, target_size=512, sigma=3, num_kp=None):
        self.index_list = index_list
        self.images_dir = images_dir
        self.seg_wrapper = seg_wrapper
        self.transform = transform
        self.target_size = target_size
        self.sigma = sigma
        self.num_kp = num_kp

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        rec = self.index_list[idx]
        img_path = str(rec["base_dir"] / rec['file_name'])

        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        kps = flat_to_kp_array(rec['keypoints'])
        mask = self.seg_wrapper.get_person_mask(img)
        canvas, transformed_kps = apply_mask_and_resize(img, kps, mask, target_size=self.target_size)
        # Albumentations keypoints: list of (x,y) or (-1,-1) for invisible
        kps_xy = []
        for x,y,v in transformed_kps:
            if v > 0:
                kps_xy.append((float(x), float(y)))
            else:
                kps_xy.append((-1.0, -1.0))
        if self.transform:
            augmented = self.transform(image=canvas, keypoints=kps_xy)
            img_aug = augmented['image']
            kp_aug_xy = augmented['keypoints']
            final_kps = np.zeros_like(transformed_kps)
            for i,(xy,orig) in enumerate(zip(kp_aug_xy, transformed_kps)):
                if xy[0] < 0 or xy[1] < 0:
                    final_kps[i] = [0,0,0]
                else:
                    final_kps[i] = [xy[0], xy[1], float(orig[2])]
        else:
            img_aug = canvas
            final_kps = transformed_kps
        heatmaps = make_heatmaps(final_kps, self.target_size, self.num_kp, sigma=self.sigma)
        img_tensor = torch.from_numpy(img_aug.astype('float32')/255.0).permute(2,0,1).float()
        heat_tensor = torch.from_numpy(heatmaps).float()
        return img_tensor, heat_tensor, final_kps
