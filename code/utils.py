# utils.py
import json, os
from pathlib import Path
import numpy as np
import torch
def load_multiple_coco_annotations(json_paths, base_dirs):
    """
    json_paths: list of json annotation files
    base_dirs: list of directories containing images for each json
               (same length as json_paths)
    Returns combined_index, kp_names, num_kp
    """
    combined = []
    kp_names = None
    num_kp = None

    for json_path, base_dir in zip(json_paths, base_dirs):
        index_list, kp_names_local, num_kp_local = load_coco_annotations(json_path)

        # set global keypoint config
        if num_kp is None:
            num_kp = num_kp_local
            kp_names = kp_names_local

        # attach base_dir to each image record
        for item in index_list:
            item["base_dir"] = base_dir
        combined.extend(index_list)

    return combined, kp_names, num_kp

def load_coco_annotations(json_path):
    json_path = Path(json_path)
    with open(json_path, 'r') as f:
        coco = json.load(f)
    img_map = {im['id']: im for im in coco['images']}
    anns = [a for a in coco['annotations'] if 'keypoints' in a]
    # infer keypoint names & count if present
    kp_names = None
    num_kp = None
    for cat in coco.get('categories', []):
        if 'keypoints' in cat:
            kp_names = cat['keypoints']
            num_kp = len(kp_names)
            break
    if num_kp is None and anns:
        num_kp = len(anns[0]['keypoints']) // 3
        kp_names = [f'kp{i}' for i in range(num_kp)]
    dataset_index = []
    for ann in anns:
        imginfo = img_map[ann['image_id']]
        dataset_index.append({
            "image_id": ann["image_id"],
            "file_name": imginfo["file_name"],
            "width": imginfo["width"],
            "height": imginfo["height"],
            "keypoints": ann["keypoints"],
        })
    return dataset_index, kp_names, num_kp

def flat_to_kp_array(flat):
    arr = np.array(flat, dtype=float).reshape(-1,3)
    return arr  # shape (K,3)

def kp_array_to_flat(arr):
    return arr.reshape(-1).tolist()

def save_checkpoint(path, model, optimizer=None, extra=None):
    state = {'model_state': model.state_dict()}
    if optimizer: state['opt_state'] = optimizer.state_dict()
    if extra: state['extra'] = extra
    torch.save(state, path)

def load_checkpoint(path, model, optimizer=None, map_location=None):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt['model_state'])
    if optimizer and 'opt_state' in ckpt:
        optimizer.load_state_dict(ckpt['opt_state'])
    return ckpt.get('extra', None)
