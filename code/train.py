# train.py
import torch, time, os
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
import numpy as np

from utils import save_checkpoint, load_coco_annotations
from dataset import FaceKeypointDataset
from model import SimpleUNet
from augmentations import get_augmentations
from seg_inference import SegmentationWrapper

# CONFIG - adjust
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_DIR = PROJECT_ROOT / "data" / "train"
TEST_DIR = PROJECT_ROOT / "data" / "test"

TRAIN_JSON = TRAIN_DIR / "_annotations.coco.json"
TEST_JSON = TEST_DIR / "_annotations.coco.json"

IMAGES_DIR = DATA_DIR
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 512
BATCH = 8
EPOCHS = 10
LR = 1e-3
SIGMA = 3

# Load dataset index
index_list, kp_names, num_kp = load_coco_annotations(COCO_JSON)
print("Loaded dataset count:", len(index_list), "num_kp:", num_kp)

# Split
np.random.shuffle(index_list)
n = len(index_list)
n_train = int(0.8*n)
train_idx = index_list[:n_train]
val_idx = index_list[n_train:]

# Prepare components
seg = SegmentationWrapper(weights="yolov8n-seg.pt", device=None, conf=0.25)
train_ds = FaceKeypointDataset(train_idx, Path(IMAGES_DIR), seg, transform=get_augmentations(IMG_SIZE),
                               target_size=IMG_SIZE, sigma=SIGMA, num_kp=num_kp)
val_ds = FaceKeypointDataset(val_idx, Path(IMAGES_DIR), seg, transform=get_augmentations(IMG_SIZE),
                             target_size=IMG_SIZE, sigma=SIGMA, num_kp=num_kp)
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True)

model = SimpleUNet(in_ch=3, base=32, num_kp=num_kp).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

scaler = torch.cuda.amp.GradScaler() if DEVICE.startswith("cuda") else None

def preds_to_keypoints(preds, threshold=0.1):
    """
    preds: (B, K, H, W) numpy
    returns list of (K,3) arrays with x,y,v estimated by argmax on heatmap
    """
    batch_kps = []
    B,K,H,W = preds.shape
    for b in range(B):
        kps = np.zeros((K,3))
        for k in range(K):
            hmap = preds[b,k]
            idx = hmap.argmax()
            y = idx // W; x = idx % W
            val = float(hmap.max())
            if val < threshold:
                kps[k] = [0,0,0]
            else:
                kps[k] = [float(x), float(y), 1.0]
        batch_kps.append(kps)
    return batch_kps

def compute_nme(pred_kps, gt_kps):
    """
    pred_kps, gt_kps: (K,3) numpy arrays
    NME = mean euclidean distance over visible keypoints normalized by inter-ocular distance if available
    We'll fallback to image diagonal if no eye points known.
    """
    vis = gt_kps[:,2] > 0
    if vis.sum() == 0:
        return np.nan
    dists = np.linalg.norm(pred_kps[vis,:2] - gt_kps[vis,:2], axis=1)
    # try to find two eye keypoints heuristically: use first two keypoints if available
    if gt_kps.shape[0] >= 2:
        # inter-ocular using kp[36],kp[45] typical for 68-lmk; fallback to bbox width
        if gt_kps.shape[0] >= 68:
            left = gt_kps[36]; right = gt_kps[45]
            iod = np.linalg.norm(left[:2]-right[:2]) if left[2]>0 and right[2]>0 else None
        else:
            visible_coords = gt_kps[vis,:2]
            iod = visible_coords.max(axis=0) - visible_coords.min(axis=0)
            iod = np.linalg.norm(iod) if iod.sum()>0 else None
    else:
        iod = None
    if iod is None or iod == 0:
        iod = np.sqrt(IMG_SIZE**2 + IMG_SIZE**2)  # diag as fallback
    return dists.mean() / iod

def pck(pred_kps, gt_kps, thr=0.05):
    vis = gt_kps[:,2] > 0
    if vis.sum() == 0:
        return np.nan
    dists = np.linalg.norm(pred_kps[vis,:2] - gt_kps[vis,:2], axis=1)
    # normalization by diag
    norm = np.sqrt(IMG_SIZE**2 + IMG_SIZE**2)
    return float((dists / norm) <= thr).sum() / vis.sum()

# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    for imgs, hms, kps in tqdm(train_loader, desc=f"Train E{epoch+1}/{EPOCHS}"):
        imgs = imgs.to(DEVICE); hms = hms.to(DEVICE)
        if scaler:
            with torch.cuda.amp.autocast():
                preds = model(imgs)
                preds = torch.sigmoid(preds)
                loss = criterion(preds, hms)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
        else:
            preds = model(imgs); preds = torch.sigmoid(preds)
            loss = criterion(preds, hms)
            opt.zero_grad(); loss.backward(); opt.step()
        total_loss += loss.item() * imgs.size(0)
    avg_loss = total_loss / len(train_ds)
    # validation
    model.eval()
    val_loss = 0.0
    nmers = []; pcks = []
    with torch.no_grad():
        for imgs, hms, gt_kps in val_loader:
            imgs = imgs.to(DEVICE); hms = hms.to(DEVICE)
            preds = model(imgs)
            preds = torch.sigmoid(preds)
            l = criterion(preds, hms)
            val_loss += l.item() * imgs.size(0)
            preds_np = preds.cpu().numpy()
            batch_pred_kps = preds_to_keypoints(preds_np, threshold=0.05)
            for pb, gb in zip(batch_pred_kps, gt_kps):
                nme_val = compute_nme(pb, gb.numpy())
                pck_val = pck(pb, gb.numpy(), thr=0.05)
                if not np.isnan(nme_val):
                    nmers.append(nme_val)
                if not np.isnan(pck_val):
                    pcks.append(pck_val)
    val_loss /= len(val_ds)
    print(f"Epoch {epoch+1}: train_loss {avg_loss:.6f}, val_loss {val_loss:.6f}, val_NME {np.nanmean(nmers):.6f}, val_PCK {np.nanmean(pcks):.4f}")
    # save checkpoint
    save_checkpoint(f"checkpoint_epoch_{epoch+1}.pth", model, opt, extra={'epoch': epoch+1})
