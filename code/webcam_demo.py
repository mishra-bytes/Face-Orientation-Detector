# webcam_demo.py
import cv2
import numpy as np
import torch
from seg_inference import SegmentationWrapper
from preprocess import apply_mask_and_resize
from visualize import draw_keypoints
from model import SimpleUNet

# CONFIG
TARGET_SIZE = 512
THRESHOLD_SIM = 0.04   # NME threshold; tune lower for stricter match
MODEL_CKPT = "checkpoint_epoch_10.pth"  # path to checkpoint
TARGET_KPS_PATH = "target_kps.npy"       # numpy (K,3) in 512x512 coords
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# load segmentation & model
seg = SegmentationWrapper(weights="yolov8n-seg.pt", conf=0.25)
model = SimpleUNet(in_ch=3, base=32, num_kp=68)  # change num_kp to your K
ckpt = torch.load(MODEL_CKPT, map_location=DEVICE)
model.load_state_dict(ckpt['model_state'])
model.to(DEVICE).eval()

target_kps = np.load(TARGET_KPS_PATH)  # ensure shape (K,3)

def compute_nme_local(pred, gt):
    vis = gt[:,2] > 0
    if vis.sum() == 0: return np.inf
    dists = np.linalg.norm(pred[vis,:2] - gt[vis,:2], axis=1)
    iod = np.linalg.norm(gt[36,:2] - gt[45,:2]) if gt.shape[0] >= 46 and gt[36,2]>0 and gt[45,2]>0 else np.sqrt(TARGET_SIZE**2*2)
    return dists.mean() / iod

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

print("Press q to quit. Press c to manually capture. Auto-capture triggers when similarity <= threshold.")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    orig = frame.copy()
    mask = seg.get_person_mask(frame)
    canvas, _ = apply_mask_and_resize(frame, np.zeros((target_kps.shape[0],3)), mask, target_size=TARGET_SIZE)
    inp = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype('float32')/255.0
    inp = torch.from_numpy(inp).permute(2,0,1).unsqueeze(0).to(DEVICE).float()
    with torch.no_grad():
        pred = model(inp)
        pred = torch.sigmoid(pred).cpu().numpy()[0]  # (K,H,W)
    # convert heatmaps to keypoints
    K,H,W = pred.shape
    pred_kps = np.zeros((K,3))
    for k in range(K):
        hmap = pred[k]
        idx = hmap.argmax()
        y = idx // W; x = idx % W
        val = float(hmap.max())
        if val < 0.05:
            pred_kps[k] = [0,0,0]
        else:
            pred_kps[k] = [float(x), float(y), 1.0]
    sim = compute_nme_local(pred_kps, target_kps)
    display = draw_keypoints(canvas, pred_kps)
    cv2.putText(display, f"sim(NME): {sim:.4f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.imshow("webcam-orient", cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
    key = cv2.waitKey(1) & 0xFF
    if sim <= THRESHOLD_SIM:
        # auto-capture
        fname = f"autocap_{int(time.time())}.jpg"
        cv2.imwrite(fname, canvas)
        print("Auto-captured:", fname, "sim:", sim)
    if key == ord('q'):
        break
    if key == ord('c'):
        fname = f"manualcap_{int(time.time())}.jpg"
        cv2.imwrite(fname, canvas)
        print("Manual captured:", fname)
cap.release()
cv2.destroyAllWindows()
