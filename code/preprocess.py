# preprocess.py
import numpy as np
import cv2

def apply_mask_and_resize(img_bgr: np.ndarray, keypoints: np.ndarray, mask: np.ndarray,
                          target_size=512, pad_ratio=0.05):
    """
    img_bgr: HxWx3
    keypoints: Nx3 (x,y,v)
    mask: HxW uint8 (0/255)
    Returns: canvas (target_size^2x3 uint8), transformed_kps Nx3 (float)
    """
    H, W = img_bgr.shape[:2]
    mask_bin = (mask > 127).astype(np.uint8)
    # If mask empty, attempt bbox from visible keypoints
    valid = keypoints[:,2] > 0
    if mask_bin.sum() == 0 and valid.sum()>0:
        xs = keypoints[valid,0]
        ys = keypoints[valid,1]
        xmin = max(int(xs.min()) - 20, 0)
        xmax = min(int(xs.max()) + 20, W)
        ymin = max(int(ys.min()) - 20, 0)
        ymax = min(int(ys.max()) + 20, H)
        mask_bin[ymin:ymax, xmin:xmax] = 1

    coords = np.column_stack(np.where(mask_bin>0))
    if coords.size == 0:
        y0,x0,y1,x1 = 0,0,H,W
    else:
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        pad = int(pad_ratio * max(y1-y0, x1-x0))
        y0 = max(0, y0-pad); x0 = max(0, x0-pad)
        y1 = min(H, y1+pad); x1 = min(W, x1+pad)

    cropped_img = img_bgr[y0:y1, x0:x1].copy()
    cropped_mask = mask_bin[y0:y1, x0:x1].copy()
    cropped_img[cropped_mask==0] = 0

    ch, cw = cropped_img.shape[:2]
    if ch == 0 or cw == 0:
        # fallback: return black canvas
        canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        transformed_kp = np.zeros_like(keypoints)
        return canvas, transformed_kp

    scale = target_size / max(ch, cw)
    new_h = int(ch * scale + 0.5); new_w = int(cw * scale + 0.5)
    resized = cv2.resize(cropped_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    y_off = (target_size - new_h) // 2; x_off = (target_size - new_w) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized

    transformed_kp = keypoints.copy().astype(float)
    for i in range(len(transformed_kp)):
        x,y,v = transformed_kp[i]
        if v == 0:
            transformed_kp[i] = [0,0,0]
            continue
        x_c = x - x0; y_c = y - y0
        if x_c < 0 or x_c >= cw or y_c < 0 or y_c >= ch:
            transformed_kp[i] = [0,0,0]
            continue
        x_r = x_c * scale; y_r = y_c * scale
        transformed_kp[i,0] = x_r + x_off
        transformed_kp[i,1] = y_r + y_off
    return canvas, transformed_kp

def make_heatmaps(kps, out_size, num_kp, sigma=3):
    """
    kps: Nx3 (x,y,v) in pixel coords in final image of size out_size x out_size
    returns (num_kp, out_size, out_size) float32
    """
    hm = np.zeros((num_kp, out_size, out_size), dtype=np.float32)
    tmp_size = sigma * 3
    for idx in range(num_kp):
        x, y, v = kps[idx]
        if v == 0:
            continue
        xi = int(round(x)); yi = int(round(y))
        ul = [int(xi - tmp_size), int(yi - tmp_size)]
        br = [int(xi + tmp_size + 1), int(yi + tmp_size + 1)]
        if ul[0] >= out_size or ul[1] >= out_size or br[0] < 0 or br[1] < 0:
            continue
        size = 2*tmp_size + 1
        x_coords = np.arange(0, size, 1, np.float32)
        y_coords = x_coords[:, None]
        center = tmp_size
        g = np.exp(-((x_coords - center)**2 + (y_coords - center)**2) / (2*sigma*sigma))
        g_x = max(0, -ul[0]), min(br[0], out_size) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], out_size) - ul[1]
        img_x = max(0, ul[0]), min(br[0], out_size)
        img_y = max(0, ul[1]), min(br[1], out_size)
        hm[idx, img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
            hm[idx, img_y[0]:img_y[1], img_x[0]:img_x[1]],
            g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
        )
    return hm
