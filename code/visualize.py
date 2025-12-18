# visualize.py
import matplotlib.pyplot as plt
import cv2
import numpy as np

def draw_keypoints(image_bgr, kps, radius=3, color=(0,255,0)):
    img = image_bgr.copy()
    for x,y,v in kps:
        if v>0:
            cv2.circle(img, (int(x), int(y)), radius, color, -1)
    return img

def show_pipeline(orig_bgr, mask, masked_img, canvas, final_kps, heatmaps=None, pred_heatmaps=None, title=""):
    n = 5 + (1 if pred_heatmaps is not None else 0)
    fig, axes = plt.subplots(1, n, figsize=(4*n,4))
    axes[0].imshow(cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)); axes[0].set_title("orig"); axes[0].axis('off')
    axes[1].imshow(mask, cmap='gray'); axes[1].set_title("mask"); axes[1].axis('off')
    axes[2].imshow(cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB)); axes[2].set_title("masked"); axes[2].axis('off')
    axes[3].imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)); axes[3].set_title("padded"); axes[3].axis('off')
    kp_img = draw_keypoints(canvas, final_kps)
    axes[4].imshow(cv2.cvtColor(kp_img, cv2.COLOR_BGR2RGB)); axes[4].set_title("kp on padded"); axes[4].axis('off')
    if pred_heatmaps is not None:
        axes[5].imshow(np.sum(pred_heatmaps, axis=0)); axes[5].set_title("pred heatmap sum"); axes[5].axis('off')
    plt.suptitle(title)
    plt.show()
