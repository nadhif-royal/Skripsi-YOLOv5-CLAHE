"""
ssd_utils.py — fungsi anchor & decode untuk SSD MobileNetV2 (konfigurasi FINAL
hasil tuning: 6 feature map termasuk 40x40, skala anchor 3%-70%).
HARUS SAMA PERSIS dengan konfigurasi yang dipakai saat training model
best_S3_ssd.tflite / best_S4_ssd_clahe.tflite di Colab.
"""
import numpy as np

IMG_SIZE = 320
VARIANCES = np.array([0.1, 0.1, 0.2, 0.2], dtype=np.float32)

FEATURE_MAPS = [40, 20, 10, 5, 3, 1]
ASPECT_RATIOS = [
    [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
    [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
    [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
    [1.0, 2.0, 0.5],
    [1.0, 2.0, 0.5],
    [1.0, 2.0, 0.5],
]
NUM_BOXES_PER_CELL = [6, 6, 6, 4, 4, 4]
SCALES = [0.03, 0.144, 0.258, 0.372, 0.486, 0.6, 0.7]


def generate_anchors():
    anchors = []
    for k, g in enumerate(FEATURE_MAPS):
        sk = SCALES[k]
        sk_next = SCALES[k + 1]
        sk_prime = np.sqrt(sk * sk_next)
        ratios = ASPECT_RATIOS[k]
        for i in range(g):
            for j in range(g):
                cx = (j + 0.5) / g
                cy = (i + 0.5) / g
                for r in ratios:
                    w = sk * np.sqrt(r)
                    h = sk / np.sqrt(r)
                    anchors.append((cx, cy, w, h))
                    if r == 1.0:
                        anchors.append((cx, cy, sk_prime, sk_prime))
    return np.array(anchors, dtype=np.float32)


def to_corners(boxes_cxcywh):
    cx, cy, w, h = np.split(boxes_cxcywh, 4, axis=-1)
    xmin, ymin = cx - w / 2, cy - h / 2
    xmax, ymax = cx + w / 2, cy + h / 2
    return np.concatenate([xmin, ymin, xmax, ymax], axis=-1)


def decode_boxes(loc_preds, anchors):
    acx, acy, aw, ah = anchors[..., 0], anchors[..., 1], anchors[..., 2], anchors[..., 3]
    tx = loc_preds[..., 0] * VARIANCES[0]
    ty = loc_preds[..., 1] * VARIANCES[1]
    tw = loc_preds[..., 2] * VARIANCES[2]
    th = loc_preds[..., 3] * VARIANCES[3]
    cx = tx * aw + acx
    cy = ty * ah + acy
    w = np.exp(np.clip(tw, -10, 10)) * aw
    h = np.exp(np.clip(th, -10, 10)) * ah
    return np.stack([cx, cy, w, h], axis=-1)