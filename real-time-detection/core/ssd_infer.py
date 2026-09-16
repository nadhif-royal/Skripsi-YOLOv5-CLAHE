"""
ssd_infer.py — Kelas inferensi SSD MobileNetV2 memakai TFLite Interpreter
(CPU only, tidak ada GPU delegate ditambahkan -> murni ukur kemampuan CPU).
Antarmuka (.detect()) dibuat sama persis dengan YoloInference supaya bisa
dipakai bergantian di dashboard/benchmark tanpa ubah kode pemanggil.
"""
import cv2
import numpy as np
import tensorflow as tf
from core.ssd_utils import generate_anchors, decode_boxes, to_corners


class SSDInference:
    def __init__(self, model_path, num_classes=1, img_size=320, conf_thres=0.5, iou_thres=0.45):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.img_size = img_size
        self.num_classes = num_classes
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.anchors = generate_anchors()

    def preprocess(self, img_bgr):
        """SSD dilatih TANPA letterbox (resize langsung), jadi di sini juga
        harus resize langsung -- BUKAN pakai letterbox seperti YOLOv5."""
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (self.img_size, self.img_size))
        img_norm = (img_resized / 255.0).astype(np.float32)
        return np.expand_dims(img_norm, axis=0)

    def detect(self, img_bgr, orig_w, orig_h):
        input_tensor = self.preprocess(img_bgr)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
        self.interpreter.invoke()

        # Cari output loc (dim terakhir = 4) vs cls (dim terakhir = num_classes+1)
        # berdasarkan shape, bukan urutan -- urutan output TFLite tidak selalu
        # sama dengan urutan didefinisikan di model Keras.
        out0 = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        out1 = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
        if out0.shape[-1] == 4:
            loc_preds, cls_preds = out0, out1
        else:
            loc_preds, cls_preds = out1, out0

        probs = tf.nn.softmax(cls_preds, axis=-1).numpy()
        decoded_corners = to_corners(decode_boxes(loc_preds, self.anchors))

        final_boxes = []
        for c in range(1, self.num_classes + 1):
            scores = probs[:, c]
            mask = scores >= self.conf_thres
            if not np.any(mask):
                continue
            boxes_c = decoded_corners[mask]
            scores_c = scores[mask]

            # PENTING: cv2.dnn.NMSBoxes butuh format [x,y,w,h], bukan [x1,y1,x2,y2]
            boxes_xywh = np.column_stack([
                boxes_c[:, 0], boxes_c[:, 1],
                boxes_c[:, 2] - boxes_c[:, 0], boxes_c[:, 3] - boxes_c[:, 1],
            ])
            idx = cv2.dnn.NMSBoxes(boxes_xywh.tolist(), scores_c.tolist(), self.conf_thres, self.iou_thres)
            for i in (np.array(idx).flatten() if len(idx) > 0 else []):
                x1, y1, x2, y2 = boxes_c[i]
                rx1 = int(np.clip(x1 * orig_w, 0, orig_w))
                ry1 = int(np.clip(y1 * orig_h, 0, orig_h))
                rx2 = int(np.clip(x2 * orig_w, 0, orig_w))
                ry2 = int(np.clip(y2 * orig_h, 0, orig_h))
                final_boxes.append([rx1, ry1, rx2, ry2, float(scores_c[i])])

        return final_boxes