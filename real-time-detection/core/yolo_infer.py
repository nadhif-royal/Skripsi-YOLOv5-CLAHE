import cv2
import numpy as np
import onnxruntime as ort


class YoloInference:
    def __init__(self, model_path, conf_thres=0.5, iou_thres=0.45):
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.img_size = self.input_shape[2]  # 640

        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

    def letterbox(self, img, color=(114, 114, 114)):
        """Resize dengan mempertahankan rasio aspek + padding, PERSIS seperti
        yang dipakai YOLOv5 saat training/detect.py. Mengembalikan gambar,
        rasio skala, dan besar padding (dw, dh) untuk dipakai saat scale-back."""
        shape = img.shape[:2]  # (h, w)
        new_shape = (self.img_size, self.img_size)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return img, r, (dw, dh)

    def preprocess(self, img_bgr):
        """ Menyiapkan gambar dari kamera agar sesuai format YOLOv5 """
        img_lb, r, pad = self.letterbox(img_bgr)
        img = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return np.ascontiguousarray(img), r, pad

    def xywh2xyxy(self, x):
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2
        y[:, 1] = x[:, 1] - x[:, 3] / 2
        y[:, 2] = x[:, 0] + x[:, 2] / 2
        y[:, 3] = x[:, 1] + x[:, 3] / 2
        return y

    def non_max_suppression(self, prediction):
        """ Menghapus kotak yang tumpang tindih (NMS) """
        pred = prediction[0]
        pred = pred[pred[:, 4] > self.conf_thres]
        if len(pred) == 0:
            return []

        scores = pred[:, 4] * pred[:, 5]
        boxes_xyxy = self.xywh2xyxy(pred[:, :4])

        # PENTING: cv2.dnn.NMSBoxes butuh format [x, y, w, h] (pojok kiri-atas + lebar/tinggi),
        # BUKAN [x1, y1, x2, y2]. Konversi dulu di sini, atau NMS akan salah hitung IoU
        # dan salah menghapus deteksi objek yang saling berdekatan (dampaknya besar
        # untuk kasus spring yang saling berdempetan).
        boxes_xywh = np.column_stack([
            boxes_xyxy[:, 0],
            boxes_xyxy[:, 1],
            boxes_xyxy[:, 2] - boxes_xyxy[:, 0],
            boxes_xyxy[:, 3] - boxes_xyxy[:, 1],
        ])

        indices = cv2.dnn.NMSBoxes(boxes_xywh.tolist(), scores.tolist(), self.conf_thres, self.iou_thres)

        results = []
        if len(indices) > 0:
            for i in np.array(indices).flatten():
                results.append((boxes_xyxy[i], scores[i]))
        return results

    def detect(self, img_bgr, orig_w, orig_h):
        """ Proses utama: pra-proses -> inferensi -> pasca-proses """
        input_tensor, r, pad = self.preprocess(img_bgr)

        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        predictions = outputs[0]

        detections = self.non_max_suppression(predictions)

        dw, dh = pad
        final_boxes = []
        for box, score in detections:
            x1, y1, x2, y2 = box
            # Scale-back yang benar: kurangi padding letterbox dulu, baru bagi rasio skala
            rx1 = (x1 - dw) / r
            ry1 = (y1 - dh) / r
            rx2 = (x2 - dw) / r
            ry2 = (y2 - dh) / r

            rx1 = int(np.clip(rx1, 0, orig_w))
            ry1 = int(np.clip(ry1, 0, orig_h))
            rx2 = int(np.clip(rx2, 0, orig_w))
            ry2 = int(np.clip(ry2, 0, orig_h))

            final_boxes.append([rx1, ry1, rx2, ry2, score])

        return final_boxes