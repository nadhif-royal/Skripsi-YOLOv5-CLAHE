import cv2
import time
import numpy as np
from core.preprocess import apply_clahe_lab
from core.yolo_infer import YoloInference


def pilih_sumber_kamera():
    print("\nPilih sumber kamera:")
    print("  1. Kamera bawaan laptop")
    print("  2. Webcam USB eksternal")
    print("  3. Kamera HP via IP (mis. aplikasi IP Webcam)")

    while True:
        pilihan = input("Masukkan pilihan (1/2/3): ").strip()
        if pilihan == '1':
            return 0
        elif pilihan == '2':
            idx = input("  Masukkan indeks device USB (biasanya 1, coba 2/3 kalau gagal): ").strip()
            try:
                return int(idx)
            except ValueError:
                print("  Indeks harus angka, coba lagi.")
        elif pilihan == '3':
            ip = input("  Masukkan URL stream IP kamera HP (mis. http://192.168.1.10:8080/video): ").strip()
            if ip:
                return ip
            print("  URL tidak boleh kosong.")
        else:
            print("Pilihan tidak valid, masukkan 1, 2, atau 3.")


def buka_kamera(sumber):
    cap = cv2.VideoCapture(sumber)
    if not cap.isOpened():
        print(f"❌ Gagal membuka kamera dari sumber: {sumber}")
        return None
    print(f"✅ Kamera berhasil dibuka: {sumber}")
    return cap


def main():
    print("=" * 50)
    print("DASHBOARD KOMPARASI MODEL (S1 vs S2)")
    print("FULL VIEW - ANTI CROP")
    print("=" * 50)

    sumber = pilih_sumber_kamera()
    cap = buka_kamera(sumber)
    while cap is None:
        print("Coba pilih ulang sumber kamera.")
        sumber = pilih_sumber_kamera()
        cap = buka_kamera(sumber)

    print("\n[INFO] Memuat Model YOLOv5...")
    model_s1 = YoloInference('models/best_S1_yolov5.onnx', conf_thres=0.15, iou_thres=0.45)
    model_s2 = YoloInference('models/best_S2_yolov5_clahe.onnx', conf_thres=0.15, iou_thres=0.45)

    cv2.namedWindow("Dashboard Evaluasi S1 vs S2", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Gagal membaca kamera.")
            time.sleep(1)
            continue

        orig_h, orig_w = frame.shape[:2]

        # 1. SKENARIO 1 (ORIGINAL)
        start_t1 = time.time()
        boxes_s1 = model_s1.detect(frame, orig_w, orig_h)
        fps_s1 = 1.0 / (time.time() - start_t1)

        frame_s1 = frame.copy()
        for box in boxes_s1:
            x1, y1, x2, y2, score = box
            cv2.rectangle(frame_s1, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(frame_s1, f"{score:.2f}", (x1, max(y1 - 10, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame_s1, f"S1: Asli | Deteksi: {len(boxes_s1)} | FPS: {fps_s1:.1f}",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        # 2. SKENARIO 2 (CLAHE)
        start_t2 = time.time()
        img_clahe = apply_clahe_lab(frame)
        boxes_s2 = model_s2.detect(img_clahe, orig_w, orig_h)
        fps_s2 = 1.0 / (time.time() - start_t2)

        frame_s2 = img_clahe.copy()
        for box in boxes_s2:
            x1, y1, x2, y2, score = box
            cv2.rectangle(frame_s2, (x1, y1), (x2, y2), (0, 255, 255), 3)
            cv2.putText(frame_s2, f"{score:.2f}", (x1, max(y1 - 10, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame_s2, f"S2: CLAHE | Deteksi: {len(boxes_s2)} | FPS: {fps_s2:.1f}",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        # 3. GABUNGKAN
        target_w = 640
        target_h = int((orig_h / orig_w) * target_w)
        frame_s1_rsz = cv2.resize(frame_s1, (target_w, target_h))
        frame_s2_rsz = cv2.resize(frame_s2, (target_w, target_h))
        dashboard = np.hstack((frame_s1_rsz, frame_s2_rsz))

        cv2.imshow("Dashboard Evaluasi S1 vs S2", dashboard)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"bukti_komparasi_{int(time.time())}.jpg"
            cv2.imwrite(filename, dashboard)
            print(f"[*] Screenshot disimpan: {filename}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()