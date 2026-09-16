"""
benchmark_stability.py (versi multi-kondisi, UI diperbaiki) — Split panel
S1 | S2 berdampingan seperti dashboard_komparasi.py, supaya kamu bisa lihat
kedua model sekaligus saat merekam sesi benchmark.
"""
import cv2
import time
import csv
import os
import numpy as np
from core.preprocess import apply_clahe_lab
from core.yolo_infer import YoloInference

N_FRAMES = 300
LOG_DIR = "logs_stabilitas"
WINDOW_NAME = "Benchmark Stabilitas - S1 vs S2"


def main():
    os.makedirs(LOG_DIR, exist_ok=True)

    kondisi = input("Label kondisi pencahayaan (mis. Normal / Senter_Dekat / Senter_Jauh / LampuHP): ").strip()
    kondisi = kondisi.replace(" ", "_") if kondisi else "TanpaLabel"

    gt_input = input("Jumlah spring sebenarnya (ground truth) sesi ini: ").strip()
    GROUND_TRUTH_COUNT = int(gt_input)

    print("[INFO] Memuat model...")
    model_s1 = YoloInference('models/best_S1_yolov5.onnx', conf_thres=0.15, iou_thres=0.45)
    model_s2 = YoloInference('models/best_S2_yolov5_clahe.onnx', conf_thres=0.15, iou_thres=0.45)

    cam_idx = input("Indeks/URL kamera (0=laptop, 1=USB, atau URL IP): ").strip()
    try:
        cam_idx = int(cam_idx)
    except ValueError:
        pass
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        print("❌ Gagal membuka kamera.")
        return

    print(f"\nKondisi: {kondisi} | Ground Truth: {GROUND_TRUTH_COUNT} spring")
    print("Posisikan pencahayaan sesuai skenario (mis. senter menyorot), spring DIAM.")
    print("Tekan 'r' untuk MULAI merekam, 'q' untuk batal.")

    log_s1, log_s2 = [], []
    fps_s1_list, fps_s2_list = [], []
    recording = False
    frame_count = 0

    # Window bisa di-resize manual + kita set ukuran awal yang wajar (bukan kotak kecil ter-zoom)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 480)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        orig_h, orig_w = frame.shape[:2]

        # --- S1: Original ---
        t1 = time.time()
        boxes_s1 = model_s1.detect(frame, orig_w, orig_h)
        fps1 = 1.0 / (time.time() - t1)

        frame_s1 = frame.copy()
        for x1, y1, x2, y2, score in boxes_s1:
            cv2.rectangle(frame_s1, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_s1, f"{score:.2f}", (x1, max(y1 - 5, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # --- S2: CLAHE L*a*b* ---
        t2 = time.time()
        img_clahe = apply_clahe_lab(frame)
        boxes_s2 = model_s2.detect(img_clahe, orig_w, orig_h)
        fps2 = 1.0 / (time.time() - t2)

        frame_s2 = img_clahe.copy()
        for x1, y1, x2, y2, score in boxes_s2:
            cv2.rectangle(frame_s2, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame_s2, f"{score:.2f}", (x1, max(y1 - 5, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        status = f"REKAM {frame_count}/{N_FRAMES}" if recording else "TEKAN 'r' UTK MULAI"
        cv2.putText(frame_s1, f"[{kondisi}] S1 Original | Deteksi:{len(boxes_s1)} | FPS:{fps1:.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        cv2.putText(frame_s2, f"[{kondisi}] S2 CLAHE | Deteksi:{len(boxes_s2)} | FPS:{fps2:.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        cv2.putText(frame_s1, status, (10, orig_h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        # Gabungkan 2 panel berdampingan (S1 | S2), diskalakan biar pas di layar
        target_w = 640
        target_h = int((orig_h / orig_w) * target_w)
        frame_s1_rsz = cv2.resize(frame_s1, (target_w, target_h))
        frame_s2_rsz = cv2.resize(frame_s2, (target_w, target_h))
        dashboard = np.hstack((frame_s1_rsz, frame_s2_rsz))

        cv2.imshow(WINDOW_NAME, dashboard)

        if recording:
            log_s1.append(len(boxes_s1))
            log_s2.append(len(boxes_s2))
            fps_s1_list.append(fps1)
            fps_s2_list.append(fps2)
            frame_count += 1
            if frame_count >= N_FRAMES:
                print("\n✅ Perekaman selesai!")
                break

        key = cv2.waitKey(1) & 0xFF
        if key == ord('r') and not recording:
            recording = True
            print("🔴 Mulai merekam...")
        elif key == ord('q'):
            print("Dibatalkan.")
            cap.release()
            cv2.destroyAllWindows()
            return

    cap.release()
    cv2.destroyAllWindows()

    csv_path = os.path.join(LOG_DIR, f"{kondisi}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "kondisi", "ground_truth", "count_s1", "count_s2", "fps_s1", "fps_s2"])
        for i in range(len(log_s1)):
            writer.writerow([i, kondisi, GROUND_TRUTH_COUNT, log_s1[i], log_s2[i], fps_s1_list[i], fps_s2_list[i]])
    print(f"✅ Data tersimpan: {csv_path}")
    print(f"   Rata-rata S1: {np.mean(log_s1):.2f} | Rata-rata S2: {np.mean(log_s2):.2f} (GT: {GROUND_TRUTH_COUNT})")
    print("\nJalankan sesi lain untuk kondisi berbeda, lalu jalankan aggregate_results.py setelah semua selesai.")


if __name__ == "__main__":
    main()