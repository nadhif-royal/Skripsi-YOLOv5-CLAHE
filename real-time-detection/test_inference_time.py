"""
test_inference_time.py — Mengukur Waktu Inferensi CPU per Citra (ms) untuk
KEEMPAT skenario (S-1, S-2, S-3, S-4) di Tabel 6.1 skripsi.

Definisi "waktu inferensi" di sini = waktu penuh dari gambar mentah masuk
sampai hasil deteksi (bounding box) keluar -- mencakup preprocessing
(termasuk CLAHE untuk S-2/S-4), forward pass model, dan NMS. Ini konsisten
dengan cara FPS dihitung di dashboard_komparasi.py & benchmark_stability.py
sebelumnya (satu definisi "waktu proses" dipakai di seluruh skripsi).

Semua model dipaksa berjalan di CPU:
  - YOLOv5 ONNX  -> onnxruntime providers=['CPUExecutionProvider']
  - SSD TFLite   -> tf.lite.Interpreter (tanpa delegate GPU)
"""
import os
import glob
import time
import csv
import cv2
import numpy as np

from core.preprocess import apply_clahe_lab
from core.yolo_infer import YoloInference
from core.ssd_infer import SSDInference

# ==== KONFIGURASI ====
TEST_IMG_DIR = "test_images_static"   # folder berisi citra uji representatif
WARMUP_RUNS = 10                       # jumlah run awal yang DIBUANG (cold-start)
REPEATS_PER_IMAGE = 5                  # ulangi tiap citra beberapa kali utk hasil stabil
OUTPUT_CSV = "hasil_waktu_inferensi_cpu.csv"

MODELS = {
    "S-1": {
        "label": "YOLOv5s (Original)",
        "loader": lambda: YoloInference('models/best_S1_yolov5.onnx', conf_thres=0.25, iou_thres=0.45),
        "clahe": False,
    },
    "S-2": {
        "label": "YOLOv5s + CLAHE L*a*b* (Usulan)",
        "loader": lambda: YoloInference('models/best_S2_yolov5_clahe.onnx', conf_thres=0.25, iou_thres=0.45),
        "clahe": True,
    },
    "S-3": {
        "label": "SSD MobileNetV2 (Original)",
        "loader": lambda: SSDInference('models/best_S3_ssd.tflite', num_classes=1, conf_thres=0.25, iou_thres=0.45),
        "clahe": False,
    },
    "S-4": {
        "label": "SSD MobileNetV2 + CLAHE L*a*b*",
        "loader": lambda: SSDInference('models/best_S4_ssd_clahe.tflite', num_classes=1, conf_thres=0.25, iou_thres=0.45),
        "clahe": True,
    },
}


def load_test_images(folder):
    paths = sorted(
        glob.glob(os.path.join(folder, "*.jpg")) +
        glob.glob(os.path.join(folder, "*.jpeg")) +
        glob.glob(os.path.join(folder, "*.png"))
    )
    if not paths:
        raise FileNotFoundError(
            f"Tidak ada gambar ditemukan di '{folder}'. "
            f"Isi folder ini dengan beberapa citra uji (mis. 14 citra uji statis kamu)."
        )
    images = []
    for p in paths:
        img = cv2.imread(p)
        if img is not None:
            images.append(img)
    print(f"[INFO] {len(images)} citra uji dimuat dari '{folder}'")
    return images


def benchmark_scenario(name, cfg, images):
    print(f"\n[INFO] Memuat model {name}: {cfg['label']} ...")
    model = cfg["loader"]()
    use_clahe = cfg["clahe"]

    def run_once(img_bgr):
        h, w = img_bgr.shape[:2]
        t0 = time.perf_counter()
        infer_input = apply_clahe_lab(img_bgr) if use_clahe else img_bgr
        _ = model.detect(infer_input, w, h)
        return (time.perf_counter() - t0) * 1000.0  # ms

    # --- Warm-up (dibuang, supaya tidak kena overhead cold-start) ---
    print(f"[INFO] Warm-up {WARMUP_RUNS}x ...")
    for _ in range(WARMUP_RUNS):
        run_once(images[0])

    # --- Pengukuran sesungguhnya ---
    times_ms = []
    for img in images:
        for _ in range(REPEATS_PER_IMAGE):
            times_ms.append(run_once(img))

    times_ms = np.array(times_ms)
    
    # Ekstraksi Metrik Mean, Standar Deviasi, dan Persentil
    mean_ms = float(np.mean(times_ms))
    std_ms = float(np.std(times_ms))
    p50_ms = float(np.percentile(times_ms, 50))  # Median
    p95_ms = float(np.percentile(times_ms, 95))  # P95 (Batas atas stabilitas)
    
    result = {
        "skenario": name,
        "label": cfg["label"],
        "n_pengukuran": len(times_ms),
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "min_ms": float(np.min(times_ms)),
        "max_ms": float(np.max(times_ms)),
        "p50_ms": p50_ms,
        "p95_ms": p95_ms,
    }
    
    # Menampilkan hasil di terminal sesuai format yang diinginkan
    print(f"[HASIL] {name}: {cfg['label']}")
    print(f"   Mean: {mean_ms:.2f} ms")
    print(f"   P50 (Median): {p50_ms:.2f} ms")
    print(f"   P95 (Batas atas): {p95_ms:.2f} ms")
    
    return result


def main():
    print("=" * 60)
    print("PENGUJIAN WAKTU INFERENSI CPU PER CITRA (ms)")
    print("Tabel 6.1 - Hasil Komparasi Pelatihan Arsitektur Pendeteksian Objek")
    print("=" * 60)

    images = load_test_images(TEST_IMG_DIR)

    all_results = []
    for name, cfg in MODELS.items():
        try:
            res = benchmark_scenario(name, cfg, images)
            all_results.append(res)
        except Exception as e:
            print(f"[ERROR] Gagal benchmark {name}: {e}")
            print("        Pastikan file model ada di folder 'models/' dan namanya sesuai MODELS di atas.")

    print("\n" + "=" * 80)
    print("RINGKASAN — Waktu Inferensi CPU per Citra (ms)")
    print("=" * 80)
    print(f"{'Skenario':<10}{'Label':<35}{'Mean (ms)':>10}{'P50 (ms)':>10}{'P95 (ms)':>10}")
    for r in all_results:
        print(f"{r['skenario']:<10}{r['label']:<35}{r['mean_ms']:>10.2f}{r['p50_ms']:>10.2f}{r['p95_ms']:>10.2f}")

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "skenario", "label", "n_pengukuran", "mean_ms", "std_ms", "min_ms", "max_ms", "p50_ms", "p95_ms"
        ])
        writer.writeheader()
        for r in all_results:
            writer.writerow(r)
    print(f"\n✅ Hasil lengkap tersimpan di: {OUTPUT_CSV}")
    print("   Ingat: Masukkan angka P50 dan P95 untuk Skenario 2 (S-2) ke dalam narasi Bab 6.1.1 skripsi Anda.")


if __name__ == "__main__":
    main()