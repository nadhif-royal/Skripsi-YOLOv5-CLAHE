
"""
benchmark_stability.py

Benchmark kestabilan deteksi S1 vs S2 menggunakan 300 frame webcam.

Folder output dibuat otomatis berdasarkan Ground Truth:
    logs_stabilitas_15/
    logs_stabilitas_25/
    logs_stabilitas_30/

Setiap sesi disimpan dengan nama file unik agar tidak menimpa CSV lama.

CSV memuat:
    session_id, frame, kondisi, ground_truth,
    count_s1, count_s2, fps_s1, fps_s2
"""

from datetime import datetime
from pathlib import Path
import csv
import time

import cv2
import numpy as np

from core.preprocess import apply_clahe_lab
from core.yolo_infer import YoloInference


# ============================================================
# 1. KONFIGURASI EKSPERIMEN
# ============================================================

N_FRAMES = 300

ALLOWED_GROUND_TRUTH = {15, 25, 30}

ALLOWED_CONDITIONS = {
    "normal": "Normal",
    "senter_dekat": "Senter_Dekat",
    "senter_jauh": "Senter_Jauh",
}

WINDOW_NAME = "Benchmark Stabilitas - S1 vs S2"

CONF_THRESHOLD = 0.15
IOU_THRESHOLD = 0.45

TARGET_PANEL_WIDTH = 640

BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# 2. INPUT DAN VALIDASI
# ============================================================

def ask_ground_truth() -> int:
    """Meminta GT dan hanya menerima 15, 25, atau 30 komponen."""

    while True:
        raw_value = input(
            "Jumlah spring sebenarnya (15 / 25 / 30): "
        ).strip()

        try:
            value = int(raw_value)
        except ValueError:
            print(
                "Input harus berupa angka 15, 25, atau 30."
            )
            continue

        if value not in ALLOWED_GROUND_TRUTH:
            print(
                "Ground Truth yang diperbolehkan hanya "
                "15, 25, atau 30."
            )
            continue

        return value


def ask_condition() -> str:
    """
    Menerima pilihan angka menu atau label kondisi manual.

    1 = Normal
    2 = Senter_Dekat
    3 = Senter_Jauh
    """

    menu_options = {
        "1": "Normal",
        "2": "Senter_Dekat",
        "3": "Senter_Jauh",
    }

    print("\nPilih kondisi pencahayaan:")
    print("  1. Normal")
    print("  2. Senter_Dekat")
    print("  3. Senter_Jauh")

    while True:
        raw_value = input(
            "Pilihan kondisi (1-3): "
        ).strip().lower()

        # Pilihan numerik sesuai nomor menu.
        if raw_value in menu_options:
            return menu_options[raw_value]

        # Tetap mendukung input label manual.
        normalized = raw_value.replace(" ", "_")

        if normalized in ALLOWED_CONDITIONS:
            return ALLOWED_CONDITIONS[normalized]

        print(
            "Pilihan tidak valid. Masukkan 1, 2, 3, "
            "Normal, Senter_Dekat, atau Senter_Jauh."
        )


def ask_camera_source():
    """
    Menerima indeks kamera numerik atau URL kamera IP.
    Default menggunakan kamera laptop (0).
    """

    raw_value = input(
        "\nIndeks/URL kamera (0=laptop, 1=USB, atau URL IP) [0]: "
    ).strip()

    if not raw_value:
        return 0

    try:
        return int(raw_value)
    except ValueError:
        return raw_value


def safe_fps(elapsed_seconds: float) -> float:
    """Menghindari pembagian dengan nol pada pengukuran FPS."""

    return 1.0 / max(elapsed_seconds, 1e-9)


# ============================================================
# 3. PENYIMPANAN CSV
# ============================================================

def save_session_csv(
    output_path: Path,
    session_id: str,
    condition: str,
    ground_truth: int,
    log_s1: list[int],
    log_s2: list[int],
    fps_s1_list: list[float],
    fps_s2_list: list[float],
) -> None:
    """
    Menyimpan satu sesi secara atomik setelah validasi
    bahwa seluruh log berisi tepat 300 frame.
    """

    lengths = {
        len(log_s1),
        len(log_s2),
        len(fps_s1_list),
        len(fps_s2_list),
    }

    if lengths != {N_FRAMES}:
        raise ValueError(
            "Penyimpanan dibatalkan: semua log harus berisi "
            f"tepat {N_FRAMES} frame. Panjang log saat ini: "
            f"S1={len(log_s1)}, "
            f"S2={len(log_s2)}, "
            f"FPS_S1={len(fps_s1_list)}, "
            f"FPS_S2={len(fps_s2_list)}."
        )

    # File sementara mencegah CSV final terbentuk setengah jadi.
    temp_path = output_path.with_suffix(".csv.tmp")

    with temp_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "session_id",
            "frame",
            "kondisi",
            "ground_truth",
            "count_s1",
            "count_s2",
            "fps_s1",
            "fps_s2",
        ])

        for index in range(N_FRAMES):
            writer.writerow([
                session_id,
                index,
                condition,
                ground_truth,
                log_s1[index],
                log_s2[index],
                f"{fps_s1_list[index]:.6f}",
                f"{fps_s2_list[index]:.6f}",
            ])

    # Nama file unik sehingga sesi sebelumnya tidak tertimpa.
    temp_path.replace(output_path)


# ============================================================
# 4. BENCHMARK UTAMA
# ============================================================

def main() -> None:

    # --------------------------------------------------------
    # A. INPUT PARAMETER EKSPERIMEN
    # --------------------------------------------------------

    ground_truth = ask_ground_truth()
    condition = ask_condition()
    camera_source = ask_camera_source()

    # ID unik berdasarkan waktu hingga mikrodetik.
    session_id = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    # Folder output otomatis berdasarkan Ground Truth.
    output_dir = (
        BASE_DIR / f"logs_stabilitas_{ground_truth}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = output_dir / (
        f"GT{ground_truth}_{condition}_{session_id}.csv"
    )

    # --------------------------------------------------------
    # B. INISIALISASI MODEL S1 DAN S2
    # --------------------------------------------------------

    print("\n[INFO] Memuat model S1 dan S2...")

    model_s1 = YoloInference(
        str(
            BASE_DIR / "models" / "best_S1_yolov5.onnx"
        ),
        conf_thres=CONF_THRESHOLD,
        iou_thres=IOU_THRESHOLD,
    )

    model_s2 = YoloInference(
        str(
            BASE_DIR / "models" / "best_S2_yolov5_clahe.onnx"
        ),
        conf_thres=CONF_THRESHOLD,
        iou_thres=IOU_THRESHOLD,
    )

    # --------------------------------------------------------
    # C. INISIALISASI WEBCAM
    # --------------------------------------------------------

    cap = cv2.VideoCapture(camera_source)

    if not cap.isOpened():
        cap.release()
        raise RuntimeError(
            f"Gagal membuka kamera dengan sumber: {camera_source}"
        )

    print("\n" + "=" * 60)
    print("PENGATURAN BENCHMARK STABILITAS")
    print(f"Ground Truth       : {ground_truth} peer spring")
    print(f"Kondisi            : {condition}")
    print(f"Target frame       : {N_FRAMES}")
    print(f"Confidence         : {CONF_THRESHOLD}")
    print(f"IoU                : {IOU_THRESHOLD}")
    print(f"Kamera             : {camera_source}")
    print(f"Folder output      : {output_dir}")
    print(f"Nama file output   : {output_path.name}")
    print("=" * 60)

    print(
        "Atur pencahayaan dan pastikan peer spring tidak bergerak."
    )
    print("Tekan 'r' untuk mulai merekam 300 frame.")
    print("Tekan 'q' untuk membatalkan sesi tanpa menyimpan CSV.")

    # --------------------------------------------------------
    # D. INISIALISASI LOG PENGAMATAN
    # --------------------------------------------------------

    log_s1: list[int] = []
    log_s2: list[int] = []

    fps_s1_list: list[float] = []
    fps_s2_list: list[float] = []

    recording = False
    camera_failed = False

    # --------------------------------------------------------
    # E. INISIALISASI JENDELA VISUALISASI
    # --------------------------------------------------------

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL
    )

    cv2.resizeWindow(
        WINDOW_NAME,
        1280,
        480
    )

    try:

        # ====================================================
        # 5. LOOP UTAMA AKUISISI DAN INFERENSI
        # ====================================================

        while True:

            # ------------------------------------------------
            # F. AKUISISI SATU FRAME WEBCAM
            # ------------------------------------------------

            ret, frame = cap.read()

            if not ret or frame is None:
                print(
                    "[ERROR] Frame kamera gagal dibaca. "
                    "Sesi dibatalkan."
                )
                camera_failed = True
                break

            orig_h, orig_w = frame.shape[:2]

            # =================================================
            # G. INFERENSI S1 - YOLOv5s ORIGINAL
            # =================================================

            start_s1 = time.perf_counter()

            boxes_s1 = model_s1.detect(
                frame,
                orig_w,
                orig_h
            )

            elapsed_s1 = (
                time.perf_counter() - start_s1
            )

            fps1 = safe_fps(elapsed_s1)

            # Visualisasi hasil S1.
            frame_s1 = frame.copy()

            for x1, y1, x2, y2, score in boxes_s1:

                p1 = (int(x1), int(y1))
                p2 = (int(x2), int(y2))

                cv2.rectangle(
                    frame_s1,
                    p1,
                    p2,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame_s1,
                    f"{score:.2f}",
                    (int(x1), max(int(y1) - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )

            # =================================================
            # H. PREPROCESSING CLAHE DAN INFERENSI S2
            # =================================================

            start_s2 = time.perf_counter()

            # Preprocessing CLAHE pada kanal L* CIE L*a*b*.
            img_clahe = apply_clahe_lab(frame)

            boxes_s2 = model_s2.detect(
                img_clahe,
                orig_w,
                orig_h
            )

            # Waktu S2 mencakup preprocessing + inferensi.
            elapsed_s2 = (
                time.perf_counter() - start_s2
            )

            fps2 = safe_fps(elapsed_s2)

            # Visualisasi hasil S2.
            frame_s2 = img_clahe.copy()

            for x1, y1, x2, y2, score in boxes_s2:

                p1 = (int(x1), int(y1))
                p2 = (int(x2), int(y2))

                cv2.rectangle(
                    frame_s2,
                    p1,
                    p2,
                    (0, 255, 255),
                    2
                )

                cv2.putText(
                    frame_s2,
                    f"{score:.2f}",
                    (int(x1), max(int(y1) - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1,
                )

            # =================================================
            # I. PENCATATAN DATA SETELAH RECORDING DIAKTIFKAN
            # =================================================

            if recording:

                log_s1.append(
                    len(boxes_s1)
                )

                log_s2.append(
                    len(boxes_s2)
                )

                fps_s1_list.append(fps1)
                fps_s2_list.append(fps2)

            current_count = len(log_s1)

            status = (
                f"REKAM {current_count}/{N_FRAMES}"
                if recording
                else "TEKAN 'r' UNTUK MULAI"
            )

            # ------------------------------------------------
            # J. INFORMASI VISUAL PADA PANEL S1
            # ------------------------------------------------

            cv2.putText(
                frame_s1,
                (
                    f"[{condition}] S1 Original | "
                    f"Deteksi: {len(boxes_s1)} | "
                    f"FPS: {fps1:.1f}"
                ),
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (0, 0, 255),
                2,
            )

            # ------------------------------------------------
            # K. INFORMASI VISUAL PADA PANEL S2
            # ------------------------------------------------

            cv2.putText(
                frame_s2,
                (
                    f"[{condition}] S2 CLAHE | "
                    f"Deteksi: {len(boxes_s2)} | "
                    f"FPS: {fps2:.1f}"
                ),
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (0, 0, 255),
                2,
            )

            cv2.putText(
                frame_s1,
                status,
                (10, orig_h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (0, 165, 255),
                2,
            )

            # =================================================
            # L. RESIZE DAN SPLIT-SCREEN
            # =================================================

            target_h = max(
                1,
                int(
                    (orig_h / orig_w)
                    * TARGET_PANEL_WIDTH
                ),
            )

            frame_s1_resized = cv2.resize(
                frame_s1,
                (TARGET_PANEL_WIDTH, target_h)
            )

            frame_s2_resized = cv2.resize(
                frame_s2,
                (TARGET_PANEL_WIDTH, target_h)
            )

            dashboard = np.hstack(
                (
                    frame_s1_resized,
                    frame_s2_resized
                )
            )

            cv2.imshow(
                WINDOW_NAME,
                dashboard
            )

            # =================================================
            # M. KONTROL KEYBOARD
            # =================================================

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print(
                    "\n[INFO] Sesi dibatalkan oleh pengguna. "
                    "CSV tidak disimpan."
                )
                break

            if key == ord("r") and not recording:

                recording = True

                print(
                    "\n[INFO] Perekaman dimulai. "
                    "Jangan ubah posisi objek/kamera."
                )

            # ------------------------------------------------
            # N. PENGHENTIAN OTOMATIS SETELAH 300 FRAME
            # ------------------------------------------------

            if recording and len(log_s1) >= N_FRAMES:

                print(
                    "\n[INFO] 300 frame berhasil direkam."
                )

                break

    finally:

        # ====================================================
        # 6. PEMBERSIHAN SUMBER DAYA
        # ====================================================

        cap.release()
        cv2.destroyAllWindows()

    # ========================================================
    # 7. VALIDASI KELENGKAPAN HASIL PEREKAMAN
    # ========================================================

    if camera_failed or len(log_s1) != N_FRAMES:

        print(
            f"[WARNING] Tidak ada CSV yang disimpan. "
            f"Frame terekam: {len(log_s1)}/{N_FRAMES}."
        )

        return

    # ========================================================
    # 8. EKSPOR DATA KE CSV
    # ========================================================

    save_session_csv(
        output_path=output_path,
        session_id=session_id,
        condition=condition,
        ground_truth=ground_truth,
        log_s1=log_s1,
        log_s2=log_s2,
        fps_s1_list=fps_s1_list,
        fps_s2_list=fps_s2_list,
    )

    # ========================================================
    # 9. RINGKASAN HASIL PEREKAMAN
    # ========================================================

    print(
        "\n[OK] Perekaman selesai dan CSV berhasil disimpan."
    )

    print(f"File      : {output_path}")
    print(f"Sesi      : {session_id}")
    print(f"GT        : {ground_truth}")
    print(f"Kondisi   : {condition}")
    print(f"Frame     : {len(log_s1)}")

    print(
        f"Rata-rata deteksi S1: {np.mean(log_s1):.3f} | "
        f"MAE S1: "
        f"{np.mean(np.abs(np.array(log_s1) - ground_truth)):.3f}"
    )

    print(
        f"Rata-rata deteksi S2: {np.mean(log_s2):.3f} | "
        f"MAE S2: "
        f"{np.mean(np.abs(np.array(log_s2) - ground_truth)):.3f}"
    )

    print(
        "\nLanjutkan sesi berikutnya dengan kondisi/kepadatan "
        "yang belum direkam."
    )


# ============================================================
# 10. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
