import cv2
import time
import os
from core.preprocess import apply_clahe_lab
from core.yolo_infer import YoloInference
from ui.hitl_interface import HITLInterface

SAVE_DIR = "hasil_inspeksi"


def input_target():
    while True:
        try:
            target_input = input(
                "Masukkan target jumlah peer spring (mis. 20, 25, 30) "
                "atau ketik 0 jika tidak tahu jumlah pastinya: "
            )
            target_count = int(target_input)
            if target_count >= 0:
                return target_count
            print("Target tidak boleh negatif!")
        except ValueError:
            print("Masukkan angka yang valid!")


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
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    print(f"✅ Kamera berhasil dibuka: {sumber}")
    return cap


def main():
    print("=" * 50)
    print("SISTEM INSPEKSI PEER SPRING (REAL-TIME)")
    print("Skripsi Nadhif Rif'at Rasendriya")
    print("=" * 50)

    os.makedirs(SAVE_DIR, exist_ok=True)

    target_count = input_target()
    mode_bebas = (target_count == 0)

    sumber = pilih_sumber_kamera()
    cap = buka_kamera(sumber)
    while cap is None:
        print("Coba pilih ulang sumber kamera.")
        sumber = pilih_sumber_kamera()
        cap = buka_kamera(sumber)

    print("\n[INFO] Memuat Model AI (Harap Tunggu)...")
    model_s1 = YoloInference('models/best_S1_yolov5.onnx', conf_thres=0.15, iou_thres=0.45)
    model_s2 = YoloInference('models/best_S2_yolov5_clahe.onnx', conf_thres=0.15, iou_thres=0.45)
    ui = HITLInterface()

    current_scenario = 1  # mulai dari YOLOv5 Original
    scenario_names = {
        1: "S1: YOLOv5 (Original)",
        2: "S2: YOLOv5 + CLAHE (USULAN)",
    }

    is_frozen = False
    display_frame = None

    print("\n[INFO] Sistem Siap! Membuka Kamera...")
    print("Tekan '2' untuk aktifkan CLAHE, '1' untuk kembali ke Original.")
    if mode_bebas:
        print("Mode BEBAS (target tidak ditentukan): [S] simpan screenshot, [Q] keluar.")
    else:
        print(f"Mode TARGET ({target_count} part): freeze otomatis saat tepat, [Y]/[N] saat freeze, [Q] keluar.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Gagal membaca frame dari kamera.")
            break

        if not is_frozen:
            orig_h, orig_w = frame.shape[:2]
            start_time = time.time()

            if current_scenario == 1:
                boxes = model_s1.detect(frame, orig_w, orig_h)
            else:  # current_scenario == 2
                img_clahe = apply_clahe_lab(frame)
                boxes = model_s2.detect(img_clahe, orig_w, orig_h)

            fps = 1.0 / (time.time() - start_time)

            display_frame, trigger_freeze = ui.draw(
                frame.copy(), boxes, fps, scenario_names[current_scenario], target_count
            )

            if trigger_freeze and not mode_bebas:
                is_frozen = True

        if display_frame is not None:
            cv2.imshow("Skripsi Nadhif - Sistem Inspeksi Peer Spring", display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif not is_frozen:
            if key == ord('1'):
                current_scenario = 1
            elif key == ord('2'):
                current_scenario = 2
            elif mode_bebas and key == ord('s'):
                filename = os.path.join(SAVE_DIR, f"screenshot_{int(time.time())}.jpg")
                cv2.imwrite(filename, display_frame)
                print(f"[*] Screenshot disimpan: {filename}")

        elif is_frozen:
            if key == ord('y'):
                filename = os.path.join(SAVE_DIR, f"validasi_{int(time.time())}.jpg")
                cv2.imwrite(filename, display_frame)
                print(f"[HITL LOG] Inspeksi Divalidasi OK. Target tercapai: {target_count}. Disimpan: {filename}")
                is_frozen = False
            elif key == ord('n'):
                print("[HITL LOG] Inspeksi Ditolak. Mengulangi proses...")
                is_frozen = False

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()