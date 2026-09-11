# 🎓 Skripsi: YOLOv5s + CLAHE untuk Deteksi Objek Mikro

Repositori ini memuat berkas pemodelan, bobot jaringan (*weights*), dan hasil evaluasi metrik dari penelitian skripsi berjudul **"Analisis Pengaruh Preprocessing CLAHE (Contrast Limited Adaptive Histogram Equalization) terhadap Akurasi Deteksi Objek Mikro pada Kemasan Transparan Menggunakan YOLOv5"**.

Penelitian ini dikembangkan untuk mengotomatisasi inspeksi kuantitas komponen *peer spring* di lini produksi industri manufaktur menggunakan *computer vision*, dengan pendekatan *Controlled Autonomy* (Human-in-the-Loop).

### 🏛️ Identitas Akademik
* **Peneliti:** Nadhif Rif'at Rasendriya (NIM: 235150201111074)
* **Dosen Pembimbing 1:** Tirana Noor Fatyanosa, S.Kom., M.Kom., Ph.D.
* **Dosen Pembimbing 2:** Dr. Eng. Irawati Nurmala Sari, S.Kom., M.Sc.
* **Institusi:** Program Studi Teknik Informatika, Fakultas Ilmu Komputer, Universitas Brawijaya
* **Tahun:** 2026

---

## 📌 Latar Belakang & Solusi
Implementasi algoritme deteksi objek (seperti YOLOv5) di lingkungan industri manufaktur sering kali terhambat oleh pantulan cahaya spekular (**glare**) dari material kemasan plastik transparan. Efek *glare* (terutama dari *spotlight* atau lampu industri) memicu *over-saturation* dan *pixel clipping* yang mengaburkan garis tepi (*edges*) objek berukuran mikro yang saling bertumpuk.

**Solusi yang Diusulkan:**
Penelitian ini mengimplementasikan metode prapemrosesan **CLAHE** secara eksklusif pada saluran *Lightness* (L*) di dalam ruang warna **CIE L\*a\*b\***. Modul ini secara adaptif meratakan kontras piksel yang terkena *glare* ekstrem dan merestorasi fitur tepi komponen sebelum matriks citra diumpankan ke arsitektur YOLOv5s.

---

## 📂 Struktur Repositori

Repositori ini disusun berdasarkan skenario pengujian komparasi (Ablation Study) sebagai berikut:

```text
Skripsi-YOLOv5-CLAHE/
│
├── models/
│   ├── S1_YOLOv5s_Original/    # Bobot model dasar (tanpa CLAHE) berformat .pt dan .onnx
│   └── S2_YOLOv5s_CLAHE/       # Bobot model usulan (CIE L*a*b* + CLAHE) berformat .pt dan .onnx
│
├── training_results/
│   ├── S1_YOLOv5s_Original/    # Metrik pelatihan Skenario 1 (Loss, PR Curve, Confusion Matrix)
│   └── S2_YOLOv5s_CLAHE/       # Metrik pelatihan Skenario 2 (Loss, PR Curve, Confusion Matrix)
│
├── LICENSE
└── README.md

```

---

## 📊 Hasil Pengujian (Fase Pelatihan)

Kedua model telah dilatih selama 150 *epochs* dengan resolusi citra 640x640 menggunakan GPU NVIDIA Tesla T4.

* **Skenario 1 (Baseline - Tanpa CLAHE):** Mencapai konvergensi optimal dengan tingkat mAP@0.5 sebesar 99.2%.
* **Skenario 2 (Metode Usulan - CLAHE L*a*b*):** Mencapai metrik yang setara dengan mAP@0.5 sebesar 99.2%, diiringi metrik *Precision* 98.1% dan *Recall* 96.6%.

> **Catatan Analitik:** Meskipun skor mAP saat pelatihan terlihat identik, keunggulan metode usulan (Skenario 2) terbukti secara mutlak pada saat pengujian inferensi menghadapi **kondisi iluminasi ekstrem (Senter Industri / Spotlight)**. Pada kondisi tersebut, model *baseline* (S1) mengalami kegagalan ekstraksi dan *missed detection* (False Negative tinggi), sementara model dengan prapemrosesan CLAHE berhasil mempertahankan lokalisasi akurat berkat restorasi fitur piksel.

---

## 🚀 Rencana Pembaruan Mendatang (To-Do List)

Penelitian ini masih terus berjalan. Repositori akan diperbarui dengan modul-modul berikut:

* [ ] Penambahan Skenario 3 & 4 (Model Baseline SSD MobileNet V2).
* [ ] Kode sumber inferensi waktu nyata (`main.py`) menggunakan OpenCV dan ONNXRuntime CPU.
* [ ] Implementasi antarmuka GUI untuk sistem peringatan *Controlled Autonomy* (Otonomi Tingkat 2).

---

## 📜 Lisensi

Didistribusikan di bawah Lisensi MIT. Lihat `LICENSE` untuk informasi lebih lanjut. Hak cipta akademik tetap tunduk pada regulasi Fakultas Ilmu Komputer, Universitas Brawijaya.
