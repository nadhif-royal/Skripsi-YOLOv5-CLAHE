# 🎓 Skripsi: Analisis Pengaruh Preprocessing CLAHE terhadap Akurasi Deteksi Objek Mikro pada Kemasan Transparan Menggunakan YOLOv5

Repositori ini memuat *source code* komputasi lokal, skrip pengujian stabilitas waktu nyata (*real-time*), modul evaluasi latensi CPU, serta artefak hasil pelatihan dari penelitian skripsi berjudul:
**"Analisis Pengaruh Preprocessing CLAHE (Contrast Limited Adaptive Histogram Equalization) terhadap Akurasi Deteksi Objek Mikro pada Kemasan Transparan Menggunakan YOLOv5"**.

Penelitian ini mengevaluasi intervensi prapemrosesan citra di tingkat piksel untuk membantu otomatisasi inspeksi kuantitas komponen mikro (*peer spring*) di dalam kemasan plastik transparan tertutup menggunakan *computer vision* berbasis *Controlled Autonomy* (*Human-in-the-Loop*).

### 🏛️ Identitas Akademik
* **Peneliti:** Nadhif Rif'at Rasendriya (NIM: 235150201111074)
* **Dosen Pembimbing 1:** Tirana Noor Fatyanosa, S.Kom., M.Kom., Ph.D.
* **Dosen Pembimbing 2:** Dr. Eng. Irawati Nurmala Sari, S.Kom., M.Sc.
* **Institusi:** Program Studi Teknik Informatika, Departemen Teknik Informatika, Fakultas Ilmu Komputer, Universitas Brawijaya
* **Tahun:** 2026

---

## 📌 Latar Belakang & Pendekatan Metodologis

Inspeksi kuantitas komponen mikro (*peer spring* berwarna hitam) di dalam kemasan plastik transparan tertutup secara manual rentan terhadap *human error* akibat kelelahan visual. Namun, penerapan detektor objek satu tahap (*one-stage detector*) seperti YOLOv5s pada kemasan plastik fleksibel menghadapi kendala optik berupa pantulan spekular (**glare**) yang mengaburkan gradien tepi (*edges*) komponen dan memicu kegagalan deteksi (*missed detection* / *false negative*).

### 💡 Metodologi dan Arsitektur Usulan
1. **Isolasi Saluran Luminansi & Replikasi Tiga Kanal `(L*, L*, L*)`:**
   Citra masukan berformat BGR ditransformasikan ke ruang warna **CIE L\*a\*b\*** untuk memisahkan saluran *Lightness* ($L^*$) dari saluran kromatik ($a^*$ dan $b^*$). Algoritme **CLAHE** diterapkan secara eksklusif pada saluran $L^*$. Untuk menekan beban komputasi konversi balik serta mengeliminasi bias warna latar belakang, saluran $L^*$ hasil CLAHE direplikasi menjadi tensor tiga kanal identik `(L*, L*, L*)` bertampilan *grayscale* (saluran $a^*$ dan $b^*$ tidak disertakan ke dalam masukan model).
2. **Kalibrasi Parameter CLAHE & Ambang Batas Inferensi:**
   * **Fase Pelatihan (Cloud GPU):** Dilatih selama 150 *epochs* menggunakan parameter CLAHE awal `clipLimit = 4.0` dan `tileGridSize = (8, 8)`.
   * **Fase Kalibrasi Parameter (Grid Search):** Melalui pencarian kisi pada subhimpunan kalibrasi (10 citra sampel), diperoleh konfigurasi operasional terpilih `clipLimit = 3.0` dan `tileGridSize = (4, 4)` yang dibekukan (*freeze*) untuk tahap inferensi.
   * **Konfigurasi Inferensi Lokal:** Menggunakan penyesuaian skala *letterbox* ($640 \times 640$), perbaikan format koordinat matriks `xyxy` ke `xywh` pada *Non-Maximum Suppression* (`iou_thres = 0.45`), serta ambang kepercayaan terkalibrasi (`conf_thres = 0.15` pada pengujian *real-time* dan `conf_thres = 0.50` pada pengujian citra statis).
3. **Evaluasi Berpasangan Secara Berurutan (*Paired Sequential Inference*):**
   Pada pengujian waktu nyata (`benchmark_stability.py`), setiap *frame* sumber yang diambil dari *webcam* dievaluasi secara berpasangan dan berurutan: *frame* asli diinferensikan terlebih dahulu oleh model Skenario 1 (YOLOv5s Original), kemudian *frame* sumber yang sama diproses menggunakan fungsi CLAHE `(L*, L*, L*)` sebelum diinferensikan oleh model Skenario 2 (YOLOv5s + CLAHE).
4. **Paradigma *Human-in-the-Loop* (HITL) — *Controlled Autonomy*:**
   Sistem dirancang sebagai *assistive inspection tool* (Otonomi Tingkat 2) pada `main.py`, di mana layar otomatis mengunci (*freeze*) ketika jumlah deteksi mencapai target kuantitas dan mewajibkan validasi akhir (*approval*) dari operator manusia.

---

## 📂 Struktur Direktori Repositori

Repositori ini disusun secara modular sesuai dengan implementasi pada Bab 5 dan Bab 6 naskah skripsi:

```text
Skripsi-YOLOv5-CLAHE/
│
├── core/
│   ├── __init__.py
│   ├── yolo_infer.py                 # Modul inferensi YOLOv5s (ONNXRuntime CPU, Letterbox, & NMS xywh fix)
│   ├── ssd_infer.py                  # Modul inferensi baseline SSD MobileNet V2 (TFLite Interpreter)
│   ├── ssd_utils.py                  # Modul utilitas decode anchor boxes arsitektur SSD
│   └── preprocess.py                 # Modul prapemrosesan utama apply_clahe_lab (clipLimit=3.0, grid=(4,4))
│
├── ui/
│   ├── __init__.py
│   └── hitl_interface.py             # Antarmuka Human-in-the-Loop (Interrupt State, Tolerance ±1, & Freeze Frame)
│
├── models/                           # Direktori penempatan bobot model lokal (.onnx / .tflite)
│   ├── S1_YOLOv5s_Original/
│   ├── S2_YOLOv5s_CLAHE/
│   ├── S3_SSDMobileNetV2_Original/
│   └── S4_SSDMobileNetV2_CLAHE/
│
├── logs_stabilitas_15/               # Log CSV pengujian real-time 300 frames (Ground Truth 15 komponen)
├── logs_stabilitas_25/               # Log CSV pengujian real-time 300 frames (Ground Truth 25 komponen)
├── logs_stabilitas_30/               # Log CSV pengujian real-time 300 frames (Ground Truth 30 komponen)
│
├── training_results/
│   ├── S1_YOLOv5s_Original/          # Grafik loss, PR Curve, dan Confusion Matrix Skenario 1
│   ├── S2_YOLOv5s_CLAHE/             # Grafik loss, PR Curve, dan Confusion Matrix Skenario 2
│   ├── S3_SSDMobileNetV2_Original/   # Grafik evaluasi dan Confusion Matrix Skenario 3 (Baseline)
│   └── S4_SSDMobileNetV2_CLAHE/      # Grafik evaluasi dan Confusion Matrix Skenario 4 (Baseline + CLAHE)
│
├── main.py                           # Skrip utama simulasi inspeksi industri berbasis HITL (Controlled Autonomy)
├── benchmark_stability.py            # Skrip evaluasi real-time berpasangan (Paired Sequential Inference, 300 frames)
├── aggregate_results.py              # Skrip agregasi metrik Mean, MAE, Akurasi Toleransi ±2, & generator grafik
├── test_inference_time.py            # Skrip benchmark latensi end-to-end CPU murni (10 warm-up + 14x5 repeats)
├── requirements.txt                  # Daftar dependensi pustaka Python lokal
├── LICENSE                           # Lisensi MIT
└── README.md                         # Dokumentasi utama repositori
```

---

## 🔗 Lampiran Repositori Publik (Cloud Assets)

Mengingat adanya batas ukuran berkas pada GitHub, kumpulan dataset uji, bobot model penuh (`.pt`, `.onnx`, `.h5`, `.tflite`), rekaman video pengujian waktu nyata, serta *Google Colab Notebook* diarsipkan secara terpusat melalui tautan berikut:

1. **[📹 Video Dokumentasi Pengujian Waktu Nyata (Real-Time Screen Record)](https://clips.id/Nadhif-PengujianSkripsi-RealTime)**
   Memuat rekaman pengujian *split-screen* berpasangan antara S1 (YOLOv5s Original) dan S2 (YOLOv5s + CLAHE) pada kepadatan **15, 25, dan 30 komponen** di bawah kondisi pencahayaan **Normal, Senter Dekat, dan Senter Jauh**, serta demonstrasi antarmuka HITL (`main.py`) pada Mode Bebas (`Input = 0`) dan Mode Target (`Input = 30`).
2. **[📈 Dataset Log Stabilitas CSV & Grafik Evaluasi MAE (Google Drive)](https://drive.google.com/drive/folders/1hndbMgwL2fO-lGjY_wkUEWH637GvmQEg?usp=sharing)**
   Memuat berkas mentah `.csv` hasil perekaman 300 *frames* per sesi (dilengkapi `session_id` dan *timestamp*), tabel ringkasan hasil agregasi `aggregate_results.py`, serta grafik komparasi MAE untuk GT 15, GT 25, dan GT 30.
3. **[🖼️ Sampel 14 Citra Pengujian Statis / Held-Out Testing Set (Google Drive)](https://drive.google.com/drive/folders/13wG-AGiHUivgMisPS-bNFXvVsS0wy6bY?usp=sharing)**
   Memuat 14 citra uji statis terpisah (Ground Truth = 21 komponen per citra) yang terdiri atas 1 citra pencahayaan normal, 7 citra paparan *flash* ponsel (*diffuse light*), dan 6 citra paparan senter terpusat (*spotlight glare*).
4. **[🧠 Direktori Bobot Model / Model Weights (Google Drive)](https://drive.google.com/drive/folders/1p5D9UnSyJRGhxSn4LNEkaTxk--lECZj7?usp=sharing)**
   Memuat bobot hasil pelatihan format PyTorch (`.pt`), Keras/TensorFlow (`.h5`), ONNX (`.onnx`), dan TFLite (`.tflite`) untuk keempat skenario eksperimen (S1–S4).
5. **[📓 Skrip Pelatihan Cloud GPU — Google Colab Notebooks](https://drive.google.com/drive/folders/1DPiOl_iNzKCL4KGYS1WMPMphwxsC7d7n?usp=sharing)**
   * **Colab Notebook YOLOv5s (Skenario 1 & 2):** [Buka di Google Colab](https://colab.research.google.com/drive/1UshlVQg8fD8HT0ifzf8cPL6JieFDiI4k?usp=sharing)
   * **Colab Notebook SSD MobileNet V2 (Skenario 3 & 4):** [Buka di Google Colab](https://colab.research.google.com/drive/168uMF3CZ34_b07u5n7I3PpvO0yCoiQlJ?usp=sharing)

---

## 📊 Ringkasan Hasil Pengujian & Analisis Empiris

### 1. Evaluasi Pelatihan Arsitektur & Latensi Inferensi CPU (Tabel 6.1)
Pelatihan dilaksanakan pada Google Colab (GPU NVIDIA Tesla T4) selama 150 *epochs* menggunakan 302 citra teranotasi dengan skema evaluasi *in-sample* (`train=val`). Pengukuran waktu inferensi dilakukan secara terpisah pada CPU lokal (Intel Core i5-12450HX, `CPUExecutionProvider` / tanpa delegasi GPU) melalui skrip `test_inference_time.py` dengan **10 putaran pemanasan (*warm-up runs*)** diikuti **5 pengulangan (*repeats*) pada 14 citra uji statis** (total 70 pengukuran *steady-state* per skenario):

| Skenario | Prapemrosesan | Arsitektur | Precision (%) | Recall (%) | mAP@0.5 (%) | Rata-rata Waktu Inferensi CPU per Citra |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **S-1** | Tanpa CLAHE (Citra Asli BGR) | YOLOv5s | 98,3 | 96,4 | 99,2 | **63,85 ms** (15,66 FPS) |
| **S-2** | **CIE L\*a\*b\* `(L*,L*,L*)` + CLAHE** | **YOLOv5s** | **98,1** | **96,6** | **99,2** | **104,01 ms** (9,61 FPS) |
| **S-3** | Tanpa CLAHE (Citra Asli BGR) | SSD MobileNet V2 | 94,8 | 24,4 | 24,4 | **32,79 ms** (30,50 FPS) |
| **S-4** | CIE L\*a\*b\* `(L*,L*,L*)` + CLAHE | SSD MobileNet V2 | 89,4 | 24,8 | 24,6 | **93,66 ms** (10,68 FPS) |

* **Catatan Distribusi Latensi S-2:** Pada metode usulan (S-2), rata-rata waktu inferensi *end-to-end* (mencakup *letterbox resizing*, konversi CIE L\*a\*b\*, eksekusi CLAHE, *forward pass* ONNX, hingga NMS) tercatat sebesar **104,01 ms** (9,61 FPS), dengan nilai **Median (P50) sebesar 120,44 ms** dan **Persentil ke-95 (P95) sebesar 174,92 ms** (pembacaan latensi sekitar ~160 ms pada sesi pendahuluan berada di dalam rentang distribusi P50–P95 ini). Seluruh nilai latensi berada jauh di bawah batas toleransi siklus inspeksi per kemasan ($\le 3\text{ detik}$).
* **Catatan Metodologis `train=val`:** Skor `mAP@0.5` sebesar 99,2% pada S-1 dan S-2 merepresentasikan konvergensi pembelajaran *in-sample* pada distribusi data latih, sehingga pengujian ketangguhan model terhadap *unseen data* dibuktikan secara terpisah melalui pengujian 14 citra statis *held-out* dan pengujian video *real-time*.

---

### 2. Pengujian Silang Ruang Warna pada 14 Citra Uji Statis (Tabel 6.2 & Tabel 6.3)
Pengujian eksploratif pada 14 citra uji statis (*Ground Truth* = 21 komponen per citra; `clipLimit = 3.0`, `tileGridSize = (4, 4)`, `conf_thres = 0.50`) membandingkan representasi masukan tiga kanal hasil CLAHE pada saluran *Value* HSV `(V, V, V)` dan saluran *Lightness* CIE L\*a\*b\* `(L*, L*, L*)`:

| Kategori Iluminasi | Jumlah Citra ($N$) | MAE S1 + `(V,V,V)` | MAE S1 + `(L*,L*,L*)` | MAE S2 + `(V,V,V)` | MAE S2 + `(L*,L*,L*)` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Normal** | 1 | 0,00 | 1,00 | 0,00 | **0,00** |
| **Flash Ponsel (*Diffuse*)** | 7 | 1,43 | 1,29 | 1,43 | **1,29** |
| **Senter (*Spotlight Glare*)** | 6 | 2,83 | 2,50 | 1,33 | **0,50** |
| **Total / Global MAE** | **14** | **1,93** | **1,79** | **1,29** | **0,86** |

---

### 3. Evaluasi Stabilitas Inferensi Waktu Nyata / *Real-Time* (300 Frames per Sesi)
Pengujian waktu nyata dilakukan menggunakan *webcam* eksternal (`conf_thres = 0.15`, `iou_thres = 0.45`) melalui mekanisme *paired sequential inference* selama 300 *frames* per kondisi pencahayaan pada tiga tingkat kepadatan komponen:

#### A. Kepadatan Rendah — Ground Truth 15 Komponen (Tabel 6.6)
*(Perekaman validasi final: 12 November 2026 pukul 09.06 WIB)*

| Kondisi Pencahayaan | Model Arsitektur | Rata-rata Terdeteksi (Mean) | Skor MAE | Akurasi Toleransi $\pm 2$ (%) |
| :--- | :--- | :---: | :---: | :---: |
| **Normal** | S1 (YOLOv5s Asli) | 15,17 | 0,173 | **100,00%** |
| **Normal** | S2 (YOLOv5s + CLAHE) | 15,16 | **0,160** | 99,67% |
| **Senter Dekat (*Diffuse*)** | S1 (YOLOv5s Asli) | 15,91 | **1,070** | **90,33%** |
| **Senter Dekat (*Diffuse*)** | S2 (YOLOv5s + CLAHE) | 16,90 | 1,940 | 70,67% |
| **Senter Jauh (*Spotlight*)** | S1 (YOLOv5s Asli) | 11,38 | 3,763 | 34,67% |
| **Senter Jauh (*Spotlight*)** | S2 (YOLOv5s + CLAHE) | 14,63 | **1,863** | **67,33%** |

#### B. Kepadatan Standar — Ground Truth 25 Komponen (Tabel 6.4)
*(Perekaman: 15 September 2026 pukul 11.17 WIB)*

| Kondisi Pencahayaan | Model Arsitektur | Rata-rata Terdeteksi (Mean) | Skor MAE | Akurasi Toleransi $\pm 2$ (%) |
| :--- | :--- | :---: | :---: | :---: |
| **Normal** | S1 (YOLOv5s Asli) | 26,12 | 1,243 | 91,33% |
| **Normal** | S2 (YOLOv5s + CLAHE) | 25,86 | **0,937** | **95,00%** |
| **Senter Dekat (*Diffuse*)** | S1 (YOLOv5s Asli) | 24,24 | 1,320 | **93,67%** |
| **Senter Dekat (*Diffuse*)** | S2 (YOLOv5s + CLAHE) | 24,10 | **1,203** | 93,33% |
| **Senter Jauh (*Spotlight*)** | S1 (YOLOv5s Asli) | 20,35 | 4,657 | 29,33% |
| **Senter Jauh (*Spotlight*)** | S2 (YOLOv5s + CLAHE) | 21,40 | **3,687** | **37,00%** |

#### C. Kepadatan Tinggi / Uji Stres Oklusi — Ground Truth 30 Komponen (Tabel 6.5)
*(Perekaman: 16 September 2026 pukul 09.21 WIB)*

| Kondisi Pencahayaan | Model Arsitektur | Rata-rata Terdeteksi (Mean) | Skor MAE | Akurasi Toleransi $\pm 2$ (%) |
| :--- | :--- | :---: | :---: | :---: |
| **Normal** | S1 (YOLOv5s Asli) | 29,39 | **0,773** | **100,00%** |
| **Normal** | S2 (YOLOv5s + CLAHE) | 29,25 | 0,907 | 99,67% |
| **Senter Dekat (*Diffuse*)** | S1 (YOLOv5s Asli) | 30,99 | 1,423 | 86,33% |
| **Senter Dekat (*Diffuse*)** | S2 (YOLOv5s + CLAHE) | 30,63 | **1,173** | **90,33%** |
| **Senter Jauh (*Spotlight*)** | S1 (YOLOv5s Asli) | 26,53 | 3,557 | 40,67% |
| **Senter Jauh (*Spotlight*)** | S2 (YOLOv5s + CLAHE) | 27,25 | **2,827** | **51,33%** |

### 🔍 Kesimpulan Utama Eksperimen
* **Efektivitas pada *Spotlight Glare* Terpusat:** Prapemrosesan CLAHE pada saluran $L^*$ secara konsisten menurunkan galat penghitungan (MAE) pada kondisi **Senter Jauh (*Spotlight*)** di seluruh tingkat kepadatan objek: turun dari **3,763 menjadi 1,863** pada GT 15, dari **4,657 menjadi 3,687** pada GT 25, dan dari **3,557 menjadi 2,827** pada GT 30.
* **Sifat Kondisional & *Trade-Off* Prapemrosesan:** CLAHE bukan merupakan peningkat akurasi yang berlaku secara universal. Pada kepadatan rendah (GT 15) dengan pencahayaan menyebar jarak dekat (**Senter Dekat**) maupun kondisi **Normal** pada GT 30, pemerataan kontras lokal berisiko memicu amplifikasi tekstur nonobjek (*over-enhancement* pada lipatan plastik atau bayangan) sehingga meningkatkan nilai MAE (dari 1,070 menjadi 1,940 pada GT 15 Senter Dekat) di samping memberikan tambahan beban komputasi CPU sekitar ~40 ms per citra.

---

## 🛠️ Panduan Instalasi & Eksekusi Lokal

Bagi peneliti atau penguji yang ingin menjalankan ulang simulasi dan evaluasi di lingkungan komputasi lokal:

1. **Kloning repositori ini:**
   ```bash
   git clone https://github.com/nadhif-royal/Skripsi-YOLOv5-CLAHE.git
   cd Skripsi-YOLOv5-CLAHE
   ```

2. **Instalasi pustaka pendukung (disarankan menggunakan Python 3.10 / 3.11):**
   ```bash
   pip install -r requirements.txt
   ```

3. **Penyiapan berkas bobot model (*Model Weights*):**
   Unduh berkas model `.onnx` dan `.tflite` dari tautan Google Drive pada bagian **Lampiran Repositori Publik**, lalu tempatkan pada subdirektori yang sesuai di dalam folder `models/`.

4. **Menjalankan modul pengujian:**
   ```bash
   # 1. Simulasi stasiun inspeksi industri dengan antarmuka Human-in-the-Loop (HITL)
   python main.py

   # 2. Perekaman benchmark stabilitas berpasangan S1 vs S2 secara berurutan (300 frames)
   python benchmark_stability.py

   # 3. Agregasi log CSV (GT 15, 25, 30) untuk menghitung Mean, MAE, Akurasi Toleransi ±2, & grafik
   python aggregate_results.py

   # 4. Pengujian latensi inferensi end-to-end murni CPU (ms & FPS)
   python test_inference_time.py
   ```

---

## 📜 Lisensi & Sitasi Akademik

Seluruh kode sumber dalam repositori ini didistribusikan di bawah **Lisensi MIT** (lihat berkas `LICENSE` untuk rincian lebih lanjut). Apabila Anda menggunakan kode program, dataset, atau temuan dari repositori ini sebagai rujukan akademik, silakan sitasi karya skripsi berikut:

> Rasendriya, N.R., 2026. *Analisis Pengaruh Preprocessing CLAHE (Contrast Limited Adaptive Histogram Equalization) terhadap Akurasi Deteksi Objek Mikro pada Kemasan Transparan Menggunakan YOLOv5*. Skripsi. Program Studi Teknik Informatika, Fakultas Ilmu Komputer, Universitas Brawijaya.