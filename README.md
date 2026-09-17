# 🎓 Skripsi: YOLOv5s + CLAHE untuk Deteksi Objek Mikro

Repositori ini memuat *source code* komputasi lokal, skrip *notebook* pelatihan, model bobot (*weights*), serta hasil evaluasi metrik dari penelitian skripsi berjudul:
**"Analisis Pengaruh Preprocessing CLAHE (Contrast Limited Adaptive Histogram Equalization) terhadap Akurasi Deteksi Objek Mikro pada Kemasan Transparan Menggunakan YOLOv5"**.

Penelitian ini dikembangkan untuk mengotomatisasi inspeksi kuantitas komponen mikro (*peer spring*) di lini produksi industri manufaktur menggunakan *computer vision*, dengan mengadopsi pendekatan *Controlled Autonomy* (mekanisme *Human-in-the-Loop*).

### 🏛️ Identitas Akademik
* **Peneliti:** Nadhif Rif'at Rasendriya (NIM: 235150201111074)
* **Dosen Pembimbing 1:** Tirana Noor Fatyanosa, S.Kom., M.Kom., Ph.D.
* **Dosen Pembimbing 2:** Dr. Eng. Irawati Nurmala Sari, S.Kom., M.Sc.
* **Institusi:** Program Studi Teknik Informatika, Fakultas Ilmu Komputer, Universitas Brawijaya
* **Tahun:** 2026

---

## 📌 Latar Belakang & Solusi
Implementasi algoritme pendeteksi objek (*one-stage detector*) seperti YOLO di lingkungan riil industri manufaktur sering kali terhambat oleh disrupsi anomali optik. Material kemasan plastik transparan fleksibel memantulkan cahaya lampu secara terpusat, memicu pantulan spekular (**glare**). Efek *glare* ekstrem ini menyebabkan *over-saturation* dan *pixel clipping* pada matriks digital yang secara radikal mengaburkan garis tepi (*edges*) dan fitur geometri dari objek berukuran mikro yang saling bertumpuk di baliknya.

**💡 Solusi yang Diusulkan:**
Penelitian ini mengusulkan intervensi di tingkat piksel melalui algoritme prapemrosesan **CLAHE**. Operasi pemerataan histogram ini dieksekusi secara eksklusif pada saluran *Lightness* (L\*) di dalam ruang warna **CIE L\*a\*b\*** untuk mereduksi efek *glare* ekstrem dan merestorasi fitur tepi komponen tanpa mendistorsi integritas warna asli objek, sebelum matriks tersebut diumpankan ke dalam jaringan konvolusional YOLOv5s.

---

## 📂 Struktur Repositori

Repositori ini disusun secara modular berdasarkan skenario komparasi pengujian sebagai berikut:

```text
Skripsi-YOLOv5-CLAHE/
│
├── core/
│   ├── yolo_infer.py         # Modul utama arsitektur inferensi YOLOv5s ONNX Runtime
│   ├── ssd_infer.py          # Modul arsitektur inferensi baseline SSD TFLite Interpreter
│   ├── ssd_utils.py          # Modul penyangga anchors & decode boxes untuk arsitektur SSD
│   └── preprocess.py         # Modul fungsi prapemrosesan utama `apply_clahe_lab`
│
├── ui/
│   └── hitl_interface.py     # Modul antarmuka Human-in-the-Loop (Interrupt State & Freeze Frame)
│
├── training_results/
│   ├── S1_YOLOv5s_Original/  # Evaluasi Skenario 1 (Loss, PR Curve, Confusion Matrix)
│   ├── S2_YOLOv5s_CLAHE/     # Evaluasi Skenario 2 (Loss, PR Curve, Confusion Matrix)
│   ├── S3_SSDMobileNetV2_Original/ # Evaluasi Skenario 3 (Baseline Konvensional)
│   └── S4_SSDMobileNetV2_CLAHE/    # Evaluasi Skenario 4 (Baseline Ter-CLAHE)
│
├── main.py                   # Skrip komputasi utama untuk simulasi inspeksi industri
├── test_inference_time.py    # Modul uji benchmark latensi/waktu inferensi murni CPU (ms)
├── benchmark_stability.py    # Modul pengujian split-screen (S1 vs S2) waktu nyata (300 frames)
├── aggregate_results.py      # Kalkulator Mean Absolute Error (MAE) dari output stabilitas
├── LICENSE
└── README.md

```

*(Catatan: Mengingat batasan penyimpanan GitHub, beberapa aset besar seperti video *screen record*, matriks *Training Log*, file *Jupyter Notebook*, dan cadangan *weights* didistribusikan secara eksternal melalui layanan *Cloud* pada tautan di bawah).*

---

## 🔗 Lampiran Repositori Publik (Cloud Assets)

Sebagai upaya mendukung transparansi riset (*open science*) dan kemudahan reproduksi eksperimen, seluruh instrumen tambahan dapat diakses melalui tautan berikut:

1. **[📹 Video Demo Pengujian Waktu Nyata (Real-Time)](https://www.google.com/search?q=https://clips.id/Nadhif-PengujianSkripsi-RealTime)**
Memuat rekaman layar pengujian komparasi model (YOLOv5 Asli vs YOLOv5 CLAHE) menghadapi variasi pencahayaan (Normal, Senter Dekat, Senter Jauh) pada kepadatan objek 15, 25, dan 30 *part*. Termasuk demonstrasi sistem `main.py` saat simulasi industri sesungguhnya (Target Tidak Ditentukan vs Target 30).
2. **[📈 Metrik Log & Visualisasi MAE (Google Drive)](https://drive.google.com/drive/folders/1hndbMgwL2fO-lGjY_wkUEWH637GvmQEg?usp=sharing)**
Direktori berisi arsip `.csv` rekam jejak jumlah deteksi absolut per bingkai (300 *frames*), kalkulasi matematis *Mean Absolute Error* (MAE), persentase toleransi, serta cuplikan terminal latensi komputasi inferensi murni CPU.
3. **[🧠 Direktori Penuh Model Weights (Google Drive)](https://drive.google.com/drive/folders/1p5D9UnSyJRGhxSn4LNEkaTxk--lECZj7?usp=sharing)**
Mengingat batasan ukuran unggahan GitHub, seluruh hasil model terlatih yang mencakup format *Native* (`.pt`, `.h5`) dan format terekspor (`.onnx`, `.tflite`) untuk keempat skenario pengujian diarsip dalam direktori ini.
4. **[📓 Source Code Cloud Training - Colab Notebook (Google Drive)](https://drive.google.com/drive/folders/1DPiOl_iNzKCL4KGYS1WMPMphwxsC7d7n?usp=sharing)**
Backup berkas skrip komputasi berformat `.ipynb` untuk pelacakan alur data secara absolut.
* *Preview Colab YOLOv5s:* [Link Skenario 1 & 2](https://colab.research.google.com/drive/1UshlVQg8fD8HT0ifzf8cPL6JieFDiI4k?usp=sharing)
* *Preview Colab SSD MobileNet V2:* [Link Skenario 3 & 4](https://colab.research.google.com/drive/168uMF3CZ34_b07u5n7I3PpvO0yCoiQlJ?usp=sharing)



---

## 📊 Ringkasan Hasil Pengujian

1. **Fase Pelatihan (Cloud GPU Computing):**
* **Skenario 1 (YOLOv5s Asli):** Mencapai konvergensi mAP@0.5 sebesar 99.2%.
* **Skenario 2 (YOLOv5s + CLAHE L*a*b*):** Mencapai konvergensi yang identik (mAP@0.5 99.2%) diiringi *Precision* 98.1% dan *Recall* 96.6%.
* **Skenario 3 & 4 (SSD MobileNet V2):** Mengalami kegagalan ekstraksi spasial (*under-detection*) fatal dengan *Recall* tertahan di kisaran 24%.


2. **Inferensi Waktu Nyata / Real-Time (CPU Only):**
* Meskipun mAP S-1 dan S-2 identik saat dilatih di lingkungan terkontrol, ketangguhan *State-of-the-Art* sesungguhnya diuji pada inferensi *live* menggunakan *webcam*.
* Saat sensor kamera dihadapkan pada **kondisi pantulan spekular ekstrem (Senter Jauh/Spotlight)**, Skenario 1 (YOLOv5s Asli) mengalami lonjakan *Mean Absolute Error* (MAE) hingga **4.657** akibat *pixel clipping*.
* Sebaliknya, metode usulan Skenario 2 (YOLOv5s + CLAHE) terbukti krusial menyelamatkan jaringan saraf dari kebutaan dengan menekan angka MAE secara absolut menjadi **3.687**, serta menjaga rasio akurasi deteksi ganda di lingkungan padat objek (30 *part*).


3. **Efisiensi Komputasi:**
* Kecepatan inferensi jaringan saraf usulan (S-2) menggunakan CPU eksekutor terukur di angka **~160.37 ms** per siklus penuh (termasuk *letterbox resizing*, konversi CLAHE L*a*b*, dan fungsi *Non-Maximum Suppression*). Waktu ini masih berada jauh di bawah toleransi industri manufaktur (maks. 3 detik per siklus inspeksi).



---

## 🛠️ Instalasi & Eksekusi Lokal

Bagi peneliti atau pengembang yang ingin mereplikasi eksperimen evaluasi *real-time* ini di stasiun kerja lokal:

1. Kloning repositori ini:
```bash
git clone [https://github.com/nadhif-royal/Skripsi-YOLOv5-CLAHE.git](https://github.com/nadhif-royal/Skripsi-YOLOv5-CLAHE.git)
cd Skripsi-YOLOv5-CLAHE

```


2. Pastikan lingkungan virtual Python terkonfigurasi dengan versi pustaka yang kompatibel:
```bash
pip install -r requirements.txt

```


3. Unduh berkas `best.onnx` dari Lampiran Drive di atas dan letakkan di dalam folder `models/S2_YOLOv5s_CLAHE/`.
4. Jalankan pengujian:
```bash
python main.py                     # Untuk simulasi antarmuka pabrik dengan HITL
python benchmark_stability.py      # Untuk menguji MAE antara 2 model secara paralel

```



---

## 📜 Lisensi & Publikasi Akademik

Seluruh kode sumber didistribusikan di bawah **Lisensi MIT** (Lihat `LICENSE` untuk rincian).
Pengutipan komersial, replikasi data, maupun modifikasi turunan dalam ranah akademik sangat diizinkan dengan tetap merujuk pada hak cipta pelaporan skripsi kepada Universitas Brawijaya.
