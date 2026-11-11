
"""
aggregate_results.py

Fungsi:
1. Membaca CSV pengujian stabilitas GT 15, 25, dan 30.
2. Memvalidasi jumlah frame, kondisi, Ground Truth, dan nilai numerik.
3. Menghitung metrik secara terpisah untuk setiap GT, kondisi, dan model.
4. Menyimpan tabel hasil agregasi dan grafik MAE individual.
5. TIDAK menghitung pooled MAE lintas kepadatan.

Struktur folder:
    logs_stabilitas_15/
    logs_stabilitas_25/
    logs_stabilitas_30/

Catatan:
- GT 25 dan GT 30 wajib tersedia pada tahap audit awal.
- GT 15 bersifat opsional sampai tiga sesi pengujiannya selesai.
- Setelah GT 15 selesai, ubah REQUIRED_GT menjadi (15, 25, 30)
  agar ketiga kelompok wajib tersedia pada proses final.
- CSV lama tanpa kolom session_id tetap didukung.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==================================================
# 1. KONFIGURASI
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

FOLDERS = {
    15: BASE_DIR / "logs_stabilitas_15",
    25: BASE_DIR / "logs_stabilitas_25",
    30: BASE_DIR / "logs_stabilitas_30",
}

# Sementara: GT 25 dan GT 30 wajib.
# GT 15 otomatis diproses jika tiga CSV-nya sudah tersedia.
# Setelah pengujian GT 15 selesai, ubah menjadi (15, 25, 30).
REQUIRED_GT = (15, 25, 30)

EXPECTED_CONDITIONS = (
    "Normal",
    "Senter_Dekat",
    "Senter_Jauh",
)

N_FRAMES = 300
TOLERANCE = 2

OUTPUT_DIR = BASE_DIR / "hasil_agregasi"

REQUIRED_COLUMNS = {
    "frame",
    "kondisi",
    "ground_truth",
    "count_s1",
    "count_s2",
    "fps_s1",
    "fps_s2",
}

NUMERIC_COLUMNS = (
    "frame",
    "ground_truth",
    "count_s1",
    "count_s2",
    "fps_s1",
    "fps_s2",
)

MODEL_COLUMNS = {
    "S1_Original": "count_s1",
    "S2_CLAHE_LAB": "count_s2",
}


# ==================================================
# 2. NORMALISASI LABEL KONDISI
# ==================================================

def normalize_condition(value):
    """Menyeragamkan penulisan label kondisi pencahayaan."""

    normalized = str(value).strip().lower().replace(" ", "_")

    aliases = {
        "normal": "Normal",
        "senter_dekat": "Senter_Dekat",
        "senter_jauh": "Senter_Jauh",
    }

    return aliases.get(normalized)


# ==================================================
# 3. MEMBACA DAN MEMVALIDASI CSV PER GROUND TRUTH
# ==================================================

def load_density_data(ground_truth, folder):
    """Membaca tiga CSV untuk satu GT dan memvalidasi isinya."""

    if not folder.exists():
        raise FileNotFoundError(
            f"Folder tidak ditemukan: {folder}"
        )

    csv_files = sorted(folder.glob("*.csv"))

    if len(csv_files) == 0:
        raise FileNotFoundError(
            f"Tidak ada CSV di folder: {folder}"
        )

    if len(csv_files) != len(EXPECTED_CONDITIONS):
        raise ValueError(
            f"Folder {folder.name} harus berisi tepat "
            f"{len(EXPECTED_CONDITIONS)} CSV, satu untuk setiap "
            f"kondisi. Ditemukan: {len(csv_files)}."
        )

    data_frames = []
    found_conditions = set()

    print(f"\nMemeriksa folder: {folder.name}")

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)

        # 3A. Memeriksa kelengkapan kolom.
        missing_columns = REQUIRED_COLUMNS - set(df.columns)

        if missing_columns:
            raise ValueError(
                f"{csv_path.name}: kolom wajib tidak ditemukan: "
                f"{sorted(missing_columns)}"
            )

        # 3B. Menyeragamkan label kondisi dalam CSV.
        normalized_conditions = df["kondisi"].map(normalize_condition)

        if normalized_conditions.isna().any():
            invalid_values = df.loc[
                normalized_conditions.isna(), "kondisi"
            ].unique().tolist()
            raise ValueError(
                f"{csv_path.name}: label kondisi tidak dikenal: "
                f"{invalid_values}"
            )

        df["kondisi"] = normalized_conditions
        unique_conditions = df["kondisi"].unique()

        if len(unique_conditions) != 1:
            raise ValueError(
                f"{csv_path.name}: satu CSV harus hanya memuat "
                "satu kondisi pencahayaan."
            )

        condition = unique_conditions[0]

        if condition in found_conditions:
            raise ValueError(
                f"Duplikasi kondisi {condition} ditemukan di "
                f"folder {folder.name}. Pastikan hanya ada satu "
                "CSV terpilih untuk setiap kondisi."
            )

        found_conditions.add(condition)

        # 3C. Mengonversi seluruh kolom numerik.
        for column in NUMERIC_COLUMNS:
            df[column] = pd.to_numeric(
                df[column], errors="raise"
            )

        numeric_values = df[list(NUMERIC_COLUMNS)].to_numpy(
            dtype=float
        )

        if not np.isfinite(numeric_values).all():
            raise ValueError(
                f"{csv_path.name}: ditemukan nilai NaN atau infinity."
            )

        # 3D. Memastikan Ground Truth konsisten dan sesuai folder.
        gt_values = df["ground_truth"].unique()

        if len(gt_values) != 1 or not np.isclose(
            gt_values[0], ground_truth
        ):
            raise ValueError(
                f"{csv_path.name}: Ground Truth dalam CSV harus "
                f"selalu {ground_truth}. Ditemukan: {gt_values}."
            )

        # 3E. Memastikan tepat 300 baris/frame.
        if len(df) != N_FRAMES:
            raise ValueError(
                f"{csv_path.name}: ditemukan {len(df)} baris, "
                f"seharusnya {N_FRAMES}."
            )

        # Nomor frame wajib berupa bilangan bulat 0 sampai 299.
        frame_values = df["frame"].to_numpy(dtype=float)

        if not np.allclose(frame_values, np.round(frame_values)):
            raise ValueError(
                f"{csv_path.name}: nomor frame harus berupa bilangan bulat."
            )

        df["frame"] = df["frame"].astype(int)

        if sorted(df["frame"].tolist()) != list(range(N_FRAMES)):
            raise ValueError(
                f"{csv_path.name}: nomor frame harus lengkap dan unik "
                f"dari 0 sampai {N_FRAMES - 1}."
            )

        # 3F. Jumlah deteksi harus berupa bilangan bulat non-negatif.
        for column in ("count_s1", "count_s2"):
            values = df[column].to_numpy(dtype=float)

            if (values < 0).any() or not np.allclose(
                values, np.round(values)
            ):
                raise ValueError(
                    f"{csv_path.name}: {column} harus berupa "
                    "bilangan bulat non-negatif."
                )

            df[column] = df[column].astype(int)

        # 3G. FPS harus positif.
        for column in ("fps_s1", "fps_s2"):
            if (df[column] <= 0).any():
                raise ValueError(
                    f"{csv_path.name}: nilai {column} harus lebih besar "
                    "dari nol."
                )

        # 3H. Mendukung CSV baru (memiliki session_id)
        # dan CSV lama (belum memiliki session_id).
        if "session_id" in df.columns:
            session_ids = df["session_id"].dropna().astype(str).unique()

            if len(session_ids) != 1 or df["session_id"].isna().any():
                raise ValueError(
                    f"{csv_path.name}: satu CSV harus memiliki tepat "
                    "satu session_id yang konsisten."
                )

            df["session_id"] = str(session_ids[0])
        else:
            # Nama file digunakan sebagai identitas sesi untuk CSV lama.
            df["session_id"] = csv_path.stem

        # Menambahkan metadata untuk pelacakan sumber data.
        df["density"] = int(ground_truth)
        df["source_file"] = csv_path.name

        data_frames.append(df)

        print(
            f"  OK: {csv_path.name} | "
            f"GT={ground_truth} | "
            f"Kondisi={condition} | "
            f"Frame={len(df)}"
        )

    # 3I. Memastikan tiga kondisi telah tersedia seluruhnya.
    expected = set(EXPECTED_CONDITIONS)

    if found_conditions != expected:
        missing_conditions = expected - found_conditions
        raise ValueError(
            f"Folder {folder.name} belum lengkap. "
            f"Kondisi yang hilang: {sorted(missing_conditions)}"
        )

    return pd.concat(data_frames, ignore_index=True)


# ==================================================
# 4. MEMUAT SEMUA DATA YANG TERSEDIA
# ==================================================

def load_all_data():
    """Memuat GT wajib dan GT opsional jika tiga CSV tersedia."""

    all_data = []

    for ground_truth, folder in FOLDERS.items():
        csv_exists = (
            folder.exists()
            and any(folder.glob("*.csv"))
        )

        if ground_truth in REQUIRED_GT or csv_exists:
            df = load_density_data(ground_truth, folder)
            all_data.append(df)
        else:
            print(
                f"\n[INFO] GT {ground_truth} belum tersedia. "
                "Kelompok ini dilewati sementara."
            )

    if not all_data:
        raise RuntimeError(
            "Tidak ada data CSV yang dapat diproses."
        )

    return pd.concat(all_data, ignore_index=True)


# ==================================================
# 5. MENGUBAH DATA KE FORMAT PANJANG DAN MENGHITUNG ERROR
# ==================================================

def create_long_data(df):
    """Memisahkan data S1/S2 menjadi satu baris per model per frame."""

    parts = []

    for model_name, count_column in MODEL_COLUMNS.items():
        temp = df[
            [
                "session_id",
                "source_file",
                "frame",
                "kondisi",
                "ground_truth",
                "density",
                count_column,
            ]
        ].copy()

        temp = temp.rename(
            columns={count_column: "count_detected"}
        )

        temp["model"] = model_name
        parts.append(temp)

    long_df = pd.concat(parts, ignore_index=True)

    # Error bertanda: positif = overcount, negatif = undercount.
    long_df["error"] = (
        long_df["count_detected"] - long_df["ground_truth"]
    )

    # MAE menggunakan nilai absolut error per frame.
    long_df["absolute_error"] = long_df["error"].abs()

    # RMSE menggunakan kuadrat error per frame.
    long_df["squared_error"] = long_df["error"] ** 2

    # Prediksi tepat sama dengan Ground Truth.
    long_df["exact"] = long_df["absolute_error"] == 0

    # Prediksi masih dalam toleransi ±2 komponen.
    long_df["within_tolerance"] = (
        long_df["absolute_error"] <= TOLERANCE
    )

    return long_df


# ==================================================
# 6. MENGHITUNG METRIK SECARA INDIVIDUAL PER GT
# ==================================================

def calculate_individual_metrics(long_df):
    """Menghitung metrik per density, kondisi, dan model."""

    results = []

    grouped = long_df.groupby(
        ["density", "kondisi", "model"],
        sort=True,
    )

    for (density, condition, model), group in grouped:
        errors = group["absolute_error"].to_numpy(dtype=float)
        squared_errors = group["squared_error"].to_numpy(dtype=float)
        counts = group["count_detected"].to_numpy(dtype=float)

        if len(group) != N_FRAMES:
            raise ValueError(
                f"Kelompok GT {density}, {condition}, {model} "
                f"memiliki {len(group)} frame, seharusnya {N_FRAMES}."
            )

        results.append({
            "ground_truth": int(density),
            "kondisi": condition,
            "model": model,
            "n_frames": len(group),
            "mean_detected": counts.mean(),
            "MAE": errors.mean(),
            "RMSE": np.sqrt(squared_errors.mean()),
            "std_dev": counts.std(ddof=0),
            "acc_exact_pct": group["exact"].mean() * 100,
            "acc_tol2_pct": group["within_tolerance"].mean() * 100,
        })

    summary = pd.DataFrame(results)

    return summary.sort_values(
        ["ground_truth", "kondisi", "model"]
    ).reset_index(drop=True)


# ==================================================
# 7. MEMBUAT GRAFIK MAE INDIVIDUAL PER KEPADATAN
# ==================================================

def create_individual_charts(summary):
    """Membuat satu grafik S1 vs S2 untuk setiap GT yang tersedia."""

    maximum_mae = float(summary["MAE"].max())

    # Skala Y seragam agar grafik GT 15/25/30 mudah dibandingkan.
    y_max = max(
        1.0,
        np.ceil((maximum_mae * 1.10) * 2.0) / 2.0,
    )

    for density in sorted(summary["ground_truth"].unique()):
        subset = summary[
            summary["ground_truth"] == density
        ].copy()

        pivot = subset.pivot(
            index="kondisi",
            columns="model",
            values="MAE",
        )

        pivot = pivot.reindex(EXPECTED_CONDITIONS)

        model_order = [
            model for model in MODEL_COLUMNS.keys()
            if model in pivot.columns
        ]
        pivot = pivot.reindex(columns=model_order)

        ax = pivot.plot(
            kind="bar",
            figsize=(9, 5),
            rot=0,
        )

        ax.set_title(
            f"Perbandingan MAE Real-Time - GT {density}"
        )
        ax.set_xlabel("Kondisi Pencahayaan")
        ax.set_ylabel("MAE (Mean Absolute Error)")
        ax.set_ylim(0, y_max)
        ax.legend(title="Model")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)

        # Menampilkan nilai MAE di atas setiap batang.
        for container in ax.containers:
            ax.bar_label(
                container,
                fmt="%.3f",
                padding=3,
            )

        plt.tight_layout()

        output_path = OUTPUT_DIR / f"grafik_MAE_GT{density}.png"
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()

        print(f"Grafik disimpan: {output_path.name}")


# ==================================================
# 8. PROGRAM UTAMA
# ==================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("AGREGASI HASIL UJI STABILITAS REAL-TIME")
    print("Analisis individual GT 15, GT 25, dan GT 30")
    print("Tanpa pooled MAE")
    print("=" * 60)

    # Memuat dan memvalidasi CSV.
    raw_data = load_all_data()
    print("\n[OK] Seluruh CSV yang diproses berhasil divalidasi.")

    # Mengubah data ke format panjang dan menghitung error per frame.
    long_data = create_long_data(raw_data)

    # Menghitung metrik secara individual.
    summary = calculate_individual_metrics(long_data)

    # Menyimpan tabel rekapitulasi.
    summary_path = OUTPUT_DIR / "hasil_per_kepadatan.csv"
    summary.to_csv(summary_path, index=False)

    # Menyimpan data detail per frame untuk audit/penelusuran.
    detail_path = OUTPUT_DIR / "detail_per_frame.csv"
    long_data.to_csv(detail_path, index=False)

    pd.set_option(
        "display.float_format",
        lambda value: f"{value:.3f}",
    )

    print("\n=== HASIL EVALUASI PER KEPADATAN ===\n")
    print(
        summary[
            [
                "ground_truth",
                "kondisi",
                "model",
                "n_frames",
                "mean_detected",
                "MAE",
                "RMSE",
                "acc_exact_pct",
                "acc_tol2_pct",
            ]
        ].to_string(index=False)
    )

    print(f"\nTabel rekapitulasi disimpan: {summary_path}")
    print(f"Data detail per frame disimpan: {detail_path}")

    # Membuat grafik MAE individual; tidak ada grafik pooled.
    create_individual_charts(summary)

    print("\n" + "=" * 60)
    print("PROSES AGREGASI SELESAI.")
    print(f"Periksa seluruh output di: {OUTPUT_DIR}")
    print("Tidak ada perhitungan pooled MAE lintas kepadatan.")
    print("=" * 60)


if __name__ == "__main__":
    main()
