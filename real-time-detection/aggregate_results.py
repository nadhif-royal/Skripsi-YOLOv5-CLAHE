
"""
aggregate_results.py

Fungsi:
1. Membaca CSV kepadatan 25 dan 30.
2. Memvalidasi jumlah frame, kondisi, dan Ground Truth.
3. Menghitung MAE, RMSE, rata-rata deteksi,
   akurasi exact, dan akurasi toleransi.
4. Menyimpan hasil per kepadatan secara terpisah.
5. Menghitung pooled MAE 15 + 25 hanya jika
   data kepadatan 15 sudah tersedia dan valid.

Struktur folder:
logs_stabilitas_25/
logs_stabilitas_30/
logs_stabilitas_15/  (opsional, untuk nanti)
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

# Saat ini GT 25 dan 30 wajib tersedia.
# GT 15 akan menjadi opsional sampai pengujian ulang selesai.
REQUIRED_GT = [25, 30]

EXPECTED_CONDITIONS = {
    "Normal",
    "Senter_Dekat",
    "Senter_Jauh",
}

N_FRAMES = 300
TOLERANCE = 2

OUTPUT_DIR = BASE_DIR / "hasil_agregasi"


# ==================================================
# 2. MEMBACA DAN MEMVALIDASI CSV
# ==================================================

def load_density_data(ground_truth, folder):

    if not folder.exists():
        raise FileNotFoundError(
            f"Folder tidak ditemukan: {folder}"
        )

    csv_files = sorted(folder.glob("*.csv"))

    if len(csv_files) == 0:
        raise FileNotFoundError(
            f"Tidak ada CSV di folder: {folder}"
        )

    if len(csv_files) != 3:
        raise ValueError(
            f"Folder {folder.name} harus berisi tepat "
            f"3 CSV. Ditemukan: {len(csv_files)}"
        )

    data_frames = []
    found_conditions = set()

    required_columns = {
        "frame",
        "kondisi",
        "ground_truth",
        "count_s1",
        "count_s2",
        "fps_s1",
        "fps_s2",
    }

    print(f"\nMemeriksa folder: {folder.name}")

    for csv_path in csv_files:

        df = pd.read_csv(csv_path)

        # Periksa kelengkapan kolom
        missing = required_columns - set(df.columns)

        if missing:
            raise ValueError(
                f"{csv_path.name}: kolom hilang: {missing}"
            )

        # Rapikan label kondisi
        df["kondisi"] = df["kondisi"].astype(str).str.strip()

        conditions = df["kondisi"].unique()

        if len(conditions) != 1:
            raise ValueError(
                f"{csv_path.name}: terdapat lebih dari "
                "satu kondisi pencahayaan."
            )

        condition = conditions[0]

        if condition not in EXPECTED_CONDITIONS:
            raise ValueError(
                f"Label kondisi tidak dikenal: {condition}"
            )

        if condition in found_conditions:
            raise ValueError(
                f"Duplikasi kondisi {condition} "
                f"di folder {folder.name}"
            )

        found_conditions.add(condition)

        # Pastikan Ground Truth konsisten
        gt_values = pd.to_numeric(
            df["ground_truth"], errors="raise"
        ).unique()

        if len(gt_values) != 1:
            raise ValueError(
                f"{csv_path.name}: Ground Truth berubah "
                "dalam satu file."
            )

        if gt_values[0] != ground_truth:
            raise ValueError(
                f"{csv_path.name}: GT dalam CSV = "
                f"{gt_values[0]}, seharusnya {ground_truth}."
            )

        # Pastikan tepat 300 frame
        if len(df) != N_FRAMES:
            raise ValueError(
                f"{csv_path.name}: ditemukan {len(df)} frame, "
                f"seharusnya {N_FRAMES}."
            )

        # Pastikan nomor frame tidak duplikat
        if df["frame"].duplicated().any():
            raise ValueError(
                f"{csv_path.name}: terdapat nomor frame duplikat."
            )

        # Konversi kolom numerik
        numeric_columns = [
            "frame",
            "ground_truth",
            "count_s1",
            "count_s2",
            "fps_s1",
            "fps_s2",
        ]

        for column in numeric_columns:
            df[column] = pd.to_numeric(
                df[column], errors="raise"
            )

        # Periksa nilai yang tidak valid
        if not np.isfinite(
            df[numeric_columns].to_numpy(dtype=float)
        ).all():
            raise ValueError(
                f"{csv_path.name}: terdapat nilai NaN atau infinity."
            )

        if (
            (df["count_s1"] < 0).any()
            or (df["count_s2"] < 0).any()
        ):
            raise ValueError(
                f"{csv_path.name}: jumlah deteksi negatif."
            )

        # Tambahkan identitas kepadatan dan sumber
        df["density"] = ground_truth
        df["source_file"] = csv_path.name

        data_frames.append(df)

        print(
            f"  OK: {csv_path.name} | "
            f"GT={ground_truth} | "
            f"Frame={len(df)}"
        )

    # Pastikan ketiga kondisi tersedia
    if found_conditions != EXPECTED_CONDITIONS:
        missing_conditions = (
            EXPECTED_CONDITIONS - found_conditions
        )

        raise ValueError(
            f"Folder {folder.name} belum lengkap. "
            f"Kondisi yang hilang: {missing_conditions}"
        )

    return pd.concat(data_frames, ignore_index=True)


# ==================================================
# 3. MEMUAT SEMUA DATA YANG TERSEDIA
# ==================================================

def load_all_data():

    all_data = []

    for gt, folder in FOLDERS.items():

        csv_exists = (
            folder.exists()
            and len(list(folder.glob("*.csv"))) > 0
        )

        if gt in REQUIRED_GT or csv_exists:

            df = load_density_data(gt, folder)
            all_data.append(df)

        else:
            print(
                f"\n[INFO] GT {gt} belum tersedia. "
                "Tahap ini dilewati."
            )

    if not all_data:
        raise RuntimeError("Tidak ada data yang dapat diproses.")

    return pd.concat(all_data, ignore_index=True)


# ==================================================
# 4. MENGUBAH DATA KE FORMAT PANJANG
# ==================================================

def create_long_data(df):

    parts = []

    for model, count_column in [
        ("S1_Original", "count_s1"),
        ("S2_CLAHE_LAB", "count_s2"),
    ]:

        temp = df[
            [
                "frame",
                "kondisi",
                "ground_truth",
                "density",
                count_column,
            ]
        ].copy()

        temp = temp.rename(
            columns={count_column: "count"}
        )

        temp["model"] = model

        parts.append(temp)

    long_df = pd.concat(parts, ignore_index=True)

    # Absolute error dihitung untuk setiap frame
    long_df["absolute_error"] = (
        long_df["count"] - long_df["ground_truth"]
    ).abs()

    long_df["squared_error"] = (
        long_df["count"] - long_df["ground_truth"]
    ) ** 2

    long_df["exact"] = (
        long_df["absolute_error"] == 0
    )

    long_df["within_tolerance"] = (
        long_df["absolute_error"] <= TOLERANCE
    )

    return long_df


# ==================================================
# 5. MENGHITUNG METRIK PER KEPADATAN
# ==================================================

def calculate_individual_metrics(long_df):

    results = []

    grouped = long_df.groupby(
        ["density", "kondisi", "model"]
    )

    for (density, condition, model), group in grouped:

        errors = group["absolute_error"].to_numpy()
        squared_errors = group["squared_error"].to_numpy()
        counts = group["count"].to_numpy()

        results.append({
            "ground_truth": density,
            "kondisi": condition,
            "model": model,
            "n_frames": len(group),
            "mean_detected": counts.mean(),
            "MAE": errors.mean(),
            "RMSE": np.sqrt(squared_errors.mean()),
            "std_dev": counts.std(),
            "acc_exact_pct": group["exact"].mean() * 100,
            "acc_tol2_pct": (
                group["within_tolerance"].mean() * 100
            ),
        })

    return pd.DataFrame(results)


# ==================================================
# 6. POOLED MAE GT 15 + GT 25
# ==================================================

def calculate_pooled_15_25(long_df):

    available_gt = set(
        long_df["ground_truth"].unique()
    )

    if not {15, 25}.issubset(available_gt):
        print(
            "\n[INFO] Pooled MAE 15 + 25 belum dihitung. "
            "Data GT 15 dan GT 25 harus tersedia."
        )
        return None

    selected = long_df[
        long_df["ground_truth"].isin([15, 25])
    ].copy()

    results = []

    for (condition, model), group in selected.groupby(
        ["kondisi", "model"]
    ):

        # Harus ada 300 frame GT15 dan 300 frame GT25
        counts_per_gt = (
            group.groupby("ground_truth").size().to_dict()
        )

        if counts_per_gt.get(15, 0) != N_FRAMES:
            raise ValueError(
                f"Pooled {condition} {model}: "
                "data GT15 tidak berjumlah 300 frame."
            )

        if counts_per_gt.get(25, 0) != N_FRAMES:
            raise ValueError(
                f"Pooled {condition} {model}: "
                "data GT25 tidak berjumlah 300 frame."
            )

        if len(group) != 600:
            raise ValueError(
                f"Pooled {condition} {model}: "
                f"total frame {len(group)}, seharusnya 600."
            )

        results.append({
            "kondisi": condition,
            "model": model,
            "n_frames": len(group),
            "pooled_MAE": group["absolute_error"].mean(),
            "pooled_RMSE": np.sqrt(
                group["squared_error"].mean()
            ),
            "acc_tol2_pct": (
                group["within_tolerance"].mean() * 100
            ),
        })

    return pd.DataFrame(results)


# ==================================================
# 7. MEMBUAT GRAFIK MAE PER KEPADATAN
# ==================================================

def create_individual_charts(summary):

    for density in sorted(summary["ground_truth"].unique()):

        subset = summary[
            summary["ground_truth"] == density
        ]

        pivot = subset.pivot(
            index="kondisi",
            columns="model",
            values="MAE"
        )

        pivot = pivot.reindex(
            ["Normal", "Senter_Dekat", "Senter_Jauh"]
        )

        ax = pivot.plot(
            kind="bar",
            figsize=(9, 5),
            rot=0
        )

        ax.set_title(
            f"Perbandingan MAE Real-Time - GT {density}"
        )

        ax.set_xlabel("Kondisi Pencahayaan")
        ax.set_ylabel("MAE (Mean Absolute Error)")
        ax.legend(title="Model")

        plt.tight_layout()

        output_path = (
            OUTPUT_DIR / f"grafik_MAE_GT{density}.png"
        )

        plt.savefig(output_path, dpi=200)
        plt.close()

        print(f"Grafik disimpan: {output_path.name}")


# ==================================================
# 8. MEMBUAT GRAFIK POOLED
# ==================================================

def create_pooled_chart(pooled):

    if pooled is None:
        return

    pivot = pooled.pivot(
        index="kondisi",
        columns="model",
        values="pooled_MAE"
    )

    pivot = pivot.reindex(
        ["Normal", "Senter_Dekat", "Senter_Jauh"]
    )

    ax = pivot.plot(
        kind="bar",
        figsize=(9, 5),
        rot=0
    )

    ax.set_title(
        "Pooled MAE: GT 15 + GT 25 (600 Frames per Condition)"
    )

    ax.set_xlabel("Kondisi Pencahayaan")
    ax.set_ylabel("Pooled MAE")
    ax.legend(title="Model")

    plt.tight_layout()

    output_path = OUTPUT_DIR / "grafik_pooled_15_25.png"

    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Grafik disimpan: {output_path.name}")


# ==================================================
# 9. PROGRAM UTAMA
# ==================================================

def main():

    OUTPUT_DIR.mkdir(exist_ok=True)

    print("=" * 55)
    print("AUDIT AGREGASI DATA REAL-TIME")
    print("=" * 55)

    df = load_all_data()

    print("\n[OK] Seluruh CSV berhasil divalidasi.")

    long_df = create_long_data(df)

    summary = calculate_individual_metrics(long_df)

    summary_path = OUTPUT_DIR / "hasil_per_kepadatan.csv"
    summary.to_csv(summary_path, index=False)

    print("\n=== HASIL MAE PER KEPADATAN ===")

    print(
        summary[
            [
                "ground_truth",
                "kondisi",
                "model",
                "n_frames",
                "MAE",
                "acc_tol2_pct",
            ]
        ].to_string(index=False)
    )

    print(
        f"\nTabel disimpan: {summary_path.name}"
    )

    create_individual_charts(summary)

    pooled = calculate_pooled_15_25(long_df)

    if pooled is not None:

        pooled_path = (
            OUTPUT_DIR / "hasil_pooled_15_25.csv"
        )

        pooled.to_csv(pooled_path, index=False)

        print("\n=== HASIL POOLED GT 15 + GT 25 ===")
        print(pooled.to_string(index=False))

        create_pooled_chart(pooled)

    print("\n" + "=" * 55)
    print("PROSES AGREGASI SELESAI.")
    print(f"Periksa hasil di: {OUTPUT_DIR}")
    print("=" * 55)


if __name__ == "__main__":
    main()
