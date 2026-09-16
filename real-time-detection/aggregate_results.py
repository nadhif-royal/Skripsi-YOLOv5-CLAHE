"""
aggregate_results.py — Gabungkan semua sesi (Normal, Senter, LampuHP, dst)
yang sudah direkam via benchmark_stability_multi.py menjadi satu tabel
perbandingan MAE/RMSE/Std/Akurasi per kondisi x model, plus grafik batang.

Jalankan SETELAH semua sesi kondisi selesai direkam.
"""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

LOG_DIR = "logs_stabilitas"
TOLERANSI = 2  # samakan dengan tolerance di hitl_interface.py


def load_all_logs():
    csv_files = glob.glob(os.path.join(LOG_DIR, "*.csv"))
    if not csv_files:
        print(f"❌ Tidak ada file CSV di folder {LOG_DIR}. Rekam sesi dulu pakai benchmark_stability_multi.py")
        return None
    dfs = [pd.read_csv(f) for f in csv_files]
    return pd.concat(dfs, ignore_index=True)


def compute_group_metrics(group, gt_col, count_col):
    gt = group[gt_col].iloc[0]
    counts = group[count_col].values
    return pd.Series({
        "n_frames": len(counts),
        "ground_truth": gt,
        "mean_detected": np.mean(counts),
        "MAE": np.mean(np.abs(counts - gt)),
        "RMSE": np.sqrt(np.mean((counts - gt) ** 2)),
        "std_dev": np.std(counts),
        "cv_percent": (np.std(counts) / np.mean(counts) * 100) if np.mean(counts) > 0 else 0,
        "acc_exact_pct": np.mean(counts == gt) * 100,
        f"acc_tol{TOLERANSI}_pct": np.mean(np.abs(counts - gt) <= TOLERANSI) * 100,
    })


def main():
    df = load_all_logs()
    if df is None:
        return

    print(f"✅ Memuat {df['kondisi'].nunique()} kondisi: {df['kondisi'].unique().tolist()}")

    # Reshape long-format: satu baris per (frame, kondisi, model, count)
    rows = []
    for _, r in df.iterrows():
        rows.append({"kondisi": r["kondisi"], "model": "S1_Original", "ground_truth": r["ground_truth"], "count": r["count_s1"]})
        rows.append({"kondisi": r["kondisi"], "model": "S2_CLAHE_LAB", "ground_truth": r["ground_truth"], "count": r["count_s2"]})
    long_df = pd.DataFrame(rows)

    summary = long_df.groupby(["kondisi", "model"]).apply(
        lambda g: compute_group_metrics(g, "ground_truth", "count")
    ).reset_index()

    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    print("\n=== TABEL PERBANDINGAN STABILITAS REAL-TIME PER KONDISI ===\n")
    print(summary.to_string(index=False))

    summary.to_csv("ringkasan_stabilitas_semua_kondisi.csv", index=False)
    print("\n✅ Tabel lengkap tersimpan: ringkasan_stabilitas_semua_kondisi.csv")

    # Grafik batang: MAE per kondisi, dikelompokkan per model -> bukti utama "CLAHE lebih tahan glare"
    pivot_mae = summary.pivot(index="kondisi", columns="model", values="MAE")
    ax = pivot_mae.plot(kind="bar", figsize=(10, 6), rot=0)
    ax.set_ylabel("MAE (Mean Absolute Error)")
    ax.set_title("Perbandingan MAE Real-time: Original vs CLAHE L*a*b* per Kondisi Pencahayaan")
    ax.legend(title="Model")
    plt.tight_layout()
    plt.savefig("grafik_mae_per_kondisi.png", dpi=150)
    plt.show()

    # Grafik batang kedua: akurasi toleransi
    pivot_acc = summary.pivot(index="kondisi", columns="model", values=f"acc_tol{TOLERANSI}_pct")
    ax2 = pivot_acc.plot(kind="bar", figsize=(10, 6), rot=0)
    ax2.set_ylabel(f"Akurasi Toleransi ±{TOLERANSI} (%)")
    ax2.set_title(f"Perbandingan Akurasi (Toleransi ±{TOLERANSI} part) per Kondisi Pencahayaan")
    ax2.legend(title="Model")
    plt.tight_layout()
    plt.savefig("grafik_akurasi_toleransi_per_kondisi.png", dpi=150)
    plt.show()

    print("\n✅ Grafik tersimpan: grafik_mae_per_kondisi.png, grafik_akurasi_toleransi_per_kondisi.png")


if __name__ == "__main__":
    main()