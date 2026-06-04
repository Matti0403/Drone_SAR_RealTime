# src/plot_metrics.py
# FlyPose-SAR — Generatore Grafici Comparativi (Fase 1: Baseline RGB)
#
# Legge tutti i file metrics_summary.json nelle sottocartelle di runs/fase1
# e genera grafici PNG pronti per la tesi.
#
# Grafici generati:
#   1. Bar chart comparativo: Box mAP@0.5 e Pose mAP@0.5 per ogni modello
#   2. Bar chart: Box mAP@0.5:0.95 e Pose mAP@0.5:0.95
#   3. Radar chart: confronto multidimensionale (Precision, Recall, mAP50, mAP50-95 Pose)
#   4. Heatmap metriche: tutti i modelli × tutte le metriche chiave
#
# Uso:
#   python src/plot_metrics.py
#   python src/plot_metrics.py --runs-dir runs/fase1 --output-dir risultati/grafici

import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

try:
    import matplotlib
    matplotlib.use("Agg")   # Backend non-interattivo (nessuna finestra)
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    MATPLOTLIB_OK = True
except ImportError:
    MATPLOTLIB_OK = False
    print("[ERRORE] matplotlib non trovato. Installa con: pip install matplotlib")
    exit(1)


# ---------------------------------------------------------------------------
# STILE GLOBALE DEI GRAFICI
# ---------------------------------------------------------------------------
COLORS = ["#2196F3", "#4CAF50", "#FF5722", "#9C27B0", "#FF9800", "#00BCD4"]
FONT_SIZE_TITLE  = 14
FONT_SIZE_LABELS = 11
FONT_SIZE_TICKS  = 9
DPI = 150   # Alta risoluzione per stampa tesi


# ---------------------------------------------------------------------------
# CARICAMENTO DATI
# ---------------------------------------------------------------------------
def load_all_metrics(runs_dir: Path) -> list:
    """Raccoglie tutti i metrics_summary.json nella directory dei run."""
    metrics_files = sorted(runs_dir.rglob("metrics_summary.json"))

    if not metrics_files:
        print(f"[AVVISO] Nessun file metrics_summary.json trovato in: {runs_dir}")
        print("  Esegui prima train.py per generare i risultati.")
        return []

    data = []
    for f in metrics_files:
        try:
            with open(f, "r", encoding="utf-8") as fp:
                entry = json.load(fp)
            if entry.get("status") == "COMPLETED":
                data.append(entry)
                print(f"  [+] Caricato: {entry.get('label', f.parent.name)}")
        except Exception as e:
            print(f"  [!] Errore lettura {f}: {e}")

    print(f"\n[OK] {len(data)} esperimenti completati trovati.")
    return data


# ---------------------------------------------------------------------------
# GRAFICO 1: BAR CHART mAP@0.5
# ---------------------------------------------------------------------------
def plot_map50_comparison(data: list, output_dir: Path):
    labels      = [d["label"].split("—")[0].strip() for d in data]
    box_map50   = [d["metrics"].get("box_mAP50", 0) for d in data]
    pose_map50  = [d["metrics"].get("pose_mAP50", 0) for d in data]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 2.5), 6))
    bars1 = ax.bar(x - width/2, box_map50,  width, label="Box mAP@0.5",  color=COLORS[0], alpha=0.85)
    bars2 = ax.bar(x + width/2, pose_map50, width, label="Pose mAP@0.5", color=COLORS[1], alpha=0.85)

    # Etichette sui bar
    for bar in bars1:
        ax.annotate(f"{bar.get_height():.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=FONT_SIZE_TICKS)
    for bar in bars2:
        ax.annotate(f"{bar.get_height():.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=FONT_SIZE_TICKS)

    ax.set_title("Confronto Box mAP@0.5 vs Pose mAP@0.5 — Fase 1 Baseline RGB",
                 fontsize=FONT_SIZE_TITLE, fontweight="bold", pad=15)
    ax.set_ylabel("mAP@0.5", fontsize=FONT_SIZE_LABELS)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FONT_SIZE_TICKS, rotation=15, ha="right")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=FONT_SIZE_LABELS)
    ax.grid(axis="y", alpha=0.4, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out_path = output_dir / "01_map50_comparison.png"
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  [+] Salvato: {out_path.name}")


# ---------------------------------------------------------------------------
# GRAFICO 2: BAR CHART mAP@0.5:0.95
# ---------------------------------------------------------------------------
def plot_map5095_comparison(data: list, output_dir: Path):
    labels        = [d["label"].split("—")[0].strip() for d in data]
    box_map5095   = [d["metrics"].get("box_mAP50_95", 0) for d in data]
    pose_map5095  = [d["metrics"].get("pose_mAP50_95", 0) for d in data]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 2.5), 6))
    bars1 = ax.bar(x - width/2, box_map5095,  width, label="Box mAP@0.5:0.95",  color=COLORS[2], alpha=0.85)
    bars2 = ax.bar(x + width/2, pose_map5095, width, label="Pose mAP@0.5:0.95", color=COLORS[3], alpha=0.85)

    for bar in bars1:
        ax.annotate(f"{bar.get_height():.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=FONT_SIZE_TICKS)
    for bar in bars2:
        ax.annotate(f"{bar.get_height():.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=FONT_SIZE_TICKS)

    ax.set_title("Confronto Box mAP@0.5:0.95 vs Pose mAP@0.5:0.95 — Fase 1 Baseline RGB",
                 fontsize=FONT_SIZE_TITLE, fontweight="bold", pad=15)
    ax.set_ylabel("mAP@0.5:0.95", fontsize=FONT_SIZE_LABELS)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FONT_SIZE_TICKS, rotation=15, ha="right")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=FONT_SIZE_LABELS)
    ax.grid(axis="y", alpha=0.4, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out_path = output_dir / "02_map5095_comparison.png"
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  [+] Salvato: {out_path.name}")


# ---------------------------------------------------------------------------
# GRAFICO 3: RADAR CHART multidimensionale
# ---------------------------------------------------------------------------
def plot_radar_chart(data: list, output_dir: Path):
    metric_keys  = ["box_precision", "box_recall", "box_mAP50", "box_mAP50_95",
                    "pose_mAP50", "pose_mAP50_95"]
    metric_labels = ["Precision\n(Box)", "Recall\n(Box)", "mAP50\n(Box)", "mAP50-95\n(Box)",
                     "mAP50\n(Pose)", "mAP50-95\n(Pose)"]

    N = len(metric_keys)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]   # chiude il poligono

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for i, d in enumerate(data):
        values = [d["metrics"].get(k, 0) for k in metric_keys]
        values += values[:1]
        color = COLORS[i % len(COLORS)]
        label = d["label"].split("—")[0].strip()
        ax.plot(angles, values, color=color, linewidth=2, linestyle="solid", label=label)
        ax.fill(angles, values, color=color, alpha=0.15)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=FONT_SIZE_TICKS)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7, color="grey")
    ax.grid(color="grey", linestyle="--", linewidth=0.5, alpha=0.7)

    ax.set_title("Radar Multidimensionale — Fase 1 Baseline RGB",
                 fontsize=FONT_SIZE_TITLE, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.15), fontsize=FONT_SIZE_TICKS)

    plt.tight_layout()
    out_path = output_dir / "03_radar_comparison.png"
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  [+] Salvato: {out_path.name}")


# ---------------------------------------------------------------------------
# GRAFICO 4: HEATMAP metriche × modelli
# ---------------------------------------------------------------------------
def plot_heatmap(data: list, output_dir: Path):
    metric_keys = [
        "box_precision", "box_recall", "box_mAP50", "box_mAP50_95",
        "pose_mAP50", "pose_mAP50_95",
        "val_box_loss", "val_pose_loss",
    ]
    metric_labels = [
        "Precision (Box)", "Recall (Box)", "mAP50 (Box)", "mAP50-95 (Box)",
        "mAP50 (Pose)", "mAP50-95 (Pose)",
        "Val Box Loss", "Val Pose Loss",
    ]

    model_labels = [d["label"].split("—")[0].strip() for d in data]
    matrix = np.array([
        [d["metrics"].get(k, 0) for k in metric_keys]
        for d in data
    ])

    fig, ax = plt.subplots(figsize=(len(metric_keys) * 1.4 + 2, len(data) * 1.0 + 2))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(metric_labels)))
    ax.set_yticks(np.arange(len(model_labels)))
    ax.set_xticklabels(metric_labels, fontsize=FONT_SIZE_TICKS, rotation=30, ha="right")
    ax.set_yticklabels(model_labels, fontsize=FONT_SIZE_TICKS)

    # Annotazioni valore nelle celle
    for i in range(len(model_labels)):
        for j in range(len(metric_labels)):
            val = matrix[i, j]
            text_color = "black" if 0.3 < val < 0.7 else "white"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=8, color=text_color, fontweight="bold")

    plt.colorbar(im, ax=ax, shrink=0.8, label="Valore metrica")
    ax.set_title("Heatmap Metriche — Fase 1 Baseline RGB",
                 fontsize=FONT_SIZE_TITLE, fontweight="bold", pad=15)

    plt.tight_layout()
    out_path = output_dir / "04_heatmap_metrics.png"
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  [+] Salvato: {out_path.name}")


# ---------------------------------------------------------------------------
# GRAFICO 5: LOSS CURVE (se disponibile il CSV di training)
# ---------------------------------------------------------------------------
def plot_loss_curves(data: list, output_dir: Path):
    """
    Legge i file results.csv generati da YOLO (uno per run)
    e plotta le curve di loss sovrapposte.
    """
    try:
        import pandas as pd
    except ImportError:
        print("  [Skip] pandas non installato — curve loss non generate.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    plotted = 0
    for i, d in enumerate(data):
        run_dir = Path(d.get("run_dir", ""))
        csv_path = run_dir / "results.csv"

        if not csv_path.exists():
            continue

        try:
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.strip()
            color = COLORS[i % len(COLORS)]
            label = d["label"].split("—")[0].strip()

            # Train loss
            if "train/box_loss" in df.columns:
                axes[0].plot(df["epoch"], df["train/box_loss"],
                             color=color, linewidth=1.5, label=f"{label} (train)")
            if "val/box_loss" in df.columns:
                axes[0].plot(df["epoch"], df["val/box_loss"],
                             color=color, linewidth=1.5, linestyle="--", label=f"{label} (val)")

            # Pose mAP
            if "metrics/mAP50(P)" in df.columns:
                axes[1].plot(df["epoch"], df["metrics/mAP50(P)"],
                             color=color, linewidth=1.5, label=label)
            plotted += 1

        except Exception as e:
            print(f"  [!] Errore lettura {csv_path}: {e}")

    if plotted == 0:
        plt.close()
        return

    axes[0].set_title("Box Loss per Epoca", fontsize=FONT_SIZE_LABELS, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=FONT_SIZE_TICKS)
    axes[0].set_ylabel("Loss", fontsize=FONT_SIZE_TICKS)
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.4, linestyle="--")
    axes[0].spines[["top", "right"]].set_visible(False)

    axes[1].set_title("Pose mAP@0.5 per Epoca", fontsize=FONT_SIZE_LABELS, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=FONT_SIZE_TICKS)
    axes[1].set_ylabel("mAP@0.5 (Pose)", fontsize=FONT_SIZE_TICKS)
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.4, linestyle="--")
    axes[1].spines[["top", "right"]].set_visible(False)

    fig.suptitle("Curve di Training/Validation — Fase 1 Baseline RGB",
                 fontsize=FONT_SIZE_TITLE, fontweight="bold")

    plt.tight_layout()
    out_path = output_dir / "05_loss_curves.png"
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  [+] Salvato: {out_path.name}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="FlyPose-SAR — Generatore grafici Fase 1")
    parser.add_argument("--runs-dir",    type=str, default="runs/fase1",
                        help="Cartella contenente i run (default: runs/fase1)")
    parser.add_argument("--output-dir",  type=str, default="risultati/grafici/fase1",
                        help="Cartella di output per i PNG (default: risultati/grafici/fase1)")
    args = parser.parse_args()

    script_dir   = Path(__file__).resolve().parent
    project_root = script_dir.parent
    runs_dir     = project_root / args.runs_dir
    output_dir   = project_root / args.output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  FLYPOSE-SAR — GENERATORE GRAFICI FASE 1")
    print("=" * 60)
    print(f"  Runs dir   : {runs_dir}")
    print(f"  Output dir : {output_dir}")
    print("")

    data = load_all_metrics(runs_dir)

    if not data:
        print("\n[ATTENZIONE] Nessun dato trovato. Esegui prima train.py.")
        return

    print("\n[*] Generazione grafici...")
    plot_map50_comparison(data, output_dir)
    plot_map5095_comparison(data, output_dir)

    if len(data) >= 2:
        plot_radar_chart(data, output_dir)
        plot_heatmap(data, output_dir)
    else:
        print("  [Skip] Radar/Heatmap richiedono almeno 2 esperimenti.")

    plot_loss_curves(data, output_dir)

    print(f"\n[DONE] Grafici salvati in: {output_dir}")
    print("  Pronti per l'inserimento nella tesi.")


if __name__ == "__main__":
    main()
