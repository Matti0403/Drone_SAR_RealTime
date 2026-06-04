# src/evaluate.py
# FlyPose-SAR — Valutazione Quantitativa Formale sul Test Set
#
# COSA FA:
#   Prende tutti i best.pt dei run completati in runs/fase1/ e li valuta
#   sul dataset di test ufficiale usando le annotazioni GT di VisDrone.
#   Produce metriche formali (Precision, Recall, mAP50, mAP50-95) per ogni
#   modello e genera grafici comparativi pronti per la tesi.
#
# DIFFERENZA CON plot_metrics.py:
#   plot_metrics.py  -> legge le metriche di VALIDAZIONE salvate durante il
#                       training (sul validation set, visto durante l'addestramento).
#   evaluate.py      -> esegue i modelli su dati MAI VISTI (test set), confronta
#                       le detection con il GT reale e calcola le metriche da zero.
#                       Questa e' la valutazione formale da riportare in tesi.
#
# PERCHE' E' IMPORTANTE LA DISTINZIONE:
#   Le metriche di validazione sono ottimistiche: il modello e' stato selezionato
#   (early stopping, best.pt) proprio in base a quelle. Le metriche sul test set
#   sono imparziali: il modello non le ha mai viste durante il training.
#   In letteratura si riportano SEMPRE le metriche sul test set.
#
# COME FUNZIONA LA VALUTAZIONE:
#   1. Prepara un dataset di test in formato YOLO Pose (stesso approccio
#      ibrido di prepare_dataset.py: box GT + keypoints teacher su crop)
#   2. Per ogni modello esegue model.val() sul test set
#   3. Raccoglie Precision, Recall, mAP50, mAP50-95 (box e pose)
#   4. Salva tutto in evaluation_report.json e genera i grafici
#
# STRUTTURA ATTESA DEL TEST SET:
#   C:\Dataset_ViSDrone_RGB\VisDrone2019-VID-test-dev\
#     annotations/   <- un .txt per sequenza
#     sequences/
#       <seq_name>/
#         0000001.jpg ...
#
# NOTA: se non hai il test set ufficiale, lo script puo' usare il val set
# come proxy (imposta USE_VAL_AS_TEST=True). Non e' ideale ma e' meglio
# di non avere nessuna valutazione formale.

import json
import torch
import shutil
import logging
import argparse
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from ultralytics import YOLO

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_OK = True
except ImportError:
    MATPLOTLIB_OK = False

# ---------------------------------------------------------------------------
# CONFIGURAZIONE
# ---------------------------------------------------------------------------

# Percorso dataset di test VisDrone
PATH_TEST = Path(r"C:\Users\MATTIA-D'AGOSTINO\Desktop\Drone_SAR_RealTime\datasets\dataset_test_official")

# Se non hai il test set ufficiale, usa il val set come proxy
USE_VAL_AS_TEST = False
PATH_VAL  = Path(r"C:\Dataset_ViSDrone_RGB\VisDrone2019-VID-val")

# Teacher per generare keypoints sul test set (stesso di prepare_dataset.py)
TEACHER_MODEL_PATH = "modelli_base/yolo11l-pose.pt"

# Parametri di filtraggio (stessi di prepare_dataset.py per coerenza)
PERSON_CATEGORIES = {1, 2}
MAX_OCCLUSION     = 1
MIN_BOX_W         = 8
MIN_BOX_H         = 8
CROP_PADDING      = 0.20
TEACHER_CONF      = 0.25
MIN_KPT_CONF_MEAN = 0.20

# Grafici
DPI         = 150
COLORS      = ["#2196F3", "#4CAF50", "#FF5722", "#9C27B0", "#FF9800"]


# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------
def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"evaluate_{ts}.log"
    logger = logging.getLogger("FlyPose_Evaluate")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ---------------------------------------------------------------------------
# PARSING ANNOTAZIONI VISDRONE (identico a prepare_dataset.py)
# ---------------------------------------------------------------------------
def parse_visdrone_annotations(ann_file: Path) -> defaultdict:
    annotations = defaultdict(list)
    if not ann_file.exists():
        return annotations
    with open(ann_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 10:
                continue
            try:
                frame_idx  = int(parts[0])
                x          = int(parts[2])
                y          = int(parts[3])
                w          = int(parts[4])
                h          = int(parts[5])
                category   = int(parts[7])
                truncation = int(parts[8])
                occlusion  = int(parts[9])
            except ValueError:
                continue
            if category not in PERSON_CATEGORIES:
                continue
            if occlusion > MAX_OCCLUSION:
                continue
            if w < MIN_BOX_W or h < MIN_BOX_H:
                continue
            annotations[frame_idx].append({
                "x": x, "y": y, "w": w, "h": h,
                "truncation": truncation, "occlusion": occlusion,
            })
    return annotations


def crop_with_padding(frame, x, y, w, h):
    fh, fw = frame.shape[:2]
    pad_x = int(w * CROP_PADDING)
    pad_y = int(h * CROP_PADDING)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(fw, x + w + pad_x)
    y2 = min(fh, y + h + pad_y)
    return frame[y1:y2, x1:x2].copy(), x1, y1


def get_keypoints_from_crop(teacher_model, crop, x_offset, y_offset,
                             frame_w, frame_h, device):
    if crop is None or crop.size == 0:
        return None
    ch, cw = crop.shape[:2]
    scale = 1.0
    if cw < 64 or ch < 64:
        scale = max(64 / cw, 64 / ch)
        crop = cv2.resize(crop, (int(cw * scale), int(ch * scale)),
                          interpolation=cv2.INTER_LINEAR)
    results = teacher_model(crop, verbose=False, device=device,
                            conf=TEACHER_CONF, imgsz=256)[0]
    if (results.keypoints is None or len(results.keypoints) == 0
            or results.boxes is None or len(results.boxes) == 0):
        return None
    confs = results.boxes.conf.cpu().numpy()
    best  = int(np.argmax(confs))
    kpts      = results.keypoints.xy.cpu().numpy()[best]
    kpts_conf = results.keypoints.conf.cpu().numpy()[best]
    if np.mean(kpts_conf) < MIN_KPT_CONF_MEAN:
        return None
    kpts_abs = []
    for kp, kconf in zip(kpts, kpts_conf):
        kx_norm = float(np.clip((kp[0] / scale + x_offset) / frame_w, 0.0, 1.0))
        ky_norm = float(np.clip((kp[1] / scale + y_offset) / frame_h, 0.0, 1.0))
        if kx_norm == 0.0 and ky_norm == 0.0:
            vis = 0
        elif float(kconf) < 0.3:
            vis = 1
        else:
            vis = 2
        kpts_abs.append((kx_norm, ky_norm, vis))
    return kpts_abs


# ---------------------------------------------------------------------------
# PREPARAZIONE TEST SET IN FORMATO YOLO POSE
#
# Stesso identico approccio di prepare_dataset.py ma sulla split "test".
# Il risultato viene salvato in datasets/dataset_test_yolo/ — una cartella
# separata da dataset_sar per non contaminare train/val.
# ---------------------------------------------------------------------------
def prepare_test_dataset(src_base: Path, dest_base: Path,
                          teacher_model, device: str,
                          logger: logging.Logger) -> bool:
    seq_dir = src_base / "sequences"
    ann_dir = src_base / "annotations"

    if not seq_dir.exists() or not ann_dir.exists():
        logger.error(f"Test set non trovato in: {src_base}")
        return False

    dest_img = dest_base / "images" / "test"
    dest_lbl = dest_base / "labels" / "test"

    # Se gia' esiste e ha file, chiedi conferma via flag
    if dest_img.exists() and any(dest_img.iterdir()):
        logger.info("[*] Test set YOLO gia' esistente — skip preparazione.")
        logger.info(f"    Per rigenerarlo elimina: {dest_base}")
        return True

    dest_img.mkdir(parents=True, exist_ok=True)
    dest_lbl.mkdir(parents=True, exist_ok=True)

    seq_dirs = sorted([d for d in seq_dir.iterdir() if d.is_dir()])
    logger.info(f"[*] Preparazione test set: {len(seq_dirs)} sequenze...")

    total_saved = 0
    iterator = tqdm(seq_dirs, desc="  [test]", unit="seq") if TQDM_AVAILABLE else seq_dirs

    for seq_path in iterator:
        seq_name = seq_path.name
        ann_file = ann_dir / f"{seq_name}.txt"
        if not ann_file.exists():
            continue

        seq_ann = parse_visdrone_annotations(ann_file)
        if not seq_ann:
            continue

        img_files = sorted([
            f for f in seq_path.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ])

        for img_path in img_files:
            try:
                frame_idx = int(img_path.stem)
            except ValueError:
                digits = "".join(c for c in img_path.stem if c.isdigit())
                frame_idx = int(digits) if digits else -1

            persons = seq_ann.get(frame_idx, [])
            if not persons:
                continue

            nuovo_nome    = f"{seq_name}_{img_path.stem}"
            dest_lbl_file = dest_lbl / f"{nuovo_nome}.txt"
            if dest_lbl_file.exists():
                continue

            frame = cv2.imread(str(img_path))
            if frame is None:
                continue

            fh, fw = frame.shape[:2]
            lines  = []

            for person in persons:
                x, y, w, h = person["x"], person["y"], person["w"], person["h"]
                crop, x_off, y_off = crop_with_padding(frame, x, y, w, h)
                kpts = get_keypoints_from_crop(
                    teacher_model, crop, x_off, y_off, fw, fh, device
                )
                if kpts is None:
                    continue

                cx = float(np.clip((x + w / 2) / fw, 0, 1))
                cy = float(np.clip((y + h / 2) / fh, 0, 1))
                wn = float(np.clip(w / fw, 0, 1))
                hn = float(np.clip(h / fh, 0, 1))

                box_str = f"0 {cx:.6f} {cy:.6f} {wn:.6f} {hn:.6f}"
                kpt_str = " ".join(f"{kx:.6f} {ky:.6f} {vis}" for kx, ky, vis in kpts)
                lines.append(f"{box_str} {kpt_str}")

            if lines:
                shutil.copy(str(img_path), str(dest_img / f"{nuovo_nome}.jpg"))
                with open(dest_lbl_file, "w") as f:
                    f.write("\n".join(lines) + "\n")
                total_saved += 1

        torch.cuda.empty_cache()

    logger.info(f"[+] Test set pronto: {total_saved} frame annotati in {dest_base}")
    return total_saved > 0


# ---------------------------------------------------------------------------
# SCOPERTA AUTOMATICA DEI RUN COMPLETATI
# ---------------------------------------------------------------------------
def find_completed_runs(runs_fase1_dir: Path, logger: logging.Logger) -> list:
    """
    Cerca tutti i metrics_summary.json in runs/fase1/ e restituisce
    la lista dei run completati con i loro metadati.
    """
    runs = []
    for summary_file in sorted(runs_fase1_dir.rglob("metrics_summary.json")):
        try:
            with open(summary_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("status") != "COMPLETED":
                continue
            best_weights = Path(data.get("best_weights", ""))
            if not best_weights.exists():
                logger.warning(f"  [!] best.pt non trovato: {best_weights}")
                continue
            runs.append({
                "label":       data["label"],
                "run_name":    data["run_name"],
                "best_weights": str(best_weights),
                "experiment_id": data["experiment_id"],
                "epochs":      data["epochs"],
                "imgsz":       data["imgsz"],
                "train_metrics": data.get("metrics", {}),
            })
            logger.info(f"  [+] Trovato: {data['label']}  ({best_weights.name})")
        except Exception as e:
            logger.warning(f"  [!] Errore lettura {summary_file}: {e}")
    return runs


# ---------------------------------------------------------------------------
# VALUTAZIONE DI UN SINGOLO MODELLO
#
# model.val() esegue una passata completa sul dataset di test:
#   - per ogni immagine: inferenza + confronto con GT
#   - accumula TP, FP, FN per ogni soglia IoU
#   - calcola Precision, Recall, mAP50, mAP50-95
#
# La differenza con le metriche di training e' che qui non c'e' nessuna
# informazione che il modello ha gia' visto: e' una valutazione completamente
# imparziale su dati nuovi.
# ---------------------------------------------------------------------------
def evaluate_model(run: dict, data_yaml: str, device: str,
                   logger: logging.Logger) -> dict:
    label    = run["label"]
    weights  = run["best_weights"]

    logger.info(f"\n  Valutazione: {label}")
    logger.info(f"  Pesi: {weights}")

    try:
        model = YOLO(weights)

        # model.val() restituisce un oggetto con tutte le metriche calcolate
        # sul dataset specificato in data_yaml, split "test"
        val_results = model.val(
            data    = data_yaml,
            split   = "test",
            imgsz   = run["imgsz"],
            device  = device,
            verbose = False,
            plots   = False,
            project = r"C:\Temp\eval_results",
            name    = run["experiment_id"],
            exist_ok= True,
        )

        # Estrai metriche dal risultato
        m = {}
        try:
            rd = val_results.results_dict
            m["box_precision"]  = float(rd.get("metrics/precision(B)",  0))
            m["box_recall"]     = float(rd.get("metrics/recall(B)",      0))
            m["box_mAP50"]      = float(rd.get("metrics/mAP50(B)",       0))
            m["box_mAP50_95"]   = float(rd.get("metrics/mAP50-95(B)",    0))
            m["pose_mAP50"]     = float(rd.get("metrics/mAP50(P)",       0))
            m["pose_mAP50_95"]  = float(rd.get("metrics/mAP50-95(P)",    0))
        except Exception as e:
            logger.warning(f"    Errore estrazione metriche: {e}")

        # F1 score: media armonica di Precision e Recall
        p = m.get("box_precision", 0)
        r = m.get("box_recall", 0)
        m["box_f1"] = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

        logger.info(f"    Box  Precision : {m.get('box_precision', 0):.4f}")
        logger.info(f"    Box  Recall    : {m.get('box_recall',    0):.4f}")
        logger.info(f"    Box  F1        : {m.get('box_f1',        0):.4f}")
        logger.info(f"    Box  mAP@0.5   : {m.get('box_mAP50',    0):.4f}")
        logger.info(f"    Box  mAP@0.5:95: {m.get('box_mAP50_95', 0):.4f}")
        logger.info(f"    Pose mAP@0.5   : {m.get('pose_mAP50',   0):.4f}")
        logger.info(f"    Pose mAP@0.5:95: {m.get('pose_mAP50_95',0):.4f}")

        return {
            "label":           label,
            "run_name":        run["run_name"],
            "experiment_id":   run["experiment_id"],
            "best_weights":    weights,
            "status":          "COMPLETED",
            "test_metrics":    m,
            "train_metrics":   run["train_metrics"],  # per confronto diretto
        }

    except Exception as e:
        logger.error(f"    [ERRORE] {label}: {e}", exc_info=True)
        return {
            "label":  label,
            "status": "FAILED",
            "reason": str(e),
        }
    finally:
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# GRAFICI
# ---------------------------------------------------------------------------
def plot_evaluation_results(results: list, output_dir: Path, logger: logging.Logger):
    if not MATPLOTLIB_OK:
        logger.warning("matplotlib non disponibile — grafici saltati.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    completed = [r for r in results if r.get("status") == "COMPLETED"]
    if not completed:
        logger.warning("Nessun risultato completato per i grafici.")
        return

    labels    = [r["label"].split("—")[0].strip() for r in completed]
    x         = np.arange(len(labels))
    width     = 0.18
    font_s    = 9
    font_l    = 11

    # ------------------------------------------------------------------
    # Grafico 1: Confronto test vs train mAP50
    # Mostra quanto le metriche di validazione durante il training
    # sovrastimano le prestazioni reali sul test set.
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, metric_key, title in [
        (axes[0], "box_mAP50",  "Box mAP@0.5"),
        (axes[1], "pose_mAP50", "Pose mAP@0.5"),
    ]:
        test_vals  = [r["test_metrics"].get(metric_key, 0)  for r in completed]
        train_vals = [r["train_metrics"].get(metric_key, 0) for r in completed]

        b1 = ax.bar(x - width/2, train_vals, width, label="Val (training)",
                    color=COLORS[0], alpha=0.75)
        b2 = ax.bar(x + width/2, test_vals,  width, label="Test (formale)",
                    color=COLORS[1], alpha=0.85)

        for bar in list(b1) + list(b2):
            ax.annotate(f"{bar.get_height():.3f}",
                        xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=7)

        ax.set_title(f"{title} — Validation vs Test", fontsize=font_l,
                     fontweight="bold")
        ax.set_ylabel(title, fontsize=font_s)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=font_s, rotation=10, ha="right")
        ax.set_ylim(0, 1.0)
        ax.legend(fontsize=font_s)
        ax.grid(axis="y", alpha=0.4, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Confronto metriche validation (training) vs test formale",
                 fontsize=font_l + 1, fontweight="bold")
    plt.tight_layout()
    p = output_dir / "01_val_vs_test_mAP50.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  [+] {p.name}")

    # ------------------------------------------------------------------
    # Grafico 2: Metriche complete sul test set
    # ------------------------------------------------------------------
    metric_keys    = ["box_precision", "box_recall", "box_f1",
                      "box_mAP50", "box_mAP50_95", "pose_mAP50", "pose_mAP50_95"]
    metric_labels  = ["Precision", "Recall", "F1",
                      "Box\nmAP50", "Box\nmAP50-95", "Pose\nmAP50", "Pose\nmAP50-95"]

    fig, ax = plt.subplots(figsize=(13, 5))
    w = 0.8 / len(completed)

    for i, r in enumerate(completed):
        vals  = [r["test_metrics"].get(k, 0) for k in metric_keys]
        xpos  = np.arange(len(metric_keys)) + i * w - (len(completed) - 1) * w / 2
        bars  = ax.bar(xpos, vals, w * 0.9, label=labels[i],
                       color=COLORS[i % len(COLORS)], alpha=0.85)
        for bar in bars:
            if bar.get_height() > 0.02:
                ax.annotate(f"{bar.get_height():.3f}",
                            xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", fontsize=6.5, rotation=90)

    ax.set_title("Metriche complete sul test set formale — Fase 1 Baseline RGB",
                 fontsize=font_l, fontweight="bold")
    ax.set_xticks(np.arange(len(metric_keys)))
    ax.set_xticklabels(metric_labels, fontsize=font_s)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Valore", fontsize=font_s)
    ax.legend(fontsize=font_s)
    ax.grid(axis="y", alpha=0.4, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    p = output_dir / "02_test_metrics_complete.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  [+] {p.name}")

    # ------------------------------------------------------------------
    # Grafico 3: Heatmap test set
    # ------------------------------------------------------------------
    all_metrics = ["box_precision", "box_recall", "box_f1",
                   "box_mAP50", "box_mAP50_95", "pose_mAP50", "pose_mAP50_95"]
    all_labels  = ["Precision", "Recall", "F1",
                   "Box mAP50", "Box mAP50-95", "Pose mAP50", "Pose mAP50-95"]

    matrix = np.array([
        [r["test_metrics"].get(k, 0) for k in all_metrics]
        for r in completed
    ])

    fig, ax = plt.subplots(figsize=(len(all_metrics) * 1.4 + 2, len(completed) + 1.5))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(all_labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(all_labels, fontsize=font_s, rotation=25, ha="right")
    ax.set_yticklabels(labels, fontsize=font_s)

    for i in range(len(labels)):
        for j in range(len(all_labels)):
            val = matrix[i, j]
            tc  = "black" if 0.25 < val < 0.75 else "white"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=8, color=tc, fontweight="bold")

    plt.colorbar(im, ax=ax, shrink=0.8, label="Valore metrica")
    ax.set_title("Heatmap metriche sul test set — Fase 1 Baseline RGB",
                 fontsize=font_l, fontweight="bold", pad=12)

    plt.tight_layout()
    p = output_dir / "03_test_heatmap.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  [+] {p.name}")

    # ------------------------------------------------------------------
    # Grafico 4: Precision-Recall scatter
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    for ax, pk, rk, mk, title in [
        (axes[0], "box_precision", "box_recall", "box_mAP50", "Box Detection"),
        (axes[1], "box_precision", "box_recall", "pose_mAP50","Pose Estimation"),
    ]:
        for i, r in enumerate(completed):
            p_val = r["test_metrics"].get(pk, 0)
            r_val = r["test_metrics"].get(rk, 0)
            m_val = r["test_metrics"].get(mk, 0)
            ax.scatter(r_val, p_val, s=120, color=COLORS[i % len(COLORS)],
                       zorder=3, label=f"{labels[i]} (mAP50={m_val:.3f})")
            ax.annotate(labels[i], (r_val, p_val),
                        textcoords="offset points", xytext=(6, 4),
                        fontsize=8, color=COLORS[i % len(COLORS)])

        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Recall", fontsize=font_s)
        ax.set_ylabel("Precision", fontsize=font_s)
        ax.set_title(f"Precision vs Recall — {title}", fontsize=font_l,
                     fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.4, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)
        # Linea iso-F1
        for f1 in [0.3, 0.5, 0.7]:
            r_range = np.linspace(0.01, 1.0, 200)
            p_iso   = f1 * r_range / (2 * r_range - f1 + 1e-9)
            mask    = (p_iso >= 0) & (p_iso <= 1)
            ax.plot(r_range[mask], p_iso[mask], "--", color="gray",
                    alpha=0.35, linewidth=0.8)
            idx = np.argmin(np.abs(r_range - 0.85))
            if mask[idx]:
                ax.text(r_range[idx], p_iso[idx], f"F1={f1}",
                        fontsize=7, color="gray", alpha=0.6)

    plt.tight_layout()
    p = output_dir / "04_precision_recall_scatter.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  [+] {p.name}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="FlyPose-SAR — Valutazione formale test set")
    parser.add_argument("--force-regen", action="store_true",
                        help="Rigenera il test set YOLO anche se esiste gia'")
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    script_dir   = Path(__file__).resolve().parent
    project_root = script_dir.parent
    log_dir      = project_root / "logs"
    runs_dir     = project_root / "runs" / "fase1"
    output_dir   = project_root / "risultati" / "grafici" / "evaluate"
    report_dir   = project_root / "risultati"

    # Percorso test set YOLO (preparato da questo script)
    src_test = PATH_VAL if USE_VAL_AS_TEST else PATH_TEST
    dest_test = Path(r"C:\Temp\dataset_test_yolo")

    # data.yaml per la valutazione (split test)
    test_yaml = project_root / "data_test.yaml"

    logger = setup_logging(log_dir)
    logger.info("=" * 62)
    logger.info("  FLYPOSE-SAR — VALUTAZIONE FORMALE SUL TEST SET")
    logger.info("=" * 62)
    logger.info(f"  Device        : {device}")
    if device == "cuda:0":
        logger.info(f"  GPU           : {torch.cuda.get_device_name(0)}")
    logger.info(f"  Test src      : {src_test}")
    logger.info(f"  Test YOLO dir : {dest_test}")
    logger.info(f"  Runs dir      : {runs_dir}")
    if USE_VAL_AS_TEST:
        logger.warning("  ATTENZIONE: uso val set come proxy del test set.")
        logger.warning("  Le metriche saranno ottimistiche (modello ha visto questi dati).")

    # ------------------------------------------------------------------
    # 1. Prepara test set in formato YOLO se necessario
    # ------------------------------------------------------------------
    if args.force_regen and dest_test.exists():
        logger.info("[*] --force-regen: elimino test set esistente...")
        shutil.rmtree(dest_test)

    logger.info("\n[*] Caricamento teacher model per test set...")
    teacher = YOLO(TEACHER_MODEL_PATH)
    teacher.to(device)

    ok = prepare_test_dataset(src_test, dest_test, teacher, device, logger)
    if not ok:
        logger.error("[ERRORE] Preparazione test set fallita. Verifica i percorsi.")
        return

    del teacher
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 2. Scrivi data_test.yaml per model.val()
    # ------------------------------------------------------------------
    test_yaml_content = f"""# data_test.yaml — usato da evaluate.py
path: {dest_test}
train: images/test
val:   images/test
test:  images/test

nc: 1
names:
  0: person

kpt_shape: [17, 3]
flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
"""
    with open(test_yaml, "w") as f:
        f.write(test_yaml_content)
    logger.info(f"[+] Scritto: {test_yaml}")

    # ------------------------------------------------------------------
    # 3. Scopri tutti i run completati
    # ------------------------------------------------------------------
    logger.info(f"\n[*] Ricerca run completati in {runs_dir}...")
    if not runs_dir.exists():
        logger.error(f"  Cartella non trovata: {runs_dir}")
        logger.error("  Esegui prima train.py")
        return

    runs = find_completed_runs(runs_dir, logger)
    if not runs:
        logger.error("  Nessun run completato trovato.")
        return

    logger.info(f"  Trovati {len(runs)} run da valutare.")

    # ------------------------------------------------------------------
    # 4. Valuta ogni modello
    # ------------------------------------------------------------------
    logger.info("\n[*] Avvio valutazione formale...")
    all_results = []
    for run in runs:
        result = evaluate_model(run, str(test_yaml), device, logger)
        all_results.append(result)

    # ------------------------------------------------------------------
    # 5. Salva report JSON
    # ------------------------------------------------------------------
    report_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = report_dir / f"evaluation_report_{ts}.json"

    completed = [r for r in all_results if r.get("status") == "COMPLETED"]
    report = {
        "generated_at":   datetime.now().isoformat(),
        "phase":          "Fase 1 — Baseline RGB",
        "test_source":    str(src_test),
        "used_val_as_test": USE_VAL_AS_TEST,
        "models_evaluated": len(completed),
        "results":        all_results,
        "ranking_by_pose_mAP50": sorted(
            [{"label": r["label"],
              "pose_mAP50":   r["test_metrics"].get("pose_mAP50", 0),
              "box_mAP50":    r["test_metrics"].get("box_mAP50", 0),
              "box_f1":       r["test_metrics"].get("box_f1", 0),
              "best_weights": r["best_weights"]}
             for r in completed],
            key=lambda x: x["pose_mAP50"], reverse=True
        ),
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # 6. Grafici
    # ------------------------------------------------------------------
    logger.info("\n[*] Generazione grafici...")
    plot_evaluation_results(all_results, output_dir, logger)

    # ------------------------------------------------------------------
    # 7. Riepilogo finale
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 62)
    logger.info("  RANKING FINALE — TEST SET FORMALE (per Pose mAP@0.5)")
    logger.info("=" * 62)
    logger.info(f"  {'Modello':<28} {'PosemAP50':>10} {'BoxmAP50':>10} {'F1':>8}")
    logger.info(f"  {'-'*58}")
    for row in report["ranking_by_pose_mAP50"]:
        logger.info(
            f"  {row['label']:<28} "
            f"{row['pose_mAP50']:>10.4f} "
            f"{row['box_mAP50']:>10.4f} "
            f"{row['box_f1']:>8.4f}"
        )
    logger.info("=" * 62)
    logger.info(f"  Report   : {report_path}")
    logger.info(f"  Grafici  : {output_dir}")
    logger.info("\n[DONE] Valutazione completata.")


if __name__ == "__main__":
    main()