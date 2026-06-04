# src/train.py
# FlyPose-SAR — Training Incrementale (Fase 1: Baseline RGB)
#
# COSA FA:
#   Esegue in sequenza il fine-tuning di 3 architetture YOLO11-Pose
#   (Nano, Small, Large) sul dataset SAR preparato da prepare_dataset.py.
#   Ogni run e' completamente indipendente: cartella dedicata con timestamp,
#   metriche salvate in JSON, grafici generati automaticamente da Ultralytics.
#
# LOGICA DI FINE-TUNING:
#   Si parte dai pesi pre-addestrati su COCO (conoscenza generica del corpo umano)
#   e si specializzano sul dataset zenitale SAR. Il modello non impara da zero:
#   i primi layer (bordi, texture) restano quasi invariati, gli ultimi layer
#   (interpretazione semantica) si adattano alla visione dall'alto.
#
# EARLY STOPPING:
#   Se per 15 epoche consecutive nessuna metrica migliora, il training si ferma.
#   Evita sprechi di GPU e overfitting. Il best.pt viene sempre salvato al picco.
#
# OUTPUT PER OGNI RUN:
#   runs/fase1/<id>_<timestamp>/
#     weights/
#       best.pt       <- pesi al picco di mAP (quello da usare in inferenza)
#       last.pt       <- pesi dell'ultima epoca
#     results.csv     <- metriche epoca per epoca (usato da plot_metrics.py)
#     metrics_summary.json  <- riepilogo finale (generato da questo script)
#     confusion_matrix.png  <- matrice di confusione
#     PR_curve.png          <- curva Precision-Recall
#     val_batch*.jpg        <- esempi di validazione con skeleton overlay

import json
import torch
import logging
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# DEFINIZIONE ESPERIMENTI
# Modifica batch se la tua GPU ha piu' o meno di 6GB VRAM.
# Regola empirica: se vai OOM (out of memory) dimezza il batch.
# ---------------------------------------------------------------------------
EXPERIMENTS = [
    {
        "id":      "fase1_nano",
        "weights": "modelli_base/yolo11n-pose.pt",
        "label":   "YOLO11n-Pose — Nano",
        "epochs":  50,
        "batch":   16,   # ~3.5 GB VRAM su 640px
        "imgsz":   640,
        "notes":   "Modello piu' leggero. Valida rapidamente il dataset e la pipeline.",
    },
    #{
    #    "id":      "fase1_small",
    #    "weights": "modelli_base/yolo11s-pose.pt",
    #    "label":   "YOLO11s-Pose — Small",
    #   "epochs":  50,
    #    "batch":   12,   # ~4.5 GB VRAM su 640px
    #    "imgsz":   640,
    #    "notes":   "Bilancio velocita'/precisione. Buon candidato per edge deploy.",
    #},
    #{
    #    "id":      "fase1_large",
    #    "weights": "modelli_base/yolo11l-pose.pt",
    #    "label":   "YOLO11l-Pose — Large",
    #    "epochs":  50,
    #    "batch":   6,    # ~5.5 GB VRAM su 640px — al limite RTX 2060
    #    "imgsz":   640,
    #    "notes":   "Massima capacita' estrattiva. Stesso modello usato come teacher.",
    #},
]

# ---------------------------------------------------------------------------
# PARAMETRI COMUNI DI TRAINING
#
# Nota sulle augmentation scelte per la visione zenitale SAR:
#
#   flipud=0.3   -> flip verticale valido: il drone vola in direzioni diverse,
#                   non esiste un "sopra" fisso come nelle immagini frontali.
#
#   degrees=10   -> rotazione moderata: il drone rollea leggermente ma non di 90
#                   gradi; rotazioni troppo grandi distorcono la geometria dello scheletro.
#
#   scale=0.5    -> scaling aggressivo: simula persone a diverse altitudini di volo.
#                   Una persona a 60m appare 3x piu' piccola che a 20m.
#
#   mosaic=1.0   -> combina 4 immagini in una: aumenta varieta' di scala e contesto.
#                   Utile con dataset relativamente piccoli come il nostro.
#
#   mixup=0.1    -> blending leggero tra coppie di immagini: riduce overfitting
#                   senza alterare troppo la struttura degli scheletri.
# ---------------------------------------------------------------------------
COMMON_PARAMS = {
    "data":        "data.yaml",
    "workers":     4,
    "patience":    15,
    "save":        True,
    "save_period": 10,
    "plots":       True,
    "val":         True,
    "verbose":     True,
    "augment":     True,
    "hsv_h":       0.015,
    "hsv_s":       0.3,
    "hsv_v":       0.4,
    "degrees":     10.0,
    "translate":   0.1,
    "scale":       0.5,
    "flipud":      0.3,
    "fliplr":      0.5,
    "mosaic":      1.0,
    "mixup":       0.1,
}


# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------
def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_{ts}.log"
    logger = logging.getLogger("FlyPose_Training")
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
# ESTRAZIONE METRICHE
#
# Ultralytics restituisce un oggetto Results dopo il training.
# Le metriche finali sono in results.results_dict con chiavi tipo:
#   "metrics/precision(B)"    -> precisione box (quante detection sono corrette)
#   "metrics/recall(B)"       -> recall box (quante persone reali vengono trovate)
#   "metrics/mAP50(B)"        -> mean Average Precision a IoU=0.50 per box
#   "metrics/mAP50-95(B)"     -> mAP mediato su IoU da 0.50 a 0.95 (piu' severo)
#   "metrics/mAP50(P)"        -> mAP50 per pose (keypoints)
#   "metrics/mAP50-95(P)"     -> mAP50-95 per pose
#
# La differenza tra mAP50 e mAP50-95:
#   mAP50    -> box/skeleton accettato se sovrapposto >= 50% al GT. Permissivo.
#   mAP50-95 -> media su soglie 50,55,60...95%. Molto piu' esigente.
#   Per tesi: riporta entrambi. mAP50 e' il confronto standard in letteratura.
# ---------------------------------------------------------------------------
def extract_metrics(results) -> dict:
    metrics = {}
    try:
        if hasattr(results, "results_dict"):
            rd = results.results_dict
            metrics["box_precision"]  = float(rd.get("metrics/precision(B)",   0))
            metrics["box_recall"]     = float(rd.get("metrics/recall(B)",       0))
            metrics["box_mAP50"]      = float(rd.get("metrics/mAP50(B)",        0))
            metrics["box_mAP50_95"]   = float(rd.get("metrics/mAP50-95(B)",     0))
            metrics["pose_mAP50"]     = float(rd.get("metrics/mAP50(P)",        0))
            metrics["pose_mAP50_95"]  = float(rd.get("metrics/mAP50-95(P)",     0))
            metrics["train_box_loss"] = float(rd.get("train/box_loss",          0))
            metrics["train_pose_loss"]= float(rd.get("train/pose_loss",         0))
            metrics["val_box_loss"]   = float(rd.get("val/box_loss",            0))
            metrics["val_pose_loss"]  = float(rd.get("val/pose_loss",           0))
    except Exception as e:
        metrics["extraction_error"] = str(e)
    return metrics


# ---------------------------------------------------------------------------
# SINGOLO ESPERIMENTO
# ---------------------------------------------------------------------------
def run_experiment(exp: dict, device: str, project_root: Path,
                   logger: logging.Logger) -> dict:
    exp_id    = exp["id"]
    label     = exp["label"]
    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name  = f"{exp_id}_{ts}"

    logger.info("")
    logger.info("=" * 62)
    logger.info(f"  {label}")
    logger.info(f"  Run : {run_name}")
    logger.info(f"  Epoche: {exp['epochs']}  Batch: {exp['batch']}  imgsz: {exp['imgsz']}")
    logger.info(f"  Note: {exp.get('notes','')}")
    logger.info("=" * 62)

    if not Path(exp["weights"]).exists():
        logger.error(f"  [SKIP] Pesi non trovati: {exp['weights']}")
        return {"id": exp_id, "label": label, "status": "SKIPPED",
                "reason": f"Pesi non trovati: {exp['weights']}"}

    try:
        model = YOLO(exp["weights"])

        results = model.train(
            **COMMON_PARAMS,
            epochs   = exp["epochs"],
            batch    = exp["batch"],
            imgsz    = exp["imgsz"],
            device   = device,
            project  = str(project_root / "runs" / "fase1"),
            name     = run_name,
            exist_ok = False,
        )

        metrics  = extract_metrics(results)
        run_dir  = project_root / "runs" / "fase1" / run_name

        summary = {
            "experiment_id": exp_id,
            "label":         label,
            "run_name":      run_name,
            "timestamp":     ts,
            "model_weights": exp["weights"],
            "epochs":        exp["epochs"],
            "batch":         exp["batch"],
            "imgsz":         exp["imgsz"],
            "device":        device,
            "notes":         exp.get("notes", ""),
            "status":        "COMPLETED",
            "metrics":       metrics,
            "run_dir":       str(run_dir),
            "best_weights":  str(run_dir / "weights" / "best.pt"),
        }

        with open(run_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(f"  [OK] Completato.")
        logger.info(f"       Box  mAP@0.5      : {metrics.get('box_mAP50',  0):.4f}")
        logger.info(f"       Box  mAP@0.5:0.95 : {metrics.get('box_mAP50_95', 0):.4f}")
        logger.info(f"       Pose mAP@0.5      : {metrics.get('pose_mAP50', 0):.4f}")
        logger.info(f"       Pose mAP@0.5:0.95 : {metrics.get('pose_mAP50_95',0):.4f}")
        logger.info(f"       Pesi migliori     : {summary['best_weights']}")
        return summary

    except Exception as e:
        logger.error(f"  [ERRORE] {exp_id}: {e}", exc_info=True)
        return {"id": exp_id, "label": label, "status": "FAILED", "reason": str(e)}


# ---------------------------------------------------------------------------
# REPORT COMPARATIVO
# ---------------------------------------------------------------------------
def save_comparison_report(all_results: list, project_root: Path,
                            logger: logging.Logger):
    report_dir = project_root / "runs" / "fase1"
    report_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    completed = [r for r in all_results if r.get("status") == "COMPLETED"]
    summary_rows = []
    for r in completed:
        m = r.get("metrics", {})
        summary_rows.append({
            "label":         r["label"],
            "run_name":      r.get("run_name", ""),
            "best_weights":  r.get("best_weights", ""),
            "box_mAP50":     m.get("box_mAP50",    0),
            "box_mAP50_95":  m.get("box_mAP50_95", 0),
            "pose_mAP50":    m.get("pose_mAP50",   0),
            "pose_mAP50_95": m.get("pose_mAP50_95",0),
        })

    report = {
        "generated_at": datetime.now().isoformat(),
        "phase":        "Fase 1 — Baseline RGB",
        "experiments":  all_results,
        "summary":      summary_rows,
    }

    report_path = report_dir / f"comparison_report_{ts}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info("")
    logger.info("=" * 62)
    logger.info("  CONFRONTO FINALE FASE 1")
    logger.info("=" * 62)
    logger.info(f"  {'Modello':<30} {'BoxmAP50':>10} {'PosemAP50':>10}")
    logger.info(f"  {'-'*52}")
    for row in summary_rows:
        logger.info(f"  {row['label']:<30} {row['box_mAP50']:>10.4f} {row['pose_mAP50']:>10.4f}")
    logger.info("=" * 62)
    logger.info(f"  Report: {report_path}")
    logger.info("  Esegui plot_metrics.py per i grafici.")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    script_dir   = Path(__file__).resolve().parent
    project_root = script_dir.parent
    log_dir      = project_root / "logs"

    logger = setup_logging(log_dir)
    logger.info("=" * 62)
    logger.info("  FLYPOSE-SAR — TRAINING INCREMENTALE (FASE 1)")
    logger.info("=" * 62)
    logger.info(f"  Device      : {device}")
    if device == "cuda:0":
        logger.info(f"  GPU         : {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"  VRAM totale : {vram:.1f} GB")
    logger.info(f"  Esperimenti : {len(EXPERIMENTS)}")

    all_results = []
    for exp in EXPERIMENTS:
        result = run_experiment(exp, device, project_root, logger)
        all_results.append(result)
        torch.cuda.empty_cache()

    save_comparison_report(all_results, project_root, logger)
    logger.info("\n[DONE] Training Fase 1 completato.")


if __name__ == "__main__":
    main()