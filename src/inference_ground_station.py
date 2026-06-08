# src/inference_ground_station.py
# FlyPose-SAR — Inferenza con pipeline SAHI (modalita' stazione di terra)
#
# COSA FA:
#   Esegue la pipeline di inferenza ottimizzata per la stazione di terra,
#   dove non ci sono vincoli di latenza real-time come a bordo drone.
#   L'obiettivo e' massimizzare la precisione su target microscopici
#   (persone viste dall'alto a grande distanza) usando la tecnica SAHI.
#
# COS'E' SAHI (Slicing Aided Hyper Inference):
#   Il problema: YOLO e' addestrato su immagini 640px. Una persona zenitale
#   a 80m di altezza occupa solo 8-15px nel frame originale (1920x1080).
#   A questa scala il modello non la "vede" perche' e' troppo piccola rispetto
#   agli oggetti nel training COCO.
#   La soluzione: dividiamo il frame in riquadri sovrapposti da 640px.
#   Ogni riquadro viene processato indipendentemente. La persona che occupava
#   15px nel frame originale ora occupa ~60px nel suo riquadro -> rilevabile.
#   Poi i risultati di tutti i riquadri vengono unificati con NMS globale.
#
#   Frame 1920x1080
#   ┌──────────┬──────────┬──────────┐
#   │  slice   │  slice   │  slice   │
#   │  0,0     │  640,0   │  1280,0  │  <- ogni slice: 640x640px
#   ├──────────┼──────────┼──────────┤     overlap 25% per non perdere
#   │  slice   │  slice   │  slice   │     persone sul bordo dei riquadri
#   │  0,480   │  640,480 │  1280,480│
#   └──────────┴──────────┴──────────┘
#
# PIPELINE COMPLETA PER OGNI FRAME:
#   1. Calcola le slice con overlap (apply_sahi_slicing)
#   2. Per ogni slice: inferenza con il modello, raccogli box+kpts
#   3. Riproietta coordinate da slice -> frame originale
#   4. Filtra false detection con filtri geometrici (aspect ratio, kpt conf)
#   5. NMS globale su tutte le box per eliminare duplicati sui bordi slice
#   6. Rendering e salvataggio frame annotato
#
# MODALITA' COMPARATIVA:
#   Lo script confronta due modelli sullo stesso test set e produce output
#   separati per ciascuno. Utile per la tesi: confronto visivo tra
#   modello COCO originale e modello fine-tuned SAR.

import os
import sys
import json
import cv2
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# CONFIGURAZIONE MODELLI DA CONFRONTARE
# Sostituisci CUSTOM_MODEL_PATH con il percorso al tuo best.pt
# ---------------------------------------------------------------------------

# -----PRIMA VERSIONE-----
#OFFICIAL_MODEL_PATH = "yolo11l-pose.pt"
#CUSTOM_MODEL_PATH = r"runs\fase1\fase1_nano\weights\best.pt"

# -----SECONDA VERSIONE-----
#OFFICIAL_MODEL_PATH = r"runs\fase1\fase1_nano\weights\best.pt"
#CUSTOM_MODEL_PATH   = r"runs\fase1\fase1_small\weights\best.pt"

# -----TERZA VERSIONE-----
OFFICIAL_MODEL_PATH = r"runs\fase1\fase1_small\weights\best.pt"
CUSTOM_MODEL_PATH   = r"runs\fase1\fase1_large\weights\best.pt"

TEST_SEQ_DIR  = None   # None = usa percorso relativo al progetto

# Parametri SAHI
SLICE_SIZE    = 640
OVERLAP_RATIO = 0.25

# Parametri inferenza
CONF          = 0.28
NMS_IOU       = 0.25

# Filtri anti-falsi-positivi
MAX_ASPECT_RATIO = 2.5   # box troppo larghe = non persona (auto, ombre orizzontali)
MIN_ASPECT_RATIO = 0.2   # box troppo alte e strette = palo, albero
MIN_KPT_CONF     = 0.20  # confidenza media minima keypoints per accettare detection


# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------
def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"inference_gs_{ts}.log"
    logger = logging.getLogger("FlyPose_GroundStation")
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
# CALCOLO SLICE SAHI
#
# Genera le coordinate (x1,y1,x2,y2) di ogni riquadro.
# L'overlap garantisce che una persona sul bordo di uno slice
# sia presente anche nel riquadro adiacente, evitando miss.
# I riquadri agli estremi vengono aggiustati per non sforare l'immagine.
# ---------------------------------------------------------------------------
def compute_slices(img_h: int, img_w: int,
                   slice_size: int = SLICE_SIZE,
                   overlap: float = OVERLAP_RATIO) -> list:
    slices = []
    step = int(slice_size * (1 - overlap))

    for y in range(0, img_h, step):
        for x in range(0, img_w, step):
            x1 = x
            y1 = y
            x2 = min(x + slice_size, img_w)
            y2 = min(y + slice_size, img_h)

            # Aggiusta per mantenere slice esattamente slice_size
            if x2 == img_w:
                x1 = max(0, img_w - slice_size)
            if y2 == img_h:
                y1 = max(0, img_h - slice_size)

            slices.append((x1, y1, x2, y2))

            if x2 == img_w:
                break
        if y2 == img_h:
            break

    return slices


# ---------------------------------------------------------------------------
# NMS MANUALE SU NUMPY
#
# Ultralytics ha il proprio NMS interno, ma qui ne serve uno globale
# che lavori su detection provenienti da slice diverse.
# Algoritmo Greedy NMS:
#   1. Ordina detection per confidenza decrescente
#   2. Prendi la piu' sicura, aggiungila ai "kept"
#   3. Elimina tutte le altre con IoU > soglia rispetto alla kept
#   4. Ripeti sulle rimanenti
# ---------------------------------------------------------------------------
def nms(boxes: np.ndarray, scores: np.ndarray,
        iou_threshold: float = NMS_IOU) -> list:
    if len(boxes) == 0:
        return []

    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas  = (x2 - x1) * (y2 - y1)
    order  = scores.argsort()[::-1]
    kept   = []

    while order.size > 0:
        i = order[0]
        kept.append(int(i))

        ix1 = np.maximum(x1[i], x1[order[1:]])
        iy1 = np.maximum(y1[i], y1[order[1:]])
        ix2 = np.minimum(x2[i], x2[order[1:]])
        iy2 = np.minimum(y2[i], y2[order[1:]])

        inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        order = order[np.where(iou <= iou_threshold)[0] + 1]

    return kept


# ---------------------------------------------------------------------------
# PIPELINE SAHI SU UN FRAME
# ---------------------------------------------------------------------------
def process_frame_sahi(model, img: np.ndarray, device: str) -> tuple:
    """
    Applica SAHI al frame e restituisce:
        (boxes_kept, scores_kept, kpts_kept)
    dove le coordinate sono assolute nel frame originale.
    """
    img_h, img_w = img.shape[:2]
    slices = compute_slices(img_h, img_w)

    all_boxes  = []
    all_scores = []
    all_kpts   = []

    for (x1, y1, x2, y2) in slices:
        slice_img = img[y1:y2, x1:x2]
        results   = model.predict(slice_img, conf=CONF, imgsz=SLICE_SIZE,
                                  classes=0, verbose=False, device=device)

        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue

            boxes      = r.boxes.xyxy.cpu().numpy()       # (N,4) in px slice
            scores     = r.boxes.conf.cpu().numpy()       # (N,)
            kpts       = (r.keypoints.xy.cpu().numpy()
                          if r.keypoints is not None else None)
            kpts_conf  = (r.keypoints.conf.cpu().numpy()
                          if r.keypoints is not None else None)

            for i in range(len(boxes)):
                bx1, by1, bx2, by2 = boxes[i]
                bw = bx2 - bx1
                bh = by2 - by1

                # Salta box degeneri
                if bw <= 0 or bh <= 0:
                    continue

                # Filtro aspect ratio: elimina falsi positivi strutturali
                # (pali verticali, ombre orizzontali, veicoli allungati)
                ar = bw / bh
                if ar > MAX_ASPECT_RATIO or ar < MIN_ASPECT_RATIO:
                    continue

                # Filtro keypoints: se il modello non e' sicuro della posa
                # la detection e' probabilmente un falso positivo
                if kpts_conf is not None and i < len(kpts_conf):
                    if np.mean(kpts_conf[i]) < MIN_KPT_CONF:
                        continue

                # Riproietta coordinate box da slice a frame originale
                all_boxes.append([bx1 + x1, by1 + y1, bx2 + x1, by2 + y1])
                all_scores.append(float(scores[i]))

                # Riproietta keypoints
                if kpts is not None and i < len(kpts):
                    kp = kpts[i].copy()
                    kp[:, 0] += x1
                    kp[:, 1] += y1
                    all_kpts.append(kp)
                else:
                    all_kpts.append(None)

    if not all_boxes:
        return [], [], []

    boxes_arr  = np.array(all_boxes)
    scores_arr = np.array(all_scores)
    keep       = nms(boxes_arr, scores_arr)

    return (
        [all_boxes[k]  for k in keep],
        [all_scores[k] for k in keep],
        [all_kpts[k]   for k in keep],
    )


# ---------------------------------------------------------------------------
# RENDERING FRAME ANNOTATO
# ---------------------------------------------------------------------------
def render_frame(img: np.ndarray, boxes: list, scores: list,
                 kpts: list, label: str) -> np.ndarray:
    """Disegna box, score e keypoints sull'immagine."""
    SKELETON = [
        (0,1),(0,2),(1,3),(2,4),        # testa
        (5,6),(5,7),(6,8),(7,9),(8,10),  # braccia
        (5,11),(6,12),(11,12),           # torso
        (11,13),(12,14),(13,15),(14,16), # gambe
    ]
    COLOR_BOX = (0, 255, 0)
    COLOR_KPT = (0, 165, 255)
    COLOR_SKL = (255, 165, 0)

    for box, score, kp in zip(boxes, scores, kpts):
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_BOX, 1)
        cv2.putText(img, f"{label} {score:.2f}",
                    (x1, max(y1 - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_BOX, 1)

        if kp is not None:
            pts = kp.astype(int)
            for pt in pts:
                if pt[0] > 0 or pt[1] > 0:
                    cv2.circle(img, tuple(pt), 2, COLOR_KPT, -1)
            for a, b in SKELETON:
                if (pts[a][0] > 0 or pts[a][1] > 0) and (pts[b][0] > 0 or pts[b][1] > 0):
                    cv2.line(img, tuple(pts[a]), tuple(pts[b]), COLOR_SKL, 1)

    return img


# ---------------------------------------------------------------------------
# PIPELINE PER UN MODELLO SU TUTTE LE SEQUENZE
# ---------------------------------------------------------------------------
def run_pipeline(model_path: str, model_label: str,
                 test_seq_dir: Path, output_base: Path,
                 device: str, logger: logging.Logger) -> dict:

    logger.info(f"\n{'='*62}")
    logger.info(f"  MODELLO: {model_label}")
    logger.info(f"  Path   : {model_path}")
    logger.info(f"{'='*62}")

    output_dir = output_base / f"gs_{model_label.replace(' ', '_')}"

    if not Path(model_path).exists() and not model_path.endswith(".pt"):
        logger.error(f"  [SKIP] Modello non trovato: {model_path}")
        return {"model": model_label, "status": "SKIPPED", "total_detections": 0}

    try:
        model = YOLO(model_path)
    except Exception as e:
        logger.error(f"  [SKIP] Errore caricamento modello: {e}")
        return {"model": model_label, "status": "ERROR", "total_detections": 0}

    sequences = sorted([d for d in test_seq_dir.iterdir() if d.is_dir()])
    total_detections = 0
    per_seq_stats = []

    for seq_path in sequences:
        seq_name = seq_path.name
        seq_out  = output_dir / seq_name
        seq_out.mkdir(parents=True, exist_ok=True)

        logger.info(f"  [*] Sequenza: {seq_name}")
        frames = sorted([f for f in seq_path.iterdir()
                         if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        seq_det = 0

        for frame_idx, frame_path in enumerate(frames):
            img = cv2.imread(str(frame_path))
            if img is None:
                continue

            boxes, scores, kpts = process_frame_sahi(model, img, device)
            seq_det += len(boxes)

            img = render_frame(img, boxes, scores, kpts, model_label)

            out_path = seq_out / f"frame_{frame_idx:07d}.jpg"
            cv2.imwrite(str(out_path), img)

        total_detections += seq_det
        per_seq_stats.append({"sequence": seq_name, "detections": seq_det})
        logger.info(f"    Detection: {seq_det}")
        import torch as _t
        _t.cuda.empty_cache()

    logger.info(f"\n  [OK] {model_label} — Totale detection: {total_detections}")
    return {
        "model":             model_label,
        "status":            "COMPLETED",
        "total_detections":  total_detections,
        "per_sequence":      per_seq_stats,
        "output_dir":        str(output_dir),
    }


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    device = "cuda:0" if __import__("torch").cuda.is_available() else "cpu"

    script_dir   = Path(__file__).resolve().parent
    project_root = script_dir.parent
    log_dir      = project_root / "logs"

    test_seq_dir = Path(TEST_SEQ_DIR) if TEST_SEQ_DIR else (
        project_root / "datasets" / "dataset_test_official" / "sequences"
    )

    ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = project_root / "runs" / "ground_station" / f"run_{ts}"

    logger = setup_logging(log_dir)
    logger.info("=" * 62)
    logger.info("  FLYPOSE-SAR — STAZIONE DI TERRA (SAHI)")
    logger.info("=" * 62)
    logger.info(f"  Device       : {device}")
    logger.info(f"  Sequenze     : {test_seq_dir}")
    logger.info(f"  Slice size   : {SLICE_SIZE}px  overlap: {OVERLAP_RATIO*100:.0f}%")
    logger.info(f"  conf={CONF}  NMS IoU={NMS_IOU}")

    if not test_seq_dir.exists():
        logger.error(f"Dataset test non trovato: {test_seq_dir}")
        sys.exit(1)

    # ----- PRIMA VERSIONE -----

    # Esegui entrambi i modelli
    #result_official = run_pipeline(
    #    OFFICIAL_MODEL_PATH, "OFFICIAL_COCO",
    #   test_seq_dir, output_base, device, logger
    #)

    #result_custom = run_pipeline(
    #    CUSTOM_MODEL_PATH, "CUSTOM_SAR",
    #    test_seq_dir, output_base, device, logger
    #)

    # ----- SECONDA VERSIONE -----
    
    #result_official = run_pipeline(
    #    OFFICIAL_MODEL_PATH, "NANO_SAR",
    #    test_seq_dir, output_base, device, logger
    #)

    #result_custom = run_pipeline(
    #    CUSTOM_MODEL_PATH, "SMALL_SAR",
    #    test_seq_dir, output_base, device, logger
    #)

    # ----- TERZA VERSIONE -----
    
    result_official = run_pipeline(
        OFFICIAL_MODEL_PATH, "SMALL_SAR",
        test_seq_dir, output_base, device, logger
    )

    result_custom = run_pipeline(
        CUSTOM_MODEL_PATH, "LARGE_SAR",
        test_seq_dir, output_base, device, logger
    )

    # Report comparativo
    report = {
        "generated_at": datetime.now().isoformat(),
        "slice_size":   SLICE_SIZE,
        "overlap":      OVERLAP_RATIO,
        "conf":         CONF,
        "nms_iou":      NMS_IOU,
        "official":     result_official,
        "custom":       result_custom,
        "delta_detections": (
            result_custom["total_detections"] - result_official["total_detections"]
        ),
    }

    output_base.mkdir(parents=True, exist_ok=True)
    report_path = output_base / "comparison_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info("")
    logger.info("=" * 62)
    logger.info("  CONFRONTO FINALE")
    logger.info("=" * 62)
    logger.info(f"  Modello COCO ufficiale : {result_official['total_detections']} detection")
    logger.info(f"  Modello custom SAR     : {result_custom['total_detections']} detection")
    logger.info(f"  Delta                  : {report['delta_detections']:+d}")
    logger.info(f"  Report                 : {report_path}")
    logger.info("=" * 62)
    logger.info("\n[DONE] Stazione di terra completata.")


if __name__ == "__main__":
    main()