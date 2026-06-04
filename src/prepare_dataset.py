# src/prepare_dataset.py
# FlyPose-SAR — Preparazione Dataset con approccio ibrido GT+Teacher
#
# STRATEGIA:
#   Le bounding box vengono lette dalle annotazioni VisDrone (ground truth reale).
#   Il teacher opera SOLO sul crop di ogni persona gia localizzata,
#   per predire i 17 keypoints COCO. I keypoints vengono riproiettati
#   in coordinate assolute del frame e normalizzati in formato YOLO Pose.
#
# STRUTTURA ATTESA:
#   VisDrone2019-VID-train/
#     annotations/   <- un .txt per ogni sequenza (stesso nome cartella)
#     sequences/
#       uav0000013_00000_v/
#         0000001.jpg
#         0000002.jpg
#
# FORMATO ANNOTAZIONI VISDRONE (una riga per oggetto per frame):
#   frame_index, target_id, x_min, y_min, width, height, score, category, truncation, occlusion
#   categoria 1=pedone, 2=persona in piedi
#
# FORMATO OUTPUT YOLO POSE:
#   cls cx cy w h  kx1 ky1 vis1  ...  kx17 ky17 vis17  (tutto normalizzato in [0,1])

import cv2
import json
import torch
import shutil
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from ultralytics import YOLO

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# ---------------------------------------------------------------------------
# PARAMETRI DI FILTRAGGIO
# ---------------------------------------------------------------------------
PERSON_CATEGORIES  = {1, 2}   # 1=pedestrian, 2=people
MAX_OCCLUSION      = 1        # scarta occlusion==2 (troppo nascosta)
MIN_BOX_W          = 8        # pixel minimi larghezza box
MIN_BOX_H          = 8        # pixel minimi altezza box
CROP_PADDING       = 0.20     # 20% padding intorno alla box per il crop
TEACHER_CONF       = 0.25     # confidenza teacher sul crop (bassa: sappiamo che c'e' una persona)
MIN_KPT_CONF_MEAN  = 0.20     # confidenza media minima keypoints per accettare


# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------
def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"prepare_dataset_{ts}.log"
    logger = logging.getLogger("FlyPose_PrepareDataset")
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
# PARSING ANNOTAZIONI VISDRONE
# ---------------------------------------------------------------------------
def parse_visdrone_annotations(ann_file: Path) -> defaultdict:
    """
    Legge un file annotazioni VisDrone-VID.
    Restituisce { frame_index: [ {x,y,w,h,truncation,occlusion}, ... ] }
    gia filtrato per categoria persona e dimensione minima.
    """
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
                "truncation": truncation,
                "occlusion": occlusion,
            })

    return annotations


# ---------------------------------------------------------------------------
# CROP CON PADDING
# ---------------------------------------------------------------------------
def crop_with_padding(frame: np.ndarray, x: int, y: int, w: int, h: int):
    """
    Ritaglia la regione con padding percentuale.
    Ritorna (crop_img, x_offset, y_offset) — offsets usati per riproiettare i kpts.
    """
    fh, fw = frame.shape[:2]
    pad_x = int(w * CROP_PADDING)
    pad_y = int(h * CROP_PADDING)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(fw, x + w + pad_x)
    y2 = min(fh, y + h + pad_y)
    return frame[y1:y2, x1:x2].copy(), x1, y1


# ---------------------------------------------------------------------------
# KEYPOINTS DAL TEACHER SU CROP
# ---------------------------------------------------------------------------
def get_keypoints_from_crop(teacher_model, crop: np.ndarray,
                             x_offset: int, y_offset: int,
                             frame_w: int, frame_h: int,
                             device: str):
    """
    Esegue il teacher sul crop e restituisce 17 keypoints normalizzati
    rispetto al frame originale, oppure None se il teacher fallisce.
    Formato ritornato: lista di (kx_norm, ky_norm, visibility).
    """
    if crop is None or crop.size == 0:
        return None

    # Upscale minimo per dare abbastanza pixel al teacher
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

    # Detection piu' sicura = la persona nel crop
    confs = results.boxes.conf.cpu().numpy()
    best  = int(np.argmax(confs))

    kpts      = results.keypoints.xy.cpu().numpy()[best]    # (17, 2) in px nel crop scalato
    kpts_conf = results.keypoints.conf.cpu().numpy()[best]  # (17,)

    if np.mean(kpts_conf) < MIN_KPT_CONF_MEAN:
        return None

    kpts_abs = []
    for kp, kconf in zip(kpts, kpts_conf):
        # Riporta dal crop scalato al crop originale
        kx_orig = kp[0] / scale
        ky_orig = kp[1] / scale

        # Riproietta in coordinate assolute del frame
        kx_frame = kx_orig + x_offset
        ky_frame = ky_orig + y_offset

        # Normalizza in [0,1]
        kx_norm = float(np.clip(kx_frame / frame_w, 0.0, 1.0))
        ky_norm = float(np.clip(ky_frame / frame_h, 0.0, 1.0))

        if kx_norm == 0.0 and ky_norm == 0.0:
            vis = 0
        elif float(kconf) < 0.3:
            vis = 1
        else:
            vis = 2

        kpts_abs.append((kx_norm, ky_norm, vis))

    return kpts_abs


# ---------------------------------------------------------------------------
# PROCESSAMENTO SPLIT
# ---------------------------------------------------------------------------
def process_split(teacher_model, src_base: Path, dest_base: Path,
                  split: str, device: str, logger: logging.Logger,
                  clean_existing: bool = False) -> dict:

    seq_dir = src_base / "sequences"
    ann_dir = src_base / "annotations"

    stats = {
        "split": split,
        "sequences_found": 0,
        "frames_processed": 0,
        "persons_found_gt": 0,
        "annotations_saved": 0,
        "frames_saved": 0,
        "keypoints_failed": 0,
        "per_sequence": {},
    }

    if not seq_dir.exists() or not ann_dir.exists():
        logger.error(f"[{split.upper()}] Cartelle non trovate in: {src_base}")
        return stats

    dest_img = dest_base / "images" / split
    dest_lbl = dest_base / "labels" / split

    if clean_existing:
        logger.info(f"[{split.upper()}] Pulizia cartelle destinazione...")
        for folder in [dest_img, dest_lbl]:
            if folder.exists():
                shutil.rmtree(folder)

    dest_img.mkdir(parents=True, exist_ok=True)
    dest_lbl.mkdir(parents=True, exist_ok=True)

    seq_dirs = sorted([d for d in seq_dir.iterdir() if d.is_dir()])
    stats["sequences_found"] = len(seq_dirs)
    logger.info(f"[{split.upper()}] Trovate {len(seq_dirs)} sequenze.")

    iterator = tqdm(seq_dirs, desc=f"  [{split}]", unit="seq") if TQDM_AVAILABLE else seq_dirs

    for seq_path in iterator:
        seq_name = seq_path.name
        ann_file = ann_dir / f"{seq_name}.txt"
        seq_stats = {
            "frames_processed": 0, "persons_gt": 0,
            "annotations_saved": 0, "frames_saved": 0, "keypoints_failed": 0,
        }

        if not ann_file.exists():
            logger.warning(f"  [{split}] Annotazione mancante per {seq_name}. Skip.")
            continue

        seq_annotations = parse_visdrone_annotations(ann_file)
        if not seq_annotations:
            logger.debug(f"  [{split}] {seq_name}: nessuna persona. Skip.")
            continue

        img_files = sorted([
            f for f in seq_path.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ])

        if not img_files:
            logger.warning(f"  [{split}] Nessuna immagine in {seq_path}. Skip.")
            continue

        for img_path in img_files:
            # Ricava l'indice frame dal nome file (es. "0000001" -> 1)
            try:
                frame_idx = int(img_path.stem)
            except ValueError:
                digits = "".join(c for c in img_path.stem if c.isdigit())
                frame_idx = int(digits) if digits else -1

            persons_in_frame = seq_annotations.get(frame_idx, [])
            if not persons_in_frame:
                continue

            nuovo_nome    = f"{seq_name}_{img_path.stem}"
            dest_lbl_file = dest_lbl / f"{nuovo_nome}.txt"

            # Resume: salta se gia processato
            if dest_lbl_file.exists():
                seq_stats["frames_saved"] += 1
                stats["frames_saved"] += 1
                continue

            frame = cv2.imread(str(img_path))
            if frame is None:
                continue

            fh, fw = frame.shape[:2]
            stats["frames_processed"] += 1
            seq_stats["frames_processed"] += 1

            yolo_pose_lines = []

            for person in persons_in_frame:
                x, y, w, h = person["x"], person["y"], person["w"], person["h"]
                stats["persons_found_gt"] += 1
                seq_stats["persons_gt"] += 1

                crop, x_off, y_off = crop_with_padding(frame, x, y, w, h)

                kpts = get_keypoints_from_crop(
                    teacher_model, crop, x_off, y_off, fw, fh, device
                )

                if kpts is None:
                    stats["keypoints_failed"] += 1
                    seq_stats["keypoints_failed"] += 1
                    continue

                # Box in formato YOLO normalizzato
                cx_norm = float(np.clip((x + w / 2) / fw, 0.0, 1.0))
                cy_norm = float(np.clip((y + h / 2) / fh, 0.0, 1.0))
                w_norm  = float(np.clip(w / fw,           0.0, 1.0))
                h_norm  = float(np.clip(h / fh,           0.0, 1.0))

                box_str = f"0 {cx_norm:.6f} {cy_norm:.6f} {w_norm:.6f} {h_norm:.6f}"
                kpt_str = " ".join(f"{kx:.6f} {ky:.6f} {vis}" for kx, ky, vis in kpts)
                yolo_pose_lines.append(f"{box_str} {kpt_str}")

            if yolo_pose_lines:
                shutil.copy(str(img_path), str(dest_img / f"{nuovo_nome}.jpg"))
                with open(dest_lbl_file, "w") as f:
                    f.write("\n".join(yolo_pose_lines) + "\n")
                stats["annotations_saved"] += len(yolo_pose_lines)
                stats["frames_saved"] += 1
                seq_stats["annotations_saved"] += len(yolo_pose_lines)
                seq_stats["frames_saved"] += 1

        stats["per_sequence"][seq_name] = seq_stats
        torch.cuda.empty_cache()

    logger.info(
        f"[{split.upper()}] Fatto — "
        f"Frame salvati: {stats['frames_saved']} | "
        f"Persone GT: {stats['persons_found_gt']} | "
        f"Annotazioni: {stats['annotations_saved']} | "
        f"Kpts falliti: {stats['keypoints_failed']}"
    )
    return stats


# ---------------------------------------------------------------------------
# REPORT
# ---------------------------------------------------------------------------
def save_report(stats_train: dict, stats_val: dict, dest_base: Path, logger: logging.Logger):
    report = {
        "generated_at": datetime.now().isoformat(),
        "approach": "GT box VisDrone + keypoints teacher su crop",
        "filters": {
            "person_categories": list(PERSON_CATEGORIES),
            "max_occlusion": MAX_OCCLUSION,
            "min_box_px": f"{MIN_BOX_W}x{MIN_BOX_H}",
            "crop_padding": CROP_PADDING,
            "teacher_conf": TEACHER_CONF,
            "min_kpt_conf_mean": MIN_KPT_CONF_MEAN,
        },
        "train": stats_train,
        "val":   stats_val,
        "totals": {
            "frames_saved":       stats_train["frames_saved"]       + stats_val["frames_saved"],
            "annotations_saved":  stats_train["annotations_saved"]  + stats_val["annotations_saved"],
            "persons_found_gt":   stats_train["persons_found_gt"]   + stats_val["persons_found_gt"],
            "keypoints_failed":   stats_train["keypoints_failed"]   + stats_val["keypoints_failed"],
        },
    }

    report_path = dest_base / "dataset_preparation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    t = report["totals"]
    kpt_rate = (1 - t["keypoints_failed"] / max(t["persons_found_gt"], 1)) * 100

    logger.info("")
    logger.info("=" * 62)
    logger.info("  RIEPILOGO FINALE")
    logger.info("=" * 62)
    logger.info(f"  Train — frame salvati    : {stats_train['frames_saved']}")
    logger.info(f"  Train — annotazioni      : {stats_train['annotations_saved']}")
    logger.info(f"  Val   — frame salvati    : {stats_val['frames_saved']}")
    logger.info(f"  Val   — annotazioni      : {stats_val['annotations_saved']}")
    logger.info(f"  Persone GT totali        : {t['persons_found_gt']}")
    logger.info(f"  Tasso successo keypoints : {kpt_rate:.1f}%")
    logger.info(f"  Report                   : {report_path}")
    logger.info("=" * 62)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    PATH_TRAIN = Path(r"C:\Dataset_ViSDrone_RGB\VisDrone2019-VID-train")
    PATH_VAL   = Path(r"C:\Dataset_ViSDrone_RGB\VisDrone2019-VID-val")

    TEACHER_MODEL_PATH = "modelli_base/yolo11l-pose.pt"
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

    script_dir   = Path(__file__).resolve().parent
    project_root = script_dir.parent
    dest_base    = project_root / "datasets" / "dataset_sar"
    log_dir      = project_root / "logs"

    logger = setup_logging(log_dir)
    logger.info("=" * 62)
    logger.info("  FLYPOSE-SAR — PREPARAZIONE DATASET (APPROCCIO IBRIDO)")
    logger.info("  Box da GT VisDrone + Keypoints da teacher su crop")
    logger.info("=" * 62)
    logger.info(f"  Device        : {DEVICE}")
    if DEVICE == "cuda:0":
        logger.info(f"  GPU           : {torch.cuda.get_device_name(0)}")
    logger.info(f"  Teacher model : {TEACHER_MODEL_PATH}")
    logger.info(f"  Train src     : {PATH_TRAIN}")
    logger.info(f"  Val src       : {PATH_VAL}")
    logger.info(f"  Destinazione  : {dest_base}")
    logger.info(f"  Filtri        : cat={PERSON_CATEGORIES}, occ<={MAX_OCCLUSION}, box>={MIN_BOX_W}x{MIN_BOX_H}px")

    logger.info("\n[*] Caricamento teacher model...")
    teacher_model = YOLO(TEACHER_MODEL_PATH)
    teacher_model.to(DEVICE)
    logger.info("[+] Teacher model pronto.\n")

    stats_train = process_split(
        teacher_model, PATH_TRAIN, dest_base,
        split="train", device=DEVICE, logger=logger,
        clean_existing=False,
    )

    stats_val = process_split(
        teacher_model, PATH_VAL, dest_base,
        split="val", device=DEVICE, logger=logger,
        clean_existing=True,
    )

    save_report(stats_train, stats_val, dest_base, logger)
    logger.info("\n[DONE] Preparazione dataset completata.")


if __name__ == "__main__":
    main()