# src/inference.py
# FlyPose-SAR — Inferenza su sequenze video (modalita' drone)
#
# COSA FA:
#   Esegue il modello addestrato su sequenze di frame reali,
#   producendo video annotati con bounding box, skeleton COCO a 17 punti
#   e ID di tracking persistente per ogni persona.
#   Pensato per girare A BORDO del drone (o in simulazione locale).
#
# DIFFERENZA CON inference_ground_station.py:
#   Questo script usa la risoluzione nativa del modello (640 o 1280px)
#   senza SAHI slicing. E' piu' veloce e adatto a hardware embedded.
#   La ground station invece fa tassellatura per massimizzare la precisione
#   su target microscopici, a costo di piu' tempo di elaborazione.
#
# TRACKING — PERCHE' BYTETRACK:
#   Ogni persona rilevata riceve un ID numerico che persiste nel tempo.
#   ByteTrack gestisce le occlusioni temporanee (persona dietro un albero
#   per 1-2 frame) meglio di SORT grazie al doppio buffer: tiene traccia
#   sia delle detection ad alta confidenza che di quelle borderline.
#   Questo riduce i "flickering" degli ID tipici delle riprese zenitali
#   dove i target entrano ed escono dall'inquadratura frequentemente.
#
# PARAMETRI CHIAVE:
#   conf=0.40   -> soglia confidenza detection. Alzare se troppi falsi positivi
#                  (ombre, veicoli). Abbassare se si perdono persone lontane.
#   iou=0.35    -> soglia NMS per sopprimere box duplicate. Valore basso = piu'
#                  aggressivo nel sopprimere. Utile in scene affollate zenitali.
#   imgsz=1280  -> risoluzione di inferenza. Piu' alta = piu' dettagli per i
#                  target piccoli, ma piu' VRAM e latenza. 640 per real-time stretto.
#
# OUTPUT:
#   runs/inferenza/<nome_sequenza>/  <- frame annotati (.jpg) o video (.mp4)
#   runs/inferenza/inference_report.json  <- statistiche per sequenza

import os
import sys
import json
import torch
import logging
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# CONFIGURAZIONE
# Modifica MODEL_PATH con il percorso al best.pt del tuo run migliore.
# Trovi il percorso nel comparison_report_*.json sotto "best_weights".
# ---------------------------------------------------------------------------
MODEL_PATH   = r"runs\fase1\fase1_small\weights\best.pt"
TEST_SEQ_DIR = None   # None = usa il percorso relativo al progetto (vedi main)

CONF_THRESHOLD = 0.40
IOU_THRESHOLD  = 0.35
IMGSZ          = 1280   # abbassa a 640 se vuoi piu' velocita'
TRACKER        = "bytetrack.yaml"
SAVE_VIDEO     = True   # True = salva video MP4, False = salva singoli frame JPG


# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------
def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"inference_{ts}.log"
    logger = logging.getLogger("FlyPose_Inference")
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
# INFERENZA SU UNA SEQUENZA
#
# model.track() combina detection + tracking in un unico passaggio.
# Internamente:
#   1. Il frame viene ridimensionato a imgsz e passato alla backbone CNN
#   2. La FPN (Feature Pyramid Network) estrae feature a piu' scale
#   3. Le teste di detection e pose stimano box, confidenza e 17 keypoints
#   4. NMS rimuove le box duplicate (soglia iou)
#   5. ByteTrack associa le detection attuali alle tracce esistenti
#      usando la distanza IoU e opzionalmente il Kalman filter per predire
#      la posizione delle tracce non viste nell'ultimo frame
# ---------------------------------------------------------------------------
def run_sequence(model, seq_path: Path, output_base: Path,
                 logger: logging.Logger) -> dict:
    seq_name = seq_path.name
    output_dir = output_base / seq_name
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"  [*] Sequenza: {seq_name}")

    seq_stats = {
        "sequence": seq_name,
        "total_detections": 0,
        "unique_track_ids": set(),
        "frames_with_detections": 0,
    }

    try:
        # model.track() e' un generatore: processa un frame alla volta.
        # Questo e' importante per la memoria: non carica tutta la sequenza in RAM.
        results_gen = model.track(
            source    = str(seq_path),
            conf      = CONF_THRESHOLD,
            iou       = IOU_THRESHOLD,
            imgsz     = IMGSZ,
            tracker   = TRACKER,
            show      = False,
            save      = SAVE_VIDEO,
            project   = str(output_base),
            name      = seq_name,
            exist_ok  = True,
            stream    = True,   # generatore frame-by-frame: fondamentale per sequenze lunghe
        )

        for frame_result in results_gen:
            # Ogni frame_result contiene:
            #   .boxes     -> bounding box con confidenza e class
            #   .keypoints -> 17 keypoints con coordinate e confidenza
            #   .boxes.id  -> ID di tracking assegnati da ByteTrack

            if frame_result.boxes is not None and len(frame_result.boxes) > 0:
                n_det = len(frame_result.boxes)
                seq_stats["total_detections"] += n_det
                seq_stats["frames_with_detections"] += 1

                # Raccoglie gli ID di tracking univoci visti nella sequenza
                if frame_result.boxes.id is not None:
                    ids = frame_result.boxes.id.cpu().numpy().astype(int)
                    seq_stats["unique_track_ids"].update(ids.tolist())

    except Exception as e:
        logger.error(f"    [ERRORE] {seq_name}: {e}", exc_info=True)
        seq_stats["error"] = str(e)

    # Converti il set in lista per la serializzazione JSON
    n_unique = len(seq_stats["unique_track_ids"])
    seq_stats["unique_track_ids"] = sorted(list(seq_stats["unique_track_ids"]))
    seq_stats["unique_persons_tracked"] = n_unique

    logger.info(
        f"    Detection totali    : {seq_stats['total_detections']}"
    )
    logger.info(
        f"    Persone tracciate   : {n_unique} ID univoci"
    )
    logger.info(
        f"    Frame con detection : {seq_stats['frames_with_detections']}"
    )
    return seq_stats


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    script_dir   = Path(__file__).resolve().parent
    project_root = script_dir.parent
    log_dir      = project_root / "logs"

    # Percorso dataset di test
    test_seq_dir = Path(TEST_SEQ_DIR) if TEST_SEQ_DIR else (
        project_root / "datasets" / "dataset_test_official" / "sequences"
    )

    # Output nella cartella runs/inferenza con timestamp
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = project_root / "runs" / "inferenza" / f"run_{ts}"

    logger = setup_logging(log_dir)
    logger.info("=" * 62)
    logger.info("  FLYPOSE-SAR — INFERENZA SU SEQUENZE (MODALITA' DRONE)")
    logger.info("=" * 62)
    logger.info(f"  Device       : {device}")
    if device == "cuda:0":
        logger.info(f"  GPU          : {torch.cuda.get_device_name(0)}")
    logger.info(f"  Modello      : {MODEL_PATH}")
    logger.info(f"  Sequenze     : {test_seq_dir}")
    logger.info(f"  Output       : {output_base}")
    logger.info(f"  conf={CONF_THRESHOLD}  iou={IOU_THRESHOLD}  imgsz={IMGSZ}  tracker={TRACKER}")

    # Verifica esistenza modello
    if not Path(MODEL_PATH).exists():
        logger.error(f"[ERRORE] Modello non trovato: {MODEL_PATH}")
        logger.error("  Aggiorna MODEL_PATH con il percorso al tuo best.pt")
        logger.error("  Lo trovi in: runs/fase1/<run_name>/weights/best.pt")
        sys.exit(1)

    # Verifica esistenza dataset di test
    if not test_seq_dir.exists():
        logger.error(f"[ERRORE] Dataset test non trovato: {test_seq_dir}")
        sys.exit(1)

    sequences = sorted([d for d in test_seq_dir.iterdir() if d.is_dir()])
    if not sequences:
        logger.error(f"[ERRORE] Nessuna sequenza trovata in: {test_seq_dir}")
        sys.exit(1)

    logger.info(f"\n[*] Trovate {len(sequences)} sequenze da processare.")
    logger.info("[*] Caricamento modello...")
    model = YOLO(MODEL_PATH)
    logger.info("[+] Modello caricato.\n")

    all_stats = []
    for seq_path in sequences:
        stats = run_sequence(model, seq_path, output_base, logger)
        all_stats.append(stats)
        torch.cuda.empty_cache()

    # Report finale
    total_det      = sum(s["total_detections"]       for s in all_stats)
    total_persons  = sum(s["unique_persons_tracked"]  for s in all_stats)
    total_frames   = sum(s["frames_with_detections"]  for s in all_stats)

    report = {
        "generated_at":          datetime.now().isoformat(),
        "model":                  MODEL_PATH,
        "conf":                   CONF_THRESHOLD,
        "iou":                    IOU_THRESHOLD,
        "imgsz":                  IMGSZ,
        "tracker":                TRACKER,
        "sequences_processed":    len(all_stats),
        "total_detections":       total_det,
        "total_unique_persons":   total_persons,
        "total_frames_with_det":  total_frames,
        "per_sequence":           all_stats,
    }

    output_base.mkdir(parents=True, exist_ok=True)
    report_path = output_base / "inference_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info("")
    logger.info("=" * 62)
    logger.info("  RIEPILOGO INFERENZA")
    logger.info("=" * 62)
    logger.info(f"  Sequenze processate   : {len(all_stats)}")
    logger.info(f"  Detection totali      : {total_det}")
    logger.info(f"  Persone tracciate     : {total_persons} ID univoci")
    logger.info(f"  Frame con detection   : {total_frames}")
    logger.info(f"  Output                : {output_base}")
    logger.info(f"  Report                : {report_path}")
    logger.info("=" * 62)
    logger.info("\n[DONE] Inferenza completata.")


if __name__ == "__main__":
    main()