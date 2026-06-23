# src/demo_realtime.py
# FlyPose-SAR — Demo real-time con webcam o video file
#
# COSA FA:
#   Apre un flusso video (webcam, file MP4, o stream RTSP) e applica
#   il modello YOLO11-Pose in tempo reale mostrando:
#   - Bounding box con confidenza e ID tracking
#   - Skeleton COCO 17 keypoints con wireframe colorato
#   - FPS e statistiche in overlay
#   - Stato posturale stimato (in piedi / a terra / supino)
#
# USO:
#   # Webcam
#   python src/demo_realtime.py
#
#   # File video
#   python src/demo_realtime.py --source runs/inferenza/run_.../uav0000073.mp4
#
#   # Stream RTSP drone
#   python src/demo_realtime.py --source rtsp://192.168.1.1/stream
#
#   # Modello specifico
#   python src/demo_realtime.py --model runs/fase1/fase1_large/weights/best.pt
#
# TASTI DURANTE LA DEMO:
#   Q / ESC  — esci
#   S        — salva screenshot
#   P        — pausa/riprendi
#   W        — toggle wireframe on/off
#   B        — toggle bounding box on/off
#   I        — toggle info overlay on/off
#   +/-      — aumenta/diminuisci soglia confidenza

import cv2
import torch
import argparse
import numpy as np
import time
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# SKELETON COCO 17 keypoints — connessioni per il wireframe
# ---------------------------------------------------------------------------
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2),           # naso → occhi
    (1, 3), (2, 4),           # occhi → orecchie
    (5, 6),                   # spalle
    (5, 7), (7, 9),           # braccio sinistro
    (6, 8), (8, 10),          # braccio destro
    (5, 11), (6, 12),         # torso
    (11, 12),                 # fianchi
    (11, 13), (13, 15),       # gamba sinistra
    (12, 14), (14, 16),       # gamba destra
]

# Colori per zona corporea (BGR)
KPT_COLORS = {
    "head":   (0,   255, 255),   # giallo — testa (0-4)
    "arms":   (0,   165, 255),   # arancione — braccia (5-10)
    "torso":  (0,   255, 0),     # verde — torso (5,6,11,12)
    "legs":   (255, 100, 0),     # blu — gambe (11-16)
}

SKELETON_COLORS = [
    (0,255,255),(0,255,255),          # naso-occhi
    (0,255,255),(0,255,255),          # occhi-orecchie
    (0,255,0),                        # spalle
    (0,165,255),(0,165,255),          # braccio sx
    (0,165,255),(0,165,255),          # braccio dx
    (0,255,0),(0,255,0),              # torso
    (0,255,0),                        # fianchi
    (255,100,0),(255,100,0),          # gamba sx
    (255,100,0),(255,100,0),          # gamba dx
]

KPT_ZONE_COLORS = [
    (0,255,255),(0,255,255),(0,255,255),(0,255,255),(0,255,255),  # 0-4 testa
    (0,165,255),(0,165,255),(0,165,255),(0,165,255),(0,165,255),(0,165,255),  # 5-10 braccia
    (0,255,0),(0,255,0),              # 11-12 fianchi
    (255,100,0),(255,100,0),(255,100,0),(255,100,0),  # 13-16 gambe
]

# ---------------------------------------------------------------------------
# STIMA STATO POSTURALE
# ---------------------------------------------------------------------------
def estimate_posture(keypoints: np.ndarray, box=None) -> str:
    """
    Stima lo stato posturale dalla geometria dello scheletro in visione ZENITALE.

    In visione zenitale la logica frontale (spalle vs fianchi in Y) non funziona
    perche' dall'alto sia una persona in piedi che una a terra hanno spalle e
    fianchi sullo stesso piano orizzontale del frame.

    Metriche corrette per visione zenitale:

    1. ASPECT RATIO della bounding box:
       - In piedi: corpo visto dall'alto e' compatto, quasi circolare → box ~ quadrata (ratio ~1)
       - A terra: corpo disteso occupa piu' spazio → box allungata (ratio > 2)

    2. DISPERSIONE dei keypoints (std delle coordinate):
       - In piedi: keypoints concentrati in area piccola (persona compatta)
       - A terra: keypoints dispersi su area piu' grande (corpo disteso)

    3. AREA relativa della box rispetto alla risoluzione:
       - Persona in piedi occupa meno pixel zenitalmente
       - Persona a terra occupa piu' pixel (corpo disteso)
    """
    if keypoints is None or len(keypoints) == 0:
        return "?"

    # Filtra keypoints validi (conf > 0.25)
    valid = keypoints[keypoints[:, 2] > 0.25]
    if len(valid) < 3:
        return "INCERTO"

    # --- METRICA 1: aspect ratio bounding box ---
    ar_score = None
    if box is not None:
        bw = abs(box[2] - box[0])
        bh = abs(box[3] - box[1])
        if bh > 0 and bw > 0:
            ar = max(bw, bh) / min(bw, bh)
            ar_score = ar   # ~1 = compatto (in piedi), >2 = allungato (a terra)

    # --- METRICA 2: dispersione keypoints ---
    kpt_std_x = np.std(valid[:, 0])
    kpt_std_y = np.std(valid[:, 1])
    kpt_spread = np.sqrt(kpt_std_x**2 + kpt_std_y**2)

    # --- METRICA 3: rapporto dispersione/box ---
    if box is not None:
        bw = abs(box[2] - box[0])
        bh = abs(box[3] - box[1])
        box_diag = np.sqrt(bw**2 + bh**2)
        spread_ratio = kpt_spread / max(box_diag, 1)
    else:
        spread_ratio = kpt_spread / 50.0  # fallback

    # --- CLASSIFICAZIONE ---
    # Combina aspect ratio e dispersione per la decisione finale

    if ar_score is not None:
        if ar_score > 2.8:
            # Box molto allungata → corpo disteso → A TERRA
            return "A TERRA"
        elif ar_score > 2.0:
            # Box moderatamente allungata + keypoints dispersi → A TERRA o FERITO
            if spread_ratio > 0.35:
                return "A TERRA"
            else:
                return "SEDUTO"
        else:
            # Box compatta → persona in piedi o rannicchiata
            if spread_ratio > 0.45:
                return "RANNICCHIATO"
            else:
                return "IN PIEDI"
    else:
        # Fallback solo su dispersione keypoints
        if spread_ratio > 0.45:
            return "A TERRA"
        elif spread_ratio > 0.30:
            return "SEDUTO"
        else:
            return "IN PIEDI"

POSTURE_COLORS = {
    "IN PIEDI":     (0, 255, 0),
    "SEDUTO":       (0, 165, 255),
    "A TERRA":      (0, 0, 255),
    "RANNICCHIATO": (255, 0, 255),
    "INCERTO":      (128, 128, 128),
    "?":            (128, 128, 128),
}

# ---------------------------------------------------------------------------
# RENDERING SKELETON
# ---------------------------------------------------------------------------
def draw_skeleton(frame: np.ndarray, keypoints: np.ndarray,
                  show_wireframe: bool = True,
                  kpt_radius: int = 4,
                  line_thickness: int = 2) -> np.ndarray:
    """
    Disegna keypoints e connessioni scheletro sul frame.
    keypoints: array (17, 3) con (x, y, conf) in pixel assoluti
    """
    if keypoints is None:
        return frame

    h, w = frame.shape[:2]
    pts  = keypoints[:, :2].astype(int)
    conf = keypoints[:, 2]

    if show_wireframe:
        for i, (a, b) in enumerate(SKELETON_CONNECTIONS):
            if conf[a] > 0.3 and conf[b] > 0.3:
                if 0 <= pts[a][0] < w and 0 <= pts[a][1] < h and \
                   0 <= pts[b][0] < w and 0 <= pts[b][1] < h:
                    color = SKELETON_COLORS[i] if i < len(SKELETON_COLORS) else (255,255,255)
                    cv2.line(frame, tuple(pts[a]), tuple(pts[b]),
                             color, line_thickness, cv2.LINE_AA)

    for i, (pt, c) in enumerate(zip(pts, conf)):
        if c > 0.3 and 0 <= pt[0] < w and 0 <= pt[1] < h:
            color = KPT_ZONE_COLORS[i] if i < len(KPT_ZONE_COLORS) else (255,255,255)
            cv2.circle(frame, tuple(pt), kpt_radius, color, -1, cv2.LINE_AA)
            cv2.circle(frame, tuple(pt), kpt_radius + 1, (0,0,0), 1, cv2.LINE_AA)

    return frame

# ---------------------------------------------------------------------------
# RENDERING BOX E INFO
# ---------------------------------------------------------------------------
def draw_detection(frame: np.ndarray, box, track_id: int,
                   conf: float, posture: str,
                   show_box: bool = True) -> np.ndarray:
    """
    Disegna bounding box e label con ID tracking e stato posturale.
    """
    if not show_box:
        return frame

    x1, y1, x2, y2 = [int(v) for v in box]
    color = POSTURE_COLORS.get(posture, (0, 255, 0))

    # Box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

    # Label
    label = f"ID:{track_id} {conf:.2f} | {posture}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    y_label = max(y1 - 6, th + 6)
    cv2.rectangle(frame, (x1, y_label - th - 4), (x1 + tw + 4, y_label + 2), color, -1)
    cv2.putText(frame, label, (x1 + 2, y_label),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    return frame

# ---------------------------------------------------------------------------
# OVERLAY INFO
# ---------------------------------------------------------------------------
def draw_info_overlay(frame: np.ndarray, fps: float, n_persons: int,
                      conf_threshold: float, model_name: str,
                      show_info: bool = True) -> np.ndarray:
    if not show_info:
        return frame

    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Sfondo semitrasparente in alto a sinistra
    cv2.rectangle(overlay, (8, 8), (300, 110), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    info_lines = [
        f"FlyPose-SAR  v1.0",
        f"Modello: {model_name}",
        f"FPS: {fps:.1f}",
        f"Persone: {n_persons}",
        f"Conf: {conf_threshold:.2f}  [+/-]",
    ]

    for i, line in enumerate(info_lines):
        color = (0, 255, 255) if i == 0 else (220, 220, 220)
        cv2.putText(frame, line, (14, 28 + i * 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    # Legenda posture
    legend = [("IN PIEDI",(0,255,0)), ("SEDUTO",(0,165,255)),
              ("A TERRA",(0,0,255)), ("INCERTO",(128,128,128))]
    for i, (label, color) in enumerate(legend):
        x = w - 130
        y = 28 + i * 18
        cv2.circle(frame, (x, y - 5), 5, color, -1)
        cv2.putText(frame, label, (x + 12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

    # Tasti
    keys = "[Q] Esci  [S] Screenshot  [P] Pausa  [W] Wire  [B] Box  [I] Info"
    cv2.putText(frame, keys, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1, cv2.LINE_AA)

    return frame

# ---------------------------------------------------------------------------
# MAIN DEMO
# ---------------------------------------------------------------------------
def run_demo(source, model_path: str, conf: float = 0.40,
             iou: float = 0.35, imgsz: int = 640,
             save_output: bool = False):

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*55}")
    print(f"  FLYPOSE-SAR — DEMO REAL-TIME")
    print(f"{'='*55}")
    print(f"  Modello : {model_path}")
    print(f"  Sorgente: {source}")
    print(f"  Device  : {device}")
    print(f"  conf={conf}  iou={iou}  imgsz={imgsz}")
    print(f"\n  Tasti: Q=esci  S=screenshot  P=pausa")
    print(f"         W=wireframe  B=box  I=info  +/-=conf\n")

    # Carica modello
    print("[*] Caricamento modello...")
    model = YOLO(model_path)
    model_name = Path(model_path).parent.parent.name
    print(f"[+] Modello caricato: {model_name}")

    # Apri sorgente video — supporta cartelle di frame JPG e file video
    frame_list = []
    frame_idx  = 0
    cap        = None

    src_path = Path(source) if isinstance(source, str) else None
    if src_path and src_path.is_dir():
        frame_list = sorted([
            f for f in src_path.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ])
        if not frame_list:
            print(f"[ERRORE] Nessun frame trovato in: {source}")
            return
        first   = cv2.imread(str(frame_list[0]))
        h, w    = first.shape[:2]
        fps_src = 30.0
        print(f"[+] Sequenza: {len(frame_list)} frame  {w}x{h}")
    else:
        src = int(source) if isinstance(source, str) and source.isdigit() else source
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            print(f"[ERRORE] Impossibile aprire: {source}")
            return
        w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_src = cap.get(cv2.CAP_PROP_FPS) or 30
        print(f"[+] Video: {w}x{h} @ {fps_src:.1f}fps")
    # Writer output
    writer = None
    if save_output:
        out_dir = Path("runs/demo")
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"demo_{ts}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps_src, (w, h))
        print(f"[+] Salvataggio in: {out_path}")

    # Stato UI
    show_wireframe = True
    show_box       = True
    show_info      = True
    paused         = False
    conf_threshold = conf

    # FPS tracking
    fps_history = []
    t_prev      = time.time()

    screenshot_dir = Path("runs/demo/screenshots")

    while True:
        # Gestione tasti
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):       # Q o ESC
            break
        elif key == ord('s'):           # Screenshot
            screenshot_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S%f")[:19]
            path = screenshot_dir / f"screenshot_{ts}.jpg"
            cv2.imwrite(str(path), frame if 'frame' in dir() else np.zeros((h,w,3), dtype=np.uint8))
            print(f"[+] Screenshot: {path}")
        elif key == ord('p'):           # Pausa
            paused = not paused
            print(f"  {'PAUSA' if paused else 'RIPRENDI'}")
        elif key == ord('w'):           # Toggle wireframe
            show_wireframe = not show_wireframe
        elif key == ord('b'):           # Toggle box
            show_box = not show_box
        elif key == ord('i'):           # Toggle info
            show_info = not show_info
        elif key == ord('+'):           # Aumenta conf
            conf_threshold = min(0.95, conf_threshold + 0.05)
        elif key == ord('-'):           # Diminuisci conf
            conf_threshold = max(0.05, conf_threshold - 0.05)

        if paused:
            cv2.imshow("FlyPose-SAR Demo", frame if 'frame' in dir() else np.zeros((h,w,3), dtype=np.uint8))
            continue

        # Leggi frame — da cartella o da VideoCapture
        if frame_list:
            if frame_idx >= len(frame_list):
                frame_idx = 0   # riparti dall'inizio
            frame = cv2.imread(str(frame_list[frame_idx]))
            frame_idx += 1
            if frame is None:
                continue
        else:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

        # Inferenza con tracking
        results = model.track(
            source          = frame,
            conf            = conf_threshold,
            iou             = iou,
            imgsz           = imgsz,
            tracker         = "bytetrack.yaml",
            persist         = True,   # mantiene tracking tra frame
            verbose         = False,
        )

        # FPS
        t_now = time.time()
        fps_inst = 1.0 / max(t_now - t_prev, 1e-6)
        t_prev   = t_now
        fps_history.append(fps_inst)
        if len(fps_history) > 30:
            fps_history.pop(0)
        fps_smooth = np.mean(fps_history)

        # Rendering detection
        n_persons = 0
        if results and results[0].boxes is not None:
            boxes  = results[0].boxes
            kpts   = results[0].keypoints
            n_det  = len(boxes)
            n_persons = n_det

            for i in range(n_det):
                box     = boxes.xyxy[i].cpu().numpy()
                conf_i  = float(boxes.conf[i].cpu())
                track_id = int(boxes.id[i].cpu()) if boxes.id is not None else i

                # Keypoints
                kp_arr = None
                if kpts is not None and i < len(kpts.data):
                    kp_arr = kpts.data[i].cpu().numpy()  # (17, 3)

                # Stima postura
                posture = estimate_posture(kp_arr, box) if kp_arr is not None else "?"

                # Disegna
                frame = draw_detection(frame, box, track_id, conf_i, posture, show_box)
                if kp_arr is not None:
                    frame = draw_skeleton(frame, kp_arr, show_wireframe)

        # Overlay info
        frame = draw_info_overlay(frame, fps_smooth, n_persons,
                                   conf_threshold, model_name, show_info)

        # Mostra
        cv2.imshow("FlyPose-SAR Demo", frame)

        # Salva output
        if writer is not None:
            writer.write(frame)

    # Cleanup
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print("\n[DONE] Demo terminata.")

# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="FlyPose-SAR — Demo real-time")
    parser.add_argument("--source",  type=str, default="0",
                        help="Sorgente video: 0=webcam, path file, URL RTSP")
    parser.add_argument("--model",   type=str,
                        default=r"runs\fase1\fase1_large\weights\best.pt",
                        help="Percorso al best.pt del modello")
    parser.add_argument("--conf",    type=float, default=0.40,
                        help="Soglia confidenza (default: 0.40)")
    parser.add_argument("--iou",     type=float, default=0.35,
                        help="Soglia NMS IoU (default: 0.35)")
    parser.add_argument("--imgsz",   type=int,   default=640,
                        help="Dimensione inferenza (default: 640)")
    parser.add_argument("--save",    action="store_true",
                        help="Salva output video in runs/demo/")
    args = parser.parse_args()

    # Converti source: se è un numero usa come indice webcam
    source = args.source
    try:
        source = int(source)
    except ValueError:
        pass

    run_demo(
        source     = source,
        model_path = args.model,
        conf       = args.conf,
        iou        = args.iou,
        imgsz      = args.imgsz,
        save_output= args.save,
    )

if __name__ == "__main__":
    main()