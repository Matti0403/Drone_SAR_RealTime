# src/demo_realtime.py
# FlyPose-SAR — Demo real-time con webcam o video file
#
# MODALITA' TERMICO (--thermal):
#   Carica il generatore G_AB e converte ogni frame RGB in termico sintetico
#   prima di darlo al modello YOLO. Utile per testare il modello fine-tunato
#   su termici senza avere un sensore LWIR fisico.
#   Il frame originale RGB viene mostrato a sinistra, il termico a destra.
#
# USO:
#   # Modalita' standard RGB
#   python src/demo_realtime.py --source 0
#
#   # Modalita' termico sintetico (converte RGB→thermal prima dell'inferenza)
#   python src/demo_realtime.py --source path/sequenza --thermal \
#       --gan-weights runs/fase2/cyclegan_run/G_AB_final.pth \
#       --model runs/fase2/flypose_thermal_large/weights/best.pt
#
#   # File video con salvataggio
#   python src/demo_realtime.py --source video.mp4 --thermal --save

import cv2
import torch
import torch.nn as nn
import argparse
import numpy as np
import time
from pathlib import Path
from datetime import datetime
from PIL import Image
import torchvision.transforms as T
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# GENERATORE GROUPNORM (compatibile con pesi Kaggle)
# ---------------------------------------------------------------------------
def get_norm_layer(num_features, num_groups=4):
    return nn.GroupNorm(min(num_groups, num_features), num_features)

class ResidualBlock(nn.Module):
    def __init__(self, dim, norm_layer):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, padding=0, bias=True),
            norm_layer(dim), nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, padding=0, bias=True),
            norm_layer(dim),
        )
    def forward(self, x): return x + self.block(x)

class ResNetGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_blocks=9):
        super().__init__()
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, 7, padding=0, bias=True),
            get_norm_layer(ngf), nn.ReLU(inplace=True),
        ]
        for mult in [1, 2]:
            layers += [
                nn.Conv2d(ngf*mult, ngf*mult*2, 3, stride=2, padding=1, bias=True),
                get_norm_layer(ngf*mult*2), nn.ReLU(inplace=True),
            ]
        for _ in range(n_blocks):
            layers.append(ResidualBlock(ngf*4, get_norm_layer))
        for mult in [4, 2]:
            layers += [
                nn.ConvTranspose2d(ngf*mult, ngf*mult//2, 3, stride=2,
                                   padding=1, output_padding=1, bias=True),
                get_norm_layer(ngf*mult//2), nn.ReLU(inplace=True),
            ]
        layers += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, 7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*layers)
    def forward(self, x): return self.model(x)


def load_gan(weights_path: str, device: str) -> ResNetGenerator:
    G = ResNetGenerator(3, 3)
    state = torch.load(weights_path, map_location=device, weights_only=False)
    # rimuovi prefisso 'module.' se presente (DataParallel)
    new_state = {k.replace('module.', ''): v for k, v in state.items()}
    G.load_state_dict(new_state)
    G.to(device).eval()
    return G


def frame_to_thermal(frame_bgr: np.ndarray, G_AB, device: str,
                     gan_size: int = 640) -> np.ndarray:
    """
    Converte un frame BGR OpenCV in termico sintetico BGR tramite G_AB.
    Ritorna frame BGR della stessa dimensione dell'input.
    """
    h, w = frame_bgr.shape[:2]
    # BGR → PIL RGB
    pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    tf  = T.Compose([
        T.Resize((gan_size, gan_size)),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    with torch.no_grad():
        t   = tf(pil).unsqueeze(0).to(device)
        out = G_AB(t)[0].cpu() * 0.5 + 0.5
    # Tensor → PIL → numpy BGR, resize alle dimensioni originali
    thermal_pil = T.ToPILImage()(out.clamp(0, 1))
    thermal_pil = thermal_pil.resize((w, h), Image.BICUBIC)
    return cv2.cvtColor(np.array(thermal_pil), cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# SKELETON COCO 17 keypoints
# ---------------------------------------------------------------------------
SKELETON_CONNECTIONS = [
    (0,1),(0,2),(1,3),(2,4),(5,6),
    (5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16),
]
SKELETON_COLORS = [
    (0,255,255),(0,255,255),(0,255,255),(0,255,255),
    (0,255,0),(0,165,255),(0,165,255),(0,165,255),(0,165,255),
    (0,255,0),(0,255,0),(0,255,0),
    (255,100,0),(255,100,0),(255,100,0),(255,100,0),
]
KPT_ZONE_COLORS = [
    (0,255,255),(0,255,255),(0,255,255),(0,255,255),(0,255,255),
    (0,165,255),(0,165,255),(0,165,255),(0,165,255),(0,165,255),(0,165,255),
    (0,255,0),(0,255,0),
    (255,100,0),(255,100,0),(255,100,0),(255,100,0),
]

# ---------------------------------------------------------------------------
# STIMA STATO POSTURALE (zenitale)
# ---------------------------------------------------------------------------
def estimate_posture(keypoints: np.ndarray, box=None) -> str:
    if keypoints is None or len(keypoints) == 0:
        return "?"
    valid = keypoints[keypoints[:, 2] > 0.25]
    if len(valid) < 3:
        return "INCERTO"

    ar_score = None
    if box is not None:
        bw = abs(box[2] - box[0])
        bh = abs(box[3] - box[1])
        if bh > 0 and bw > 0:
            ar_score = max(bw, bh) / min(bw, bh)

    kpt_std_x = np.std(valid[:, 0])
    kpt_std_y = np.std(valid[:, 1])
    kpt_spread = np.sqrt(kpt_std_x**2 + kpt_std_y**2)

    if box is not None:
        bw = abs(box[2] - box[0])
        bh = abs(box[3] - box[1])
        box_diag = np.sqrt(bw**2 + bh**2)
        spread_ratio = kpt_spread / max(box_diag, 1)
    else:
        spread_ratio = kpt_spread / 50.0

    if ar_score is not None:
        if ar_score > 2.8:
            return "A TERRA"
        elif ar_score > 2.0:
            return "A TERRA" if spread_ratio > 0.35 else "SEDUTO"
        else:
            return "RANNICCHIATO" if spread_ratio > 0.45 else "IN PIEDI"
    else:
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
# RENDERING
# ---------------------------------------------------------------------------
def draw_skeleton(frame, keypoints, show_wireframe=True, kpt_radius=4, line_thickness=2):
    if keypoints is None:
        return frame
    h, w = frame.shape[:2]
    pts  = keypoints[:, :2].astype(int)
    conf = keypoints[:, 2]
    if show_wireframe:
        for i, (a, b) in enumerate(SKELETON_CONNECTIONS):
            if conf[a] > 0.3 and conf[b] > 0.3:
                if 0<=pts[a][0]<w and 0<=pts[a][1]<h and 0<=pts[b][0]<w and 0<=pts[b][1]<h:
                    color = SKELETON_COLORS[i] if i < len(SKELETON_COLORS) else (255,255,255)
                    cv2.line(frame, tuple(pts[a]), tuple(pts[b]), color, line_thickness, cv2.LINE_AA)
    for i, (pt, c) in enumerate(zip(pts, conf)):
        if c > 0.3 and 0<=pt[0]<w and 0<=pt[1]<h:
            color = KPT_ZONE_COLORS[i] if i < len(KPT_ZONE_COLORS) else (255,255,255)
            cv2.circle(frame, tuple(pt), kpt_radius, color, -1, cv2.LINE_AA)
            cv2.circle(frame, tuple(pt), kpt_radius+1, (0,0,0), 1, cv2.LINE_AA)
    return frame

def draw_detection(frame, box, track_id, conf, posture, show_box=True):
    if not show_box:
        return frame
    x1,y1,x2,y2 = [int(v) for v in box]
    color = POSTURE_COLORS.get(posture, (0,255,0))
    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2, cv2.LINE_AA)
    label = f"ID:{track_id} {conf:.2f} | {posture}"
    (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    y_label = max(y1-6, th+6)
    cv2.rectangle(frame, (x1, y_label-th-4), (x1+tw+4, y_label+2), color, -1)
    cv2.putText(frame, label, (x1+2, y_label), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 1, cv2.LINE_AA)
    return frame

def draw_info_overlay(frame, fps, n_persons, conf_threshold, model_name,
                      thermal_mode=False, show_info=True):
    if not show_info:
        return frame
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (8,8), (320,130), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    mode_str = "TERMICO SINTETICO" if thermal_mode else "RGB"
    info_lines = [
        f"FlyPose-SAR  [{mode_str}]",
        f"Modello: {model_name}",
        f"FPS: {fps:.1f}",
        f"Persone: {n_persons}",
        f"Conf: {conf_threshold:.2f}  [+/-]",
    ]
    colors = [(0,255,255) if i==0 else (220,220,220) for i in range(len(info_lines))]
    if thermal_mode:
        colors[0] = (0,200,255)  # arancione per modalità termica
    for i, (line, color) in enumerate(zip(info_lines, colors)):
        cv2.putText(frame, line, (14, 28+i*17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    legend = [("IN PIEDI",(0,255,0)),("SEDUTO",(0,165,255)),
              ("A TERRA",(0,0,255)),("INCERTO",(128,128,128))]
    for i, (label, color) in enumerate(legend):
        x = w - 130
        y = 28 + i*18
        cv2.circle(frame, (x, y-5), 5, color, -1)
        cv2.putText(frame, label, (x+12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

    keys = "[Q] Esci  [S] Screenshot  [P] Pausa  [W] Wire  [B] Box  [I] Info  [T] Toggle view"
    cv2.putText(frame, keys, (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180,180,180), 1, cv2.LINE_AA)
    return frame


def make_side_by_side(rgb_frame, thermal_frame, detections_on='thermal'):
    """
    Affianca RGB (sinistra) e Thermal (destra) con label.
    Le detection vengono disegnate sul frame indicato da detections_on.
    """
    h, w = rgb_frame.shape[:2]
    combined = np.zeros((h, w*2, 3), dtype=np.uint8)
    combined[:, :w]  = rgb_frame
    combined[:, w:]  = thermal_frame

    # Label domini
    cv2.putText(combined, "RGB Originale",    (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(combined, "Thermal Sintetico", (w+10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,200,255), 1, cv2.LINE_AA)
    # Linea di separazione
    cv2.line(combined, (w,0), (w,h), (100,100,100), 2)
    return combined


# ---------------------------------------------------------------------------
# MAIN DEMO
# ---------------------------------------------------------------------------
def run_demo(source, model_path: str, gan_weights: str = None,
             thermal_mode: bool = False, conf: float = 0.40,
             iou: float = 0.35, imgsz: int = 640,
             save_output: bool = False, side_by_side: bool = True):

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*55}")
    print(f"  FLYPOSE-SAR — DEMO REAL-TIME")
    print(f"{'='*55}")
    print(f"  Modello     : {model_path}")
    print(f"  Sorgente    : {source}")
    print(f"  Device      : {device}")
    print(f"  Modo        : {'TERMICO SINTETICO' if thermal_mode else 'RGB'}")
    if thermal_mode:
        print(f"  GAN weights : {gan_weights}")
    print(f"  conf={conf}  iou={iou}  imgsz={imgsz}\n")

    # Carica GAN se richiesta
    G_AB = None
    if thermal_mode:
        if not gan_weights or not Path(gan_weights).exists():
            print(f"[ERRORE] --gan-weights non trovato: {gan_weights}")
            return
        print("[*] Caricamento generatore G_AB...")
        G_AB = load_gan(gan_weights, device)
        print("[+] G_AB pronto.")

    # Carica YOLO
    print("[*] Caricamento modello YOLO...")
    model      = YOLO(model_path)
    model_name = Path(model_path).parent.parent.name
    print(f"[+] YOLO caricato: {model_name}\n")

    # Apri sorgente
    frame_list = []
    frame_idx  = 0
    cap        = None

    src_path = Path(source) if isinstance(source, str) else None
    if src_path and src_path.is_dir():
        frame_list = sorted([f for f in src_path.iterdir()
                              if f.suffix.lower() in {".jpg",".jpeg",".png"}])
        if not frame_list:
            print(f"[ERRORE] Nessun frame in: {source}")
            return
        first = cv2.imread(str(frame_list[0]))
        h, w  = first.shape[:2]
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
        out_w = w*2 if (thermal_mode and side_by_side) else w
        out_path = out_dir / f"demo_{'thermal' if thermal_mode else 'rgb'}_{ts}.mp4"
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (out_w, h))
        print(f"[+] Salvataggio: {out_path}")

    # Stato UI
    show_wireframe  = True
    show_box        = True
    show_info       = True
    show_sidebyside = side_by_side and thermal_mode
    paused          = False
    conf_threshold  = conf
    fps_history     = []
    t_prev          = time.time()
    screenshot_dir  = Path("runs/demo/screenshots")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('s'):
            screenshot_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(str(screenshot_dir / f"screenshot_{ts}.jpg"), display_frame)
            print(f"[+] Screenshot salvato")
        elif key == ord('p'):
            paused = not paused
            print(f"  {'PAUSA' if paused else 'RIPRENDI'}")
        elif key == ord('w'):
            show_wireframe = not show_wireframe
        elif key == ord('b'):
            show_box = not show_box
        elif key == ord('i'):
            show_info = not show_info
        elif key == ord('t'):
            show_sidebyside = not show_sidebyside
            print(f"  Vista: {'side-by-side' if show_sidebyside else 'solo thermal'}")
        elif key == ord('+'):
            conf_threshold = min(0.95, conf_threshold + 0.05)
        elif key == ord('-'):
            conf_threshold = max(0.05, conf_threshold - 0.05)

        if paused:
            cv2.imshow("FlyPose-SAR Demo", display_frame if 'display_frame' in dir() else np.zeros((h,w,3),dtype=np.uint8))
            continue

        # Leggi frame
        if frame_list:
            if frame_idx >= len(frame_list):
                frame_idx = 0
            rgb_frame = cv2.imread(str(frame_list[frame_idx]))
            frame_idx += 1
            if rgb_frame is None:
                continue
        else:
            ret, rgb_frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

        # Conversione termica
        if thermal_mode and G_AB is not None:
            inference_frame = frame_to_thermal(rgb_frame, G_AB, device)
        else:
            inference_frame = rgb_frame.copy()

        # Inferenza YOLO sul frame (termico o RGB)
        results = model.track(
            source   = inference_frame,
            conf     = conf_threshold,
            iou      = iou,
            imgsz    = imgsz,
            tracker  = "bytetrack.yaml",
            persist  = True,
            verbose  = False,
        )

        # FPS
        t_now = time.time()
        fps_history.append(1.0 / max(t_now - t_prev, 1e-6))
        t_prev = t_now
        if len(fps_history) > 30:
            fps_history.pop(0)
        fps_smooth = np.mean(fps_history)

        # Rendering detection sul frame di inferenza
        n_persons = 0
        annotated = inference_frame.copy()
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            kpts  = results[0].keypoints
            n_persons = len(boxes)
            for i in range(n_persons):
                box      = boxes.xyxy[i].cpu().numpy()
                conf_i   = float(boxes.conf[i].cpu())
                track_id = int(boxes.id[i].cpu()) if boxes.id is not None else i
                kp_arr   = kpts.data[i].cpu().numpy() if kpts is not None and i < len(kpts.data) else None
                posture  = estimate_posture(kp_arr, box) if kp_arr is not None else "?"
                annotated = draw_detection(annotated, box, track_id, conf_i, posture, show_box)
                if kp_arr is not None:
                    annotated = draw_skeleton(annotated, kp_arr, show_wireframe)

        # Costruisce il frame da mostrare
        if thermal_mode and show_sidebyside:
            display_frame = make_side_by_side(rgb_frame, annotated)
        else:
            display_frame = annotated

        display_frame = draw_info_overlay(
            display_frame, fps_smooth, n_persons,
            conf_threshold, model_name, thermal_mode, show_info)

        cv2.imshow("FlyPose-SAR Demo", display_frame)
        if writer is not None:
            writer.write(display_frame)

    if cap:
        cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("\n[DONE] Demo terminata.")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="FlyPose-SAR — Demo real-time")
    parser.add_argument("--source",      type=str, default="0",
                        help="Sorgente: 0=webcam, path cartella frame, path video")
    parser.add_argument("--model",       type=str,
                        default=r"runs\fase1\fase1_large\weights\best.pt",
                        help="Percorso best.pt del modello YOLO")
    parser.add_argument("--thermal",     action="store_true",
                        help="Attiva modalita' termico sintetico (richiede --gan-weights)")
    parser.add_argument("--gan-weights", type=str,
                        default=r"runs\fase2\cyclegan_run\G_AB_final.pth",
                        help="Percorso G_AB_final.pth per la conversione RGB→Thermal")
    parser.add_argument("--conf",        type=float, default=0.40)
    parser.add_argument("--iou",         type=float, default=0.35)
    parser.add_argument("--imgsz",       type=int,   default=640)
    parser.add_argument("--save",        action="store_true",
                        help="Salva video output in runs/demo/")
    parser.add_argument("--no-sidebyside", action="store_true",
                        help="Mostra solo il termico senza RGB affiancato")
    args = parser.parse_args()

    source = args.source
    try:
        source = int(source)
    except ValueError:
        pass

    run_demo(
        source       = source,
        model_path   = args.model,
        gan_weights  = args.gan_weights,
        thermal_mode = args.thermal,
        conf         = args.conf,
        iou          = args.iou,
        imgsz        = args.imgsz,
        save_output  = args.save,
        side_by_side = not args.no_sidebyside,
    )

if __name__ == "__main__":
    main()