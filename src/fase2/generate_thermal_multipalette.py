# generate_thermal_multipalette.py
# FlyPose-SAR — Generazione dataset termico multi-palette
#
# COSA FA:
#   1. Carica G_AB_final.pth (CycleGAN allenata su HIT-UAV)
#   2. Converte tutti i frame di dataset_sar in thermal grayscale
#   3. Applica 5 palette diverse su ogni frame
#   4. Copia le label identiche per ogni palette (box GT + 17 keypoints)
#
# OUTPUT:
#   datasets/dataset_sar_thermal_multipalette/
#     white_hot/  images/train/ + labels/train/
#     black_hot/  images/train/ + labels/train/
#     iron_red/   images/train/ + labels/train/
#     rainbow1/   images/train/ + labels/train/
#     hot_iron/   images/train/ + labels/train/
#
# USO:
#   python src/fase2/generate_thermal_multipalette.py

import cv2
import torch
import torch.nn as nn
import shutil
import time
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as T

# ---------------------------------------------------------------------------
# CONFIGURAZIONE
# ---------------------------------------------------------------------------
_PROJECT_ROOT     = Path(__file__).resolve().parent.parent.parent
G_AB_WEIGHTS      = str(_PROJECT_ROOT / "runs/fase2/cyclegan_run/G_AB_final.pth")
DATASET_SAR       = str(_PROJECT_ROOT / "datasets/dataset_sar")
OUTPUT_ROOT       = str(_PROJECT_ROOT / "datasets/dataset_sar_thermal_multipalette")
SPLITS            = ["train", "val"]
SUBSAMPLE         = 4     # prende 1 frame ogni N (riduce ridondanza video)
                          # None = usa tutti i frame
# ---------------------------------------------------------------------------

# Palette DJI Mavic 3T / Matrice 4T
PALETTES = {
    "white_hot" : None,               # grayscale puro — DJI White Hot
    "black_hot" : "INVERT",           # grayscale invertito — DJI Black Hot
    "iron_red"  : cv2.COLORMAP_INFERNO,  # DJI Iron Red (default)
    "rainbow1"  : cv2.COLORMAP_JET,      # DJI Rainbow 1
    "hot_iron"  : cv2.COLORMAP_HOT,      # DJI Hot Iron
}

# ---------------------------------------------------------------------------
# ARCHITETTURA RESNET CYCLEGAN (GroupNorm — compatibile con G_AB_final.pth)
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

# ---------------------------------------------------------------------------
# APPLICA PALETTE
# ---------------------------------------------------------------------------
def apply_palette(gray_np: np.ndarray, palette) -> np.ndarray:
    """
    gray_np: numpy array HxW uint8 (grayscale)
    palette: None=white_hot, 'INVERT'=black_hot, cv2.COLORMAP_*=colorato
    Ritorna: numpy array HxWx3 uint8 BGR
    """
    if palette is None:
        return cv2.cvtColor(gray_np, cv2.COLOR_GRAY2BGR)
    elif palette == "INVERT":
        inv = 255 - gray_np
        return cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)
    else:
        return cv2.applyColorMap(gray_np, palette)

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    weights_path = Path(G_AB_WEIGHTS)
    src_base     = Path(DATASET_SAR)
    out_base     = Path(OUTPUT_ROOT)

    # Trova automaticamente i pesi se il path non esiste
    if not weights_path.exists():
        print(f"[*] {weights_path} non trovato, ricerca automatica...")
        candidates = sorted(
            _PROJECT_ROOT.rglob("G_AB_final.pth"),
            key=lambda p: p.stat().st_mtime, reverse=True
        )
        if candidates:
            weights_path = candidates[0]
            print(f"[+] Trovato: {weights_path}")
        else:
            print("[ERRORE] G_AB_final.pth non trovato.")
            print("         Verifica che runs/fase2/cyclegan_run/G_AB_final.pth esista.")
            return

    # Carica ResNet CycleGAN
    print(f"[*] Caricamento G_AB CycleGAN...")
    G = ResNetGenerator(3, 3).to(device)
    state = torch.load(weights_path, map_location=device, weights_only=False)
    new_state = {k.replace("module.", ""): v for k, v in state.items()}
    G.load_state_dict(new_state)
    G.eval()
    print(f"[+] Modello caricato da: {weights_path}")

    tf = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    total_generated = 0

    for split in SPLITS:
        src_img = src_base / "images" / split
        src_lbl = src_base / "labels" / split

        if not src_img.exists():
            print(f"[SKIP] {src_img} non trovato")
            continue

        imgs = sorted(src_img.glob("*.jpg"))
        if SUBSAMPLE:
            imgs = imgs[::SUBSAMPLE]

        print(f"\n[*] Split: {split} — {len(imgs)} frame (subsample 1/{SUBSAMPLE})")

        # Crea cartelle output per tutte le palette
        for palette_name in PALETTES:
            (out_base / palette_name / "images" / split).mkdir(parents=True, exist_ok=True)
            (out_base / palette_name / "labels" / split).mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        for i, p in enumerate(imgs):
            # Genera grayscale UNA SOLA VOLTA per frame
            img_pil = Image.open(p).convert("RGB")

            with torch.no_grad():
                t       = tf(img_pil).unsqueeze(0).to(device)
                out     = G(t)[0].cpu() * 0.5 + 0.5
                out_pil = T.ToPILImage()(out.clamp(0, 1))
                out_pil = out_pil.resize((640, 640), Image.BICUBIC)
                gray_np = cv2.cvtColor(np.array(out_pil), cv2.COLOR_RGB2GRAY)

            # Applica tutte le palette sullo stesso grayscale
            for palette_name, palette_id in PALETTES.items():
                dst_img = out_base / palette_name / "images" / split / p.name
                if dst_img.exists():
                    continue
                colored = apply_palette(gray_np, palette_id)
                cv2.imwrite(str(dst_img), colored, [cv2.IMWRITE_JPEG_QUALITY, 90])
                lbl = src_lbl / (p.stem + ".txt")
                if lbl.exists():
                    shutil.copy(lbl, out_base / palette_name / "labels" / split / lbl.name)

            total_generated += len(PALETTES)

            if i % 200 == 0 and i > 0:
                rate = i / (time.time() - t0)
                eta  = (len(imgs) - i) / rate
                print(f"  [{i:>5}/{len(imgs)}] rate={rate:.1f} frame/s | ETA={eta/60:.1f}m")

        print(f"  [OK] {split} completato in {(time.time()-t0)/60:.1f} minuti")

    print(f"\n[DONE] Totale immagini generate: {total_generated}")
    print(f"       Output in: {out_base}")
    print("\nRiepilogo per palette:")
    for palette_name in PALETTES:
        d = out_base / palette_name / "images" / "train"
        n = len(list(d.glob("*.jpg"))) if d.exists() else 0
        print(f"  {palette_name:12s}: {n} immagini train")


if __name__ == "__main__":
    main()