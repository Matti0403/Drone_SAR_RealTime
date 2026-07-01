# generate_thermal_multipalette.py
# FlyPose-SAR — Generazione dataset termico multi-palette
#
# COSA FA:
#   1. Carica G_AB_final.pth (CycleGAN allenata su HIT-UAV)
#   2. Converte tutti i frame di dataset_sar (train/val) in thermal grayscale
#   3. Converte anche le sequenze di test ufficiali VisDrone
#   4. Applica 5 palette diverse su ogni frame
#   5. Copia le label identiche per ogni palette (box GT + 17 keypoints)
#
# OUTPUT:
#   datasets/dataset_sar_thermal_multipalette/
#     white_hot/  images/train/ + labels/train/ + images/val/ + images/test/<seq>/
#     black_hot/  ...
#     iron_red/   ...
#     rainbow1/   ...
#     hot_iron/   ...
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
_PROJECT_ROOT  = Path(__file__).resolve().parent.parent.parent
G_AB_WEIGHTS   = str(_PROJECT_ROOT / "runs/fase2/cyclegan_run/G_AB_final.pth")
DATASET_SAR    = str(_PROJECT_ROOT / "datasets/dataset_sar")
DATASET_TEST   = str(_PROJECT_ROOT / "datasets/dataset_test_official/sequences")
OUTPUT_ROOT    = str(_PROJECT_ROOT / "datasets/dataset_sar_thermal_multipalette")
SPLITS         = ["train", "val"]   # split di dataset_sar
CONVERT_TEST   = True               # converti anche le sequenze di test
SUBSAMPLE      = 4                  # 1 frame ogni N per train/val
SUBSAMPLE_TEST = None               # None = tutti i frame del test (no subsample)
# ---------------------------------------------------------------------------

PALETTES = {
    "white_hot" : None,
    "black_hot" : "INVERT",
    "iron_red"  : cv2.COLORMAP_INFERNO,
    "rainbow1"  : cv2.COLORMAP_JET,
    "hot_iron"  : cv2.COLORMAP_HOT,
}

# ---------------------------------------------------------------------------
# ARCHITETTURA RESNET CYCLEGAN (GroupNorm)
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
    if palette is None:
        return cv2.cvtColor(gray_np, cv2.COLOR_GRAY2BGR)
    elif palette == "INVERT":
        return cv2.cvtColor(255 - gray_np, cv2.COLOR_GRAY2BGR)
    else:
        return cv2.applyColorMap(gray_np, palette)

# ---------------------------------------------------------------------------
# CONVERTI UN BATCH DI IMMAGINI
# ---------------------------------------------------------------------------
def convert_images(G, tf, device, imgs, src_lbl, out_base,
                   split, subsample=None, label_subdir=None):
    """
    Converte una lista di immagini e le salva in tutte le palette.
    split: cartella di destinazione (es. 'train', 'val', 'test/uav0000073')
    label_subdir: se None usa split, altrimenti path custom per le label
    """
    if subsample:
        imgs = imgs[::subsample]
    if not imgs:
        return 0

    # Crea cartelle output
    for palette_name in PALETTES:
        (out_base / palette_name / "images" / split).mkdir(parents=True, exist_ok=True)
        if src_lbl and src_lbl.exists():
            (out_base / palette_name / "labels" / split).mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for i, p in enumerate(imgs):
        img_pil = Image.open(p).convert("RGB")

        with torch.no_grad():
            t       = tf(img_pil).unsqueeze(0).to(device)
            out     = G(t)[0].cpu() * 0.5 + 0.5
            out_pil = T.ToPILImage()(out.clamp(0, 1))
            out_pil = out_pil.resize((640, 640), Image.BICUBIC)
            gray_np = cv2.cvtColor(np.array(out_pil), cv2.COLOR_RGB2GRAY)

        for palette_name, palette_id in PALETTES.items():
            dst_img = out_base / palette_name / "images" / split / p.name
            if dst_img.exists():
                continue
            colored = apply_palette(gray_np, palette_id)
            cv2.imwrite(str(dst_img), colored, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if src_lbl and src_lbl.exists():
                lbl = src_lbl / (p.stem + ".txt")
                if lbl.exists():
                    shutil.copy(lbl, out_base / palette_name / "labels" / split / lbl.name)

        if i % 200 == 0 and i > 0:
            rate = i / (time.time() - t0)
            eta  = (len(imgs) - i) / rate
            print(f"    [{i:>5}/{len(imgs)}] rate={rate:.1f} img/s | ETA={eta/60:.1f}m")

    elapsed = time.time() - t0
    print(f"    [OK] {len(imgs)} frame in {elapsed/60:.1f}m")
    return len(imgs) * len(PALETTES)

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    weights_path = Path(G_AB_WEIGHTS)
    src_base     = Path(DATASET_SAR)
    test_base    = Path(DATASET_TEST)
    out_base     = Path(OUTPUT_ROOT)

    if not weights_path.exists():
        print(f"[*] Ricerca automatica G_AB_final.pth...")
        candidates = sorted(_PROJECT_ROOT.rglob("G_AB_final.pth"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            weights_path = candidates[0]
            print(f"[+] Trovato: {weights_path}")
        else:
            print("[ERRORE] G_AB_final.pth non trovato.")
            return

    print(f"[*] Caricamento G_AB CycleGAN...")
    G = ResNetGenerator(3, 3).to(device)
    state = torch.load(weights_path, map_location=device, weights_only=False)
    G.load_state_dict({k.replace("module.", ""): v for k, v in state.items()})
    G.eval()
    print(f"[+] Modello caricato.")

    tf = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    total = 0

    # ── Train e Val ───────────────────────────────────────────────────────────
    for split in SPLITS:
        src_img = src_base / "images" / split
        src_lbl = src_base / "labels" / split
        if not src_img.exists():
            print(f"[SKIP] {src_img} non trovato")
            continue
        imgs = sorted(src_img.glob("*.jpg"))
        print(f"\n[*] {split.upper()} — {len(imgs)} frame (subsample 1/{SUBSAMPLE})")
        total += convert_images(G, tf, device, imgs, src_lbl, out_base,
                                split=split, subsample=SUBSAMPLE)

    # ── Sequenze Test ─────────────────────────────────────────────────────────
    if CONVERT_TEST and test_base.exists():
        sequences = sorted([d for d in test_base.iterdir() if d.is_dir()])
        print(f"\n[*] TEST — {len(sequences)} sequenze (no subsample)")
        for seq in sequences:
            imgs = sorted([f for f in seq.iterdir()
                           if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}])
            if not imgs:
                continue
            split_path = f"test/{seq.name}"
            print(f"  Sequenza: {seq.name} — {len(imgs)} frame")
            # Le sequenze test non hanno label in dataset_sar
            # cerchiamo in dataset_test_official/annotations se esiste
            lbl_dir = seq.parent.parent / "annotations" / seq.name
            src_lbl = lbl_dir if lbl_dir.exists() else None
            total += convert_images(G, tf, device, imgs, src_lbl, out_base,
                                    split=split_path, subsample=SUBSAMPLE_TEST)
    elif CONVERT_TEST:
        print(f"\n[SKIP] Test dir non trovata: {test_base}")

    print(f"\n[DONE] Totale immagini generate: {total}")
    print(f"       Output in: {out_base}")
    print("\nRiepilogo per palette:")
    for palette_name in PALETTES:
        n_train = len(list((out_base / palette_name / "images" / "train").glob("*.jpg"))) \
                  if (out_base / palette_name / "images" / "train").exists() else 0
        n_test  = sum(
            len(list(d.glob("*.jpg")))
            for d in (out_base / palette_name / "images" / "test").glob("*")
            if d.is_dir()
        ) if (out_base / palette_name / "images" / "test").exists() else 0
        print(f"  {palette_name:12s}: {n_train} train  |  {n_test} test")


if __name__ == "__main__":
    main()