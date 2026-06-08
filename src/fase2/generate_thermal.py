# src/fase2/generate_thermal.py
# FlyPose-SAR — Generazione Dataset Termico Sintetico
#
# COSA FA:
#   Applica il generatore G_AB addestrato a TUTTE le immagini di dataset_sar
#   per produrre il dataset termico sintetico dataset_sar_thermal.
#   Le annotazioni (labels/) vengono copiate invariate — sono già corrette
#   perché la geometria spaziale è preservata dalla CycleGAN.
#
# OUTPUT:
#   datasets/dataset_sar_thermal/
#     images/train/   <- frame termici sintetici
#     images/val/     <- frame termici sintetici
#     labels/train/   <- COPIATE da dataset_sar (identiche)
#     labels/val/     <- COPIATE da dataset_sar (identiche)
#
# PERCHE' LE ANNOTAZIONI RESTANO VALIDE:
#   La CycleGAN traduce lo STILE dell'immagine (colori, texture, "firma termica")
#   ma non cambia la GEOMETRIA (posizione e forma degli oggetti).
#   Una persona in alto a sinistra nel frame RGB sarà in alto a sinistra
#   anche nell'immagine termica. Quindi box e keypoints restano corretti.

import torch
import shutil
import logging
from pathlib import Path
from datetime import datetime
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from fase2.cyclegan_model import ResNetGenerator

try:
    from tqdm import tqdm
    TQDM_OK = True
except ImportError:
    TQDM_OK = False


# ---------------------------------------------------------------------------
# CONFIGURAZIONE
# ---------------------------------------------------------------------------
G_AB_WEIGHTS    = "runs/fase2/cyclegan_<timestamp>/G_AB_final.pth"  # aggiorna con il tuo run
DATASET_SAR     = "datasets/dataset_sar"
DATASET_THERMAL = "datasets/dataset_sar_thermal"
IMG_SIZE        = 256
BATCH_SIZE      = 8     # più alto = più veloce la generazione


# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------
def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"generate_thermal_{ts}.log"
    logger = logging.getLogger("FlyPose_GenerateThermal")
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
# CARICA GENERATORE
# ---------------------------------------------------------------------------
def load_generator(weights_path: str, device: str) -> ResNetGenerator:
    G_AB = ResNetGenerator(input_nc=3, output_nc=3)
    state = torch.load(weights_path, map_location=device)
    G_AB.load_state_dict(state)
    G_AB.to(device)
    G_AB.eval()
    return G_AB


# ---------------------------------------------------------------------------
# TRASFORMAZIONI
# ---------------------------------------------------------------------------
def get_transform(img_size: int):
    return T.Compose([
        T.Resize((img_size, img_size),
                 interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Riporta da [-1,1] a [0,1] per salvare come immagine."""
    return (tensor * 0.5 + 0.5).clamp(0, 1)


# ---------------------------------------------------------------------------
# GENERAZIONE PER UNA SPLIT
# ---------------------------------------------------------------------------
def generate_split(G_AB: ResNetGenerator, src_base: Path, dst_base: Path,
                   split: str, device: str, transform,
                   logger: logging.Logger) -> int:

    src_img = src_base / "images" / split
    src_lbl = src_base / "labels" / split
    dst_img = dst_base / "images" / split
    dst_lbl = dst_base / "labels" / split

    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    # Copia le labels invariate
    lbl_files = list(src_lbl.glob("*.txt"))
    for lbl in lbl_files:
        shutil.copy(str(lbl), str(dst_lbl / lbl.name))

    logger.info(f"  [+] Labels copiate: {len(lbl_files)}")

    # Genera immagini termiche
    img_files = sorted([
        f for f in src_img.glob("*.jpg")
    ])

    logger.info(f"  [*] Generazione {len(img_files)} immagini thermiche [{split}]...")

    generated = 0
    iterator  = tqdm(img_files, desc=f"  [{split}]", unit="img") if TQDM_OK else img_files

    with torch.no_grad():
        for img_path in iterator:
            dst_path = dst_img / img_path.name

            # Salta se già generata
            if dst_path.exists():
                generated += 1
                continue

            # Carica e trasforma
            img = Image.open(img_path).convert("RGB")
            orig_w, orig_h = img.size

            tensor = transform(img).unsqueeze(0).to(device)

            # Genera immagine termica
            fake_thermal = G_AB(tensor)

            # Riporta alle dimensioni originali e salva
            fake_thermal = denormalize(fake_thermal.squeeze(0))
            fake_pil = TF.to_pil_image(fake_thermal)
            fake_pil = fake_pil.resize((orig_w, orig_h),
                                        Image.BICUBIC)
            fake_pil.save(str(dst_path), quality=95)
            generated += 1

    logger.info(f"  [OK] {split}: {generated} immagini generate")
    return generated


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    script_dir   = Path(__file__).resolve().parent.parent.parent
    project_root = script_dir
    log_dir      = project_root / "logs"
    logger       = setup_logging(log_dir)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    src_base = project_root / DATASET_SAR
    dst_base = project_root / DATASET_THERMAL

    logger.info("=" * 62)
    logger.info("  FLYPOSE-SAR — GENERAZIONE DATASET TERMICO SINTETICO")
    logger.info("=" * 62)
    logger.info(f"  Device    : {device}")
    logger.info(f"  Sorgente  : {src_base}")
    logger.info(f"  Output    : {dst_base}")
    logger.info(f"  Pesi G_AB : {G_AB_WEIGHTS}")

    # Trova automaticamente l'ultimo run della CycleGAN se non specificato
    weights_path = project_root / G_AB_WEIGHTS
    if not weights_path.exists():
        logger.info("  [*] Ricerca automatica ultimo run CycleGAN...")
        runs_dir = project_root / "runs" / "fase2"
        candidates = sorted([
            f for f in runs_dir.rglob("G_AB_final.pth")
        ], key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            weights_path = candidates[0]
            logger.info(f"  [+] Trovato: {weights_path}")
        else:
            logger.error("  Nessun modello G_AB trovato. Esegui prima train_cyclegan.py")
            return

    logger.info("\n[*] Caricamento generatore G_AB...")
    transform = get_transform(IMG_SIZE)
    G_AB      = load_generator(str(weights_path), device)
    logger.info("[+] Generatore pronto.")

    # Genera train e val
    total = 0
    for split in ["train", "val"]:
        logger.info(f"\n[*] Split: {split}")
        n = generate_split(G_AB, src_base, dst_base, split,
                           device, transform, logger)
        total += n

    # Copia data.yaml adattato per il dataset termico
    src_yaml = project_root / "data.yaml"
    dst_yaml = dst_base / "data_thermal.yaml"
    if src_yaml.exists():
        content = src_yaml.read_text()
        content = content.replace("dataset_sar", "dataset_sar_thermal")
        dst_yaml.write_text(content)
        logger.info(f"\n[+] data_thermal.yaml creato: {dst_yaml}")

    logger.info(f"\n[DONE] Totale immagini generate: {total}")
    logger.info(f"       Dataset termico in: {dst_base}")
    logger.info("       Prossimo step: train.py con data=dataset_sar_thermal/data_thermal.yaml")


if __name__ == "__main__":
    main()