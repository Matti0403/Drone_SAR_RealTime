# src/fase2/plot_cyclegan.py
# FlyPose-SAR — Grafici e visualizzazioni CycleGAN (Fase 2)
#
# Genera:
#   1. Loss curves: G_loss, D_A_loss, D_B_loss, cycle_loss per epoca
#   2. Griglia visuale: RGB originale | Thermal sintetico | RGB ricostruito
#   3. Istogramma intensità: confronto distribuzione pixel RGB vs Thermal

import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MPL_OK = True
except ImportError:
    MPL_OK = False

from fase2.cyclegan_model import ResNetGenerator
from fase2.cyclegan_dataset import get_transforms
from PIL import Image
import torchvision.transforms.functional as TF


DPI = 150


def plot_loss_curves(history: dict, output_dir: Path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    axes[0,0].plot(history["epoch"], history["G_loss"],    color="#2196F3", lw=1.5)
    axes[0,0].set_title("Generator Loss (G_AB + G_BA)", fontweight="bold")
    axes[0,0].set_xlabel("Epoch"); axes[0,0].grid(alpha=0.4, ls="--")
    axes[0,0].spines[["top","right"]].set_visible(False)

    axes[0,1].plot(history["epoch"], history["D_A_loss"],  color="#FF5722",
                   lw=1.5, label="D_A (RGB)")
    axes[0,1].plot(history["epoch"], history["D_B_loss"],  color="#FF9800",
                   lw=1.5, label="D_B (Thermal)")
    axes[0,1].set_title("Discriminator Loss", fontweight="bold")
    axes[0,1].set_xlabel("Epoch"); axes[0,1].legend()
    axes[0,1].grid(alpha=0.4, ls="--")
    axes[0,1].spines[["top","right"]].set_visible(False)

    cyc = [a+b for a,b in zip(history["cycle_A_loss"], history["cycle_B_loss"])]
    axes[1,0].plot(history["epoch"], cyc, color="#4CAF50", lw=1.5)
    axes[1,0].set_title("Cycle Consistency Loss (A+B)", fontweight="bold")
    axes[1,0].set_xlabel("Epoch"); axes[1,0].grid(alpha=0.4, ls="--")
    axes[1,0].spines[["top","right"]].set_visible(False)

    axes[1,1].plot(history["epoch"], history["identity_loss"], color="#9C27B0", lw=1.5)
    axes[1,1].set_title("Identity Loss", fontweight="bold")
    axes[1,1].set_xlabel("Epoch"); axes[1,1].grid(alpha=0.4, ls="--")
    axes[1,1].spines[["top","right"]].set_visible(False)

    fig.suptitle("CycleGAN Training — Loss Curves (Fase 2)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    p = output_dir / "01_cyclegan_loss_curves.png"
    plt.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  [+] {p.name}")


def plot_visual_grid(G_AB: ResNetGenerator, G_BA: ResNetGenerator,
                     sample_paths: list, output_dir: Path, device: str):
    """
    Griglia visuale: RGB | Thermal sintetico | RGB ricostruito
    per N immagini campione. Fondamentale per valutare visivamente
    la qualità della traduzione per la tesi.
    """
    transform   = get_transforms(256, augment=False)
    denorm      = lambda t: (t * 0.5 + 0.5).clamp(0, 1)

    n   = min(len(sample_paths), 4)
    fig, axes = plt.subplots(n, 3, figsize=(10, n * 3))
    if n == 1:
        axes = axes[None, :]

    col_titles = ["RGB Originale", "Thermal Sintetico\n(G_AB output)", "RGB Ricostruito\n(G_BA(G_AB(x)))"]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=10, fontweight="bold")

    with torch.no_grad():
        for i, img_path in enumerate(sample_paths[:n]):
            img    = Image.open(img_path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)

            fake_B = G_AB(tensor)        # thermal sintetico
            rec_A  = G_BA(fake_B)        # ricostruito RGB

            rgb_np     = denorm(tensor.squeeze(0)).permute(1,2,0).cpu().numpy()
            thermal_np = denorm(fake_B.squeeze(0)).permute(1,2,0).cpu().numpy()
            rec_np     = denorm(rec_A.squeeze(0)).permute(1,2,0).cpu().numpy()

            axes[i, 0].imshow(rgb_np)
            axes[i, 1].imshow(thermal_np)
            axes[i, 2].imshow(rec_np)

            for j in range(3):
                axes[i, j].axis("off")

    fig.suptitle("Qualità traduzione CycleGAN — Fase 2",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    p = output_dir / "02_visual_grid.png"
    plt.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  [+] {p.name}")


def main():
    script_dir   = Path(__file__).resolve().parent.parent.parent
    project_root = script_dir
    output_dir   = project_root / "risultati" / "grafici" / "fase2"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print("=" * 62)
    print("  FLYPOSE-SAR — GRAFICI FASE 2 (CycleGAN)")
    print("=" * 62)

    if not MPL_OK:
        print("[ERRORE] matplotlib non disponibile")
        return

    # Trova l'ultimo run della CycleGAN
    runs_dir = project_root / "runs" / "fase2"
    history_files = sorted(
        runs_dir.rglob("training_history.json"),
        key=lambda p: p.stat().st_mtime, reverse=True
    )

    if not history_files:
        print("[AVVISO] Nessun training_history.json trovato.")
        print("         Esegui prima train_cyclegan.py")
        return

    history_path = history_files[0]
    run_dir = history_path.parent
    print(f"  Run: {run_dir.name}")

    with open(history_path) as f:
        history = json.load(f)

    # Loss curves
    print("\n[*] Generazione grafici...")
    plot_loss_curves(history, output_dir)

    # Visual grid (se i pesi esistono)
    g_ab_path = run_dir / "G_AB_final.pth"
    g_ba_path = run_dir / "G_BA_final.pth"

    if g_ab_path.exists() and g_ba_path.exists():
        G_AB = ResNetGenerator(3, 3)
        G_BA = ResNetGenerator(3, 3)
        G_AB.load_state_dict(torch.load(g_ab_path, map_location=device))
        G_BA.load_state_dict(torch.load(g_ba_path, map_location=device))
        G_AB.to(device).eval()
        G_BA.to(device).eval()

        # Campiona 4 immagini dal dataset
        sample_dir = project_root / "datasets" / "dataset_sar" / "images" / "val"
        samples = sorted(sample_dir.glob("*.jpg"))[:4]

        if samples:
            plot_visual_grid(G_AB, G_BA, samples, output_dir, device)
        else:
            print("  [Skip] Nessuna immagine val trovata per visual grid")
    else:
        print("  [Skip] Pesi G_AB/G_BA non trovati — visual grid saltata")

    print(f"\n[DONE] Grafici salvati in: {output_dir}")


if __name__ == "__main__":
    main()