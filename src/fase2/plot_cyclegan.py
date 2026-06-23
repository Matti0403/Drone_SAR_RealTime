# src/fase2/plot_cyclegan.py
# FlyPose-SAR — Grafici e visualizzazioni CycleGAN (Fase 2)

import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MPL_OK = True
except ImportError:
    MPL_OK = False

from PIL import Image
import torchvision.transforms as T

DPI = 150


# ---------------------------------------------------------------------------
# GroupNorm — stesso del notebook Kaggle
# ---------------------------------------------------------------------------
def get_norm_layer(num_features, num_groups=4):
    return nn.GroupNorm(min(num_groups, num_features), num_features)


# ---------------------------------------------------------------------------
# ResNetGenerator con GroupNorm — compatibile con i pesi Kaggle
# ---------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, dim, norm_layer):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
            norm_layer(dim), nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
            norm_layer(dim),
        )
    def forward(self, x): return x + self.block(x)

class ResNetGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_blocks=9):
        super().__init__()
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=True),
            get_norm_layer(ngf), nn.ReLU(inplace=True),
        ]
        for mult in [1, 2]:
            layers += [
                nn.Conv2d(ngf*mult, ngf*mult*2, kernel_size=3, stride=2, padding=1, bias=True),
                get_norm_layer(ngf*mult*2), nn.ReLU(inplace=True),
            ]
        for _ in range(n_blocks):
            layers.append(ResidualBlock(ngf*4, get_norm_layer))
        for mult in [4, 2]:
            layers += [
                nn.ConvTranspose2d(ngf*mult, ngf*mult//2, kernel_size=3, stride=2,
                                   padding=1, output_padding=1, bias=True),
                get_norm_layer(ngf*mult//2), nn.ReLU(inplace=True),
            ]
        layers += [nn.ReflectionPad2d(3),
                   nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                   nn.Tanh()]
        self.model = nn.Sequential(*layers)
    def forward(self, x): return self.model(x)


# ---------------------------------------------------------------------------
# TRASFORMAZIONI
# ---------------------------------------------------------------------------
def get_transforms(img_size=256):
    return T.Compose([
        T.Resize(img_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


# ---------------------------------------------------------------------------
# LOSS CURVES
# Gestisce sia le chiavi del notebook Kaggle {'G','D_A','D_B','cycle','identity'}
# che le chiavi originali {'G_loss','D_A_loss','D_B_loss',...}
# ---------------------------------------------------------------------------
def plot_loss_curves(history: dict, output_dir: Path):
    # Normalizza le chiavi
    def get(key_new, key_old):
        return history.get(key_new, history.get(key_old, []))

    epochs   = history.get("epoch", list(range(1, len(history.get("G", history.get("G_loss", []))) + 1)))
    g_loss   = get("G", "G_loss")
    da_loss  = get("D_A", "D_A_loss")
    db_loss  = get("D_B", "D_B_loss")
    cyc_loss = get("cycle", None)
    if cyc_loss is None:
        cyc_a = history.get("cycle_A_loss", [0]*len(epochs))
        cyc_b = history.get("cycle_B_loss", [0]*len(epochs))
        cyc_loss = [a+b for a,b in zip(cyc_a, cyc_b)]
    idt_loss = get("identity", "identity_loss")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    axes[0,0].plot(epochs, g_loss,   color="#2196F3", lw=1.5)
    axes[0,0].set_title("Generator Loss (G_AB + G_BA)", fontweight="bold")
    axes[0,0].set_xlabel("Epoch"); axes[0,0].grid(alpha=0.4, ls="--")
    axes[0,0].spines[["top","right"]].set_visible(False)

    axes[0,1].plot(epochs, da_loss, color="#FF5722", lw=1.5, label="D_A (RGB)")
    axes[0,1].plot(epochs, db_loss, color="#FF9800", lw=1.5, label="D_B (Thermal)")
    axes[0,1].set_title("Discriminator Loss", fontweight="bold")
    axes[0,1].set_xlabel("Epoch"); axes[0,1].legend()
    axes[0,1].grid(alpha=0.4, ls="--")
    axes[0,1].spines[["top","right"]].set_visible(False)

    axes[1,0].plot(epochs, cyc_loss, color="#4CAF50", lw=1.5)
    axes[1,0].set_title("Cycle Consistency Loss", fontweight="bold")
    axes[1,0].set_xlabel("Epoch"); axes[1,0].grid(alpha=0.4, ls="--")
    axes[1,0].spines[["top","right"]].set_visible(False)

    axes[1,1].plot(epochs, idt_loss, color="#9C27B0", lw=1.5)
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


# ---------------------------------------------------------------------------
# VISUAL GRID
# ---------------------------------------------------------------------------
def plot_visual_grid(G_AB, G_BA, sample_paths, output_dir, device):
    transform = get_transforms(256)
    denorm    = lambda t: (t * 0.5 + 0.5).clamp(0, 1)

    n = min(len(sample_paths), 4)
    fig, axes = plt.subplots(n, 3, figsize=(10, n * 3))
    if n == 1:
        axes = axes[None, :]

    for j, title in enumerate(["RGB Originale",
                                "Thermal Sintetico\n(G_AB output)",
                                "RGB Ricostruito\n(G_BA(G_AB(x)))"]):
        axes[0, j].set_title(title, fontsize=10, fontweight="bold")

    with torch.no_grad():
        for i, img_path in enumerate(sample_paths[:n]):
            img    = Image.open(img_path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)
            fake_B = G_AB(tensor)
            rec_A  = G_BA(fake_B)

            for j, t in enumerate([tensor, fake_B, rec_A]):
                axes[i, j].imshow(denorm(t.squeeze(0)).permute(1,2,0).cpu().numpy())
                axes[i, j].axis("off")

    fig.suptitle("Qualità traduzione CycleGAN — Fase 2",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    p = output_dir / "02_visual_grid.png"
    plt.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  [+] {p.name}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
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

    # Trova l'ultimo run
    runs_dir = project_root / "runs" / "fase2"
    history_files = sorted(
        runs_dir.rglob("training_history.json"),
        key=lambda p: p.stat().st_mtime, reverse=True
    )
    if not history_files:
        print("[ERRORE] Nessun training_history.json trovato in runs/fase2/")
        return

    history_path = history_files[0]
    run_dir      = history_path.parent
    print(f"  Run: {run_dir.name}")

    with open(history_path) as f:
        history = json.load(f)

    print(f"  Epoche trovate: {len(history.get('epoch', history.get('G', [])))}")
    print("\n[*] Generazione loss curves...")
    plot_loss_curves(history, output_dir)

    # Visual grid
    g_ab_path = run_dir / "G_AB_final.pth"
    g_ba_path = run_dir / "G_BA_final.pth"

    if g_ab_path.exists() and g_ba_path.exists():
        print("[*] Caricamento generatori...")
        G_AB = ResNetGenerator(3, 3)
        G_BA = ResNetGenerator(3, 3)

        # Gestisce sia pesi con DataParallel wrapper che senza
        def load_state(model, path):
            state = torch.load(path, map_location=device)
            # Rimuovi prefisso 'module.' se presente (DataParallel)
            new_state = {}
            for k, v in state.items():
                new_state[k.replace("module.", "")] = v
            model.load_state_dict(new_state)
            return model

        G_AB = load_state(G_AB, g_ab_path).to(device).eval()
        G_BA = load_state(G_BA, g_ba_path).to(device).eval()
        print("[+] Generatori caricati.")

        sample_dir = project_root / "datasets" / "dataset_sar" / "images" / "val"
        samples    = sorted(sample_dir.glob("*.jpg"))[:4]

        if samples:
            print("[*] Generazione visual grid...")
            plot_visual_grid(G_AB, G_BA, samples, output_dir, device)
        else:
            print("  [Skip] Nessuna immagine val trovata")
    else:
        print(f"  [Skip] Pesi non trovati in: {run_dir}")

    print(f"\n[DONE] Grafici in: {output_dir}")


if __name__ == "__main__":
    main()