# src/fase2/train_cyclegan.py
# FlyPose-SAR — Training CycleGAN RGB → Thermal
#
# COSA FA:
#   Addestra la CycleGAN per tradurre immagini RGB zenitali di VisDrone
#   in immagini termiche sintetiche nel dominio LWIR.
#
# LOSS FUNCTIONS:
#
#   1. Adversarial Loss (LSGAN):
#      Il generatore cerca di ingannare il discriminatore.
#      Usiamo MSELoss invece di BCELoss (LSGAN, Mao et al. 2017):
#      più stabile, evita il vanishing gradient nella fase iniziale.
#      G_AB cerca di fare: D_B(G_AB(A)) ≈ 1 (immagine "reale")
#
#   2. Cycle Consistency Loss:
#      G_BA(G_AB(A)) ≈ A  (forward cycle)
#      G_AB(G_BA(B)) ≈ B  (backward cycle)
#      Peso lambda=10 (dal paper originale).
#      Garantisce che la traduzione preservi il contenuto dell'immagine.
#
#   3. Identity Loss:
#      G_AB(B) ≈ B  (se passi già un'immagine termica, non cambiarla)
#      Peso lambda_identity=0.5 * lambda = 5
#      Preserva la colorazione/tono quando l'input è già nel dominio target.
#
# OTTIMIZZATORE:
#   Adam con lr=0.0002, beta1=0.5 (standard per GAN).
#   Learning rate costante per le prime n/2 epoche, poi decay lineare.
#
# METRICHE DI QUALITA' (salvate per la tesi):
#   - FID (Frechet Inception Distance): distanza tra distribuzione reale e generata
#   - SSIM (Structural Similarity): similarità strutturale tra originale e ricostruito
#   - Loss curves: G_loss, D_loss, cycle_loss per epoca

import os
import json
import torch
import torch.nn as nn
import itertools
import logging
from pathlib import Path
from datetime import datetime
from torch.optim import lr_scheduler

from fase2.cyclegan_model import (
    build_generator, build_discriminator, ImageBuffer
)
from fase2.cyclegan_dataset import get_dataloader

try:
    from tqdm import tqdm
    TQDM_OK = True
except ImportError:
    TQDM_OK = False


# ---------------------------------------------------------------------------
# CONFIGURAZIONE
# ---------------------------------------------------------------------------

# Percorsi dataset
DIR_A_TRAIN = "datasets/dataset_sar/images/train"    # RGB zenitale VisDrone
DIR_B_TRAIN = r"C:\Dataset_BIRDSAI_Thermal\images"   # Thermal BIRDSAI zenitale
MAX_IMAGES_B = 5000   # subset distribuito: 1 frame ogni ~8 su 40.661 totali

# Iperparametri (valori standard dal paper CycleGAN originale)
IMG_SIZE        = 256     # dimensione immagini durante il training
BATCH_SIZE      = 1       # batch=1 standard per CycleGAN (InstanceNorm richiede ≥1)
N_EPOCHS        = 100     # epoche totali
N_EPOCHS_DECAY  = 100     # epoche con lr decay (totale = N_EPOCHS + N_EPOCHS_DECAY)
LR              = 0.0002  # learning rate iniziale
BETA1           = 0.5     # Adam beta1
LAMBDA_CYCLE    = 10.0    # peso cycle consistency loss
LAMBDA_IDENTITY = 5.0     # peso identity loss (0.5 * LAMBDA_CYCLE)
N_WORKERS       = 4
SAVE_FREQ       = 10      # salva checkpoint ogni N epoche


# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------
def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"cyclegan_train_{ts}.log"
    logger = logging.getLogger("FlyPose_CycleGAN")
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
# SCHEDULER LEARNING RATE
# Con decay lineare dopo N_EPOCHS epoche
# ---------------------------------------------------------------------------
def get_scheduler(optimizer, n_epochs: int, n_epochs_decay: int):
    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch - n_epochs) / float(n_epochs_decay + 1)
        return lr_l
    return lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)


# ---------------------------------------------------------------------------
# TRAINING LOOP
# ---------------------------------------------------------------------------
def train(project_root: Path, logger: logging.Logger):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = project_root / "runs" / "fase2" / f"cyclegan_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 62)
    logger.info("  FLYPOSE-SAR — TRAINING CYCLEGAN (FASE 2)")
    logger.info("=" * 62)
    logger.info(f"  Device      : {device}")
    if device == "cuda:0":
        logger.info(f"  GPU         : {torch.cuda.get_device_name(0)}")
        logger.info(f"  VRAM        : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    logger.info(f"  Run dir     : {run_dir}")
    logger.info(f"  Img size    : {IMG_SIZE}px")
    logger.info(f"  Epoche      : {N_EPOCHS} + {N_EPOCHS_DECAY} (decay)")
    logger.info(f"  Lambda cycle: {LAMBDA_CYCLE}")

    # ------------------------------------------------------------------
    # MODELLI
    # ------------------------------------------------------------------
    # G_AB: RGB → Thermal
    # G_BA: Thermal → RGB
    # D_A:  discriminatore dominio A (RGB)
    # D_B:  discriminatore dominio B (Thermal)
    G_AB = build_generator(input_nc=3, output_nc=3, device=device)
    G_BA = build_generator(input_nc=3, output_nc=3, device=device)
    D_A  = build_discriminator(input_nc=3, device=device)
    D_B  = build_discriminator(input_nc=3, device=device)

    logger.info(f"  Parametri G_AB: {sum(p.numel() for p in G_AB.parameters())/1e6:.1f}M")
    logger.info(f"  Parametri D_B:  {sum(p.numel() for p in D_B.parameters())/1e6:.1f}M")

    # ------------------------------------------------------------------
    # LOSS
    # ------------------------------------------------------------------
    criterion_gan      = nn.MSELoss()       # LSGAN: più stabile di BCE
    criterion_cycle    = nn.L1Loss()        # cycle consistency
    criterion_identity = nn.L1Loss()        # identity loss

    # ------------------------------------------------------------------
    # OPTIMIZER
    # ------------------------------------------------------------------
    # I generatori vengono ottimizzati insieme (stesso optimizer)
    opt_G = torch.optim.Adam(
        itertools.chain(G_AB.parameters(), G_BA.parameters()),
        lr=LR, betas=(BETA1, 0.999)
    )
    opt_D_A = torch.optim.Adam(D_A.parameters(), lr=LR, betas=(BETA1, 0.999))
    opt_D_B = torch.optim.Adam(D_B.parameters(), lr=LR, betas=(BETA1, 0.999))

    # ------------------------------------------------------------------
    # SCHEDULER
    # ------------------------------------------------------------------
    sched_G   = get_scheduler(opt_G,   N_EPOCHS, N_EPOCHS_DECAY)
    sched_D_A = get_scheduler(opt_D_A, N_EPOCHS, N_EPOCHS_DECAY)
    sched_D_B = get_scheduler(opt_D_B, N_EPOCHS, N_EPOCHS_DECAY)

    # ------------------------------------------------------------------
    # REPLAY BUFFER
    # ------------------------------------------------------------------
    buffer_fake_A = ImageBuffer(max_size=50)
    buffer_fake_B = ImageBuffer(max_size=50)

    # ------------------------------------------------------------------
    # DATALOADER
    # ------------------------------------------------------------------
    logger.info("\n[*] Caricamento dataset...")
    dataloader = get_dataloader(
        dir_A       = str(project_root / DIR_A_TRAIN),
        dir_B       = DIR_B_TRAIN,
        batch_size  = BATCH_SIZE,
        img_size    = IMG_SIZE,
        num_workers = N_WORKERS,
        augment     = True,
        max_images_b= MAX_IMAGES_B,
    )
    logger.info(f"[+] Dataset caricato: {len(dataloader)} batch per epoca")

    # ------------------------------------------------------------------
    # HISTORY METRICHE
    # ------------------------------------------------------------------
    history = {
        "epoch": [], "G_loss": [], "D_A_loss": [], "D_B_loss": [],
        "cycle_A_loss": [], "cycle_B_loss": [], "identity_loss": [], "lr": []
    }

    total_epochs = N_EPOCHS + N_EPOCHS_DECAY

    # ------------------------------------------------------------------
    # LOOP PRINCIPALE
    # ------------------------------------------------------------------
    for epoch in range(1, total_epochs + 1):
        G_AB.train(); G_BA.train(); D_A.train(); D_B.train()

        epoch_G = epoch_DA = epoch_DB = 0.0
        epoch_cyc_A = epoch_cyc_B = epoch_idt = 0.0
        n_batches = 0

        iterator = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs}",
                        leave=False) if TQDM_OK else dataloader

        for batch in iterator:
            real_A = batch["A"].to(device)
            real_B = batch["B"].to(device)

            # ----------------------------------------------------------
            # STEP 1: Aggiorna i GENERATORI
            # Congela i discriminatori durante l'update dei generatori
            # ----------------------------------------------------------
            for p in D_A.parameters(): p.requires_grad_(False)
            for p in D_B.parameters(): p.requires_grad_(False)

            opt_G.zero_grad()

            # Identity loss: G_AB(B) deve essere ≈ B
            idt_A = G_AB(real_B)
            loss_idt_A = criterion_identity(idt_A, real_B) * LAMBDA_IDENTITY
            idt_B = G_BA(real_A)
            loss_idt_B = criterion_identity(idt_B, real_A) * LAMBDA_IDENTITY

            # Traduzione
            fake_B = G_AB(real_A)   # RGB → Thermal sintetico
            fake_A = G_BA(real_B)   # Thermal → RGB sintetico

            # Adversarial loss: D_B(fake_B) deve essere ≈ 1
            target_real = torch.ones_like(D_B(fake_B))
            loss_G_AB = criterion_gan(D_B(fake_B), target_real)
            loss_G_BA = criterion_gan(D_A(fake_A), target_real)

            # Cycle consistency: G_BA(G_AB(A)) ≈ A
            rec_A = G_BA(fake_B)
            rec_B = G_AB(fake_A)
            loss_cyc_A = criterion_cycle(rec_A, real_A) * LAMBDA_CYCLE
            loss_cyc_B = criterion_cycle(rec_B, real_B) * LAMBDA_CYCLE

            loss_G = (loss_G_AB + loss_G_BA +
                      loss_cyc_A + loss_cyc_B +
                      loss_idt_A + loss_idt_B)
            loss_G.backward()
            opt_G.step()

            # ----------------------------------------------------------
            # STEP 2: Aggiorna DISCRIMINATORE D_B (dominio Thermal)
            # ----------------------------------------------------------
            for p in D_B.parameters(): p.requires_grad_(True)
            opt_D_B.zero_grad()

            # Loss su immagini reali
            pred_real = D_B(real_B)
            loss_D_real = criterion_gan(pred_real, torch.ones_like(pred_real))

            # Loss su immagini false (dal buffer)
            fake_B_buf = buffer_fake_B.push_and_pop(fake_B.detach())
            pred_fake  = D_B(fake_B_buf)
            loss_D_fake = criterion_gan(pred_fake, torch.zeros_like(pred_fake))

            loss_D_B = (loss_D_real + loss_D_fake) * 0.5
            loss_D_B.backward()
            opt_D_B.step()

            # ----------------------------------------------------------
            # STEP 3: Aggiorna DISCRIMINATORE D_A (dominio RGB)
            # ----------------------------------------------------------
            for p in D_A.parameters(): p.requires_grad_(True)
            opt_D_A.zero_grad()

            pred_real = D_A(real_A)
            loss_D_real = criterion_gan(pred_real, torch.ones_like(pred_real))

            fake_A_buf = buffer_fake_A.push_and_pop(fake_A.detach())
            pred_fake  = D_A(fake_A_buf)
            loss_D_fake = criterion_gan(pred_fake, torch.zeros_like(pred_fake))

            loss_D_A = (loss_D_real + loss_D_fake) * 0.5
            loss_D_A.backward()
            opt_D_A.step()

            # Accumula per statistiche epoca
            epoch_G    += loss_G.item()
            epoch_DA   += loss_D_A.item()
            epoch_DB   += loss_D_B.item()
            epoch_cyc_A += loss_cyc_A.item()
            epoch_cyc_B += loss_cyc_B.item()
            epoch_idt  += (loss_idt_A + loss_idt_B).item()
            n_batches  += 1

        # --------------------------------------------------------------
        # Fine epoca: scheduler step + logging
        # --------------------------------------------------------------
        sched_G.step(); sched_D_A.step(); sched_D_B.step()

        avg_G    = epoch_G    / n_batches
        avg_DA   = epoch_DA   / n_batches
        avg_DB   = epoch_DB   / n_batches
        avg_cycA = epoch_cyc_A / n_batches
        avg_cycB = epoch_cyc_B / n_batches
        avg_idt  = epoch_idt  / n_batches
        cur_lr   = opt_G.param_groups[0]["lr"]

        logger.info(
            f"  Epoch {epoch:3d}/{total_epochs} | "
            f"G={avg_G:.3f} D_A={avg_DA:.3f} D_B={avg_DB:.3f} | "
            f"Cyc={avg_cycA+avg_cycB:.3f} Idt={avg_idt:.3f} | "
            f"lr={cur_lr:.6f}"
        )

        # Aggiorna history
        history["epoch"].append(epoch)
        history["G_loss"].append(avg_G)
        history["D_A_loss"].append(avg_DA)
        history["D_B_loss"].append(avg_DB)
        history["cycle_A_loss"].append(avg_cycA)
        history["cycle_B_loss"].append(avg_cycB)
        history["identity_loss"].append(avg_idt)
        history["lr"].append(cur_lr)

        # Salva checkpoint periodico
        if epoch % SAVE_FREQ == 0 or epoch == total_epochs:
            ckpt_path = run_dir / f"checkpoint_epoch{epoch:03d}.pth"
            torch.save({
                "epoch":    epoch,
                "G_AB":     G_AB.state_dict(),
                "G_BA":     G_BA.state_dict(),
                "D_A":      D_A.state_dict(),
                "D_B":      D_B.state_dict(),
                "opt_G":    opt_G.state_dict(),
                "opt_D_A":  opt_D_A.state_dict(),
                "opt_D_B":  opt_D_B.state_dict(),
                "history":  history,
            }, ckpt_path)
            logger.info(f"  [SAVE] Checkpoint: {ckpt_path.name}")

    # ------------------------------------------------------------------
    # SALVA MODELLI FINALI E HISTORY
    # ------------------------------------------------------------------
    torch.save(G_AB.state_dict(), run_dir / "G_AB_final.pth")
    torch.save(G_BA.state_dict(), run_dir / "G_BA_final.pth")
    torch.save(D_A.state_dict(),  run_dir / "D_A_final.pth")
    torch.save(D_B.state_dict(),  run_dir / "D_B_final.pth")

    with open(run_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    logger.info(f"\n[DONE] Training completato. Modelli salvati in: {run_dir}")
    return run_dir


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    script_dir   = Path(__file__).resolve().parent.parent.parent
    project_root = script_dir
    log_dir      = project_root / "logs"
    logger       = setup_logging(log_dir)
    train(project_root, logger)


if __name__ == "__main__":
    main()