# FlyPose-SAR — Documentazione Tecnica Fase 2: ThermalGAN
# Stato: IN CORSO (training CycleGAN su Kaggle)

## Obiettivo

Estendere il sistema FlyPose-SAR al dominio termico LWIR per operatività
in condizioni di fumo e visibilità degradata tipiche degli incendi boschivi.
Il fumo denso rende lo spettro RGB completamente inutilizzabile — i sensori
termici vedono attraverso il fumo rilevando la firma termica dei corpi.

## Problema risolto

Non esistono dataset aerei annotati con keypoints nel dominio termico.
La soluzione adottata è una CycleGAN che traduce i frame RGB di VisDrone
in immagini termiche sintetiche. Poiché la cycle consistency loss preserva
la geometria spaziale, le annotazioni (box GT + 17 keypoints) restano valide
sulle immagini tradotte — eliminando completamente il costo di annotazione
manuale nel dominio LWIR.

---

## Dataset

| Dataset | Dominio | Immagini | Uso |
|---------|---------|----------|-----|
| VisDrone dataset_sar (images/train) | A — RGB zenitale | 20.420 frame | Input per traduzione |
| LLVIP infrared/train | B — LWIR reale | 12.025 (sub 5K) | Insegna la firma termica umana |

**LLVIP:** Low-Light Visible-Infrared Image Pairs
Kaggle: `afradhossain/llvip-dataset` → `LLVIP/infrared/train/`

---

## Architettura CycleGAN

| Componente | Architettura | Ruolo |
|---|---|---|
| G_AB (generatore) | ResNet-9blocks + GroupNorm | Traduce RGB → Thermal sintetico |
| G_BA (generatore) | ResNet-9blocks + GroupNorm | Traduce Thermal → RGB (ciclo inverso) |
| D_A (discriminatore) | PatchGAN 70×70 | Distingue RGB reale da ricostruito |
| D_B (discriminatore) | PatchGAN 70×70 | Distingue Thermal reale da generato |
| ImageBuffer | 50 immagini | Stabilizza il discriminatore |

### Loss functions

- **Adversarial Loss (LSGAN)** — MSELoss: più stabile di BCE, evita vanishing gradient
- **Cycle Consistency Loss** — G_BA(G_AB(A)) ≈ A e G_AB(G_BA(B)) ≈ B. Peso λ=10
- **Identity Loss** — G_AB(B) ≈ B. Peso λ_idt=5

### Iperparametri

| Parametro | Valore | Motivazione |
|---|---|---|
| img_size | 256px | Standard CycleGAN |
| batch_size | 4 | GroupNorm + 2×T4, stabile in VRAM |
| n_epochs | 100 + 100 decay | 200 totali |
| lr | 0.0002 | Standard Adam per GAN |
| beta1 | 0.5 | Adam beta1 per GAN |
| lambda_cycle | 10.0 | Peso cycle loss |
| lambda_identity | 5.0 | 0.5 × lambda_cycle |
| norm_layer | GroupNorm | Compatibile con batch > 1 e DataParallel |
| MAX_STEPS_PER_EPOCH | 500 | Limita steps per sessione Kaggle |

---

## Training su Kaggle

### Configurazione

- **Hardware:** GPU T4 x2 (DataParallel)
- **Dataset:** dataset_sar + llvip-dataset
- **Notebook:** `thermalgan_final.ipynb`
- **Checkpoint ogni:** 5 epoche

### Prima sessione

Esegui celle in ordine: **1 → 2 → 3 → 4 → 5 → 6 → 7 → 8**

### Sessioni successive (resume)

Esegui celle: **1 → 2 → 3 → 4 → 5 → RESUME → 7 → 8**

Prima di eseguire la cella RESUME:
1. Carica lo zip della sessione precedente su Kaggle come dataset
2. Aggiorna `RESUME_CHECKPOINT` con il percorso corretto:
```python
RESUME_CHECKPOINT = '/kaggle/input/<nome-dataset>/checkpoint_epochXXX.pth'
```

Per trovare il percorso esatto dei dataset aggiunti:
```python
from pathlib import Path
for d in Path('/kaggle/input').iterdir():
    print(d.name)
    for sub in d.iterdir():
        print(f"  {sub.name}")
        if sub.is_dir():
            for f in sub.iterdir():
                print(f"    {f.name}")
```

### Cella 8 — salva zip prima di chiudere

Sempre eseguire la cella 8 prima di chiudere la sessione.
Scarica `cyclegan_results.zip` dal pannello Output → scarica.

---

## Come capire quando fermarsi

Non serve arrivare a 200 epoche. Ferma quando la **visual grid** è
soddisfacente. Dopo ogni sessione:

```bash
python src/fase2/plot_cyclegan.py
```

**Criteri di accettazione:**
- Zone calde visibili su testa e torso delle persone nel thermal sintetico
- Sfondo scuro/freddo
- Ciclo A→B→A ricostruisce l'immagine originale in modo riconoscibile
- SSIM ciclo A→B→A > 0.75

**Loss come indicatore:**
- G_loss scende e si stabilizza (~2.5-3.5) ✓
- D_A e D_B intorno a 0.18-0.25 (equilibrio) ✓
- Cyc_loss scende costantemente ✓

---

## Pipeline completa Fase 2

### Step 1 — Training CycleGAN su Kaggle
```
thermalgan_final.ipynb
→ G_AB_final.pth + checkpoint_epochXXX.pth
```

### Step 2 — Verifica qualità traduzione (in locale)
```bash
python src/fase2/plot_cyclegan.py
```
Controlla visual grid. Se SSIM > 0.75 procedi.

### Step 3 — Generazione dataset termico sintetico
```bash
python src/fase2/generate_thermal.py
```
Applica G_AB a tutti i 22.671 frame di dataset_sar.
Output: `datasets/dataset_sar_thermal/` con stesse annotazioni.

**Nota:** se hai rigenerato dataset_sar con il fix v2 dei keypoints,
la GAN non va riallenata — le immagini RGB sono identiche,
sono cambiate solo le label. Esegui generate_thermal.py
sul nuovo dataset_sar e otterrai dataset_sar_thermal con
annotazioni corrette.

### Step 4 — Fine-tuning YOLO sul dominio termico
```bash
# Aggiorna data.yaml per puntare a dataset_sar_thermal
# Poi lancia il training su Kaggle con thermalgan_final.ipynb
python src/train.py
```
Parti dai pesi best.pt della Fase 1 (transfer learning progressivo).
Traina Nano, Small e Large sul dataset termico.

### Step 5 — Valutazione formale
```bash
python src/evaluate.py
```
Confronto modelli RGB (Fase 1) vs modelli Thermal (Fase 2).
Metrica chiave: delta mAP50 RGB→Thermal.

---

## Struttura file Fase 2

```
src/fase2/
├── __init__.py
├── cyclegan_model.py      # ResNet-9, PatchGAN, ImageBuffer, GroupNorm
├── cyclegan_dataset.py    # UnpairedDataset dominio A+B, subsample
├── train_cyclegan.py      # training loop locale (alternativa a notebook)
├── generate_thermal.py    # applica G_AB a dataset_sar
└── plot_cyclegan.py       # loss curves + visual grid qualità
```

---

## Note su Windows — Bug Apostrofo

Stesso problema della Fase 1 — il path `MATTIA-D'AGOSTINO` causa crash.
Per generate_thermal.py e plot_cyclegan.py assicurati che i percorsi
di input/output puntino a `C:\Temp\` tramite junction points.

---

## Stato attuale

| Attività | Stato |
|---|---|
| Architettura CycleGAN implementata | ✓ Completata |
| Dataset LLVIP confermato | ✓ Completata |
| Training CycleGAN su Kaggle | ⟳ In corso (~epoca 10/200) |
| Verifica qualità visual grid | ◷ Da fare |
| Generazione dataset_sar_thermal | ◷ Da fare |
| Fine-tuning YOLO termico | ◷ Da fare |
| Valutazione RGB vs Thermal | ◷ Da fare |

---

## Riferimenti

- Zhu et al. (2017) — CycleGAN: Unpaired Image-to-Image Translation. ICCV 2017.
- Mao et al. (2017) — LSGAN: Least Squares Generative Adversarial Networks. ICCV 2017.
- Shrivastava et al. (2017) — Learning from Simulated Images (ReplayBuffer). CVPR 2017.
- Wang et al. (2021) — LLVIP: Low-Light Visible-Infrared Image Pairs. ICCV 2021.