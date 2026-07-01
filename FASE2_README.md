# FlyPose-SAR — Documentazione Tecnica Fase 2: ThermalGAN + Pix2Pix Multi-Palette
# Stato: COMPLETATA

## Obiettivo

Estendere il sistema FlyPose-SAR al dominio termico LWIR per operativita'
in condizioni di fumo e visibilita' degradata tipiche degli incendi boschivi.
Il fumo denso rende lo spettro RGB completamente inutilizzabile — i sensori
termici vedono attraverso il fumo rilevando la firma termica dei corpi.
Hardware di riferimento del cliente: DJI Mavic 3T e DJI Matrice 4T.

## Problema risolto

Non esistono dataset aerei annotati con keypoints nel dominio termico.
La soluzione adottata traduce i frame RGB di VisDrone in immagini termiche
sintetiche tramite reti generative (CycleGAN). Poiche' la geometria spaziale
e' preservata, le annotazioni (box GT + 17 keypoints) restano valide sulle
immagini tradotte — eliminando completamente il costo di annotazione manuale
nel dominio LWIR.

---

## Evoluzione della pipeline

Il piano originale prevedeva CycleGAN con dominio B = LLVIP. Dopo analisi
sperimentale la pipeline e' stata aggiornata in tre step:

| Step | Intervento | Risultato |
|---|---|---|
| 1 | CycleGAN + LLVIP | Non converge — mismatch geometrico frontale vs zenitale |
| 2 | CycleGAN + HIT-UAV | Converge in 50 epoche. Bias luminanza residuo identificato |
| 3 | Pix2Pix + KAIST | Risultato negativo — KAIST notturno incompatibile con VisDrone diurno |
| 4 (finale) | CycleGAN HIT-UAV + colormap post-processing | Pipeline adottata per il dataset multi-palette |

### Nota sul ruolo della Pix2Pix — Risultato negativo documentato

La Pix2Pix su KAIST ha costituito un percorso esplorativo con risultato
negativo: KAIST e' acquisito di notte da veicolo in scene urbane stradali —
distribuzione troppo distante da VisDrone zenitale diurno. Il generatore
produceva frame quasi uniformemente scuri su scene di test.
Il risultato ha confermato che il problema non e' l'architettura della GAN
ma la disponibilita' di dati paired nel dominio corretto.
Il multi-palette e' stato ottenuto applicando cv2.applyColorMap() sull'output
grayscale della CycleGAN, senza necessita' di riallenare la rete.

---

## Dataset

| Dataset | Dominio | Immagini | Uso | Esito |
|---------|---------|----------|-----|-------|
| VisDrone dataset_sar (images/train) | A — RGB zenitale | 20.420 frame | Input CycleGAN | Usato |
| LLVIP infrared/train | B — LWIR reale | 12.025 (sub 5K) | Dominio B originale | Abbandonato (mismatch geometrico) |
| HIT-UAV images/train | B — LWIR zenitale | ~2.000 frame | Dominio B finale | Usato |
| KAIST Multispectral Pedestrian | Paired RGB+LWIR | 95k coppie | Training Pix2Pix | Risultato negativo |

---

## Architettura CycleGAN (soluzione adottata)

| Componente | Architettura | Ruolo |
|---|---|---|
| G_AB (generatore) | ResNet-9blocks + GroupNorm | Traduce RGB → Thermal sintetico |
| G_BA (generatore) | ResNet-9blocks + GroupNorm | Traduce Thermal → RGB (ciclo inverso) |
| D_A (discriminatore) | PatchGAN 70×70 | Distingue RGB reale da ricostruito |
| D_B (discriminatore) | PatchGAN 70×70 | Distingue Thermal reale da generato |
| ImageBuffer | 50 immagini | Stabilizza il discriminatore |

### Loss functions

- **Adversarial Loss (LSGAN)** — MSELoss: piu' stabile di BCE, evita vanishing gradient
- **Cycle Consistency Loss** — G_BA(G_AB(A)) ≈ A e G_AB(G_BA(B)) ≈ B. Peso λ=10
- **Identity Loss** — G_AB(B) ≈ B. Peso λ_idt=5

### Iperparametri CycleGAN

| Parametro | Valore | Motivazione |
|---|---|---|
| img_size | 256px | Standard CycleGAN |
| batch_size | 8 (4/GPU × 2×T4) | GroupNorm + DataParallel stabile in VRAM |
| n_epochs | 50 (fermato) | Convergenza a epoca 40, miglioramento marginale dopo |
| lr | 0.0002 | Standard Adam per GAN |
| beta1 | 0.5 | Adam beta1 per GAN |
| lambda_cycle | 10.0 | Peso cycle loss |
| lambda_identity | 5.0 | 0.5 × lambda_cycle |
| norm_layer | GroupNorm | Compatibile con batch > 1 e DataParallel |
| Crop strategy | Person-aware 70% | 70% crop centrati su persona GT |

### Loss finali CycleGAN (epoca 50)

- G_loss: 1.395
- D_loss: 0.232
- Cycle_loss: 0.011

---

## Architettura Pix2Pix (esplorazione, risultato negativo)

| Componente | Architettura | Differenza vs CycleGAN |
|---|---|---|
| G | UNet con skip connections | Supervisione diretta coppia reale; no ciclo inverso |
| D | PatchGAN 70×70 condizionato (6ch) | Riceve coppia (RGB+thermal); giudica coerenza coppia |
| Loss G | LSGAN + L1 × 100 | L1 confronto diretto output vs thermal reale |
| Loss D | LSGAN su coppie reali vs false | Stessa formula ma su coppie, non immagini singole |

---

## Pipeline multi-palette (5 palette DJI)

I sensori DJI Mavic 3T e Matrice 4T permettono di cambiare la palette in
tempo reale da DJI Pilot 2. Per garantire detection robusta su qualsiasi
palette il dataset sintetico viene generato in 5 varianti:

| Palette | Colormap applicata | Palette DJI corrispondente |
|---|---|---|
| white_hot | Grayscale puro | White Hot |
| black_hot | Inversione grayscale | Black Hot |
| iron_red | cv2.COLORMAP_INFERNO | Iron Red (default DJI) |
| rainbow1 | cv2.COLORMAP_JET | Rainbow 1 |
| hot_iron | cv2.COLORMAP_HOT | Hot Iron |

---

## Risultati

### Fase 2 — Fine-tuning YOLO Thermal (CycleGAN sintetico)

| Metrica | Fase 1 RGB | Fase 2 Thermal | Delta |
|---|---|---|---|
| Box mAP@0.5 | 0.5961 | 0.7994 | +34% |
| Pose mAP@0.5 | 0.3852 | 0.5505 | +43% |
| Box Precision | 0.6492 | 0.7928 | +22% |
| Box Recall | 0.5626 | 0.7593 | +35% |

### Fase 2b — Fine-tuning YOLO Multi-Palette

| Metrica | Fase 1 RGB | Fase 2b Multi-Palette | Delta |
|---|---|---|---|
| Box mAP@0.5 | 0.5961 | 0.8541 | +43% |
| Pose mAP@0.5 | 0.3852 | 0.5790 | +50% |
| Precision | 0.6492 | 0.8578 | +32% |
| Recall | 0.5626 | 0.7814 | +39% |

**Nota importante:** valutazione su val set sintetico generato dalla stessa
CycleGAN del training. Le metriche misurano coerenza interna del dominio
sintetico, non la performance su termico reale DJI.

### Bias della luminanza — limite strutturale identificato

La CycleGAN impara la correlazione statistica luminanza RGB / intensita'
termica invece della vera firma LWIR dei corpi. Questo introduce:
- Falsi positivi su strutture luminose non biologiche mappate come calde
- Falsi negativi su persone con vestiti scuri mappate come fredde

Non risolvibile con piu' dati sintetici — richiede dati termici reali DJI.
Richiesta inoltrata formalmente alla Regione Calabria.

---

## File e struttura

```
src/fase2/
  cyclegan_model.py                    # ResNet-9, PatchGAN, ImageBuffer, GroupNorm
  cyclegan_dataset.py                  # UnpairedDataset dominio A+B, subsample
  generate_thermal.py                  # applica G_AB CycleGAN a dataset_sar (grayscale)
  generate_thermal_multipalette.py     # genera dataset con 5 palette DJI
  plot_cyclegan.py                     # loss curves + visual grid qualita'

runs/fase2/
  cyclegan_run/
    G_AB_final.pth                     # generatore CycleGAN RGB→Thermal
    G_BA_final.pth                     # generatore inverso (non usato in inferenza)
    checkpoint_epoch050.pth            # checkpoint completo
    training_history.json              # loss per epoca
  pix2pix_run_grayscale/
    G_pix2pix_final.pth                # generatore Pix2Pix (risultato negativo, non usato)
    training_history.json
  flypose_thermal_large/
    weights/best.pt                    # modello YOLO thermal CycleGAN (Fase 2)
  flypose_multipalette_large/
    weights/best.pt                    # modello YOLO thermal multi-palette (Fase 2b) ← PRINCIPALE

datasets/
  dataset_sar/                         # RGB originale con annotazioni
  dataset_sar_thermal/                 # thermal CycleGAN (grayscale)
  dataset_sar_thermal_multipalette/    # 5 palette DJI
    white_hot/ black_hot/ iron_red/ rainbow1/ hot_iron/
      images/train/ images/val/ images/test/<seq>/
      labels/train/ labels/val/
```

---

## Demo Launcher — sistema di visualizzazione real-time

Launcher GUI (tkinter, dark mode) per la demo in tempo reale.

```powershell
cd C:\Temp\FlyPose
.\venv\Scripts\python.exe src/demo_launcher.py
```

**Funzionalita':**
- Selezione modalita' RGB (Fase 1) o Thermal Multi-Palette (Fase 2b)
- Scelta palette DJI tra le 5 supportate (radio button)
- Lista sequenze test con indicatore pre-conversione disponibile (✓/○)
- Soglia confidenza variabile (slider 0.05-0.90)
- Vista side-by-side RGB|Thermal con frame sincronizzati
- In modalita' pre-convertita: zero latenza GAN, FPS massimi

**Tasti durante la demo:**

| Tasto | Funzione |
|---|---|
| Q / ESC | Esci |
| S | Screenshot |
| P | Pausa / Riprendi |
| W | Toggle wireframe scheletro |
| B | Toggle bounding box |
| I | Toggle info overlay |
| T | Toggle side-by-side |
| +/- | Aumenta/Diminuisci soglia confidenza |

---

## Note su Windows — Bug Apostrofo

Il path `MATTIA-D'AGOSTINO` causa crash in alcuni tool.
Soluzione: junction point su `C:\Temp\FlyPose\`.

```powershell
# Crea junction point (eseguire come amministratore)
New-Item -ItemType Junction -Path "C:\Temp\FlyPose" `
  -Target "C:\Users\MATTIA-D'AGOSTINO\Desktop\Drone_SAR_RealTime"

# Lancia sempre da qui
cd C:\Temp\FlyPose
.\venv\Scripts\python.exe src/demo_launcher.py
```

---

## Stato attuale

| Attivita' | Stato |
|---|---|
| Architettura CycleGAN implementata | ✓ Completata |
| Dataset LLVIP testato (risultato negativo) | ✓ Documentato |
| Sostituzione dominio B con HIT-UAV | ✓ Completata |
| Training CycleGAN 50 epoche su Kaggle T4×2 | ✓ Completata |
| Verifica qualita' visual grid (SSIM > 0.75) | ✓ Completata |
| Generazione dataset_sar_thermal (22.671 frame) | ✓ Completata |
| Fine-tuning YOLO Large termico (Fase 2) | ✓ Completata — Box mAP +34% |
| Training Pix2Pix su KAIST (esplorazione) | ✓ Completata — risultato negativo |
| Generazione dataset multi-palette 5 palette DJI | ✓ Completata |
| Fine-tuning YOLO Large multi-palette (Fase 2b) | ✓ Completata — Box mAP +43% |
| Conversione sequenze test in 5 palette | ✓ Completata |
| Demo Launcher GUI con side-by-side | ✓ Completata |
| Validazione su termico reale DJI | ⏳ In attesa dati Regione Calabria |

---

## Riferimenti

- Zhu et al. (2017) — CycleGAN: Unpaired Image-to-Image Translation. ICCV 2017.
- Isola et al. (2017) — Pix2Pix: Image-to-Image Translation with Conditional GANs. CVPR 2017.
- Hwang et al. (2015) — KAIST Multispectral Pedestrian Detection Benchmark. CVPR 2015.
- Wang et al. (2021) — LLVIP: Low-Light Visible-Infrared Image Pairs. ICCV 2021.
- Mao et al. (2017) — LSGAN: Least Squares GAN. ICCV 2017.
- Shrivastava et al. (2017) — Learning from Simulated Images (ReplayBuffer). CVPR 2017.
- Xu et al. (2020) — BIRDSAI: Detection and Tracking in Aerial Infrared Imagery. WACV 2020.
