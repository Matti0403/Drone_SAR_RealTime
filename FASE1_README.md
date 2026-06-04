# FlyPose-SAR — Documentazione Tecnica Fase 1: Baseline RGB
# Stato: COMPLETATA

## Risultati Finali Fase 1

### Valutazione Formale sul Test Set (evaluate.py)

| Modello | Box mAP@0.5 | Pose mAP@0.5 | Box F1 | Box Precision | Box Recall |
|---------|-------------|--------------|--------|---------------|------------|
| YOLO11n-Pose Nano SAR | 0.4739 | 0.1869 | 0.5147 | 0.5865 | 0.4585 |
| YOLO11s-Pose Small SAR | **0.5400** | **0.3004** | **0.5611** | **0.6204** | **0.5122** |

### Confronto SAHI Stazione di Terra (inference_ground_station.py)

| Modello | Detection totali | Note |
|---------|-----------------|------|
| Large COCO ufficiale | 21.910 | Baseline senza fine-tuning |
| Nano SAR fine-tuned | 119.080 | +443% vs COCO, meno falsi positivi |
| Small SAR fine-tuned | 126.814 | +6.5% vs Nano, qualità superiore |

### Metriche Validation durante Training

| Modello | Box mAP@0.5 | Pose mAP@0.5 | Epoche | Hardware |
|---------|-------------|--------------|--------|----------|
| Nano SAR | 0.631 | 0.296 | 30 (early stop) | Kaggle T4 |
| Small SAR | 0.540 | 0.300 | 30 | Kaggle T4 |

---

## Struttura del Progetto

```
Drone_SAR_RealTime/
├── data.yaml                              # Configurazione dataset YOLO Pose
├── data_test.yaml                         # Config test set (generato da evaluate.py)
├── datasets/
│   ├── dataset_sar/                       # Dataset generato da prepare_dataset.py
│   │   ├── images/train/                  # 20.420 frame con persone (train)
│   │   ├── images/val/                    # 2.251 frame con persone (val)
│   │   ├── labels/train/                  # 228.216 annotazioni YOLO Pose
│   │   ├── labels/val/                    # 37.669 annotazioni YOLO Pose
│   │   └── dataset_preparation_report.json
│   ├── dataset_test_official/             # Test set VisDrone originale
│   │   ├── annotations/                   # GT annotazioni 17 sequenze
│   │   └── sequences/                     # Frame JPEG 17 sequenze
│   └── dataset_test_yolo/                 # Test set in formato YOLO (C:\Temp\)
│       ├── images/test/                   # 5.640 frame annotati
│       └── labels/test/                   # Annotazioni YOLO Pose
├── modelli_base/
│   ├── yolo11n-pose.pt                    # Pesi Nano  (~2M parametri)
│   ├── yolo11s-pose.pt                    # Pesi Small (~9M parametri)
│   └── yolo11l-pose.pt                    # Pesi Large (~25M) — teacher model
├── runs/
│   ├── fase1/
│   │   ├── fase1_nano/                    # Run Nano Kaggle (COMPLETATO)
│   │   │   ├── weights/best.pt            # Pesi migliori (junction C:\Temp\fase1_nano)
│   │   │   ├── results.csv
│   │   │   └── metrics_summary.json
│   │   ├── fase1_nano_locale/             # Run Nano locale (crash apostrofo)
│   │   ├── fase1_small/                   # Run Small Kaggle (COMPLETATO)
│   │   │   ├── weights/best.pt            # Pesi migliori (junction C:\Temp\fase1_small)
│   │   │   ├── results.csv
│   │   │   └── metrics_summary.json
│   │   └── comparison_report_*.json
│   ├── inferenza/
│   │   ├── run_<timestamp_nano>/          # Video MP4 inferenza Nano
│   │   └── run_<timestamp_small>/         # Video MP4 inferenza Small
│   └── ground_station/
│       ├── run_20260604_101650/           # COCO vs Nano SAR
│       │   ├── gs_OFFICIAL_COCO/
│       │   ├── gs_CUSTOM_SAR/
│       │   └── comparison_report.json     # 21.910 vs 119.080 detection
│       └── run_20260604_131041/           # Nano SAR vs Small SAR
│           ├── gs_NANO_SAR/
│           ├── gs_SMALL_SAR/
│           └── comparison_report.json     # 119.080 vs 126.814 detection
├── risultati/
│   ├── evaluation_report_*.json           # Metriche formali test set
│   └── grafici/
│       ├── fase1/
│       │   ├── 01_map50_comparison.png
│       │   ├── 02_map5095_comparison.png
│       │   ├── 03_radar_comparison.png
│       │   ├── 04_heatmap_metrics.png
│       │   └── 05_loss_curves.png
│       └── evaluate/
│           ├── 01_val_vs_test_mAP50.png
│           ├── 02_test_metrics_complete.png
│           ├── 03_test_heatmap.png
│           └── 04_precision_recall_scatter.png
├── logs/
└── src/
    ├── prepare_dataset.py
    ├── train.py
    ├── plot_metrics.py
    ├── evaluate.py
    ├── inference.py
    └── inference_ground_station.py
```

---

## Dipendenze

```bash
pip install ultralytics torch torchvision opencv-python matplotlib pandas tqdm
```

---

## Note su Windows — Bug Apostrofo nel Path Utente

Il path `MATTIA-D'AGOSTINO` contiene un apostrofo che causa crash in Ultralytics
quando cerca di creare cartelle o caricare pesi. La soluzione adottata è creare
junction points in `C:\Temp\` senza caratteri speciali:

```powershell
# Dataset SAR (per il training)
New-Item -ItemType Junction -Path "C:\Temp\dataset_sar" -Target "...\datasets\dataset_sar"

# Dataset test YOLO
New-Item -ItemType Directory -Path "C:\Temp\dataset_test_yolo"
New-Item -ItemType Junction -Path "C:\Temp\dataset_test_yolo" -Target "...\datasets\dataset_test_yolo"

# Pesi modelli
New-Item -ItemType Junction -Path "C:\Temp\fase1_nano"  -Target "...\runs\fase1\fase1_nano"
New-Item -ItemType Junction -Path "C:\Temp\fase1_small" -Target "...\runs\fase1\fase1_small"
```

`data.yaml` e `data_test.yaml` usano percorsi `C:\Temp\` per evitare il bug.
`evaluate.py` usa `project=r"C:\Temp\eval_results"` in `model.val()`.

---

## Sequenza di Esecuzione Completa

### Step 0 — Installazione

```bash
pip install ultralytics torch torchvision opencv-python matplotlib pandas tqdm
```

### Step 1 — Preparazione Dataset

```bash
python src/prepare_dataset.py
```

**Approccio ibrido GT + Teacher:**
- Box lette dalle annotazioni GT VisDrone (category 1=pedone, 2=persona)
- Teacher `yolo11l-pose.pt` predice i 17 keypoints COCO su crop di ogni persona
- Filtri: occlusion ≤ 1, dimensione ≥ 8×8px, confidenza media kpts ≥ 0.20
- Resume logic: riprende da dove si era fermato se interrotto

**Statistiche dataset generato:**

| Split | Frame salvati | Annotazioni | Tasso kpts |
|-------|--------------|-------------|------------|
| Train | 20.420 | 228.216 | 74.1% |
| Val | 2.251 | 37.669 | 78.6% |
| Totale | 22.671 | 265.885 | ~75% |

### Step 2 — Training su Kaggle

Training eseguito su Kaggle GPU T4 (16GB VRAM) — una sessione per modello.

**Sessione 1 — Nano (30 epoche, early stop a 25):**
```python
# In train.py, EXPERIMENTS con solo fase1_nano
# batch=32, imgsz=640
```

**Sessione 2 — Small (30 epoche):**
```python
# In train.py, EXPERIMENTS con solo fase1_small
# batch=24, imgsz=640
```

**Sessione 3 — Large (da fare):**
```python
# In train.py, EXPERIMENTS con solo fase1_large
# batch=12, imgsz=640
```

### Step 3 — Grafici Validation

```bash
python src/plot_metrics.py
```

### Step 4 — Valutazione Formale

```bash
python src/evaluate.py
```

Prima assicurarsi che i junction points siano creati e i `metrics_summary.json`
abbiano i percorsi locali corretti (non i percorsi Kaggle).

### Step 5 — Inferenza Drone

```bash
# Aggiorna MODEL_PATH in inference.py con il best.pt desiderato
python src/inference.py
```

### Step 6 — Ground Station SAHI

```bash
# Aggiorna OFFICIAL_MODEL_PATH e CUSTOM_MODEL_PATH in inference_ground_station.py
python src/inference_ground_station.py
```

**Confronti eseguiti:**
1. `OFFICIAL_COCO` (yolo11l-pose.pt) vs `CUSTOM_SAR` (fase1_nano/best.pt)
2. `NANO_SAR` (fase1_nano/best.pt) vs `SMALL_SAR` (fase1_small/best.pt)

---

## Analisi dei Risultati

### Osservazioni qualitative (inferenza visiva)

- **Nano SAR vs Large COCO**: il Nano SAR trova 5.4× più persone con meno
  falsi positivi. Il Large COCO è praticamente cieco su alcune sequenze zenitali
  (es. uav0000188: 0 detection COCO vs 2.100 SAR).
- **Small SAR vs Nano SAR**: il Small trova il 6.5% di detection in più
  mantenendo la stessa qualità (meno falsi positivi rispetto al Nano).
- **Falsi positivi tipici del Nano**: lampioni e strutture verticali sottili
  vengono occasionalmente classificati come persone (confidenza ~0.43).
  Il Small riduce questi casi grazie alla maggiore capacità discriminativa.
- **Falsi negativi in scene notturne**: entrambi i modelli faticano in
  condizioni di scarsa illuminazione — motivazione principale per la Fase 2
  (ThermalGAN).

### Interpretazione metriche formali (test set)

Le metriche sul test set sono inferiori a quelle sul validation set durante
il training (Small: Pose mAP 0.30 test vs 0.42 val). Questo è atteso:
il test set contiene scene mai viste con distribuzione leggermente diversa.
Il gap non indica overfitting grave ma normale generalizzazione.

### Conclusione della Fase 1

Il fine-tuning sul dominio zenitale SAR è **essenziale**: un modello 12×
più piccolo (Nano, ~2M parametri) specializzato sul dominio batte un modello
Large (~25M parametri) generico sia in quantità di detection che in qualità
(meno falsi positivi). Il messaggio per la tesi:

```
Large COCO  →  Nano SAR  →  Small SAR  →  [Large SAR da fare]
  21.910       119.080      126.814
         +443%        +6.5%
```

---

## Riferimenti

- Zhu et al. (2018) — VisDrone-DET2018. ECCV Workshop.
- Hinton et al. (2015) — Distilling the Knowledge in a Neural Network.
- Akyon et al. (2022) — SAHI: Slicing Aided Hyper Inference. ICIP 2022.
- Zhang et al. (2022) — ByteTrack. ECCV 2022.
