# FlyPose-SAR — Documentazione Tecnica Fase 1: Baseline RGB
# Stato: COMPLETATA

## Risultati Finali Fase 1

### Valutazione Formale sul Test Set (evaluate.py)

Test set: 5.640 frame su 17 sequenze VisDrone — dati mai visti durante il training.

| Modello | Box mAP@0.5 | Pose mAP@0.5 | Box F1 | Precision | Recall |
|---------|-------------|--------------|--------|-----------|--------|
| YOLO11n-Pose Nano SAR | 0.4739 | 0.1869 | 0.5147 | 0.5865 | 0.4585 |
| YOLO11s-Pose Small SAR | 0.5400 | 0.3004 | 0.5611 | 0.6204 | 0.5122 |
| **YOLO11l-Pose Large SAR** | **0.5961** | **0.3852** | **0.6028** | **0.6492** | **0.5626** |

### Confronto SAHI Stazione di Terra (inference_ground_station.py)

| Modello | Detection totali | Delta | Note |
|---------|-----------------|-------|------|
| Large COCO ufficiale | 21.910 | baseline | Senza fine-tuning |
| Nano SAR fine-tuned | 119.080 | +443% | Meno FP del Large COCO |
| Small SAR fine-tuned | 126.814 | +6.5% vs Nano | Qualità superiore al Nano |
| **Large SAR fine-tuned** | **144.041** | **+13.6% vs Small** | Migliore su tutte le metriche |

### Inferenza con ByteTrack (inference.py)

| Modello | Detection totali | Persone tracciate (ID) | Frame con detection |
|---------|-----------------|----------------------|---------------------|
| Nano SAR | 67.631 | 347 | 5.675 |
| Small SAR | 78.956 | 390 | 5.844 |
| **Large SAR** | **96.144** | **457** | **5.888** |

### Metriche Validation durante Training

| Modello | Box mAP@0.5 | Pose mAP@0.5 | Epoche | Hardware |
|---------|-------------|--------------|--------|----------|
| Nano SAR | 0.631 | 0.296 | 30 | Kaggle T4 |
| Small SAR | 0.678 | 0.434 | 30 | Kaggle T4 |
| Large SAR | 0.723 | 0.528 | 30 | Kaggle T4 |

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
│   │   ├── fase1_nano_locale/             # Run Nano locale (crash apostrofo — ignorare)
│   │   ├── fase1_small/                   # Run Small Kaggle (COMPLETATO)
│   │   │   ├── weights/best.pt            # Pesi migliori (junction C:\Temp\fase1_small)
│   │   │   ├── results.csv
│   │   │   └── metrics_summary.json
│   │   ├── fase1_large/                   # Run Large Kaggle (COMPLETATO)
│   │   │   ├── weights/best.pt            # Pesi migliori (junction C:\Temp\fase1_large)
│   │   │   ├── results.csv
│   │   │   └── metrics_summary.json
│   │   └── comparison_report_*.json
│   ├── inferenza/
│   │   ├── run_<timestamp_nano>/          # Video MP4 inferenza Nano
│   │   ├── run_<timestamp_small>/         # Video MP4 inferenza Small
│   │   └── run_<timestamp_large>/         # Video MP4 inferenza Large
│   └── ground_station/
│       ├── run_20260604_101650/           # COCO vs Nano SAR
│       │   ├── gs_OFFICIAL_COCO/
│       │   ├── gs_CUSTOM_SAR/
│       │   └── comparison_report.json     # 21.910 vs 119.080 detection
│       ├── run_20260604_131041/           # Nano SAR vs Small SAR
│       │   ├── gs_NANO_SAR/
│       │   ├── gs_SMALL_SAR/
│       │   └── comparison_report.json     # 119.080 vs 126.814 detection
│       └── run_20260608_094944/           # Small SAR vs Large SAR
│           ├── gs_SMALL_SAR/
│           ├── gs_LARGE_SAR/
│           └── comparison_report.json     # 126.814 vs 144.041 detection
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
    ├── inference_ground_station.py
    └── fase2/
        ├── __init__.py
        ├── cyclegan_model.py
        ├── cyclegan_dataset.py
        ├── train_cyclegan.py
        ├── generate_thermal.py
        └── plot_cyclegan.py
```

---

## Dipendenze

```bash
pip install ultralytics torch torchvision opencv-python matplotlib pandas tqdm
```

---

## Note su Windows — Bug Apostrofo nel Path Utente

Il path `MATTIA-D'AGOSTINO` contiene un apostrofo che causa crash in Ultralytics.
Soluzione: junction points in `C:\Temp\` senza caratteri speciali.

```powershell
# Dataset SAR
New-Item -ItemType Junction -Path "C:\Temp\dataset_sar" -Target "...\datasets\dataset_sar"
# Dataset test YOLO
New-Item -ItemType Directory -Path "C:\Temp\dataset_test_yolo"
New-Item -ItemType Junction -Path "C:\Temp\dataset_test_yolo" -Target "...\datasets\dataset_test_yolo"
# Pesi modelli
New-Item -ItemType Junction -Path "C:\Temp\fase1_nano"  -Target "...\runs\fase1\fase1_nano"
New-Item -ItemType Junction -Path "C:\Temp\fase1_small" -Target "...\runs\fase1\fase1_small"
New-Item -ItemType Junction -Path "C:\Temp\fase1_large" -Target "...\runs\fase1\fase1_large"
```

`data.yaml` e `data_test.yaml` usano percorsi `C:\Temp\`.
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
- Box dalle annotazioni GT VisDrone (category 1=pedone, 2=persona)
- Teacher `yolo11l-pose.pt` predice i 17 keypoints su crop di ogni persona
- Filtri: occlusion ≤ 1, dimensione ≥ 8×8px, confidenza media kpts ≥ 0.20

| Split | Frame | Annotazioni | Tasso kpts |
|-------|-------|-------------|------------|
| Train | 20.420 | 228.216 | 74.1% |
| Val | 2.251 | 37.669 | 78.6% |
| Totale | 22.671 | 265.885 | ~75% |

### Step 2 — Training su Kaggle (una sessione per modello)

```bash
# Sessione 1 — Nano: batch=32, 30 epoche 
# Sessione 2 — Small: batch=24, 30 epoche
# Sessione 3 — Large: batch=16, 30 epoche
python src/train.py   # con EXPERIMENTS configurato per un solo modello
```

### Step 3 — Grafici Validation
```bash
python src/plot_metrics.py
```

### Step 4 — Valutazione Formale
```bash
python src/evaluate.py
# Richiede junction points e metrics_summary.json con percorsi locali
```

### Step 5 — Inferenza Drone
```bash
# Aggiorna MODEL_PATH in inference.py
python src/inference.py
```

### Step 6 — Ground Station SAHI
```bash
# Aggiorna OFFICIAL_MODEL_PATH e CUSTOM_MODEL_PATH
python src/inference_ground_station.py
```

**Confronti eseguiti:**
1. Large COCO vs Nano SAR → run_20260604_101650 (21.910 vs 119.080)
2. Nano SAR vs Small SAR → run_20260604_131041 (119.080 vs 126.814)
3. Small SAR vs Large SAR → run_20260608_094944 (126.814 vs 144.041)

---

## Analisi dei Risultati

### Progressione SAHI — Messaggio chiave della Fase 1

```
Large COCO  →  Nano SAR  →  Small SAR  →  Large SAR
  21.910       119.080      126.814      144.041
         +443%        +6.5%       +13.6%
```

Un modello 12× più piccolo (Nano) specializzato sul dominio zenitale supera il Large COCO generico sia in quantità (+443%) che in qualità (meno falsi positivi). La progressione Nano→Small→Large dimostra che architetture più grandi continuano a migliorare nello stesso dominio specializzato.

### Osservazioni qualitative

- **Falsi positivi del Nano**: lampioni e strutture verticali a ~0.43 confidenza. Small e Large riducono questi casi.
- **Falsi negativi notturni**: tutti i modelli RGB faticano in condizioni di scarsa illuminazione — motivazione principale per la Fase 2 (ThermalGAN).
- **Sequenza uav0000188**: 0 detection Large COCO vs 2.100+ detection modelli SAR — caso emblematico del valore del fine-tuning zenitale.

### Interpretazione metriche test set

Le metriche test sono inferiori alla validation (Large: Pose mAP 0.385 test vs 0.528 val). Normale: test set contiene scene mai viste con distribuzione diversa. Non è overfitting ma generalizzazione attesa.

---

## Riferimenti

- Zhu et al. (2018) — VisDrone-DET2018. ECCV Workshop.
- Hinton et al. (2015) — Distilling the Knowledge in a Neural Network.
- Akyon et al. (2022) — SAHI: Slicing Aided Hyper Inference. ICIP 2022.
- Zhang et al. (2022) — ByteTrack. ECCV 2022.