# FlyPose-SAR — Documentazione Tecnica Fase 1: Baseline RGB

## Struttura del Progetto

```
Drone_SAR_RealTime/
├── data.yaml                              # Configurazione dataset YOLO Pose
├── datasets/
│   ├── dataset_sar/                       # Dataset generato da prepare_dataset.py
│   │   ├── images/train/                  # Frame con persone (train)
│   │   ├── images/val/                    # Frame con persone (val)
│   │   ├── labels/train/                  # Annotazioni YOLO Pose (train)
│   │   ├── labels/val/                    # Annotazioni YOLO Pose (val)
│   │   └── dataset_preparation_report.json
│   └── dataset_test_official/
│       └── sequences/                     # Sequenze di test per inferenza
├── modelli_base/
│   ├── yolo11n-pose.pt                    # Pesi Nano  (~2M parametri)
│   ├── yolo11s-pose.pt                    # Pesi Small (~9M parametri)
│   └── yolo11l-pose.pt                    # Pesi Large (~25M) — anche teacher
├── runs/
│   ├── fase1/
│   │   ├── fase1_nano_<timestamp>/
│   │   │   ├── weights/
│   │   │   │   ├── best.pt               # Pesi al picco di mAP
│   │   │   │   └── last.pt               # Pesi ultima epoca
│   │   │   ├── results.csv               # Metriche epoca per epoca
│   │   │   ├── metrics_summary.json      # Riepilogo finale (generato da train.py)
│   │   │   ├── confusion_matrix.png
│   │   │   └── PR_curve.png
│   │   ├── fase1_small_<timestamp>/
│   │   ├── fase1_large_<timestamp>/
│   │   └── comparison_report_<timestamp>.json  # Confronto tutti i run
│   ├── inferenza/
│   │   └── run_<timestamp>/
│   │       ├── <seq_name>/               # Frame annotati o video MP4
│   │       └── inference_report.json
│   └── ground_station/
│       └── run_<timestamp>/
│           ├── gs_OFFICIAL_COCO/         # Output modello COCO
│           ├── gs_CUSTOM_SAR/            # Output modello custom
│           └── comparison_report.json
├── risultati/
│   └── grafici/
│       └── fase1/
│           ├── 01_map50_comparison.png
│           ├── 02_map5095_comparison.png
│           ├── 03_radar_comparison.png
│           ├── 04_heatmap_metrics.png
│           └── 05_loss_curves.png
├── logs/                                  # Log timestampati di tutti gli script
└── src/
    ├── prepare_dataset.py                 # Step 1 — genera dataset_sar
    ├── train.py                           # Step 2 — training Nano/Small/Large
    ├── plot_metrics.py                    # Step 3 — grafici comparativi
    ├── inference.py                       # Step 4 — inferenza modalità drone
    └── inference_ground_station.py        # Step 5 — inferenza SAHI comparativa
```

---

## Dipendenze

```bash
pip install ultralytics torch torchvision opencv-python matplotlib pandas tqdm
```

Se hai CUDA 12.x verifica che torch usi la versione GPU:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.cuda.is_available())"  # deve stampare True
```

---

## Sequenza di Esecuzione Completa

### Step 0 — Verifica che tutto funzioni (opzionale ma consigliato)

```bash
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0))"
python -c "from ultralytics import YOLO; print('Ultralytics OK')"
```

---

### Step 1 — Preparazione Dataset

```bash
python src/prepare_dataset.py
```

**Approccio ibrido GT + Teacher:**

Le annotazioni VisDrone contengono già le bounding box delle persone (ground truth
reale, rilevate manualmente). Lo script le legge direttamente, senza reinventarle.
Il teacher model (`yolo11l-pose.pt`) viene usato esclusivamente per predire i
17 keypoints COCO su ognuno dei crop ritagliati attorno a ogni persona già localizzata.
I keypoints vengono poi riproiettati in coordinate assolute del frame originale
e normalizzati nel formato YOLO Pose.

**Formato annotazioni VisDrone in ingresso** (una riga per oggetto per frame):
```
frame_index, target_id, x_min, y_min, width, height, score, category, truncation, occlusion
```
- `category=1` → pedone, `category=2` → persona generica (le uniche usate)
- `occlusion=2` → scartata (troppo nascosta per keypoints affidabili)

**Formato annotazioni YOLO Pose in uscita** (una riga per persona):
```
cls cx cy w h  kx1 ky1 vis1  kx2 ky2 vis2  ...  kx17 ky17 vis17
```
Tutto normalizzato in `[0,1]`. `vis=2` visibile, `vis=1` parziale, `vis=0` assente.

**Parametri configurabili** (in cima a `prepare_dataset.py`):

| Parametro | Default | Significato |
|-----------|---------|-------------|
| `PERSON_CATEGORIES` | `{1, 2}` | Categorie VisDrone accettate |
| `MAX_OCCLUSION` | `1` | Scarta occlusion > 1 |
| `MIN_BOX_W/H` | `8px` | Scarta persone troppo piccole |
| `CROP_PADDING` | `0.20` | Padding 20% intorno alla box per il crop |
| `TEACHER_CONF` | `0.25` | Confidenza teacher sul crop |
| `MIN_KPT_CONF_MEAN` | `0.20` | Confidenza media minima keypoints |

**Output:**
```
datasets/dataset_sar/
  images/train/   labels/train/
  images/val/     labels/val/
  dataset_preparation_report.json   ← statistiche per sequenza
logs/prepare_dataset_<timestamp>.log
```

**Note operative:**
- Il dataset viene preparato **una volta sola**. Tutti e tre i training della Fase 1
  usano lo stesso `dataset_sar`.
- Se lo script viene interrotto, può essere rilanciatoo: salta automaticamente
  i frame già processati (resume logic).
- Il report JSON contiene il tasso di successo dei keypoints. Se scende sotto
  il 70% considera di abbassare `MIN_KPT_CONF_MEAN` a `0.15`.

**Tempo stimato:** 2–6 ore a seconda della dimensione del dataset e della GPU.

---

### Step 2 — Training Incrementale

```bash
python src/train.py
```

Lo script esegue in sequenza i tre esperimenti definiti in `EXPERIMENTS`.
**Consiglio: la prima volta commenta Small e Large e lancia solo Nano.**
Se dopo 50 epoche il `pose_mAP50` supera 0.30, il dataset è sano e puoi
procedere con gli altri due archivi.

**Perché tre architetture sullo stesso dataset?**

L'unica variabile tra i tre esperimenti è la capacità della rete. Il dataset,
gli iperparametri e le augmentation sono identici. Il confronto permette di
identificare il punto ottimale tra velocità (Nano, adatto a bordo drone) e
precisione (Large, adatto a stazione di terra) per il dominio SAR zenitale specifico.

| Modello | Parametri | Batch | VRAM ~  | Tempo ~  | Uso consigliato |
|---------|-----------|-------|---------|----------|-----------------|
| Nano    | ~2M       | 16    | 3.5 GB  | 1–2 ore  | Bordo drone, real-time |
| Small   | ~9M       | 12    | 4.5 GB  | 2–3 ore  | Bilanciato, edge deploy |
| Large   | ~25M      | 6     | 5.5 GB  | 5–8 ore  | Stazione di terra |

**Augmentation scelte per la visione zenitale SAR:**

- `flipud=0.3` — il flip verticale è valido: il drone vola in direzioni diverse,
  non esiste un "sopra" fisso come nelle immagini frontali standard.
- `degrees=10.0` — rotazione moderata: il drone rollea leggermente ma non di 90°;
  rotazioni eccessive distorcono la geometria dello scheletro.
- `scale=0.5` — scaling aggressivo: simula persone a diverse altitudini di volo.
  Una persona a 60m appare circa 3× più piccola che a 20m.
- `mosaic=1.0` — combina 4 immagini: aumenta varietà di scala e contesto,
  particolarmente utile con dataset di dimensioni moderate.

**Early stopping:** se per 15 epoche consecutive nessuna metrica migliora,
il training si ferma automaticamente. Il `best.pt` viene sempre salvato al picco.

**Se vai OOM (out of memory) su Large:**
- Riduci `batch` da 6 a 4 in `EXPERIMENTS`
- Se persiste, riduci `imgsz` da 640 a 512

**Output per ogni run:**
```
runs/fase1/fase1_<id>_<timestamp>/
  weights/best.pt           ← da usare in inferenza
  weights/last.pt
  results.csv               ← metriche epoca per epoca
  metrics_summary.json      ← riepilogo finale
  confusion_matrix.png
  PR_curve.png
  val_batch*.jpg            ← esempi validazione con skeleton
runs/fase1/comparison_report_<timestamp>.json
logs/training_<timestamp>.log
```

**Metriche chiave da leggere nei risultati:**

| Metrica | Significato |
|---------|-------------|
| `box_mAP50` | % persone trovate con box sovrapposta ≥50% al GT |
| `pose_mAP50` | come sopra ma valuta anche la correttezza dei keypoints |
| `box_mAP50_95` | versione più severa: media su soglie IoU 50→95% |
| `box_recall` | quante persone reali vengono trovate (falsi negativi) |
| `box_precision` | quante detection sono corrette (falsi positivi) |

---

### Step 3 — Grafici Comparativi

```bash
python src/plot_metrics.py
```

Oppure con percorsi espliciti:

```bash
python src/plot_metrics.py --runs-dir runs/fase1 --output-dir risultati/grafici/fase1
```

Può essere eseguito dopo ogni singolo training, non è necessario aspettare
tutti e tre i run. I grafici si aggiornano automaticamente ad ogni esecuzione.

**Grafici prodotti in `risultati/grafici/fase1/`:**

| File | Contenuto |
|------|-----------|
| `01_map50_comparison.png` | Bar chart: Box mAP@0.5 vs Pose mAP@0.5 per modello |
| `02_map5095_comparison.png` | Bar chart: mAP@0.5:0.95 (metrica più severa) |
| `03_radar_comparison.png` | Radar 6 metriche simultanee (≥2 run necessari) |
| `04_heatmap_metrics.png` | Heatmap: tutti i modelli × tutte le metriche |
| `05_loss_curves.png` | Curve loss e mAP per epoca (da results.csv) |

---

### Step 4 — Inferenza Modalità Drone

```bash
python src/inference.py
```

Prima aggiorna `MODEL_PATH` in `inference.py` con il percorso al `best.pt`
del run che vuoi testare (trovi il percorso in `metrics_summary.json`
sotto la chiave `"best_weights"`).

**Cosa fa:** esegue `model.track()` su ogni sequenza di test. Internamente:
frame → backbone CNN → Feature Pyramid Network → teste detection+pose →
NMS → ByteTrack (assegna ID persistenti alle persone nel tempo).

**Parametri configurabili** (in cima a `inference.py`):

| Parametro | Default | Effetto |
|-----------|---------|---------|
| `CONF_THRESHOLD` | `0.40` | Alzare = meno falsi positivi, abbassare = più recall |
| `IOU_THRESHOLD` | `0.35` | Abbassare = più aggressivo nel sopprimere duplicati |
| `IMGSZ` | `1280` | Abbassare a 640 per più velocità |
| `SAVE_VIDEO` | `True` | False = salva frame JPG singoli |

**Output:**
```
runs/inferenza/run_<timestamp>/
  <seq_name>/              ← frame annotati o video MP4
  inference_report.json    ← detection totali, ID univoci per sequenza
logs/inference_<timestamp>.log
```

**Cosa osservare nell'output visivo:**
- Skeleton completo vs troncato sulle persone lontane
- ID di tracking stabili (non flickering tra frame)
- Falsi positivi su ombre o veicoli
- Persone mancate ai bordi del frame

---

### Step 5 — Inferenza Stazione di Terra con SAHI

```bash
python src/inference_ground_station.py
```

Prima aggiorna `CUSTOM_MODEL_PATH` con il percorso al `best.pt` del tuo
modello Large fine-tuned (o del modello con il mAP più alto).

**Differenza fondamentale rispetto a `inference.py`:**

`inference.py` processa il frame intero a 1280px — veloce, per bordo drone.
`inference_ground_station.py` applica SAHI (Slicing Aided Hyper Inference):
divide il frame in riquadri da 640px sovrapposti del 25%, processa ogni
riquadro indipendentemente, riproietta le coordinate e unifica con NMS manuale.
Una persona che a 1280px occupa 12px, nel suo riquadro occupa ~50px → rilevabile.

**Confronto automatico tra due modelli:**
Lo script gira in sequenza il modello COCO ufficiale e il tuo modello SAR
custom sugli stessi dati, producendo output separati e un `comparison_report.json`
con il delta di detection — dati pronti per la tesi.

**Parametri configurabili:**

| Parametro | Default | Effetto |
|-----------|---------|---------|
| `SLICE_SIZE` | `640` | Dimensione ogni riquadro SAHI |
| `OVERLAP_RATIO` | `0.25` | Sovrapposizione tra riquadri adiacenti |
| `CONF` | `0.28` | Leggermente più basso che in inference.py: SAHI già filtra |
| `NMS_IOU` | `0.25` | Soglia NMS globale post-assemblag gio |

**Output:**
```
runs/ground_station/run_<timestamp>/
  gs_OFFICIAL_COCO/        ← frame annotati modello COCO
  gs_CUSTOM_SAR/           ← frame annotati modello SAR
  comparison_report.json   ← detection totali e delta per la tesi
logs/inference_gs_<timestamp>.log
```

---

## Giustificazione delle Scelte per la Tesi

**Approccio ibrido GT + Teacher (prepare_dataset.py)**

Le bounding box di VisDrone sono annotate manualmente — usarle direttamente
elimina i falsi positivi che si generano quando il teacher deve rilevare persone
in un frame intero zenitale (ombre, veicoli e strutture verticali vengono
spesso classificati come persone a bassa confidenza). Il teacher opera solo sul
crop dove sa già che c'è una persona, riducendo il carico semantico del task
a una sola domanda: *come è orientato questo corpo?*

**Knowledge Distillation via pseudo-labeling**

La tecnica di usare un modello Large come teacher per generare le annotazioni
di keypoints su cui addestrare modelli più piccoli è documentata in letteratura
come forma di Knowledge Distillation (Hinton et al., 2015). In questo caso
non si trasferisce conoscenza via soft labels ma via annotazioni sintetiche di
alta qualità — una variante denominata *pseudo-labeling* (Lee, 2013).

**VisDrone-VID come benchmark**

Dataset standard per aerial Small Object Detection (Zhu et al., 2018).
Contiene sequenze video da UAV a diverse altitudini con annotazioni dense
di pedoni e veicoli. La scelta è giustificata dalla disponibilità pubblica,
dalla varietà di condizioni (giorno, densità folla, altitudine variabile)
e dall'uso come benchmark in decine di paper recenti sul tema.

**SAHI per l'inferenza**

Slicing Aided Hyper Inference (Akyon et al., 2022 — github.com/obss/sahi)
risolve la limitazione fondamentale dei detector YOLO su target microscopici:
la rete è addestrata su oggetti che occupano almeno il 2–3% dell'immagine,
mentre persone viste a 80m occupano meno dello 0.1%. La tassellatura riporta
i target a una scala visibile senza modificare l'architettura del modello.

**ByteTrack per il tracking**

ByteTrack (Zhang et al., 2022) mantiene due buffer: uno per le detection ad alta
confidenza e uno per quelle borderline. Questo lo rende superiore a SORT nelle
scene zenitali dove le persone escono ed entrano dall'inquadratura frequentemente
e le occlusioni temporanee sono comuni (persone sotto alberi o tettoie).

---

## Riferimenti

- Zhu et al. (2018) — *VisDrone-DET2018: The Vision Meets Drone Object Detection Challenge*
- Hinton et al. (2015) — *Distilling the Knowledge in a Neural Network*
- Lee (2013) — *Pseudo-Label: The Simple and Efficient Semi-Supervised Learning Method*
- Akyon et al. (2022) — *Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection*
- Zhang et al. (2022) — *ByteTrack: Multi-Object Tracking by Associating Every Detection Box*

---

### Step 6 — Valutazione Formale sul Test Set

```bash
python src/evaluate.py
```

Per forzare la rigenerazione del test set YOLO (se hai cambiato parametri):

```bash
python src/evaluate.py --force-regen
```

**Differenza fondamentale rispetto a `plot_metrics.py`:**

`plot_metrics.py` mostra le metriche di validazione registrate durante il
training — quelle su cui il modello è stato selezionato (early stopping,
best.pt). Sono metriche ottimistiche per definizione.

`evaluate.py` esegue i modelli su dati **mai visti**, confronta le detection
con le annotazioni GT reali e calcola le metriche da zero. Questa è la
valutazione formale che si riporta in tesi.

**Cosa fa lo script in sequenza:**

1. Prepara il test set in formato YOLO Pose (stesso approccio ibrido di
   `prepare_dataset.py`: box GT + keypoints teacher su crop). Il risultato
   va in `datasets/dataset_test_yolo/` — separato da `dataset_sar`.
2. Scrive `data_test.yaml` per puntare `model.val()` al test set.
3. Scopre automaticamente tutti i `best.pt` nei run completati di `runs/fase1/`.
4. Per ogni modello esegue `model.val()` sul test set e raccoglie le metriche.
5. Genera grafici e report JSON con il ranking finale.

**Se non hai il test set ufficiale** (`VisDrone2019-VID-test-dev`), imposta
`USE_VAL_AS_TEST = True` in cima allo script. Le metriche saranno meno
imparziali (il modello ha già visto il val set durante il training) ma
comunque più informative del solo validation loss.

**Output:**

```
datasets/dataset_test_yolo/          ← test set in formato YOLO (generato una volta)
risultati/
  evaluation_report_<timestamp>.json ← ranking e metriche per ogni modello
  grafici/evaluate/
    01_val_vs_test_mAP50.png         ← confronto val (training) vs test formale
    02_test_metrics_complete.png     ← tutte le metriche per ogni modello
    03_test_heatmap.png              ← heatmap modelli × metriche
    04_precision_recall_scatter.png  ← scatter P/R con curve iso-F1
logs/evaluate_<timestamp>.log
```

**Metriche aggiuntive rispetto al training:**

| Metrica | Formula | Significato |
|---------|---------|-------------|
| `box_f1` | `2·P·R / (P+R)` | Media armonica Precision/Recall. Unica cifra riassuntiva |
| Curva iso-F1 | — | Nel grafico scatter: le linee tratteggiate mostrano i punti con lo stesso F1 |
| Delta val→test | — | Quanto le metriche calano da validation a test: indica overfitting |

**Sequenza completa Fase 1 aggiornata:**

```
Step 0  pip install ...
Step 1  python src/prepare_dataset.py     ← genera dataset_sar (una volta)
Step 2  python src/train.py               ← Nano → Small → Large
Step 3  python src/plot_metrics.py        ← grafici metriche validation
Step 4  python src/evaluate.py            ← valutazione formale test set
Step 5  python src/inference.py           ← inferenza qualitativa drone
Step 6  python src/inference_ground_station.py  ← SAHI comparativa
```
