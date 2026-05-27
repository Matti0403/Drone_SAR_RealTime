# src/train.py
import os
import torch
from ultralytics import YOLO

def main():
    print("=== AVVIO TRAINING DATASET SAR — BASELINE RGB ===")
    
    # 1. Verifica l'hardware per essere sicuri al 100% che usi la GPU
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[*] Hardware rilevato per il training: {device}")
    if device == "cuda:0":
        print(f"[+] GPU: {torch.cuda.get_device_name(0)}")
    
    # 2. Carichiamo il modello di partenza (YOLO11-Nano Pose)
    # È la versione più leggera, ideale per il real-time su drone
    model = YOLO("modelli_base/yolo11n-pose.pt")
    # 3. Lanciamo l'addestramento
    # Settiamo i parametri bilanciandoli per la tua RTX 2060 (6GB VRAM)
    results = model.train(
        data="data.yaml",      # Il file di configurazione modificato con percorso relativo
        epochs=50,             # Numero di epoche
        imgsz=640,             # Dimensione standard delle immagini
        batch=16,              # Quante immagini elaborare insieme
        device=device,         # Forza l'uso della GPU CUDA
        workers=4,             # Thread paralleli per caricare i dati
        project="runs",        # I risultati andranno semplicemente nella cartella "runs" principale
        name="yolo11n_sar_rgb",# Nome dell'esperimento corrente
        exist_ok=True,         # Evita la creazione di cartelle duplicate (es. -2, -3)
        plots=True             # Genera automaticamente i grafici delle metriche per la tesi
    )
    
    print("=== TRAINING COMPLETATO CON SUCCESSO! ===")

if __name__ == "__main__":
    main()