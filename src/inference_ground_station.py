# src/inference_ground_station.py
import os
import sys
from pathlib import Path
from ultralytics import YOLO

def main():
    print("\n=== PIPELINE STAZIONE DI TERRA: MODELLO LARGE (MAXIMUM CAPACITY) ===")
    
    # Cartella temporanea sicura per Windows (anti-apostrofo)
    safe_root = "C:\\Temp\\Drone_SAR_Inference"
    project_output = os.path.join(safe_root, "runs", "ground_station_large")
    
    # Percorso assoluto del dataset
    test_sequences_dir = r"C:\Users\MATTIA-D'AGOSTINO\Desktop\Drone_SAR_RealTime\datasets\dataset_test_official\sequences"
    
    # CARICAMENTO MODELLO LARGE: Il massimo della precisione per la famiglia YOLO11
    # Nota: Al primo avvio scaricherà automaticamente 'yolo11l-pose.pt' (~90MB+)
    model_path = "yolo11l-pose.pt" 
        
    if not os.path.exists(test_sequences_dir):
        print(f"[ERRORE] Cartella test non trovata in: {test_sequences_dir}")
        sys.exit(1)
        
    sequences = sorted([d for d in os.listdir(test_sequences_dir) if os.path.isdir(os.path.join(test_sequences_dir, d))])
    print(f"[^] Rilevate {len(sequences)} sequenze video in arrivo dal downlink del drone.")
    
    print("[*] Inizializzazione Backbone Large sulla GPU della Stazione di Terra...")
    model = YOLO(model_path) 
    
    for seq_name in sequences:
        print(f"\n[*] Analisi flusso video ad alta risoluzione: {seq_name}")
        source_path = os.path.join(test_sequences_dir, seq_name)
        
        # CONFIGURAZIONE STRATEGICA LARGE
        results = model.track(
            source=source_path,
            conf=0.45,                # Alziamo leggermente la confidenza: il Large è molto più sicuro e pulito
            iou=0.30,                 # Stringiamo l'IoU per separare nettamente le persone vicine
            imgsz=1280,               # Sfruttiamo i 1280px per dare ossigeno ai dettagli piccoli
            tracker="bytetrack.yaml", # ByteTrack per blindare gli ID delle persone
            show=False,
            save=True,
            project=project_output,
            name=seq_name,
            exist_ok=True
        )
        
        # Consuma il generatore per forzare l'elaborazione del video
        for _ in results:
            pass
                
    print("\n=== ELABORAZIONE CON MODELLO LARGE COMPLETATA! ===")
    print(f"[OK] I video finali ad altissima fedeltà sono in: {project_output}")

if __name__ == "__main__":
    main()