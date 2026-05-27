# src/inference_ground_station.py
import os
import sys
from pathlib import Path
from ultralytics import YOLO

def main():
    print("\n=== PIPELINE STAZIONE DI TERRA: MODELLO MEDIUM (HIGH CAPACITY) ===")
    
    # Cartella temporanea sicura per aggirare l'apostrofo nel percorso utente di Windows
    safe_root = "C:\\Temp\\Drone_SAR_Inference"
    project_output = os.path.join(safe_root, "runs", "ground_station_medium")
    
    # Percorso assoluto del dataset (usiamo la stringa raw r"..." per i backslash)
    test_sequences_dir = r"C:\Users\MATTIA-D'AGOSTINO\Desktop\Drone_SAR_RealTime\datasets\dataset_test_official\sequences"
    
    # CARICAMENTO MODELLO MEDIUM: Massima capacità di astrazione delle feature
    # Nota: Al primo avvio scaricherà automaticamente 'yolo11m-pose.pt' da internet
    model_path = "yolo11m-pose.pt" 
        
    if not os.path.exists(test_sequences_dir):
        print(f"[ERRORE] Cartella test non trovata in: {test_sequences_dir}")
        sys.exit(1)
        
    sequences = sorted([d for d in os.listdir(test_sequences_dir) if os.path.isdir(os.path.join(test_sequences_dir, d))])
    print(f"[^] Rilevate {len(sequences)} sequenze video trasmesse dal drone.")
    
    print("[*] Inizializzazione Backbone Medium sulla GPU della Stazione di Terra...")
    model = YOLO(model_path) 
    
    for seq_name in sequences:
        print(f"\n[*] Ricezione e analisi flusso sequenza: {seq_name}")
        source_path = os.path.join(test_sequences_dir, seq_name)
        
        # CONFIGURAZIONE ULTRA-ACCURATEZZA STAZIONE DI TERRA
        results = model.track(
            source=source_path,
            conf=0.40,                # Confidenza a 0.40: ottimale per la densità del modello Medium
            iou=0.35,                 # Riduce drasticamente i duplicati nelle folle
            imgsz=1280,               # Risoluzione a 1280px per estrarre pixel utili dai target lontani
            tracker="bytetrack.yaml", # Algoritmo ByteTrack per la continuità degli ID nelle occlusioni
            show=False,
            save=True,
            project=project_output,
            name=seq_name,
            exist_ok=True
        )
        
        # Consuma il generatore per forzare l'elaborazione del video
        for _ in results:
            pass
                
    print("\n=== ELABORAZIONE CON MODELLO MEDIUM COMPLETATA! ===")
    print(f"[OK] I risultati ad alta precisione sono in: {project_output}")

if __name__ == "__main__":
    main()