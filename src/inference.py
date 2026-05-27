# src/inference.py
import os
import sys
from pathlib import Path
from ultralytics import YOLO

def main():
    print("\n=== COCKPIT INFERENZA: VALUTAZIONE SU TEST SET UFFICIALE VID ===")
    
    # Rileviamo la cartella root del progetto attuale per leggere il dataset
    base_dir = Path(__file__).resolve().parent.parent
    test_sequences_dir = os.path.join(str(base_dir), "datasets", "dataset_test_official", "sequences")
    
    # PERCORSI NEUTRALI SU C:\Temp (Bypassa il bug dell'apostrofo su PyTorch)
    safe_root = "C:\\Temp\\Drone_SAR_Inference"
    model_path = os.path.join(safe_root, "modelli_base", "yolo11n-pose-sar-best.pt")
    project_output = os.path.join(safe_root, "runs")
    
    # Controlli di sicurezza prima di partire
    if not os.path.exists(model_path):
        print(f"[-] ERRORE: Sposta il modello in: {model_path}")
        sys.exit(1)
        
    if not os.path.exists(test_sequences_dir):
        print(f"[ERRORE] Cartella test non trovata in: {test_sequences_dir}")
        sys.exit(1)
        
    sequences = sorted([d for d in os.listdir(test_sequences_dir) if os.path.isdir(os.path.join(test_sequences_dir, d))])
    print(f"[^] Rilevate {len(sequences)} sequenze video per il test.")
    
    # Inizializzazione Rete (Ora legge da percorso sicuro senza apostrofo!)
    print("[*] Inizializzazione rete neurale...")
    model = YOLO(model_path) 
    
    # Loop sulle sequenze
    for seq_name in sequences:
        print(f"\n[*] Elaborazione sequenza: {seq_name}")
        
        source_path = os.path.join(test_sequences_dir, seq_name)
        
        # Eseguiamo il tracking con confidenza a 0.70 contro i falsi positivi
        results = model.track(
            source=source_path,
            conf=0.25,        
            iou=0.45,
            imgsz=640,
            show=False,
            save=True,
            project=project_output,  # Salva i risultati dentro C:\Temp\Drone_SAR_Inference\runs
            name=seq_name,
            exist_ok=True
        )
        
        # Forziamo lo svuotamento del generatore per scrivere su disco
        for _ in results:
            pass
                
    print("\n=== VALUTAZIONE COMPLETA DEL TEST SET COMPLETATA! ===")
    print(f"[OK] Trovi le sequenze elaborate e ripulite in: {project_output}")

if __name__ == "__main__":
    main()