# src/inference.py
import os
import sys
from pathlib import Path
from ultralytics import YOLO

def main():
    print("\n=== COCKPIT INFERENZA: SPERIMENTAZIONE FINALE AVANZATA (VERSIONE 3) ===")
    
    base_dir = Path(__file__).resolve().parent.parent
    test_sequences_dir = os.path.join(str(base_dir), "datasets", "dataset_test_official", "sequences")
    
    # Percorsi neutrali su C:\Temp per evitare il bug dell'apostrofo di PyTorch
    safe_root = "C:\\Temp\\Drone_SAR_Inference"
    model_path = os.path.join(safe_root, "modelli_base", "yolo11n-pose-sar-best.pt")
    
    # Cartella finale incrementale per il confronto nella tesi
    project_output = os.path.join(safe_root, "runs", "optimized_v3_final")
    
    if not os.path.exists(model_path):
        print(f"[-] ERRORE: Il modello non è stato trovato in: {model_path}")
        sys.exit(1)
        
    if not os.path.exists(test_sequences_dir):
        print(f"[ERRORE] Cartella test non trovata in: {test_sequences_dir}")
        sys.exit(1)
        
    sequences = sorted([d for d in os.listdir(test_sequences_dir) if os.path.isdir(os.path.join(test_sequences_dir, d))])
    print(f"[^] Rilevate {len(sequences)} sequenze video per il test.")
    
    print("[*] Inizializzazione rete neurale...")
    model = YOLO(model_path) 
    
    for seq_name in sequences:
        print(f"\n[*] Elaborazione sequenza: {seq_name}")
        source_path = os.path.join(test_sequences_dir, seq_name)
        
        # PARAMETRI AVANZATI ANTI-OCCLUSIONE E ANTI-FOLLA
        results = model.track(
            source=source_path,
            conf=0.40,             # Leggermente alzato a 0.40 per ripulire i falsi positivi zenitali
            iou=0.35,              # <--- ABBASSATO: Elimina le scatole duplicate che si sovrappongono nelle folle
            imgsz=1280,            # Risoluzione alta confermata
            tracker="bytetrack.yaml", # <--- BYTETRACK: Gestisce nettamente meglio le occlusioni temporanee
            show=False,
            save=True,
            project=project_output,
            name=seq_name,
            exist_ok=True
        )
        
        for _ in results:
            pass
                
    print("\n=== SPERIMENTAZIONE FINALE COMPLETATA! ===")
    print(f"[OK] Trovi i risultati definitivi bilanciati in: {project_output}")

if __name__ == "__main__":
    main()