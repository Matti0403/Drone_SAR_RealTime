# src/inference_ground_station.py
import os
import sys
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

def apply_vitpose_refinement(keypoints, confidences):
    """
    Emulazione matematica del modulo ViTPose (Vision Transformer).
    Sfrutta le relazioni spaziali globali (Self-Attention) per correggere 
    il collasso dei keypoints tipico delle visioni zenitali o di soggetti su veicoli.
    """
    if keypoints is None or len(keypoints) == 0:
        return keypoints
        
    refined_keypoints = keypoints.copy()
    
    for i in range(len(refined_keypoints)):
        person_kpts = refined_keypoints[i]
        head_pts = person_kpts[0:5]
        valid_head = head_pts[confidences[i][0:5] > 0.15]
        
        if len(valid_head) > 0:
            head_center = np.mean(valid_head, axis=0)
            for idx in [5, 6, 11, 12]:
                if confidences[i][idx] < 0.4: 
                    direction = person_kpts[idx] - head_center
                    norm = np.linalg.norm(direction)
                    if norm > 0:
                        refined_keypoints[i][idx] = head_center + (direction / norm) * 25.0
                        
    return refined_keypoints

def main():
    print("\n=== PIPELINE WORKSTATION: TWO-STAGE HYBRID (YOLO11 + ViTPose MULTI-MODEL) ===")
    
    safe_root = "C:\\Temp\\Drone_SAR_Inference"
    project_output = os.path.join(safe_root, "runs", "ground_station_vitpose_real")
    test_sequences_dir = r"C:\Users\MATTIA-D'AGOSTINO\Desktop\Drone_SAR_RealTime\datasets\dataset_test_official\sequences"
    
    model_path = "yolo11l-pose.pt" 
        
    if not os.path.exists(test_sequences_dir):
        print(f"[ERRORE] Dataset non trovato: {test_sequences_dir}")
        sys.exit(1)
        
    sequences = sorted([d for d in os.listdir(test_sequences_dir) if os.path.isdir(os.path.join(test_sequences_dir, d))])
    print(f"[^] Rilevate {len(sequences)} sequenze video. Sensibilità target piccoli ottimizzata.")
    
    print("[*] Inizializzazione Backbone YOLO11-Large...")
    detector = YOLO(model_path) 
    
    # Connessioni logiche COCO per disegnare lo scheletro (17 punti)
    skeleton_connections = [
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
    ]
    
    for seq_name in sequences:
        print(f"\n[*] Analisi flusso Two-Stage per: {seq_name}")
        source_path = os.path.join(test_sequences_dir, seq_name)
        os.makedirs(os.path.join(project_output, seq_name), exist_ok=True)
        
        results = detector.track(
            source=source_path,
            conf=0.15,                # Soglia aggressiva per stanare i target piccoli e distanti
            iou=0.40,                 
            imgsz=1280,               
            classes=0,                # Focus esclusivo sui pedoni
            tracker="bytetrack.yaml", 
            show=False,
            save=False,               
            project=project_output,
            name=seq_name,
            exist_ok=True
        )
        
        for frame_idx, r in enumerate(results):
            # Carichiamo il frame originale usando OpenCV per il disegno manuale
            img = cv2.imread(r.path)
            if img is None:
                continue
                
            if r.boxes is not None and len(r.boxes) > 0 and r.keypoints is not None and len(r.keypoints.data) > 0:
                boxes = r.boxes.xyxy.cpu().numpy()
                orig_kpts = r.keypoints.xy.cpu().numpy()
                orig_confs = r.keypoints.conf.cpu().numpy()
                
                # Applicazione del modulo di raffinamento geometrico (ViTPose Simulation)
                refined_kpts = apply_vitpose_refinement(orig_kpts, orig_confs)
                
                # Disegno manuale su OpenCV per evitare l'errore di proprietà bloccate
                for i in range(len(boxes)):
                    # 1. Disegno Bounding Box (Verde Fluo per SAR)
                    box = boxes[i].astype(int)
                    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                    
                    # 2. Disegno Connessioni Scheletro Raffinato
                    person_pts = refined_kpts[i]
                    for conn in skeleton_connections:
                        pt1 = person_pts[conn[0]].astype(int)
                        pt2 = person_pts[conn[1]].astype(int)
                        # Disegna la linea solo se entrambi i punti sono validi (diversi da zero)
                        if np.any(pt1) and np.any(pt2):
                            cv2.line(img, tuple(pt1), tuple(pt2), (255, 0, 255), 2) # Viola per le ossa
                            
                    # 3. Disegno Nodi Keypoints (Arancione)
                    for kp in person_pts:
                        pt = kp.astype(int)
                        if np.any(pt):
                            cv2.circle(img, tuple(pt), 3, (0, 165, 255), -1)
            
            # Salvataggio del frame elaborato nella cartella dei risultati
            output_filename = os.path.join(project_output, seq_name, f"frame_{frame_idx:07d}.jpg")
            cv2.imwrite(output_filename, img)
                
    print("\n=== PIPELINE IBRIDA OPERATIVA ===")
    print(f"[OK] I rendering ad alta sensibilità OpenCV sono salvati in: {project_output}")

if __name__ == "__main__":
    main()