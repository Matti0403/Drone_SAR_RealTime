# src/inference_ground_station.py
import os
import sys
import cv2
import numpy as np
from ultralytics import YOLO

def apply_sahi_slicing(img, slice_size=640, overlap_ratio=0.25):
    """ Seziona l'immagine in riquadri sovrapposti per la Small Object Detection """
    h, w, _ = img.shape
    slices = []
    step = int(slice_size * (1 - overlap_ratio))
    
    for y in range(0, h, step):
        for x in range(0, w, step):
            x_min, y_min = x, y
            x_max = min(x + slice_size, w)
            y_max = min(y + slice_size, h)
            
            if x_max == w: x_min = max(0, w - slice_size)
            if y_max == h: y_min = max(0, h - slice_size)
            
            slices.append((x_min, y_min, x_max, y_max))
            if x_max == w: break
        if y_max == h: break
    return slices

def nms_boxes(boxes, scores, iou_threshold=0.30):
    """ Non-Maximum Suppression rigoroso per evitare accumuli di box sulle ombre """
    if len(boxes) == 0: return []
    boxes, scores = np.array(boxes), np.array(scores)
    
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]
    return keep

def main():
    print("\n=== STAZIONE DI TERRA: SAHI CON FILTRAGGIO GEOMETRICO ANTI-FALSI POSITIVI ===")
    
    safe_root = "C:\\Temp\\Drone_SAR_Inference"
    project_output = os.path.join(safe_root, "runs", "ground_station_sahi_filtered")
    test_sequences_dir = r"C:\Users\MATTIA-D'AGOSTINO\Desktop\Drone_SAR_RealTime\datasets\dataset_test_official\sequences"
    
    model_path = "yolo11l-pose.pt" 
    if not os.path.exists(test_sequences_dir):
        print(f"[ERRORE] Dataset non trovato: {test_sequences_dir}")
        sys.exit(1)
        
    sequences = sorted([d for d in os.listdir(test_sequences_dir) if os.path.isdir(os.path.join(test_sequences_dir, d))])
    detector = YOLO(model_path) 
    
    skeleton_connections = [
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
    ]
    
    for seq_name in sequences:
        print(f"\n[*] Elaborazione sequenza: {seq_name}")
        seq_path = os.path.join(test_sequences_dir, seq_name)
        os.makedirs(os.path.join(project_output, seq_name), exist_ok=True)
        
        frames = sorted([f for f in os.listdir(seq_path) if f.endswith(('.jpg', '.png'))])
        
        for frame_idx, frame_name in enumerate(frames):
            img = cv2.imread(os.path.join(seq_path, frame_name))
            if img is None: continue
            
            slices = apply_sahi_slicing(img, slice_size=640, overlap_ratio=0.25)
            global_boxes, global_scores, global_kpts = [], [], []
            
            for (x_min, y_min, x_max, y_max) in slices:
                slice_img = img[y_min:y_max, x_min:x_max]
                # Portiamo la confidenza a 0.28: un buon compromesso per non catturare rumore di fondo
                results = detector.predict(slice_img, conf=0.28, imgsz=640, classes=0, verbose=False)
                
                for r in results:
                    if r.boxes is not None and len(r.boxes) > 0:
                        boxes = r.boxes.xyxy.cpu().numpy()
                        scores = r.boxes.conf.cpu().numpy()
                        kpts = r.keypoints.xy.cpu().numpy() if r.keypoints is not None else []
                        kpts_conf = r.keypoints.conf.cpu().numpy() if r.keypoints is not None else []
                        
                        for i in range(len(boxes)):
                            w_box = boxes[i][2] - boxes[i][0]
                            h_box = boxes[i][3] - boxes[i][1]
                            
                            # --- FILTRO 1: ANOMALIE DI ASPECT RATIO ---
                            # Le ombre lunghe o i marciapiedi creano box con rapporti esasperati (es. larghissimi o altissimi)
                            if w_box <= 0 or h_box <= 0: continue
                            aspect_ratio = w_box / h_box
                            if aspect_ratio > 2.5 or aspect_ratio < 0.2: 
                                continue # Salta il falso positivo geometrico
                                
                            # --- FILTRO 2: VERIFICA ENTROPIA KEYPOINTS ---
                            # Se la confidenza media dei punti estratti è quasi nulla, è un falso positivo
                            if len(kpts_conf) > i:
                                mean_kpt_conf = np.mean(kpts_conf[i])
                                if mean_kpt_conf < 0.20: 
                                    continue # Salta l'oggetto inanimato scambiato per persona
                            
                            global_boxes.append([
                                boxes[i][0] + x_min, boxes[i][1] + y_min,
                                boxes[i][2] + x_min, boxes[i][3] + y_min
                            ])
                            global_scores.append(scores[i])
                            
                            if len(kpts) > i:
                                kpt_trans = kpts[i].copy()
                                kpt_trans[:, 0] += x_min
                                kpt_trans[:, 1] += y_min
                                global_kpts.append(kpt_trans)
                            else:
                                global_kpts.append([])

            # NMS Globale per ripulire le sovrapposizioni delle slice
            keep_indices = nms_boxes(global_boxes, global_scores, iou_threshold=0.25)
            
            # Rendering dei soli target validati dai filtri
            for idx in keep_indices:
                box = np.array(global_boxes[idx]).astype(int)
                score = global_scores[idx]
                
                # Box verde brillante solo per ciò che supera i vincoli geometrico-strutturali
                cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                cv2.putText(img, f"Target: {score:.2f}", (box[0], box[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                
                if idx < len(global_kpts) and len(global_kpts[idx]) > 0:
                    for kp in global_kpts[idx]:
                        pt = kp.astype(int)
                        if np.any(pt):
                            cv2.circle(img, tuple(pt), 2, (0, 165, 255), -1)
            
            output_filename = os.path.join(project_output, seq_name, f"frame_{frame_idx:07d}.jpg")
            cv2.imwrite(output_filename, img)
            
            if frame_idx % 20 == 0:
                print(f"   -> Frame {frame_idx}/{len(frames)} filtrati ed elaborati...")
                
    print("\n[OK] Pipeline SAHI filtrata geometricamente completata!")

if __name__ == "__main__":
    main()