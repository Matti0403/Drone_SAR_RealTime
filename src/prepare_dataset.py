# src/prepare_dataset.py
import cv2
import os
import torch
import shutil
from pathlib import Path
from ultralytics import YOLO

def process_vid_train(teacher_model, src_base, dest_base, device):
    print("[*] Fase 1: Elaborazione sequenze di TRAIN (VisDrone2019-VID-train)...")
    img_dir = src_base / "sequences"
    ann_dir = src_base / "annotations"
    
    if not img_dir.exists() or not ann_dir.exists():
        print(f"[ERRORE Train] Percorsi non trovati. Verifica le cartelle in {src_base}")
        return

    ann_files = sorted(list(ann_dir.glob("*.txt")))
    for idx, ann_file in enumerate(ann_files):
        seq_name = ann_file.stem
        seq_img_dir = img_dir / seq_name
        
        if not seq_img_dir.exists():
            continue
            
        print(f"  [Train] Sequenza {idx+1}/{len(ann_files)}: {seq_name}")
        
        img_files = sorted(list(seq_img_dir.glob("*.jpg")))
        if not img_files:
            img_files = sorted(list(seq_img_dir.iterdir()))
            img_files = [f for f in img_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
            
        if not img_files:
            print(f"    [Avviso] Nessuna immagine trovata in {seq_img_dir}")
            continue

        for img_path in img_files:
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
                
            results = teacher_model(frame, verbose=False, device=device, conf=0.30)[0]
            
            yolo_pose_lines = []
            if results.keypoints is not None and len(results.keypoints) > 0:
                boxes = results.boxes.xywhn.cpu().numpy()
                kpts = results.keypoints.xyn.cpu().numpy()
                
                for b_idx, box in enumerate(boxes):
                    cls = 0  
                    box_str = f"{cls} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}"
                    kpt_str = "".join([f" {kp[0]:.6f} {kp[1]:.6f} {2 if (kp[0]>0 and kp[1]>0) else 0}" for kp in kpts[b_idx]])
                    yolo_pose_lines.append(box_str + kpt_str)
                    
            if yolo_pose_lines:
                nuovo_nome_base = f"{seq_name}_{img_path.stem}"
                dest_img = dest_base / "images" / "train" / f"{nuovo_nome_base}.jpg"
                dest_lbl = dest_base / "labels" / "train" / f"{nuovo_nome_base}.txt"
                
                shutil.copy(str(img_path), str(dest_img))
                with open(dest_lbl, 'w') as out_f:
                    out_f.write("\n".join(yolo_pose_lines) + "\n")
                    
        torch.cuda.empty_cache()

def process_vid_val(teacher_model, src_base, dest_base, device):
    print("\n[*] Fase 2: Elaborazione sequenze di VALIDAZIONE (VisDrone2019-VID-val)...")
    img_dir = src_base / "sequences"
    ann_dir = src_base / "annotations"
    
    if not img_dir.exists() or not ann_dir.exists():
        print(f"[ERRORE Val] Percorsi non trovati. Verifica le cartelle in {src_base}")
        return

    # Svuotiamo la vecchia cartella di validazione errata (DET) per evitare file orfani
    print("[*] Pulizia preventiva delle vecchie cartelle 'val'...")
    for folder in [dest_base / "images" / "val", dest_base / "labels" / "val"]:
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True, exist_ok=True)

    ann_files = sorted(list(ann_dir.glob("*.txt")))
    contatore_val = 0

    for idx, ann_file in enumerate(ann_files):
        seq_name = ann_file.stem
        seq_img_dir = img_dir / seq_name
        
        if not seq_img_dir.exists():
            continue
            
        print(f"  [Val] Sequenza {idx+1}/{len(ann_files)}: {seq_name}")
        
        img_files = sorted(list(seq_img_dir.glob("*.jpg")))
        if not img_files:
            img_files = sorted(list(seq_img_dir.iterdir()))
            img_files = [f for f in img_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
            
        if not img_files:
            print(f"    [Avviso] Nessuna immagine trovata in {seq_img_dir}")
            continue

        for img_path in img_files:
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
                
            results = teacher_model(frame, verbose=False, device=device, conf=0.30)[0]
            
            yolo_pose_lines = []
            if Bag := results.keypoints is not None and len(results.keypoints) > 0:
                boxes = results.boxes.xywhn.cpu().numpy()
                kpts = results.keypoints.xyn.cpu().numpy()
                
                for b_idx, box in enumerate(boxes):
                    cls = 0  
                    box_str = f"{cls} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}"
                    kpt_str = "".join([f" {kp[0]:.6f} {kp[1]:.6f} {2 if (kp[0]>0 and kp[1]>0) else 0}" for kp in kpts[b_idx]])
                    yolo_pose_lines.append(box_str + kpt_str)
                    
            if yolo_pose_lines:
                nuovo_nome_base = f"{seq_name}_{img_path.stem}"
                dest_img = dest_base / "images" / "val" / f"{nuovo_nome_base}.jpg"
                dest_lbl = dest_base / "labels" / "val" / f"{nuovo_nome_base}.txt"
                
                shutil.copy(str(img_path), str(dest_img))
                with open(dest_lbl, 'w') as out_f:
                    out_f.write("\n".join(yolo_pose_lines) + "\n")
                contatore_val += 1
                    
        torch.cuda.empty_cache()
    print(f"\n[+] Validazione aggiornata. Generate {contatore_val} annotazioni con pose reali.")

def main():
    print("[*] =========================================================")
    print("[*] PIPELINE COMPLETA GENERAZIONE DATASET (TRAIN + VAL VID)")
    print("[*] =========================================================")
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    teacher_model = YOLO("modelli_base/yolo11l-pose.pt")
    teacher_model.to(device)
    
    path_train_src = Path(r"C:\Dataset_ViSDrone_RGB\VisDrone2019-VID-train")
    path_val_src = Path(r"C:\Dataset_ViSDrone_RGB\VisDrone2019-VID-val")
    
    script_dir = Path(__file__).resolve().parent
    dest_base = script_dir.parent / "datasets" / "dataset_sar"
    
    # Crea le cartelle se non esistono
    for split in ['train', 'val']:
        (dest_base / "images" / split).mkdir(parents=True, exist_ok=True)
        (dest_base / "labels" / split).mkdir(parents=True, exist_ok=True)

    process_vid_train(teacher_model, path_train_src, dest_base, device)
    
    # Esecuzione immediata del solo aggiornamento Validation Video
    process_vid_val(teacher_model, path_val_src, dest_base, device)
    
    print(f"\n[OK] Pipeline conclusa. I file estratti si trovano in: {dest_base}")

if __name__ == "__main__":
    main()