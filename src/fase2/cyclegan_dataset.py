# src/fase2/cyclegan_dataset.py
# FlyPose-SAR — Dataset per CycleGAN (immagini non accoppiate)
#
# COSA FA:
#   Carica coppie NON accoppiate di immagini:
#   - Dominio A: frame RGB da dataset_sar (persone zenitali VisDrone)
#   - Dominio B: immagini termiche reali da BIRDSAI Thermal
#
#   "Non accoppiate" significa che non serve avere la stessa scena
#   in entrambi i domini — la CycleGAN impara la traduzione
#   di stile senza corrispondenze pixel-to-pixel.
#
# BIRDSAI Thermal:
#   Dataset di sequenze termiche aeree zenitali.
#   Struttura: C:\Dataset_BIRDSAI_Thermal\images\<seq_name>\*.jpg
#   Contiene 40.661 frame JPG in sottocartelle per sequenza.
#   Usiamo un subset di 5.000 immagini campionate in modo distribuito
#   tra tutte le sequenze per massimizzare la varieta' di scene.
#
# SUBSAMPLING DOMINIO B:
#   Con 40.661 frame e batch=1, ogni epoca richiederebbe 40.661 step —
#   inutilmente lento. Con max_images_b=5000 prendiamo 1 frame ogni ~8,
#   distribuito uniformemente su tutte le sequenze. La GAN vede comunque
#   una varieta' sufficiente di scene termiche zenitali.
#
# PREPROCESSING:
#   Entrambi i domini vengono ridimensionati a 256x256.
#   Normalizzazione in [-1, 1] per compatibilita' con Tanh del generatore.
#   Le immagini termiche BIRDSAI sono in scala di grigi (1 canale) ma le
#   convertiamo a 3 canali replicando per compatibilita' con il generatore.

import os
import random
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


# ---------------------------------------------------------------------------
# TRASFORMAZIONI
# ---------------------------------------------------------------------------
def get_transforms(img_size: int = 256, augment: bool = True):
    """
    Trasformazioni per le immagini di training.
    Load size leggermente più grande (286) poi crop casuale a img_size (256)
    — tecnica standard dal paper CycleGAN originale.
    """
    load_size = int(img_size * 1.12)  # 286 se img_size=256

    if augment:
        transform_list = [
            T.Resize(load_size, interpolation=T.InterpolationMode.BICUBIC),
            T.RandomCrop(img_size),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    else:
        transform_list = [
            T.Resize(img_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]

    return T.Compose(transform_list)


# ---------------------------------------------------------------------------
# DATASET NON ACCOPPIATO
# ---------------------------------------------------------------------------
class UnpairedDataset(Dataset):
    """
    Dataset per CycleGAN con immagini non accoppiate.

    Carica immagini da due cartelle indipendenti (dominio A e dominio B).
    Ad ogni accesso restituisce una coppia casuale (img_A, img_B).
    Le dimensioni dei due dataset possono essere diverse — il dataset
    più corto viene ciclato (modulo).

    Args:
        dir_A: cartella immagini dominio A (RGB zenitale VisDrone)
        dir_B: cartella immagini dominio B (thermal FLIR)
        img_size: dimensione di output in pixel
        augment: applica augmentation (True per train, False per val/test)
    """
    IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

    def __init__(self, dir_A: str, dir_B: str,
                 img_size: int = 256, augment: bool = True,
                 max_images_b: int = 5000):
        """
        Args:
            dir_A:        cartella immagini dominio A (RGB zenitale VisDrone)
            dir_B:        cartella immagini dominio B (thermal BIRDSAI)
            img_size:     dimensione output in pixel (default 256)
            augment:      applica augmentation (True per train)
            max_images_b: numero massimo immagini dominio B.
                          Con BIRDSAI (40.661 frame) usare 5000 per velocizzare
                          il training mantenendo varieta' sufficiente di scene.
                          None = usa tutte le immagini.
        """
        self.dir_A     = Path(dir_A)
        self.dir_B     = Path(dir_B)
        self.transform = get_transforms(img_size, augment)

        self.paths_A = sorted([
            p for p in self.dir_A.rglob("*")
            if p.suffix.lower() in self.IMG_EXTENSIONS
        ])
        self.paths_B = sorted([
            p for p in self.dir_B.rglob("*")
            if p.suffix.lower() in self.IMG_EXTENSIONS
        ])

        if not self.paths_A:
            raise FileNotFoundError(f"Nessuna immagine trovata in dominio A: {dir_A}")
        if not self.paths_B:
            raise FileNotFoundError(f"Nessuna immagine trovata in dominio B: {dir_B}")

        # Subsample distribuito dominio B:
        # Prende 1 frame ogni N in modo uniforme su tutto il dataset ordinato.
        # Con dataset BIRDSAI ordinato per sequenza, questo garantisce che ogni
        # sequenza contribuisca proporzionalmente al subset finale.
        if max_images_b and len(self.paths_B) > max_images_b:
            step = len(self.paths_B) // max_images_b
            self.paths_B = self.paths_B[::step][:max_images_b]
            print(f"[Dataset] Dominio B subsample: 1 ogni {step} frame")

        # Il dataset ha lunghezza pari al dominio più grande
        self.size = max(len(self.paths_A), len(self.paths_B))
        print(f"[Dataset] Dominio A (RGB):     {len(self.paths_A)} immagini")
        print(f"[Dataset] Dominio B (Thermal): {len(self.paths_B)} immagini")

    def __len__(self):
        return self.size

    def _load_rgb(self, path: Path) -> torch.Tensor:
        """Carica immagine e la converte in RGB a 3 canali."""
        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return self.transform(img)

    def __getitem__(self, idx):
        # Cicla se l'indice supera la dimensione del dominio
        path_A = self.paths_A[idx % len(self.paths_A)]
        # Dominio B: indice casuale per evitare correlazioni artificiali
        path_B = self.paths_B[random.randint(0, len(self.paths_B) - 1)]

        return {
            "A":      self._load_rgb(path_A),
            "B":      self._load_rgb(path_B),
            "path_A": str(path_A),
            "path_B": str(path_B),
        }


# ---------------------------------------------------------------------------
# DATASET PER GENERAZIONE (solo dominio A, nessuna augmentation)
# ---------------------------------------------------------------------------
class SingleDomainDataset(Dataset):
    """
    Dataset per la fase di generazione: applica G_AB a tutte le immagini
    del dominio A (dataset_sar completo) per creare il dataset termico.
    Non ha augmentation — vogliamo le immagini originali tradotte fedelmente.
    """
    IMG_EXTENSIONS = {".jpg", ".jpeg", ".png"}

    def __init__(self, dir_A: str, img_size: int = 256):
        self.dir_A     = Path(dir_A)
        self.transform = get_transforms(img_size, augment=False)

        self.paths = sorted([
            p for p in self.dir_A.rglob("*")
            if p.suffix.lower() in self.IMG_EXTENSIONS
        ])

        if not self.paths:
            raise FileNotFoundError(f"Nessuna immagine in: {dir_A}")

        print(f"[SingleDomain] {len(self.paths)} immagini da tradurre")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img  = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return {
            "A":    self.transform(img),
            "path": str(path),
            "stem": path.stem,
            "parent": str(path.parent.name),
        }


# ---------------------------------------------------------------------------
# FACTORY DATALOADER
# ---------------------------------------------------------------------------
def get_dataloader(dir_A: str, dir_B: str, batch_size: int = 1,
                   img_size: int = 256, num_workers: int = 4,
                   augment: bool = True,
                   max_images_b: int = 5000) -> DataLoader:
    dataset = UnpairedDataset(dir_A, dir_B, img_size, augment, max_images_b)
    return DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = num_workers,
        pin_memory  = True,
        drop_last   = True,
    )