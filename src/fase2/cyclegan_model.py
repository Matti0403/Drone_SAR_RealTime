# src/fase2/cyclegan_model.py
# FlyPose-SAR — Architettura CycleGAN per traduzione RGB → Thermal
#
# COSA FA:
#   Definisce tutti i componenti dell'architettura CycleGAN:
#   - Generatore (ResNet-based): traduce immagini da un dominio all'altro
#   - Discriminatore (PatchGAN): distingue immagini reali da generate
#   - Funzioni di loss: adversarial + cycle consistency + identity
#
# ARCHITETTURA SCELTA: ResNet-9blocks
#   Standard per CycleGAN su immagini 256x256.
#   9 blocchi residui = buon bilanciamento tra capacità e velocità.
#   Alternativa: ResNet-6blocks (più veloce, leggermente meno qualità).
#
# DISCRIMINATORE: PatchGAN 70x70
#   Invece di classificare l'intera immagine come reale/falsa,
#   classifica patch locali 70x70. Questo produce immagini più nitide
#   perché il discriminatore si concentra su texture locali (la firma
#   termica di un corpo, non la composizione globale della scena).
#
# CYCLE CONSISTENCY LOSS:
#   Il vincolo fondamentale: G_BA(G_AB(x)) ≈ x
#   Se traduci RGB→Thermal→RGB devi tornare all'immagine originale.
#   Questo impedisce alla GAN di "inventare" contenuto che non c'era.
#   Lambda=10 è il peso standard dal paper originale (Zhu et al. 2017).

import torch
import torch.nn as nn
import functools


# ---------------------------------------------------------------------------
# BLOCCO RESIDUO — cuore del generatore
# ---------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    """
    Blocco residuo con reflection padding.
    Reflection padding evita artefatti ai bordi dell'immagine
    (comune con zero-padding nelle GAN).
    """
    def __init__(self, dim: int, norm_layer):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=False),
            norm_layer(dim),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=False),
            norm_layer(dim),
        )

    def forward(self, x):
        return x + self.block(x)   # connessione residua


# ---------------------------------------------------------------------------
# GENERATORE RESNET
# ---------------------------------------------------------------------------
class ResNetGenerator(nn.Module):
    """
    Generatore CycleGAN basato su ResNet.

    Architettura:
        Encoder (downsampling) → Blocchi residui → Decoder (upsampling)

    Input:  immagine RGB 3 canali (o thermal 1 canale se necessario)
    Output: immagine tradotta stesso numero di canali

    n_blocks=9: standard per immagini 256x256
    n_blocks=6: per immagini più piccole o training più veloce
    """
    def __init__(self, input_nc: int = 3, output_nc: int = 3,
                 ngf: int = 64, n_blocks: int = 9,
                 norm_layer=nn.InstanceNorm2d):
        super().__init__()

        # InstanceNorm è preferita a BatchNorm nelle GAN perché
        # normalizza indipendentemente per ogni immagine nel batch,
        # producendo risultati più stabili con batch piccoli
        if isinstance(norm_layer, functools.partial):
            use_bias = (norm_layer.func == nn.InstanceNorm2d)
        else:
            use_bias = (norm_layer == nn.InstanceNorm2d)

        layers = []

        # --- ENCODER: 3 strati di downsampling ---
        # Reflection padding per evitare artefatti ai bordi
        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(inplace=True),
        ]

        # Downsampling ×2 e ×2
        for mult in [1, 2]:
            layers += [
                nn.Conv2d(ngf * mult, ngf * mult * 2,
                          kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(ngf * mult * 2),
                nn.ReLU(inplace=True),
            ]

        # --- BOTTLENECK: n blocchi residui ---
        mult = 4
        for _ in range(n_blocks):
            layers.append(ResidualBlock(ngf * mult, norm_layer))

        # --- DECODER: 2 strati di upsampling ---
        for mult in [4, 2]:
            layers += [
                nn.ConvTranspose2d(ngf * mult, ngf * mult // 2,
                                   kernel_size=3, stride=2, padding=1,
                                   output_padding=1, bias=use_bias),
                norm_layer(ngf * mult // 2),
                nn.ReLU(inplace=True),
            ]

        # Output layer
        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh(),  # output in [-1, 1], normalizzato come input
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# ---------------------------------------------------------------------------
# DISCRIMINATORE PATCHGAN
# ---------------------------------------------------------------------------
class PatchGANDiscriminator(nn.Module):
    """
    Discriminatore PatchGAN 70×70.

    Classifica patch locali dell'immagine invece dell'immagine intera.
    L'output è una mappa di probabilità (non uno scalare singolo):
    ogni valore nella mappa risponde alla domanda "questo patch è reale?"

    Vantaggi rispetto al discriminatore globale:
    - Meno parametri → training più veloce
    - Penalizza incoerenze locali di texture → immagini più nitide
    - Funziona bene con immagini di dimensioni variabili
    """
    def __init__(self, input_nc: int = 3, ndf: int = 64, n_layers: int = 3,
                 norm_layer=nn.InstanceNorm2d):
        super().__init__()

        if isinstance(norm_layer, functools.partial):
            use_bias = (norm_layer.func == nn.InstanceNorm2d)
        else:
            use_bias = (norm_layer == nn.InstanceNorm2d)

        layers = [
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            layers += [
                nn.Conv2d(nf_prev, nf, kernel_size=4, stride=2,
                          padding=1, bias=use_bias),
                norm_layer(nf),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        nf_prev = nf
        nf = min(nf * 2, 512)
        layers += [
            nn.Conv2d(nf_prev, nf, kernel_size=4, stride=1,
                      padding=1, bias=use_bias),
            norm_layer(nf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, 1, kernel_size=4, stride=1, padding=1),
            # Nessuna sigmoid: usiamo MSELoss (LSGAN) che è più stabile
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# ---------------------------------------------------------------------------
# REPLAY BUFFER
# ---------------------------------------------------------------------------
class ImageBuffer:
    """
    Buffer che memorizza le ultime N immagini generate.
    Il discriminatore viene addestrato su immagini sia recenti che passate,
    non solo sull'ultimo batch. Questo riduce le oscillazioni del training
    e previene il mode collapse (la GAN che genera sempre la stessa immagine).
    Tecnica introdotta da Shrivastava et al. (2017).
    """
    def __init__(self, max_size: int = 50):
        self.max_size = max_size
        self.buffer   = []

    def push_and_pop(self, images: torch.Tensor) -> torch.Tensor:
        result = []
        for img in images:
            img = img.unsqueeze(0)
            if len(self.buffer) < self.max_size:
                self.buffer.append(img)
                result.append(img)
            else:
                if torch.rand(1).item() > 0.5:
                    idx = torch.randint(0, self.max_size, (1,)).item()
                    result.append(self.buffer[idx].clone())
                    self.buffer[idx] = img
                else:
                    result.append(img)
        return torch.cat(result, dim=0)


# ---------------------------------------------------------------------------
# INIZIALIZZAZIONE PESI
# ---------------------------------------------------------------------------
def init_weights(net: nn.Module, init_type: str = "normal",
                 init_gain: float = 0.02):
    """
    Inizializzazione dei pesi con distribuzione normale (mean=0, std=init_gain).
    Fondamentale per il training stabile delle GAN:
    pesi troppo grandi → esplosione dei gradienti
    pesi troppo piccoli → vanishing gradient
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and ("Conv" in classname or "Linear" in classname):
            if init_type == "normal":
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif "BatchNorm2d" in classname or "InstanceNorm2d" in classname:
            if m.weight is not None:
                nn.init.normal_(m.weight.data, 1.0, init_gain)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)
    return net


# ---------------------------------------------------------------------------
# FACTORY: crea modelli pronti all'uso
# ---------------------------------------------------------------------------
def build_generator(input_nc=3, output_nc=3, ngf=64,
                    n_blocks=9, device="cpu") -> ResNetGenerator:
    net = ResNetGenerator(input_nc, output_nc, ngf, n_blocks)
    init_weights(net)
    return net.to(device)


def build_discriminator(input_nc=3, ndf=64,
                        n_layers=3, device="cpu") -> PatchGANDiscriminator:
    net = PatchGANDiscriminator(input_nc, ndf, n_layers)
    init_weights(net)
    return net.to(device)