"""Leitura dos crops de preparar.py (por memmap: a máquina tem 8 GB) e aumento de dados no dispositivo de treino."""
import math
import os
import shutil
import sys
import zipfile

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from expressao.comum import ROOT                      # noqa: E402
from idade_genero.modelo_torch import np2t            # noqa: E402

DIR = os.path.join(ROOT, "data", "geral")


def load(name):
    """-> dict; X fica em memmap (extraído do .npz para um .npy da primeira vez), o resto em memória."""
    path, npy = os.path.join(DIR, name + ".npz"), os.path.join(DIR, name + "_X.npy")
    with zipfile.ZipFile(path) as z:
        inside = "X.npy" in z.namelist()              # formato antigo: X dentro do .npz
    if inside and (not os.path.exists(npy) or os.path.getmtime(npy) < os.path.getmtime(path)):
        with zipfile.ZipFile(path) as z, z.open("X.npy") as src, open(npy, "wb") as dst:
            shutil.copyfileobj(src, dst, 1 << 22)
    d = np.load(path)
    out = {k: d[k] for k in d.files if k != "X"}
    out["X"] = np.load(npy, mmap_mode="r")
    return out


def dev_split(y, frac=0.1, seed=0):
    """Máscara das imagens do AffectNet train reservadas para escolher a época e o limiar (a validação oficial fica intacta)."""
    rng, m = np.random.RandomState(seed), np.zeros(len(y), bool)
    for c in np.unique(y):
        i = rng.permutation(np.where(y == c)[0])
        m[i[:int(frac * len(i))]] = True
    return m


def batch(X, idx, dev):
    """BGR uint8 -> RGB float NCHW 0..255 no dispositivo."""
    x = np.ascontiguousarray(X[np.sort(idx)][np.argsort(np.argsort(idx))][..., ::-1].transpose(0, 3, 1, 2))
    return np2t(x, np.uint8).to(dev).float()


def vary_margin(x, bits, other, dev):
    """AffectNet: troca a margem (píxeis sem imagem real) por refletida (40%), preta (20%) ou a margem de outra cara
    com fundo real do UTKFace (40%), para a expressão não depender dela."""
    m = np2t(np.unpackbits(bits, axis=1).reshape(len(bits), 1, 112, 112), np.uint8).to(dev).float()
    r = torch.rand(len(x), 1, 1, 1, device=dev)
    j = torch.randint(0, len(other), (len(x),), device=dev)
    fill = torch.where(r < 0.4, x, torch.where(r < 0.6, torch.zeros_like(x), other[j]))
    return x * m + fill * (1 - m)


def augment(x):
    n, dev = len(x), x.device
    x = torch.where((torch.rand(n, device=dev) < 0.5)[:, None, None, None], x.flip(3), x)
    s = 1 + (torch.rand(n, device=dev) * 2 - 1) * 0.08
    r = (torch.rand(n, device=dev) * 2 - 1) * math.radians(8)
    t = (torch.rand(n, 2, device=dev) * 2 - 1) * 0.08
    th = torch.stack([torch.stack([torch.cos(r) / s, -torch.sin(r) / s, t[:, 0]], 1),
                      torch.stack([torch.sin(r) / s, torch.cos(r) / s, t[:, 1]], 1)], 1)
    x = F.grid_sample(x, F.affine_grid(th, list(x.shape), align_corners=False), padding_mode="zeros", align_corners=False)
    g = x.mean(1, keepdim=True)
    x = torch.where(torch.rand(n, 1, 1, 1, device=dev) < 0.1, g.expand_as(x), x)            # cinzento
    c = 1 + (torch.rand(n, 1, 1, 1, device=dev) * 2 - 1) * 0.25
    b = (torch.rand(n, 1, 1, 1, device=dev) * 2 - 1) * 25
    return ((x - 127.5) * c + 127.5 + b).clamp(0, 255)
