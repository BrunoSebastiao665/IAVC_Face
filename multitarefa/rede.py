"""Rede multitarefa (notas da aula): um backbone partilhado e um ramo convolucional por tarefa.

    cara 112x112 RGB ─► TRONCO ArcFace (w600k_mbf, congelado) ─► mapa 256x14x14
                          ├─► ramo IDENTIDADE  (original, congelado)  ─► embedding 512-D (igual ao ArcFace)
                          ├─► ramo GÉNERO      (cópia afinada)        ─► pooling ─► MLP ─► 2 classes
                          ├─► ramo IDADE       (cópia afinada)        ─► pooling ─► MLP ─► 117 classes (DLDL, 0..116 anos)
                          └─► ramo EXPRESSÃO   (cópia afinada)        ─► pooling ─► MLP ─► 8 classes

Cada ramo de atributos é uma cópia das camadas do ArcFace que vão do mapa 256x14x14 até ao mapa 512x7x7
(o 6.º andar + a conv 1x1 de expansão). Arranca com os pesos do ArcFace e é afinado só para a sua tarefa.
O ramo da identidade não é treinado: o embedding fica invariante à expressão, como o ArcFace original.
Diferença para geral/rede.py (modo 'ramo'): lá a idade e o género partilham um ramo; aqui cada tarefa tem o seu.
"""
import copy
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from geral.rede import MAX_AGE, split_backbone          # noqa: E402

TASKS = {"genero": 2, "idade": MAX_AGE + 1, "expressao": 8}


def mlp(d, o, drop=0.3):
    return nn.Sequential(nn.Dropout(drop), nn.Linear(d, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(drop), nn.Linear(256, o))


def pool(m):
    """média + máximo espaciais do mapa 512x7x7 -> 1024"""
    return torch.cat([m.mean((2, 3)), m.amax((2, 3))], 1)


class MultiNet(nn.Module):
    def __init__(self, path):
        super().__init__()
        self.trunk, self.id_branch, at = split_backbone(path)
        for p in list(self.trunk.parameters()) + list(self.id_branch.parameters()):
            p.requires_grad_(False)
        self.br = nn.ModuleDict({t: copy.deepcopy(at) for t in TASKS})
        self.head = nn.ModuleDict({t: mlp(1024, n) for t, n in TASKS.items()})
        self.register_buffer("ages", torch.arange(MAX_AGE + 1, dtype=torch.float32))

    def train(self, flag=True):
        super().train(flag)
        self.trunk.eval(); self.id_branch.eval()         # congelados, incluindo a BatchNorm
        return self

    def features(self, x):
        """x: RGB 0..255 [N,3,112,112] -> mapa do tronco [N,256,14,14] (sem gradiente)."""
        with torch.no_grad():
            return self.trunk((x - 127.5) / 127.5)

    def forward(self, x, tasks=("identidade",) + tuple(TASKS), f=None):
        f = self.features(x) if f is None else f
        out = {}
        if "identidade" in tasks:
            with torch.no_grad():
                out["identidade"] = F.normalize(self.id_branch(f), dim=1)
        for t in tasks:
            if t in TASKS:
                out[t] = self.head[t](pool(self.br[t](f)))
        return out

    def years(self, a):
        return F.softmax(a, 1) @ self.ages

    def age_loss(self, a, age, sigma=2.0):
        """DLDL: alvo gaussiano à volta da idade real + um pouco de L1 na idade esperada (igual a geral/rede.py)."""
        t = torch.exp(-0.5 * ((self.ages[None] - age[:, None]) / sigma) ** 2)
        t = t / t.sum(1, keepdim=True)
        return -(t * F.log_softmax(a, 1)).sum(1).mean() + F.l1_loss(self.years(a), age) / 10

    def task_params(self, t):
        return list(self.br[t].parameters()), list(self.head[t].parameters())


class Export(nn.Module):
    """Mesmas saídas que geral/rede.py, para servir em geral/modelo.py, na galeria e na app sem mudanças."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        o = self.net(x)
        return (o["identidade"], F.softmax(o["genero"], 1), self.net.years(o["idade"])[:, None],
                F.softmax(o["expressao"], 1), F.softmax(o["idade"], 1))


def export(net, path):
    net = net.cpu().eval()
    names = ["embedding", "gender_prob", "age", "expr_prob", "age_dist"]
    torch.onnx.export(Export(net), torch.zeros(2, 3, 112, 112), path, opset_version=13, input_names=["data"],
                      output_names=names, dynamic_axes={k: {0: "N"} for k in ["data"] + names})
