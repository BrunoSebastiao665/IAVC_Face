"""A rede geral: um backbone ArcFace partilhado e uma saída por tarefa.

    cara 112x112 ─► TRONCO (w600k_mbf, congelado) ─┬─► ramo original (congelado) ─► embedding 512-D ─► identidade
                                                    └─► RAMO DE ATRIBUTOS (treinado) ─► idade | género | expressão

mode:
  emb     as cabeças usam SÓ o embedding final de 512-D (a ideia mais simples)
  ramo    as cabeças usam um ramo próprio que parte do mapa 14x14x256 do tronco. O ramo é uma cópia das últimas camadas
          do ArcFace (arranca com os seus pesos) e é afinado; o ramo da identidade fica intacto.

Em ambos o embedding é bit a bit o do ArcFace original, logo a identidade não piora por se treinar o resto.
Idade: DLDL (117 classes, 0..116 anos, alvo gaussiano), porque foi das mais estáveis na fase 3 e dá a distribuição.
"""
import numpy as np
import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
from onnx import numpy_helper

from idade_genero.modelo_torch import np2t

CUT = "473"                    # saída do 5.º andar: 256 x 14 x 14 (daqui para a frente há um ramo por tarefa)
MAX_AGE = 116


class Graph(nn.Module):
    """Executa um pedaço do grafo ONNX do w600k_mbf (Conv, PRelu, Add, Flatten, Gemm, BatchNorm)."""

    def __init__(self, nodes, w, inp, out):
        super().__init__()
        self.inp, self.out, self.plan, self.mods = inp, out, [], nn.ModuleDict()
        for n in nodes:
            name = "m" + n.output[0]
            at = {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}
            if n.op_type == "Conv":
                W = w[n.input[1]]
                m = nn.Conv2d(W.shape[1] * at["group"], W.shape[0], W.shape[2], at["strides"][0], at["pads"][0],
                              groups=at["group"], bias=len(n.input) > 2)
                m.weight.data = np2t(W)
                if len(n.input) > 2:
                    m.bias.data = np2t(w[n.input[2]])
            elif n.op_type == "PRelu":
                m = nn.PReLU(w[n.input[1]].size)
                m.weight.data = np2t(w[n.input[1]].reshape(-1))
            elif n.op_type == "Gemm":
                W = w[n.input[1]]
                assert at.get("transB", 0) == 1
                m = nn.Linear(W.shape[1], W.shape[0], bias=len(n.input) > 2)
                m.weight.data = np2t(W)
                if len(n.input) > 2:
                    m.bias.data = np2t(w[n.input[2]])
            elif n.op_type == "BatchNormalization":
                g, b, mu, va = (w[i] for i in n.input[1:5])
                m = nn.BatchNorm1d(len(g), eps=at.get("epsilon", 1e-5))
                m.weight.data, m.bias.data, m.running_mean.data, m.running_var.data = np2t(g), np2t(b), np2t(mu), np2t(va)
            elif n.op_type in ("Add", "Flatten"):
                m = None
            else:
                raise ValueError(n.op_type)
            if m is not None:
                self.mods[name] = m
            self.plan.append((n.op_type, name, [i for i in n.input if i not in w], n.output[0]))

    def forward(self, x):
        v = {self.inp: x}
        for op, name, ins, out in self.plan:
            v[out] = (v[ins[0]] + v[ins[1]] if op == "Add" else v[ins[0]].flatten(1) if op == "Flatten"
                      else self.mods[name](v[ins[0]]))
        return v[self.out]


def split_backbone(path):
    """-> (tronco, ramo da identidade, ramo de atributos) com os pesos do ArcFace."""
    g = onnx.load(path).graph
    w = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    nodes = list(g.node)
    c = next(i for i, n in enumerate(nodes) if n.output[0] == CUT) + 1
    p = next(i for i, n in enumerate(nodes) if n.output[0] == "509") + 1     # 512 x 7 x 7, antes da projeção final
    return (Graph(nodes[:c], w, g.input[0].name, CUT), Graph(nodes[c:], w, CUT, g.output[0].name),
            Graph(nodes[c:p], w, CUT, "509"))


class GeralNet(nn.Module):
    def __init__(self, path, mode="ramo", n_expr=8, drop=0.3):
        super().__init__()
        self.mode = mode
        self.trunk, self.id_branch, at = split_backbone(path)
        for p in list(self.trunk.parameters()) + list(self.id_branch.parameters()):
            p.requires_grad_(False)
        mlp = lambda d, o: nn.Sequential(nn.Dropout(drop), nn.Linear(d, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(drop), nn.Linear(256, o))
        self.n_expr, d = n_expr, 512
        if mode == "ramo":
            import copy
            self.br_ag, self.br_ex = at, copy.deepcopy(at)       # ramos separados: idade/género e expressão
            d = 1024                                             # média + máximo espacial
        self.h_age, self.h_gen, self.h_exp = mlp(d, MAX_AGE + 1), mlp(d, 2), mlp(d, n_expr)
        self.register_buffer("ages", torch.arange(MAX_AGE + 1, dtype=torch.float32))

    def train(self, flag=True):
        super().train(flag)
        self.trunk.eval(); self.id_branch.eval()                 # congelados (inclui a BatchNorm)
        return self

    def forward(self, x, want="aei"):
        """x: RGB 0..255. want: a = idade+género, e = expressão, i = identidade -> dict."""
        with torch.no_grad():
            f = self.trunk((x - 127.5) / 127.5)
            emb = F.normalize(self.id_branch(f), dim=1) if "i" in want or self.mode == "emb" else None
        pool = lambda m: torch.cat([m.mean((2, 3)), m.amax((2, 3))], 1)
        out = {}
        if "i" in want:
            out["emb"] = emb
        if "a" in want:
            z = emb * 22.6 if self.mode == "emb" else pool(self.br_ag(f))      # 22.6 = sqrt(512): componentes ~ N(0, 1)
            out["age"], out["gen"] = self.h_age(z), self.h_gen(z)
        if "e" in want:
            z = emb * 22.6 if self.mode == "emb" else pool(self.br_ex(f))
            out["exp"] = self.h_exp(z)
        return out

    def years(self, a):
        return F.softmax(a, 1) @ self.ages

    def age_loss(self, a, age):
        t = torch.exp(-0.5 * ((self.ages[None] - age[:, None]) / 2.0) ** 2)
        t = t / t.sum(1, keepdim=True)
        return -(t * F.log_softmax(a, 1)).sum(1).mean() + F.l1_loss(self.years(a), age) / 10


class Export(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        o = self.net(x)
        return o["emb"], F.softmax(o["gen"], 1), self.net.years(o["age"])[:, None], F.softmax(o["exp"], 1)


def export(net, path):
    net = net.cpu().eval()
    names = ["embedding", "gender_prob", "age", "expr_prob"]
    torch.onnx.export(Export(net), torch.zeros(2, 3, 112, 112), path, opset_version=13, input_names=["data"],
                      output_names=names, dynamic_axes={k: {0: "N"} for k in ["data"] + names})
