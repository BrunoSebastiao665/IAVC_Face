"""Fine-tuning do genderage (InsightFace) no UTKFace — multi-tarefa idade + género.

Parte dos pesos pré-treinados (models/genderage.onnx) e ajusta a rede inteira no split de treino
de data/UTKFace/utkface_96.npz (criado por preparar_utkface.py). A melhor época é escolhida pelo
MAE no split de validação; o teste NÃO é usado aqui (só em avaliar.py).

Perda = CE(género) + perda de idade da variante (ver modelo_torch.py: reg | cls | dldl | ordinal).
Otimizador AdamW + one-cycle; lr do backbone = lr/3, lr das cabeças = lr.
Aumento de dados: flip horizontal, pequena afim (escala ±10%, rotação ±10°, translação ±6%),
brilho/contraste.

Saídas:
    models/genderage_ft_<variante>.onnx
    idade_genero/resultados/treino_<variante>.csv    (curva de treino)

Uso:
    python idade_genero/finetune.py --variant reg
    python idade_genero/finetune.py --variant all --epochs 12     # as 4 variantes seguidas
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from idade_genero.age_gender import AGE_GROUPS, MODELS_DIR, ROOT, age_group   # noqa: E402
from idade_genero.modelo_torch import VARIANTS, GenderAgeNet, export_onnx, np2t  # noqa: E402

DATA = os.path.join(ROOT, "data", "UTKFace", "utkface_96.npz")
RES = os.path.join(ROOT, "idade_genero", "resultados")


def load_split(d, split):
    m = d["split"] == split
    X = np2t(d["X"][m][..., ::-1].transpose(0, 3, 1, 2), np.uint8)        # BGR->RGB, NCHW, uint8
    age = d["age"][m].astype(np.float32)
    group = np.array([age_group(a) for a in age], np.int64)
    return X, np2t(age), np2t(group, np.int64), np2t(d["gender"][m], np.int64)


def augment(x):
    """x float [N,3,96,96] 0..255 -> versão aumentada (mesma forma)."""
    n = len(x)
    flip = torch.rand(n) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(3), x)
    s = 1 + (torch.rand(n) * 2 - 1) * 0.10
    r = (torch.rand(n) * 2 - 1) * math.radians(10)
    t = (torch.rand(n, 2) * 2 - 1) * 0.12            # em coordenadas normalizadas [-1,1] -> ±6%
    theta = torch.stack([torch.stack([torch.cos(r) / s, -torch.sin(r) / s, t[:, 0]], 1),
                         torch.stack([torch.sin(r) / s, torch.cos(r) / s, t[:, 1]], 1)], 1)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)
    c = 1 + (torch.rand(n, 1, 1, 1) * 2 - 1) * 0.2
    b = (torch.rand(n, 1, 1, 1) * 2 - 1) * 20
    return ((x - 127.5) * c + 127.5 + b).clamp(0, 255)


@torch.no_grad()
def evaluate(net, X, age, gender, bs=512):
    net.eval()
    preds, gpreds = [], []
    for i in range(0, len(X), bs):
        g, a = net(X[i:i + bs].float())
        preds.append(net.age_from_output(a)); gpreds.append(g.argmax(1))
    p, gp = torch.cat(preds), torch.cat(gpreds)
    return float((p - age).abs().mean()), float((gp == gender).float().mean())


def train(variant, d, epochs, bs, lr):
    Xtr, age_tr, grp_tr, g_tr = load_split(d, 0)
    Xva, age_va, _, g_va = load_split(d, 1)
    centers = np.array([d["age"][d["split"] == 0][[age_group(a) == k for a in d["age"][d["split"] == 0]]].mean()
                        for k in range(len(AGE_GROUPS))], np.float32)
    net = GenderAgeNet(os.path.join(MODELS_DIR, "genderage.onnx"), variant, group_centers=centers)
    # backbone pré-treinado com lr 3x menor que as cabeças (que nas variantes novas começam aleatórias)
    heads = [p for n, p in net.named_parameters() if n.startswith(("fc_", "ord_bias"))]
    backbone = [p for n, p in net.named_parameters() if not n.startswith(("fc_", "ord_bias"))]
    opt = torch.optim.AdamW([{"params": backbone, "lr": lr / 3}, {"params": heads, "lr": lr}], weight_decay=1e-4)
    steps = epochs * math.ceil(len(Xtr) / bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[lr / 3, lr], total_steps=steps, pct_start=0.15)

    mae0, acc0 = evaluate(net, Xva, age_va, g_va)
    print(f"\n[{variant}] treino {len(Xtr)} | val {len(Xva)} | antes do fine-tuning: val MAE {mae0:.2f}, género acc {acc0:.3f}"
          + ("  (cabeça de idade nova, ainda aleatória)" if variant != "reg" else ""))
    log = [("epoca", "loss_genero", "loss_idade", "val_mae", "val_acc_genero", "segundos")]
    best, best_state = 1e9, None
    for ep in range(1, epochs + 1):
        net.train()
        t0, lg_sum, la_sum, nb = time.time(), 0.0, 0.0, 0
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), bs):
            idx = perm[i:i + bs]
            g, a = net(augment(Xtr[idx].float()))
            lg = F.cross_entropy(g, g_tr[idx])
            la = net.age_loss(a, age_tr[idx], grp_tr[idx])
            opt.zero_grad()
            (lg + la).backward()
            opt.step(); sched.step()
            lg_sum += float(lg); la_sum += float(la); nb += 1
        mae, acc = evaluate(net, Xva, age_va, g_va)
        dt = time.time() - t0
        log.append((ep, lg_sum / nb, la_sum / nb, mae, acc, dt))
        mark = ""
        if mae < best:
            best, best_state, mark = mae, {k: v.clone() for k, v in net.state_dict().items()}, "  *"
        print(f"  época {ep:2d}/{epochs}  loss género {lg_sum / nb:.3f}  loss idade {la_sum / nb:.3f}  "
              f"val MAE {mae:.2f}  género acc {acc:.3f}  ({dt:.0f}s){mark}")

    net.load_state_dict(best_state)
    out = os.path.join(MODELS_DIR, f"genderage_ft_{variant}.onnx")
    export_onnx(net, out)
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, f"treino_{variant}.csv"), "w") as fp:
        fp.write("\n".join(",".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in row) for row in log) + "\n")
    print(f"  melhor val MAE {best:.2f} -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="all", choices=list(VARIANTS) + ["all"])
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    torch.set_num_threads(os.cpu_count())
    d = dict(np.load(DATA))
    for v in (VARIANTS if a.variant == "all" else [a.variant]):
        train(v, d, a.epochs, a.bs, a.lr)


if __name__ == "__main__":
    main()
