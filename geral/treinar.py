"""Treina as saídas de idade, género e expressão do modelo geral; o tronco e a identidade ficam congelados.

Cada passo usa um batch do UTKFace (idade + género) e um do AffectNet train (expressão).
Escolha da época: idade/género pelo MAE no val do UTKFace; expressão pela balanced accuracy em 10% do AffectNet train
(a validação oficial do AffectNet, o teste do UTKFace e o BTFER não são usados aqui). Como os ramos são independentes,
cada um fica com a sua melhor época.

Uso: python geral/treinar.py --mode emb ; python geral/treinar.py --mode ramo
Saída: models/geral_<mode>.onnx, geral/resultados/treino_<mode>.csv
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from expressao.comum import MODELS_DIR, ROOT                                # noqa: E402
from geral.dados import augment, batch, dev_split, load, vary_margin        # noqa: E402
from geral.rede import GeralNet, export                                     # noqa: E402
from idade_genero.modelo_torch import np2t                                  # noqa: E402

RES = os.path.join(ROOT, "geral", "resultados")


@torch.no_grad()
def evaluate(net, U, va, A, dv, dev, bs=256):
    net.eval()
    e, g, c = [], [], []
    for i in range(0, len(va), bs):
        idx = va[i:i + bs]
        o = net(batch(U["X"], idx, dev), "a")
        e.append(np.abs(np.array(net.years(o["age"]).tolist()) - U["age"][idx]))
        g.append(np.array(o["gen"].argmax(1).tolist()) == U["gender"][idx])
    for i in range(0, len(dv), bs):
        c += net(batch(A["X"], dv[i:i + bs], dev), "e")["exp"].argmax(1).tolist()
    c, y = np.array(c), A["y"][dv]
    return np.concatenate(e).mean(), np.concatenate(g).mean(), np.mean([(c[y == k] == k).mean() for k in range(8)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["emb", "ramo"], default="ramo")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-ramo", type=float, default=2e-4)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--steps", type=int, default=0, help="limite de passos por época (teste rápido)")
    a = ap.parse_args()
    torch.manual_seed(0); np.random.seed(0)
    dev = torch.device(a.device)
    U, A = load("utk"), load("aff_train")
    dvm = dev_split(A["y"])
    tr, va, atr, dv = np.where(U["split"] == 0)[0], np.where(U["split"] == 1)[0], np.where(~dvm)[0], np.where(dvm)[0]
    net = GeralNet(os.path.join(MODELS_DIR, "w600k_mbf.onnx"), a.mode).to(dev)
    heads = [p for n, p in net.named_parameters() if p.requires_grad and n.startswith("h_")]
    branch = [p for n, p in net.named_parameters() if p.requires_grad and n.startswith("br_")]
    groups = [{"params": heads, "lr": a.lr}] + ([{"params": branch, "lr": a.lr_ramo}] if branch else [])
    opt = torch.optim.AdamW(groups, weight_decay=5e-3)
    bu, ba = 64, 128
    steps = a.steps or len(atr) // ba
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, [g["lr"] for g in groups], total_steps=a.epochs * steps, pct_start=0.15)
    cnt = np.bincount(A["y"][atr], minlength=8)
    cw = np2t((cnt.mean() / cnt) ** 0.5).to(dev)
    print(f"[{a.mode}] UTK treino {len(tr)} val {len(va)} | AffectNet treino {len(atr)} dev {len(dv)} | "
          f"parâmetros treinados {sum(p.numel() for p in heads + branch) / 1e6:.2f} M | {dev}", flush=True)
    log, best = ["epoca,loss_idade,loss_genero,loss_expr,val_mae,val_acc_genero,dev_balacc_expr,segundos"], {"a": 1e9, "e": -1}
    keep = {}
    group = {"a": ("h_age", "h_gen", "br_ag"), "e": ("h_exp", "br_ex")}
    for ep in range(1, a.epochs + 1):
        net.train()
        t0, S = time.time(), torch.zeros(3, device=dev)
        pu, pa = np.random.permutation(tr), np.random.permutation(atr)
        for s in range(steps):
            iu = pu[(np.arange(bu) + s * bu) % len(pu)]
            ia = pa[s * ba:(s + 1) * ba]
            xu = batch(U["X"], iu, dev)
            xa = vary_margin(batch(A["X"], ia, dev), A["mask"][ia], xu, dev)
            oa, oe = net(augment(xu), "a"), net(augment(xa), "e")
            la = net.age_loss(oa["age"], np2t(U["age"][iu]).to(dev))
            lg = F.cross_entropy(oa["gen"], np2t(U["gender"][iu], np.int64).to(dev))
            le = F.cross_entropy(oe["exp"], np2t(A["y"][ia], np.int64).to(dev), weight=cw, label_smoothing=0.1)
            opt.zero_grad()
            (la + lg + le).backward()
            torch.nn.utils.clip_grad_norm_(heads + branch, 5.0)
            opt.step(); sched.step()
            S += torch.stack([la, lg, le]).detach()
        mae, acc, ba_e = evaluate(net, U, va, A, dv, dev)
        mark = ""
        for k, better in (("a", mae < best["a"]), ("e", ba_e > best["e"])):
            if better:
                best[k] = mae if k == "a" else ba_e
                keep.update({n: v.cpu().clone() for n, v in net.state_dict().items() if n.startswith(group[k])})
                mark += " *" + k
        L = (S / steps).tolist()
        log.append(f"{ep}," + ",".join(f"{v:.4f}" for v in L + [mae, acc, ba_e]) + f",{time.time() - t0:.0f}")
        print(f"  época {ep:2d}  loss idade {L[0]:.3f} género {L[1]:.3f} expr {L[2]:.3f} | val MAE {mae:.2f} género {acc:.3f} "
              f"| dev expr {ba_e:.3f} ({time.time() - t0:.0f}s){mark}", flush=True)
    st = net.state_dict()
    st.update(keep)
    net.load_state_dict(st)
    out = os.path.join(MODELS_DIR, f"geral_{a.mode}.onnx")
    export(net, out)
    with open(os.path.join(RES, f"treino_{a.mode}.csv"), "w") as fp:
        fp.write("\n".join(log) + "\n")
    print(f"melhor val MAE {best['a']:.2f}, melhor dev expr {best['e']:.3f} -> {out}")


if __name__ == "__main__":
    main()
