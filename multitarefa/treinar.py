"""Treino por blocos da rede multitarefa: cada ramo (género, idade, expressão) tem o seu otimizador, a sua loss,
o seu escalonamento do lr e a sua melhor época. O tronco e a identidade ficam congelados.

    género    CE                                   UTKFace train   -> época escolhida pela balanced acc. no val do UTKFace
    idade     DLDL (gaussiana σ=2) + L1/10         UTKFace train   -> época escolhida pelo MAE no val do UTKFace
    expressão CE com pesos de classe + smoothing   AffectNet train -> época escolhida pela balanced acc. no dev (10% do train)

Os ramos não partilham parâmetros nem gradientes. Quando se treinam vários na mesma execução, só se poupa a passagem
pelo tronco congelado, que é igual para todos; o resultado é o mesmo que treinar cada um sozinho.
O teste do UTKFace, a validação oficial do AffectNet e o BTFER NÃO são usados aqui (ficam para avaliar.py).

Uso:  python multitarefa/treinar.py                         os 3 ramos
      python multitarefa/treinar.py --ramos expressao       só um bloco (os outros vêm de resultados/pesos.pt)
Saída: multitarefa/resultados/pesos.pt (melhor época de cada ramo), treino.csv, models/multitarefa.onnx
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
from idade_genero.modelo_torch import np2t                                  # noqa: E402
from multitarefa.rede import TASKS, MultiNet, export                        # noqa: E402

RES = os.path.join(ROOT, "multitarefa", "resultados")
PESOS = os.path.join(RES, "pesos.pt")


def bal_acc(y, p, k):
    return float(np.mean([(p[y == c] == c).mean() for c in range(k) if (y == c).any()]))


@torch.no_grad()
def evaluate(net, U, va, A, dv, dev, bs=256):
    net.eval()
    g, a, e = [], [], []
    for i in range(0, len(va), bs):
        o = net(batch(U["X"], va[i:i + bs], dev), ("genero", "idade"))
        g += o["genero"].argmax(1).tolist()
        a += net.years(o["idade"]).tolist()
    for i in range(0, len(dv), bs):
        e += net(batch(A["X"], dv[i:i + bs], dev), ("expressao",))["expressao"].argmax(1).tolist()
    return {"genero": bal_acc(U["gender"][va], np.array(g), 2),
            "idade": float(np.abs(np.array(a) - U["age"][va]).mean()),
            "expressao": bal_acc(A["y"][dv], np.array(e), 8)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ramos", nargs="+", choices=list(TASKS), default=list(TASKS))
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3, help="cabeças")
    ap.add_argument("--lr-ramo", type=float, default=2e-4, help="ramos convolucionais (afinação: lr 5x menor)")
    ap.add_argument("--steps", type=int, default=0, help="limite de passos por época (teste rápido)")
    a = ap.parse_args()
    torch.manual_seed(0); np.random.seed(0)
    torch.set_num_threads(os.cpu_count())
    dev = torch.device("cpu")
    os.makedirs(RES, exist_ok=True)

    U, A = load("utk"), load("aff_train")
    dvm = dev_split(A["y"])
    tr, va, atr, dv = np.where(U["split"] == 0)[0], np.where(U["split"] == 1)[0], np.where(~dvm)[0], np.where(dvm)[0]
    net = MultiNet(os.path.join(MODELS_DIR, "w600k_mbf.onnx")).to(dev)
    if os.path.exists(PESOS):                       # blocos já treinados noutra execução
        net.load_state_dict(torch.load(PESOS), strict=False)

    bu, ba = 64, 128
    steps = a.steps or len(atr) // ba
    opt, sched = {}, {}
    for t in a.ramos:                               # um bloco de treino por ramo
        br, hd = net.task_params(t)
        opt[t] = torch.optim.AdamW([{"params": hd, "lr": a.lr}, {"params": br, "lr": a.lr_ramo}], weight_decay=5e-3)
        sched[t] = torch.optim.lr_scheduler.OneCycleLR(opt[t], [a.lr, a.lr_ramo], total_steps=a.epochs * steps, pct_start=0.15)
    cnt = np.bincount(A["y"][atr], minlength=8)
    cw = np2t((cnt.mean() / cnt) ** 0.5).to(dev)
    print(f"ramos {a.ramos} | UTK treino {len(tr)} val {len(va)} | AffectNet treino {len(atr)} dev {len(dv)} | "
          f"{steps} passos/época x {a.epochs} épocas", flush=True)

    best = {"genero": -1.0, "idade": 1e9, "expressao": -1.0}
    better = {"genero": lambda v, b: v > b, "idade": lambda v, b: v < b, "expressao": lambda v, b: v > b}
    keep = {}
    log = ["epoca,loss_genero,loss_idade,loss_expressao,val_balacc_genero,val_mae_idade,dev_balacc_expressao,segundos"]
    need_u = {"genero", "idade"} & set(a.ramos)
    for ep in range(1, a.epochs + 1):
        net.train()
        t0, S = time.time(), {t: 0.0 for t in TASKS}
        pu, pa = np.random.permutation(tr), np.random.permutation(atr)
        for s in range(steps):
            iu = pu[(np.arange(bu) + s * bu) % len(pu)]
            xu = batch(U["X"], iu, dev)
            losses = {}
            if need_u:
                o = net(None, tuple(need_u), f=net.features(augment(xu)))
                if "genero" in o:
                    losses["genero"] = F.cross_entropy(o["genero"], np2t(U["gender"][iu], np.int64).to(dev))
                if "idade" in o:
                    losses["idade"] = net.age_loss(o["idade"], np2t(U["age"][iu]).to(dev))
            if "expressao" in a.ramos:
                ia = pa[s * ba:(s + 1) * ba]
                xa = vary_margin(batch(A["X"], ia, dev), A["mask"][ia], xu, dev)
                o = net(None, ("expressao",), f=net.features(augment(xa)))
                losses["expressao"] = F.cross_entropy(o["expressao"], np2t(A["y"][ia], np.int64).to(dev),
                                                      weight=cw, label_smoothing=0.1)
            for t in losses:
                opt[t].zero_grad()
            sum(losses.values()).backward()          # gradientes independentes: cada loss só toca no seu ramo
            for t, l in losses.items():
                torch.nn.utils.clip_grad_norm_(sum(net.task_params(t), []), 5.0)
                opt[t].step(); sched[t].step()
                S[t] += float(l)
            if s % 50 == 0:
                print(f"    passo {s}/{steps} ({(time.time() - t0) / (s + 1):.1f} s/passo)", flush=True)
        m = evaluate(net, U, va, A, dv, dev)
        mark = ""
        for t in a.ramos:
            if better[t](m[t], best[t]):
                best[t] = m[t]
                keep.update({k: v.clone() for k, v in net.state_dict().items() if k.startswith((f"br.{t}.", f"head.{t}."))})
                mark += f" *{t}"
        L = [S[t] / steps for t in TASKS]
        log.append(f"{ep}," + ",".join(f"{v:.4f}" for v in L + [m['genero'], m['idade'], m['expressao']]) + f",{time.time() - t0:.0f}")
        with open(os.path.join(RES, "treino.csv"), "w") as fp:
            fp.write("\n".join(log) + "\n")
        old = torch.load(PESOS) if os.path.exists(PESOS) else {}
        old.update(keep)
        torch.save(old, PESOS)                      # guarda já: se parar a meio, não se perde a melhor época
        print(f"  época {ep}/{a.epochs}  loss g {L[0]:.3f} i {L[1]:.3f} e {L[2]:.3f} | val género {m['genero']:.3f} "
              f"MAE {m['idade']:.2f} | dev expr {m['expressao']:.3f} ({time.time() - t0:.0f}s){mark}", flush=True)

    net.load_state_dict(torch.load(PESOS), strict=False)
    out = os.path.join(MODELS_DIR, "multitarefa.onnx")
    export(net, out)
    print(f"melhor: {best} -> {out}", flush=True)


if __name__ == "__main__":
    main()
