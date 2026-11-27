"""Multitarefa vs especialistas de idade/género nas MESMAS 4 682 caras de teste do UTKFace (fase 3):
bootstrap emparelhado da diferença, MAE e viés por faixa etária, MAE por etnia (bruto e ajustado à idade).

Cada modelo usa o seu próprio crop: multitarefa = 112x112 ArcFace (data/geral/utk.npz); FairFace = 224x224 alinhado a
partir da imagem original; MobileNet FT = 96x96 da caixa. As caras são as mesmas (mesma ordem, 0 descartadas).

Uso: python multitarefa/comparar.py -> multitarefa/resultados/comparacao_idade_genero.md
"""
import os
import sys

import cv2 as cv
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from expressao.comum import ROOT, metrics                                       # noqa: E402
from geral.dados import load                                                    # noqa: E402
from geral.modelo import GeralEstimator                                         # noqa: E402
from idade_genero.age_gender import AGE_GROUPS, AgeGenderEstimator, age_group, group_label   # noqa: E402

RES = os.path.join(ROOT, "multitarefa", "resultados")
RACES = ["White", "Black", "Asian", "Indian", "Others"]


def specialist(name, d, te):
    est = AgeGenderEstimator(name)
    g, a = [], []
    for k in range(0, len(te), 64):
        crops = []
        for i in te[k:k + 64]:
            img = cv.imread(os.path.join(ROOT, "data", "UTKFace", str(d["names"][i])))
            crops.append(est.make_crop(img, {"box": d["box"][i], "landmarks": d["kps"][i]}))
        gp, ag, _ = est.predict_crops(np.stack(crops))
        g.append(gp); a.append(ag)
    return np.concatenate(g).argmax(1), np.concatenate(a)


def main():
    U = load("utk")
    d = np.load(os.path.join(ROOT, "data", "UTKFace", "utkface_96.npz"))
    assert (d["names"] == U["names"]).all()
    te = np.where(U["split"] == 2)[0]
    ya, yg, race = U["age"][te].astype(float), U["gender"][te], U["race"][te]
    # o .npz já está na convenção dos modelos: 0 = F, 1 = M (idade_genero/preparar_utkface.py)
    _, G, A, _ = GeralEstimator("multitarefa").run(U["X"][te])
    P = {"Multitarefa": (G.argmax(1), A)}
    for nm, key in (("FairFace", "fairface"), ("MobileNet FT ordinal", "mobilenet_ft")):
        P[nm] = specialist(key, d, te)
    grp = np.array([age_group(v) for v in ya])
    w = np.bincount(grp, minlength=len(AGE_GROUPS)) / len(grp)
    rng = np.random.RandomState(0)
    B = [rng.randint(0, len(te), len(te)) for _ in range(1000)]
    md = ["# Multitarefa vs especialistas — idade e género (UTKFace teste, 4 682 caras, as mesmas para todos)\n",
          "| Modelo | MAE | RMSE | viés | CS(5) | F1 género |", "|---|---|---|---|---|---|"]
    for n, (g, a) in P.items():
        e = a - ya
        md.append(f"| {n} | {np.abs(e).mean():.2f} | {np.sqrt((e ** 2).mean()):.2f} | {e.mean():+.2f} | {(np.abs(e) <= 5).mean():.3f} | "
                  f"{metrics(yg, g, 2)['F1_macro']:.3f} |")
    md += ["\n## Bootstrap emparelhado (1 000 reamostragens das mesmas caras)\n", "| Diferença | valor | IC 95% |", "|---|---|---|"]
    gm, am = P["Multitarefa"]
    for n in ("FairFace", "MobileNet FT ordinal"):
        gs, as_ = P[n]
        dm = lambda b: np.abs(am[b] - ya[b]).mean() - np.abs(as_[b] - ya[b]).mean()
        df = lambda b: metrics(yg[b], gm[b], 2)["F1_macro"] - metrics(yg[b], gs[b], 2)["F1_macro"]
        v = [dm(b) for b in B]; f = [df(b) for b in B]
        md.append(f"| MAE multitarefa − {n} (anos) | {dm(np.arange(len(te))):+.2f} | [{np.percentile(v, 2.5):+.2f}, {np.percentile(v, 97.5):+.2f}] |")
        md.append(f"| F1 género multitarefa − {n} | {df(np.arange(len(te))):+.3f} | [{np.percentile(f, 2.5):+.3f}, {np.percentile(f, 97.5):+.3f}] |")
    md += ["\n## MAE / viés por faixa etária\n", "| Faixa | caras | " + " | ".join(P) + " |", "|---|---|" + "---|" * len(P)]
    for k in range(len(AGE_GROUPS)):
        m = grp == k
        md.append(f"| {group_label(k)} | {m.sum()} | " + " | ".join(
            f"{np.abs(a[m] - ya[m]).mean():.2f} / {(a[m] - ya[m]).mean():+.2f}" for _, a in P.values()) + " |")
    md += ["\n## Por etnia anotada: MAE bruto / MAE ajustado à idade / F1 género\n",
           "(ajustado = MAE de cada faixa dentro da etnia, pesado pela distribuição global das faixas; só faixas com ≥ 5 caras, como na fase 3)\n",
           "| Etnia | caras | " + " | ".join(P) + " |", "|---|---|" + "---|" * len(P)]
    for r, rn in enumerate(RACES):
        m = race == r
        cells = []
        for g, a in P.values():
            err = np.abs(a - ya)
            adj = sum(w[k] * err[m & (grp == k)].mean() for k in range(len(AGE_GROUPS)) if (m & (grp == k)).sum() >= 5) / \
                sum(w[k] for k in range(len(AGE_GROUPS)) if (m & (grp == k)).sum() >= 5)
            cells.append(f"{err[m].mean():.2f} / {adj:.2f} / {metrics(yg[m], g[m], 2)['F1_macro']:.3f}")
        md.append(f"| {rn} | {m.sum()} | " + " | ".join(cells) + " |")
    os.makedirs(RES, exist_ok=True)
    open(os.path.join(RES, "comparacao_idade_genero.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
