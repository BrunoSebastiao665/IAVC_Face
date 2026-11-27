"""Avaliação do HSEmotion EfficientNet-B0 e B2 (pré-treinados no AffectNet pelos autores, sem treino nosso).

  python expressao/avaliar.py      # intra (AffectNet) e depois cross (BTFER)

intra  validação oficial do AffectNet (val+test do Kaggle). Principal: sem as imagens com uma (quase) igual no
       train; ao lado, a validação completa, comparável com o que os autores reportam.
       8 classes (problema original) e 7 classes (sem Contempt: saem as imagens e a probabilidade).
cross  BTFER, 7 classes comuns, sem qualquer ajuste: a probabilidade de Contempt é ignorada e as outras
       renormalizadas. Crop principal: alinhado com o SCRFD, igual ao da app e à geometria do AffectNet (decidido
       antes de ver resultados); ao lado, a imagem tal como é distribuída e o BTFER completo sem limpeza.
Limiar de confiança da app: o menor da grelha em que a exatidão das caras rotuladas é >= 70% no 'val' (7 classes);
       a cobertura é reportada no 'test' e no BTFER. O BTFER não influencia nada.

Modelo da app: entre os que têm latência <= 40 ms por cara, o de maior balanced accuracy a 7 classes no AffectNet
sem fuga; se outro mais rápido ficar a menos de 1 p.p., o mais rápido.

Métricas: accuracy, balanced accuracy, precisão/recall/F1 macro, IC 95% (bootstrap), por classe, matriz de confusão.
Saídas em expressao/resultados/: avaliacao.md, confusao_*.png, erros_*.jpg, config_app.json
"""
import json
import os
import sys
import time

import cv2 as cv
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from expressao.comum import (AFF_NPZ, BTFER_NPZ, CLASSES, K7, MODELS, MODELS_DIR, RES, bootstrap,     # noqa: E402
                             metrics, to7)
from expressao.expression import CONFIG, ExpressionEstimator              # noqa: E402

THRS = (0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
ACC_TARGET = 0.70
LAT_MAX, MARGIN = 40.0, 0.01
AUTH = {"b0": (0.6193, 0.6494), "b2": (0.6303, 0.6629)}       # accuracy reportada pelos autores (8, 7 classes)
METR = [("Accuracy", "acc"), ("**Balanced accuracy**", "bal_acc"), ("Precisão macro", "P_macro"),
        ("Recall macro", "R_macro"), ("**F1 macro**", "F1_macro")]


def run(est, crops, bs=32):
    return np.concatenate([est.predict_crops(crops[i:i + bs]) for i in range(0, len(crops), bs)])


def latency(est, crop, n=100):
    for _ in range(10):
        est.predict_crops(crop[None])
    t = []
    for _ in range(n):
        t0 = time.perf_counter()
        est.predict_crops(crop[None])
        t.append(time.perf_counter() - t0)
    return np.median(t) * 1000, np.percentile(t, 95) * 1000


def evaluate(y, P, k):
    m = metrics(y, P.argmax(1), k)
    m["ci"] = {key: bootstrap(lambda b: metrics(y[b], P[b].argmax(1), k)[key], len(y)) for key in ("acc", "bal_acc", "F1_macro")}
    m["top2"] = (np.argsort(-P, 1)[:, :2] == y[:, None]).any(1).mean()
    m["n"] = len(y)
    return m


def table(cols):
    """cols = [(título, métricas)]"""
    md = ["| Métrica | " + " | ".join(t for t, _ in cols) + " |", "|" + "---|" * (len(cols) + 1),
          "| imagens | " + " | ".join(str(m["n"]) for _, m in cols) + " |"]
    for lab, k in METR:
        md.append(f"| {lab} | " + " | ".join(
            f"{m[k]:.3f}" + (f" [{m['ci'][k][0]:.3f}, {m['ci'][k][1]:.3f}]" if k in m["ci"] else "") for _, m in cols) + " |")
    md.append("| Top-2 accuracy | " + " | ".join(f"{m['top2']:.3f}" for _, m in cols) + " |")
    return md


def per_class(cols, k):
    n = cols[0][1]["cm"].sum(1)
    return ["| Classe | imagens | " + " | ".join(t for t, _ in cols) + " |", "|" + "---|" * (len(cols) + 2)] + [
        f"| {CLASSES[c]} | {n[c]} | " + " | ".join(f"{m['P'][c]:.3f} / {m['R'][c]:.3f} / {m['F1'][c]:.3f}" for _, m in cols) + " |"
        for c in range(k)]


def cm_md(m, k):
    cm = m["cm"]
    md = ["| real \\ prevista | " + " | ".join(CLASSES[:k]) + " | recall |", "|" + "---|" * (k + 2)]
    for i in range(k):
        md.append(f"| **{CLASSES[i]}** | " + " | ".join(f"**{v}**" if i == j else str(v) for j, v in enumerate(cm[i])) + f" | {m['R'][i]:.2f} |")
    e = sorted(((cm[i, j] / cm[i].sum(), i, j) for i in range(k) for j in range(k) if i != j), reverse=True)[:6]
    md.append("\nErros mais frequentes (% da classe real): " +
              "; ".join(f"{CLASSES[i]} → {CLASSES[j]} ({cm[i, j]}, {100 * f:.0f}%)" for f, i, j in e))
    return md


def cm_png(cols, k, path, title):
    cv.imwrite(path, cv.hconcat([cm_panel(m, k, f"{title}  {t}  bal.acc {m['bal_acc']:.3f}") for t, m in cols]))


def cm_panel(m, k, title):
    S, mx, my = 56, 92, 66
    cm = m["cm"]
    R = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    im = np.full((my + k * S + 10, mx + k * S + 10, 3), 255, np.uint8)
    cv.putText(im, title, (6, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)
    cv.putText(im, "linhas = real, colunas = prevista, % por linha", (6, 38), cv.FONT_HERSHEY_SIMPLEX, 0.4, (90, 90, 90), 1, cv.LINE_AA)
    for i in range(k):
        cv.putText(im, CLASSES[i], (4, my + i * S + 32), cv.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv.LINE_AA)
        cv.putText(im, CLASSES[i][:5], (mx + i * S + 6, my - 6), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv.LINE_AA)
        for j in range(k):
            v = R[i, j]
            x, yy = mx + j * S, my + i * S
            cv.rectangle(im, (x, yy), (x + S - 2, yy + S - 2), tuple(int(255 - v * (255 - b)) for b in (140, 80, 20)), -1)
            cv.putText(im, f"{100 * v:.0f}", (x + 16, yy + 33), cv.FONT_HERSHEY_SIMPLEX, 0.48,
                       (255, 255, 255) if v > 0.5 else (0, 0, 0), 1, cv.LINE_AA)
    return im


def errors_jpg(X, y, P, path, n=24):
    """Os n erros mais confiantes."""
    p = P.argmax(1)
    w = np.where(p != y)[0]
    w = w[np.argsort(-P[w].max(1))][:n]
    tiles = []
    for i in w:
        b = np.full((36, 140, 3), 255, np.uint8)
        cv.putText(b, f"real {CLASSES[y[i]]}", (3, 14), cv.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
        cv.putText(b, f"prev {CLASSES[p[i]]} {P[i].max():.2f}", (3, 30), cv.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 200), 1)
        tiles.append(cv.vconcat([cv.resize(X[i], (140, 140)), b]))
    while len(tiles) % 8:
        tiles.append(np.full_like(tiles[0], 255))
    cv.imwrite(path, cv.vconcat([cv.hconcat(tiles[k:k + 8]) for k in range(0, len(tiles), 8)]))


def coverage(sets, thr):
    """sets = [(título, y, P7)]"""
    md = ["| Limiar | " + " | ".join(f"{t}: cobertura / exatidão / bal. acc." for t, _, _ in sets) + " |", "|" + "---|" * (len(sets) + 1)]
    for t in THRS:
        cells = []
        for _, y, P in sets:
            k = P.max(1) >= t
            m = metrics(y[k], P[k].argmax(1), K7)
            cells.append(f"{k.mean():.2f} / {m['acc']:.3f} / {m['bal_acc']:.3f}")
        md.append(f"| {t:.1f}{' ← app' if t == thr else ''} | " + " | ".join(cells) + " |")
    return md


def paired(y, Pa, Pb, k):
    """Diferença b - a da balanced accuracy e do F1 macro, IC 95% por bootstrap emparelhado."""
    a, b = Pa.argmax(1), Pb.argmax(1)
    out = []
    for key in ("bal_acc", "F1_macro"):
        f = lambda i: metrics(y[i], b[i], k)[key] - metrics(y[i], a[i], k)[key]
        lo, hi = bootstrap(f, len(y))
        out.append(f"{f(np.arange(len(y))):+.3f} [{lo:+.3f}, {hi:+.3f}]")
    return out


def main():
    ns = list(MODELS)
    NICE = {n: MODELS[n][1].replace("EfficientNet-", "") for n in ns}
    ests = {n: ExpressionEstimator(n, n_classes=8, threshold=0.0) for n in ns}
    # ------------------------------------------------------------------ intra
    d = np.load(AFF_NPZ)
    X, y, part, twin = d["X"], d["y"].astype(int), d["part"], d["twin"]
    s7, ok = y < K7, ~twin
    P = {n: run(ests[n], X) for n in ns}
    P7 = {n: to7(P[n]) for n in ns}
    I8 = {n: evaluate(y[ok], P[n][ok], 8) for n in ns}
    I8all = {n: evaluate(y, P[n], 8) for n in ns}
    I7 = {n: evaluate(y[ok & s7], P7[n][ok & s7], 7) for n in ns}
    I7all = {n: evaluate(y[s7], P7[n][s7], 7) for n in ns}
    lat = {n: latency(ests[n], X[0]) for n in ns}
    v, tst = (part == "val") & s7 & ok, (part == "test") & s7 & ok
    thr = {}
    for n in ns:
        thr[n] = THRS[-1]
        for t in THRS:
            k = P7[n][v].max(1) >= t
            if k.any() and (P7[n][v][k].argmax(1) == y[v][k]).mean() >= ACC_TARGET:
                thr[n] = t
                break
    # escolha do modelo da app: só AffectNet e latência
    fast = [n for n in ns if lat[n][0] <= LAT_MAX] or ns
    best = max(fast, key=lambda n: I7[n]["bal_acc"])
    cand = [n for n in fast if I7[n]["bal_acc"] >= I7[best]["bal_acc"] - MARGIN]
    app = min(cand, key=lambda n: lat[n][0])
    rule = (f"entre os modelos com latência <= {LAT_MAX:.0f} ms, o de maior balanced accuracy a 7 classes no AffectNet sem fuga; "
            f"se outro mais rápido ficar a menos de {100 * MARGIN:.0f} p.p., o mais rápido")
    with open(CONFIG, "w") as fp:
        json.dump(dict(modelo=app, limiares=thr, regra_modelo=rule, regra_limiar=f"menor limiar com exatidão >= {ACC_TARGET} "
                       "no val do AffectNet (7 classes)"), fp, indent=1, ensure_ascii=False)
    print({n: (round(I7[n]["bal_acc"], 4), round(lat[n][0], 1)) for n in ns}, "app:", app, flush=True)

    # ------------------------------------------------------------------ cross
    b = np.load(BTFER_NPZ)
    yb, keep, A, S = b["y"].astype(int), b["keep"], b["aligned"], b["asis"]
    yk = yb[keep]
    Pa = {n: to7(run(ests[n], A)) for n in ns}
    Ps = {n: to7(run(ests[n], S)) for n in ns}
    C = {n: evaluate(yk, Pa[n][keep], 7) for n in ns}
    Cs = {n: evaluate(yk, Ps[n][keep], 7) for n in ns}
    Cf = {n: evaluate(yb, Pa[n], 7) for n in ns}

    cols = lambda M: [(NICE[n], M[n]) for n in ns]
    md = ["# Expressão facial — HSEmotion B0 e B2 (pré-treinados)\n",
          "- Os rótulos são **estimativas da configuração facial visível**, não evidência da emoção sentida.",
          "- Modelos treinados pelos autores no train oficial do AffectNet: B0 = `enet_b0_8_va_mtl`, B2 = `enet_b2_8`. **Sem treino nosso.**",
          "- Métricas com [IC 95%] por bootstrap (1 000 reamostragens das imagens).\n",
          "| | " + " | ".join(NICE[n] for n in ns) + " |", "|" + "---|" * (len(ns) + 1),
          "| Entrada | " + " | ".join(f"{ests[n].size}×{ests[n].size}" for n in ns) + " |",
          "| Ficheiro (MB) | " + " | ".join(f"{os.path.getsize(os.path.join(MODELS_DIR, MODELS[n][0])) / 1e6:.1f}" for n in ns) + " |",
          "| ms por cara, CPU, com resize (mediana / p95) | " + " | ".join(f"{lat[n][0]:.1f} / {lat[n][1]:.1f}" for n in ns) + " |",
          "\n## 1. Intra-dataset — validação oficial do AffectNet\n",
          f"Principal: sem as {int(twin.sum())} imagens com uma (quase) igual no train ('sem fuga').\n",
          "### 8 classes, sem fuga (principal)\n", *table(cols(I8)),
          "\n### 7 classes (sem Contempt), sem fuga\n", *table(cols(I7)),
          "\n### Validação oficial completa (para comparar com os autores)\n",
          "| Accuracy | " + " | ".join(NICE[n] for n in ns) + " |", "|" + "---|" * (len(ns) + 1),
          "| 8 classes: nós / autores | " + " | ".join(f"{I8all[n]['acc']:.4f} / {AUTH[n][0]:.4f}" for n in ns) + " |",
          "| 7 classes: nós / autores | " + " | ".join(f"{I7all[n]['acc']:.4f} / {AUTH[n][1]:.4f}" for n in ns) + " |",
          "\n### Por classe (8 classes, sem fuga): precisão / recall / F1\n", *per_class([(NICE[n], I8[n]) for n in ns], 8),
          "\n### Matrizes de confusão (8 classes, sem fuga) — figura `confusao_affectnet.png`"]
    for n in ns:
        md += [f"\n**{NICE[n]}**\n", *cm_md(I8[n], 8)]
    md += ["\n## 2. Cross-dataset — BTFER (7 classes comuns, sem ajuste)\n",
           "### Limpo, alinhado (principal)\n", *table(cols(C)),
           "\n### Sensibilidade (balanced accuracy / F1 macro)\n",
           "| | " + " | ".join(NICE[n] for n in ns) + " |", "|" + "---|" * (len(ns) + 1)]
    for lab, M in (("limpo, alinhado (principal)", C), ("limpo, imagem tal como distribuída", Cs), ("completo (2100, com repetições), alinhado", Cf)):
        md.append(f"| {lab} | " + " | ".join(f"{M[n]['bal_acc']:.3f} / {M[n]['F1_macro']:.3f}" for n in ns) + " |")
    md += ["\n### Por classe: precisão / recall / F1\n", *per_class([(NICE[n], C[n]) for n in ns], 7),
           "\n### Matrizes de confusão — figura `confusao_btfer.png`"]
    for n in ns:
        md += [f"\n**{NICE[n]}**\n", *cm_md(C[n], 7)]
    md += ["\n## 3. Generalização: AffectNet (7 classes, sem fuga) → BTFER\n",
           "| | " + " | ".join(NICE[n] for n in ns) + " |", "|" + "---|" * (len(ns) + 1)]
    for lab, k in METR:
        md.append(f"| {lab} | " + " | ".join(f"{I7[n][k]:.3f} → {C[n][k]:.3f} ({C[n][k] - I7[n][k]:+.3f})" for n in ns) + " |")
    for c in range(7):
        md.append(f"| Recall {CLASSES[c]} | " + " | ".join(f"{I7[n]['R'][c]:.3f} → {C[n]['R'][c]:.3f}" for n in ns) + " |")
    a, bb = ns
    md += [f"\n## 4. {NICE[bb]} − {NICE[a]}: a diferença é real? (bootstrap emparelhado, IC 95%)\n",
           "| Dados | Balanced accuracy | F1 macro |", "|---|---|---|"]
    for lab, yy, pa, pb, k in (("AffectNet, 8 classes, sem fuga", y[ok], P[a][ok], P[bb][ok], 8),
                               ("AffectNet, 7 classes, sem fuga", y[ok & s7], P7[a][ok & s7], P7[bb][ok & s7], 7),
                               ("BTFER, 7 classes", yk, Pa[a][keep], Pa[bb][keep], 7)):
        md.append(f"| {lab} | " + " | ".join(paired(yy, pa, pb, k)) + " |")
    md += ["\n## 5. Confiança e rótulo \"incerto\" (7 classes)\n",
           f"Limiar de cada modelo = o menor com exatidão ≥ {ACC_TARGET:.0%} no `val` do AffectNet ({int(v.sum())} imagens). "
           "A coluna AffectNet é só o `test`, que não foi usado na escolha."]
    for n in ns:
        md += [f"\n**{NICE[n]}**\n", *coverage([("AffectNet test", y[tst], P7[n][tst]), ("BTFER", yk, Pa[n][keep])], thr[n])]
    md += ["\n## 6. Modelo da app (escolhido só com AffectNet e latência)\n", f"Regra: {rule}.\n",
           f"**Escolhido: {MODELS[app][1]}** (limiar {thr[app]}). Erros mais confiantes deste modelo: `erros_affectnet.jpg`, `erros_btfer.jpg`"]
    cm_png([(NICE[n], I8[n]) for n in ns], 8, os.path.join(RES, "confusao_affectnet.png"), "AffectNet")
    cm_png([(NICE[n], C[n]) for n in ns], 7, os.path.join(RES, "confusao_btfer.png"), "BTFER")
    errors_jpg(X[ok & s7], y[ok & s7], P7[app][ok & s7], os.path.join(RES, "erros_affectnet.jpg"))
    errors_jpg(A[keep], yk, Pa[app][keep], os.path.join(RES, "erros_btfer.jpg"))
    with open(os.path.join(RES, "avaliacao.md"), "w") as fp:
        fp.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
