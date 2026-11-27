"""Avaliação de idade aparente e género no split de TESTE do UTKFace.

Compara, nas MESMAS caras de teste:
  - 2 modelos só pré-treinados (sem ver o UTKFace):
      MobileNet genderage original (InsightFace)  e  FairFace ResNet-34 (models/fairface_res34.onnx)
  - os fine-tuned no UTKFace (models/genderage_ft_*.onnx) que existirem.
Cada modelo recebe o crop que espera (metadados do ONNX): caixa 96x96 já guardada no .npz, ou crop
alinhado pelos landmarks 224x224 feito a partir das imagens originais (FairFace).

Métricas
  Idade:   MAE, RMSE, CS(5) (% com erro <= 5 anos), MAE por faixa etária,
           exatidão da faixa etária (exata e ±1 faixa), matriz de confusão das faixas (melhor modelo)
  Género:  precisão, recall e F1 POR CLASSE, F1 macro, matriz de confusão;
           efeito do limiar de confiança ("incerto"): cobertura vs exatidão
  Demografia: MAE e F1 macro do género por etnia; MAE por etnia AJUSTADO À IDADE (média do MAE por
           faixa, pesada pela distribuição global das faixas, para tirar o efeito de cada etnia ter
           idades diferentes); MAE por género
  Viés e fração de género "incerto" por faixa etária; bootstrap emparelhado FairFace vs melhor MobileNet FT
  Tempo:   ms por cara (batch 1, CPU)

Saídas em idade_genero/resultados/: idade_genero.md, idade_dispersao.png, exemplos_teste.jpg

Uso:
    python idade_genero/avaliar.py
    python idade_genero/avaliar.py --gender-threshold 0.8
"""
import argparse
import glob
import os
import sys
import time

import cv2 as cv
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from idade_genero.age_gender import (AGE_GROUPS, GENDER_LABELS, MODELS_DIR, ROOT,   # noqa: E402
                                     AgeGenderEstimator, age_group, group_label)

DATA = os.path.join(ROOT, "data", "UTKFace", "utkface_96.npz")
RES = os.path.join(ROOT, "idade_genero", "resultados")
RACES = ["White", "Black", "Asian", "Indian", "Others"]
BOOT = 1000
THRESHOLDS = (0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.95)
NICE = {"genderage": "MobileNet pré-treinada", "fairface_res34": "FairFace ResNet-34 pré-treinado",
        "genderage_ft_reg": "MobileNet FT regressão", "genderage_ft_cls": "MobileNet FT classes",
        "genderage_ft_dldl": "MobileNet FT DLDL", "genderage_ft_ordinal": "MobileNet FT ordinal"}
SHORT = {"genderage": "MN pré", "fairface_res34": "FairFace", "genderage_ft_reg": "MN-FT reg",
         "genderage_ft_cls": "MN-FT cls", "genderage_ft_dldl": "MN-FT dldl", "genderage_ft_ordinal": "MN-FT ord"}
ORDER = ["genderage", "fairface_res34"] + [f"genderage_ft_{v}" for v in ("reg", "cls", "dldl", "ordinal")]


def prf(y, p, k=2):
    """-> (P, R, F1 por classe, matriz de confusão [verdadeiro, previsto])."""
    cm = np.zeros((k, k), int)
    np.add.at(cm, (y, p), 1)
    tp = np.diag(cm).astype(float)
    P = tp / np.maximum(cm.sum(0), 1)
    R = tp / np.maximum(cm.sum(1), 1)
    F1 = 2 * P * R / np.maximum(P + R, 1e-9)
    return P, R, F1, cm


def timing(est, crop, n=200):
    est.predict_crops(crop[None])
    t0 = time.perf_counter()
    for _ in range(n):
        est.predict_crops(crop[None])
    return (time.perf_counter() - t0) / n * 1000


def scatter(results, ages, path):
    """Idade prevista vs verdadeira (um painel por modelo), desenhado com OpenCV."""
    S, M = 300, 40
    panels = []
    for name, r in results.items():
        img = np.full((S + 2 * M, S + 2 * M, 3), 255, np.uint8)
        to = lambda a, b: (int(M + a / 100 * S), int(M + S - min(b, 100) / 100 * S))
        for v in range(0, 101, 20):
            cv.line(img, to(v, 0), to(v, 100), (230, 230, 230), 1)
            cv.line(img, to(0, v), to(100, v), (230, 230, 230), 1)
            cv.putText(img, str(v), (to(v, 0)[0] - 8, S + M + 15), cv.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
            cv.putText(img, str(v), (M - 25, to(0, v)[1] + 4), cv.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
        for a, p in zip(ages, r["age"]):
            if a <= 100:
                cv.circle(img, to(a, p), 1, (200, 120, 40), -1)
        cv.line(img, to(0, 0), to(100, 100), (0, 0, 220), 1, cv.LINE_AA)
        cv.rectangle(img, to(0, 100), to(100, 0), (0, 0, 0), 1)
        cv.putText(img, f"{NICE.get(name, name)}  MAE={r['mae']:.2f}", (M - 30, 22),
                   cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv.LINE_AA)
        cv.putText(img, "real", (S // 2 + M - 10, S + 2 * M - 5), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        cv.putText(img, "prevista", (2, M - 8), cv.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
        panels.append(img)
    while len(panels) % 3:
        panels.append(np.full_like(panels[0], 255))
    cv.imwrite(path, cv.vconcat([cv.hconcat(panels[k:k + 3]) for k in range(0, len(panels), 3)]))


def examples(X, ages, genders, models, path, n=24, seed=1):
    """models: [(rótulo curto, resultados)] — uma linha de previsão por modelo."""
    idx = np.random.RandomState(seed).choice(len(X), n, replace=False)
    colors = [(0, 0, 180), (160, 60, 0), (0, 120, 0)]
    tiles = []
    for i in idx:
        t = cv.resize(X[i], (140, 140))
        bar = np.full((18 + 16 * len(models), 140, 3), 255, np.uint8)
        cv.putText(bar, f"real {ages[i]} {GENDER_LABELS[genders[i]]}", (3, 14), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        for k, (lab, r) in enumerate(models):
            g = "?" if r["gconf"][i] < r["thr"] else GENDER_LABELS[r["g"][i]]
            cv.putText(bar, f"{lab} ~{r['age'][i]:.0f} {g}", (3, 30 + 16 * k), cv.FONT_HERSHEY_SIMPLEX, 0.38,
                       colors[k % 3], 1)
        tiles.append(cv.vconcat([t, bar]))
    rows = [cv.hconcat(tiles[k:k + 8]) for k in range(0, n, 8)]
    cv.imwrite(path, cv.vconcat(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gender-threshold", type=float, default=0.75,
                    help="prob. mínima para mostrar o género (abaixo = incerto)")
    a = ap.parse_args()
    os.makedirs(RES, exist_ok=True)

    d = np.load(DATA)
    m = d["split"] == 2
    X, ages, genders, races = d["X"][m], d["age"][m].astype(int), d["gender"][m].astype(int), d["race"][m]
    names_img, boxes, kps = d["names"][m], d["box"][m], d["kps"][m] if "kps" in d else None
    groups = np.array([age_group(x) for x in ages])
    print(f"teste: {len(X)} caras")

    def crops_for(est):
        """Crops no formato do modelo: os 96x96 do .npz, ou refeitos a partir das imagens originais."""
        if est.crop == "box" and est.size == X.shape[1] and abs(est.box_factor - 1.5) < 1e-9:
            return X
        if est.crop == "align" and (kps is None or np.isnan(kps).any()):
            raise SystemExit("faltam landmarks no .npz: python idade_genero/preparar_utkface.py --add-landmarks val test")
        print(f"  a fazer crops '{est.crop}' {est.size}x{est.size} a partir das imagens originais...")
        out = np.zeros((len(X), est.size, est.size, 3), np.uint8)
        for i in range(len(X)):
            img = cv.imread(os.path.join(ROOT, "data", "UTKFace", str(names_img[i])))
            out[i] = est.make_crop(img, {"box": boxes[i], "landmarks": kps[i] if kps is not None else []})
        return out

    found = [os.path.join(MODELS_DIR, f"{n}.onnx") for n in ("genderage", "fairface_res34")]
    found = [p for p in found if os.path.exists(p)] + glob.glob(os.path.join(MODELS_DIR, "genderage_ft_*.onnx"))
    rank = lambda p: (ORDER.index(os.path.basename(p)[:-5]) if os.path.basename(p)[:-5] in ORDER else 99, p)
    results = {}
    for p in sorted(found, key=rank):
        est = AgeGenderEstimator(p, gender_threshold=a.gender_threshold)
        C = crops_for(est)
        probs, pa, gps = [], [], []
        bs = 512 if est.size <= 112 else 32
        for i in range(0, len(C), bs):
            pr, ag, gp = est.predict_crops(C[i:i + bs])
            probs.append(pr); pa.append(ag); gps.append(gp)
        probs, pa = np.concatenate(probs), np.clip(np.concatenate(pa), 0, 116)
        g, gconf = probs.argmax(1), probs.max(1)
        err = np.abs(pa - ages)
        if gps[0] is not None:                       # modelo de faixas (FairFace): faixa = argmax nativo
            pg = np.concatenate(gps).argmax(1)
        else:
            pg = np.array([age_group(int(round(x))) for x in pa])
        P, R, F1, cm = prf(genders, g)
        results[est.name] = dict(
            age=pa, g=g, gconf=gconf, thr=a.gender_threshold, pg=pg, P=P, R=R, F1=F1, cm=cm,
            mae=err.mean(), bias=(pa - ages).mean(), rmse=np.sqrt((err ** 2).mean()), cs5=(err <= 5).mean(),
            grp_acc=(pg == groups).mean(), grp_acc1=(np.abs(pg - groups) <= 1).mean(),
            ms=timing(est, C[0]), crop=f"{est.crop} {est.size}×{est.size}")
        print(f"  {est.name:<24} MAE {err.mean():.2f}  F1 macro género {F1.mean():.3f}  ({results[est.name]['ms']:.1f} ms/cara)")

    names = list(results)
    best = min(names, key=lambda n: results[n]["mae"])
    ft = [n for n in names if n.startswith("genderage_ft_")]
    best_ft = min(ft, key=lambda n: results[n]["mae"]) if ft else None
    detail = list(dict.fromkeys(n for n in (best, "fairface_res34", best_ft) if n in results))
    H = lambda first: f"| {first} | " + " | ".join(NICE.get(n, n) for n in names) + " |"
    S = lambda k=1: "|" + "---|" * (len(names) + k)
    row = lambda label, f: f"| {label} | " + " | ".join(f(results[n]) for n in names) + " |"

    md = ["# Idade aparente e género — UTKFace (split de teste)\n",
          f"- {len(X)} caras de teste (split fixo 70/10/20, seed 0), caras detetadas pelo SCRFD",
          "- **Pré-treinados** (nunca viram o UTKFace): MobileNet genderage (InsightFace) e FairFace ResNet-34. "
          "**Fine-tuned**: a MobileNet ajustada no treino do UTKFace com 4 formulações da idade",
          "- **Idade** = estimativa de idade *aparente*; **género** = classificação visual nas 2 categorias anotadas "
          "no UTKFace (female/male), não identidade de género",
          f"- Limiar de confiança do género para a aplicação: **{a.gender_threshold}** (abaixo → \"incerto\")\n",
          "## 1. Idade — resumo\n", H("Métrica"), S(),
          row("**MAE** (anos)", lambda r: f"**{r['mae']:.2f}**"),
          row("RMSE (anos)", lambda r: f"{r['rmse']:.2f}"),
          row("Erro médio com sinal (viés, anos)", lambda r: f"{r['bias']:+.2f}"),
          row("CS(5) — erro ≤ 5 anos", lambda r: f"{r['cs5']:.3f}"),
          row("Faixa etária exata", lambda r: f"{r['grp_acc']:.3f}"),
          row("Faixa etária ±1", lambda r: f"{r['grp_acc1']:.3f}"),
          row("ms / cara (CPU, batch 1)", lambda r: f"{r['ms']:.2f}"),
          row("Entrada (crop)", lambda r: r["crop"]),
          "\nDispersão prevista vs real: `idade_dispersao.png`\n",
          "## 2. Idade — MAE por faixa etária\n",
          "| Faixa | caras | " + " | ".join(NICE.get(n, n) for n in names) + " |", S(2)]
    for k in range(len(AGE_GROUPS)):
        sel = groups == k
        md.append(f"| {group_label(k)} | {int(sel.sum())} | " +
                  " | ".join(f"{np.abs(results[n]['age'][sel] - ages[sel]).mean():.2f}" if sel.any() else "—"
                             for n in names) + " |")

    G = len(AGE_GROUPS)
    for n in detail:
        cmg = np.zeros((G, G), int)
        np.add.at(cmg, (groups, results[n]["pg"]), 1)
        md += [f"\n### Matriz de confusão das faixas etárias — {NICE.get(n, n)} (linhas = real, colunas = prevista)\n",
               "| real \\ prev | " + " | ".join(group_label(k) for k in range(G)) + " | recall |",
               "|---|" + "---|" * (G + 1)]
        for k in range(G):
            md.append(f"| **{group_label(k)}** | " + " | ".join(
                f"**{cmg[k, j]}**" if j == k else str(cmg[k, j]) for j in range(G)) +
                f" | {cmg[k, k] / max(cmg[k].sum(), 1):.2f} |")

    md += ["\n## 3. Género — por classe\n", H("Métrica"), S()]
    for c, lab in enumerate(["female", "male"]):
        md += [row(f"Precisão {lab}", lambda r: f"{r['P'][c]:.3f}"),
               row(f"Recall {lab}", lambda r: f"{r['R'][c]:.3f}"),
               row(f"F1 {lab}", lambda r: f"{r['F1'][c]:.3f}")]
    md += [row("**F1 macro**", lambda r: f"**{r['F1'].mean():.3f}**"),
           row("Exatidão (só referência)", lambda r: f"{np.trace(r['cm']) / r['cm'].sum():.3f}"),
           "\n### Matrizes de confusão (linhas = real, colunas = previsto; F, M)\n",
           "| Modelo | real F → F | real F → M | real M → F | real M → M |", "|---|---|---|---|---|"]
    for n in names:
        cm = results[n]["cm"]
        md.append(f"| {NICE.get(n, n)} | {cm[0, 0]} | {cm[0, 1]} | {cm[1, 0]} | {cm[1, 1]} |")

    md += ["\n### Limiar de confiança (\"incerto\") — cobertura / F1 macro\n",
           "Cobertura = fração de caras com rótulo de género mostrado; F1 macro calculado só nessas caras.\n",
           H("Limiar"), S()]
    for t in THRESHOLDS:
        cells = []
        for n in names:
            r = results[n]
            keep = r["gconf"] >= t
            _, _, F1, _ = prf(genders[keep], r["g"][keep])
            cells.append(f"{keep.mean():.3f} / {F1.mean():.3f}")
        md.append(f"| {t:.2f}{' ← app' if abs(t - a.gender_threshold) < 1e-9 else ''} | " + " | ".join(cells) + " |")

    md += ["\n## 4. Diferenças demográficas (etnia anotada no UTKFace)\n",
           "| Etnia | caras | " + " | ".join(f"{NICE.get(n, n)} MAE / F1" for n in names) + " |", S(2)]
    for k, race in enumerate(RACES):
        sel = races == k
        cells = []
        for n in names:
            r = results[n]
            _, _, F1, _ = prf(genders[sel], r["g"][sel])
            cells.append(f"{np.abs(r['age'][sel] - ages[sel]).mean():.2f} / {F1.mean():.3f}")
        md.append(f"| {race} | {int(sel.sum())} | " + " | ".join(cells) + " |")
    w = np.bincount(groups, minlength=len(AGE_GROUPS)) / len(groups)
    md += ["\n**MAE por etnia ajustado à idade** (MAE de cada faixa etária dentro da etnia, pesado pela distribuição "
           "global das faixas; remove o efeito de cada etnia ter idades diferentes no teste):\n",
           "| Etnia | " + " | ".join(NICE.get(n, n) for n in names) + " |", S()]
    for k, race in enumerate(RACES):
        cells = []
        for n in names:
            e = np.abs(results[n]["age"] - ages)
            vals = [(w[g], e[(races == k) & (groups == g)].mean()) for g in range(len(AGE_GROUPS))
                    if ((races == k) & (groups == g)).sum() >= 5]
            cells.append(f"{sum(a * b for a, b in vals) / sum(a for a, _ in vals):.2f}")
        md.append(f"| {race} | " + " | ".join(cells) + " |")
    md += ["\n| Género anotado | caras | " + " | ".join(f"{NICE.get(n, n)} MAE" for n in names) + " |", S(2)]
    for c, lab in enumerate(["female", "male"]):
        sel = genders == c
        md.append(f"| {lab} | {int(sel.sum())} | " +
                  " | ".join(f"{np.abs(results[n]['age'][sel] - ages[sel]).mean():.2f}" for n in names) + " |")
    if "fairface_res34" in results and best_ft:
        ff, mn = results["fairface_res34"], results[best_ft]
        rng, dm, df = np.random.RandomState(0), [], []
        for _ in range(BOOT):                         # reamostra as MESMAS caras para os dois modelos
            b = rng.randint(0, len(ages), len(ages))
            dm.append(np.abs(ff["age"][b] - ages[b]).mean() - np.abs(mn["age"][b] - ages[b]).mean())
            df.append(prf(genders[b], ff["g"][b])[2].mean() - prf(genders[b], mn["g"][b])[2].mean())
        ci = lambda v: f"[{np.percentile(v, 2.5):+.3f}, {np.percentile(v, 97.5):+.3f}]"
        md += [f"\n## 5. FairFace − {NICE.get(best_ft, best_ft)}: a diferença é real?\n",
               f"Bootstrap emparelhado ({BOOT} reamostragens das caras de teste). Mede a incerteza da amostra de "
               "teste, não a variação entre treinos.\n",
               "| | Diferença | IC 95% |", "|---|---|---|",
               f"| MAE (anos; negativo = FairFace melhor) | {ff['mae'] - mn['mae']:+.2f} | {ci(dm)} |",
               f"| F1 macro do género (positivo = FairFace melhor) | {ff['F1'].mean() - mn['F1'].mean():+.3f} | {ci(df)} |"]
    md += ["\n**Viés por faixa etária** (erro médio com sinal, anos):\n",
           "| Faixa | " + " | ".join(NICE.get(n, n) for n in names) + " |", S()]
    for k in range(len(AGE_GROUPS)):
        sel = groups == k
        md.append(f"| {group_label(k)} | " +
                  " | ".join(f"{(results[n]['age'][sel] - ages[sel]).mean():+.2f}" for n in names) + " |")
    md += ["\n**Género \"incerto\" por faixa etária** (fração de caras abaixo do limiar da app) e F1 macro sem limiar:\n",
           "| Faixa | " + " | ".join(NICE.get(n, n) for n in names) + " |", S()]
    for k in range(len(AGE_GROUPS)):
        sel = groups == k
        md.append(f"| {group_label(k)} | " + " | ".join(
            f"{(results[n]['gconf'][sel] < a.gender_threshold).mean():.3f} / "
            f"{prf(genders[sel], results[n]['g'][sel])[2].mean():.3f}" for n in names) + " |")
    md.append("\nExemplos: `exemplos_teste.jpg`")

    scatter(results, ages, os.path.join(RES, "idade_dispersao.png"))
    ex = [(SHORT.get(n, n), results[n]) for n in dict.fromkeys(
        detail + [n for n in ("genderage",) if n in results])]
    examples(X, ages, genders, ex, os.path.join(RES, "exemplos_teste.jpg"))
    with open(os.path.join(RES, "idade_genero.md"), "w") as fp:
        fp.write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\n-> {RES}/idade_genero.md")


if __name__ == "__main__":
    main()
