"""Compara o modelo geral (emb e ramo) nos mesmos testes das fases 3 e 4 -> resultados/avaliacao.md

Limiar da expressão: o menor em que as caras rotuladas do 'dev' (10% do AffectNet train) acertam >= 70% (7 classes).
Modelo da app: escolhido pelo val do UTKFace + dev do AffectNet (treino_*.csv), não pelos testes.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from expressao.comum import K7, bootstrap, metrics, to7                 # noqa: E402
from geral.dados import dev_split, load                                  # noqa: E402
from geral.modelo import GeralEstimator, save_config                     # noqa: E402
from idade_genero.age_gender import AGE_GROUPS, age_group, group_label   # noqa: E402

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados")
# especialistas, nos MESMOS testes: idade_genero/resultados/idade_genero.md e expressao/resultados/avaliacao.md
SPEC = {"MobileNet FT": dict(mae=6.03, f1=0.918, ms=0.4), "FairFace": dict(mae=5.28, f1=0.942, ms=48.5),
        "HSEmotion B0": dict(a8=0.584, a7=0.615, b7=0.749, ms=11.9), "HSEmotion B2": dict(a8=0.598, a7=0.631, b7=0.745, ms=25.1)}


def gender_f1(y, p):
    return metrics(y, p, 2)["F1_macro"]


def main():
    U, E, B, A = load("utk"), load("aff_eval"), load("btfer"), load("aff_train")
    te, nt, kp = np.where(U["split"] == 2)[0], np.where(~E["twin"])[0], np.where(B["keep"])[0]
    dv = np.where(dev_split(A["y"]))[0]
    res, val = {}, {}
    for mode in ("emb", "ramo"):
        est = GeralEstimator(f"geral_{mode}")
        _, g, a, _ = est.run(U["X"][te])
        ya, yg = U["age"][te].astype(float), U["gender"][te]
        xe, xb, xd = (est.run(D["X"][i])[3] for D, i in ((E, nt), (B, kp), (A, dv)))
        ye, yb, yd = E["y"][nt], B["y"][kp], A["y"][dv]
        m7, d7 = ye < K7, yd < K7
        err = a - ya
        grp = np.array([age_group(v) for v in ya])
        r = dict(mae=np.abs(err).mean(), rmse=np.sqrt((err ** 2).mean()), bias=err.mean(), cs5=(np.abs(err) <= 5).mean(),
                 mae_ci=bootstrap(lambda b: np.abs(err[b]).mean(), len(err)), groups=[np.abs(err[grp == k]).mean() for k in range(9)],
                 gen=metrics(yg, g.argmax(1), 2), a8=metrics(ye, xe.argmax(1), 8), a7=metrics(ye[m7], to7(xe[m7]).argmax(1), 7),
                 b7=metrics(yb, to7(xb).argmax(1), 7),
                 a7_ci=bootstrap(lambda b: metrics(ye[m7][b], to7(xe[m7])[b].argmax(1), 7)["bal_acc"], int(m7.sum())),
                 b7_ci=bootstrap(lambda b: metrics(yb[b], to7(xb)[b].argmax(1), 7)["bal_acc"], len(yb)))
        P = to7(xd[d7])
        hit, conf = P.argmax(1) == yd[d7], P.max(1)
        r["thr"] = next((t for t in np.arange(0.2, 0.96, 0.05) if (conf >= t).any() and hit[conf >= t].mean() >= 0.70), 0.95)
        c7 = to7(xe[m7]).max(1) >= r["thr"]
        r["cov"], r["acc_cov"] = c7.mean(), (to7(xe[m7]).argmax(1) == ye[m7])[c7].mean()
        t, one = [], U["X"][te[:1]]
        for k in range(60):
            t0 = time.perf_counter(); est.run(one); t.append(time.perf_counter() - t0)
        r["ms"], r["mb"] = np.median(t[10:]) * 1000, os.path.getsize(os.path.join("models", f"geral_{mode}.onnx")) / 1e6
        rows = [l.split(",") for l in open(os.path.join(RES, f"treino_{mode}.csv")).read().split()[1:]]
        val[mode] = (min(float(x[4]) for x in rows), max(float(x[6]) for x in rows))
        res[mode] = r
    e, r = res["emb"], res["ramo"]
    best = "ramo" if val["ramo"][1] > val["emb"][1] else "emb"               # decide a expressão, a tarefa mais difícil para o embedding
    save_config(modelo=f"geral_{best}", limiar_expressao=float(round(res[best]["thr"], 2)))
    S = SPEC
    ci = lambda v: f"[{v[0]:.3f}, {v[1]:.3f}]"
    md = ["# Modelo geral: avaliação\n", "Mesmas imagens de teste das fases 3 e 4. `emb` = cabeças sobre o embedding de 512-D; "
          "`ramo` = ramo próprio a partir do mapa intermédio do ArcFace.\n", "## Resumo\n",
          "| Métrica (teste) | geral emb | geral ramo | especialista leve | especialista preciso |", "|---|---|---|---|---|",
          f"| Idade MAE, UTKFace ({len(te)}) | {e['mae']:.2f} | {r['mae']:.2f} {ci(r['mae_ci'])} | MobileNet FT {S['MobileNet FT']['mae']} | FairFace {S['FairFace']['mae']} |",
          f"| Idade RMSE / viés / CS(5) | {e['rmse']:.2f} / {e['bias']:+.2f} / {e['cs5']:.3f} | {r['rmse']:.2f} / {r['bias']:+.2f} / {r['cs5']:.3f} | — | — |",
          f"| Género F1 macro, UTKFace | {e['gen']['F1_macro']:.3f} | {r['gen']['F1_macro']:.3f} | MobileNet FT {S['MobileNet FT']['f1']} | FairFace {S['FairFace']['f1']} |",
          f"| Expressão bal. acc., AffectNet 8 cl. ({len(nt)}) | {e['a8']['bal_acc']:.3f} | {r['a8']['bal_acc']:.3f} | B0 {S['HSEmotion B0']['a8']} | B2 {S['HSEmotion B2']['a8']} |",
          f"| Expressão bal. acc., AffectNet 7 cl. | {e['a7']['bal_acc']:.3f} {ci(e['a7_ci'])} | {r['a7']['bal_acc']:.3f} {ci(r['a7_ci'])} | B0 {S['HSEmotion B0']['a7']} | B2 {S['HSEmotion B2']['a7']} |",
          f"| Expressão bal. acc., BTFER 7 cl. ({len(kp)}) | {e['b7']['bal_acc']:.3f} {ci(e['b7_ci'])} | {r['b7']['bal_acc']:.3f} {ci(r['b7_ci'])} | B0 {S['HSEmotion B0']['b7']} | B2 {S['HSEmotion B2']['b7']} |",
          f"| ms por cara (CPU, rede toda) | {e['ms']:.1f} | {r['ms']:.1f} | ArcFace + FT + B0 | ArcFace + FairFace + B2 |",
          f"| Tamanho | {e['mb']:.0f} MB | {r['mb']:.0f} MB | — | — |",
          f"\nValidação usada para escolher (MAE UTK val / bal. acc. dev AffectNet): emb {val['emb'][0]:.2f} / {val['emb'][1]:.3f}; "
          f"ramo {val['ramo'][0]:.2f} / {val['ramo'][1]:.3f}. **App: geral_{best}.**",
          f"\nLimiar da expressão ({best}): {res[best]['thr']:.2f}; no AffectNet de teste rotula {100 * res[best]['cov']:.0f}% das caras, "
          f"com {100 * res[best]['acc_cov']:.1f}% de acerto.\n", "## Idade: MAE por faixa\n", "| Faixa | " +
          " | ".join(group_label(k) for k in range(len(AGE_GROUPS))) + " |", "|---|" + "---|" * 9]
    for m in ("emb", "ramo"):
        md.append(f"| {m} | " + " | ".join(f"{v:.2f}" for v in res[m]["groups"]) + " |")
    md += ["\n## Género por classe (P / R / F1) e confusão\n"]
    from expressao.comum import CLASSES
    for m in ("emb", "ramo"):
        q = res[m]["gen"]
        md.append(f"- {m}: F {q['P'][0]:.3f} / {q['R'][0]:.3f} / {q['F1'][0]:.3f}; M {q['P'][1]:.3f} / {q['R'][1]:.3f} / {q['F1'][1]:.3f}; "
                  f"matriz [[F→F, F→M], [M→F, M→M]] = {q['cm'].tolist()}")
    for title, key in (("AffectNet 7 classes", "a7"), ("BTFER", "b7")):
        md += [f"\n## Expressão, {title}: P / R / F1 por classe\n", "| Classe | emb | ramo |", "|---|---|---|"]
        for c in range(7):
            md.append(f"| {CLASSES[c]} | " + " | ".join(
                f"{res[m][key]['P'][c]:.2f} / {res[m][key]['R'][c]:.2f} / {res[m][key]['F1'][c]:.2f}" for m in ("emb", "ramo")) + " |")
        for m in ("emb", "ramo"):
            q = res[m][key]
            md.append(f"\n{m}: accuracy {q['acc']:.3f}, P/R/F1 macro {q['P_macro']:.3f} / {q['R_macro']:.3f} / {q['F1_macro']:.3f}; confusão (linhas = real): {q['cm'].tolist()}")
    with open(os.path.join(RES, "avaliacao.md"), "w") as fp:
        fp.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
