"""Avaliação da rede multitarefa -> multitarefa/resultados/avaliacao.md

A. Atributos: os MESMOS testes das fases 3 e 4, comparados com os especialistas e com o modelo geral (se existir).
   idade/género: teste do UTKFace (4 682) | expressão: validação oficial do AffectNet sem gémeas, 8 e 7 classes,
   e BTFER (7 classes, cross-dataset, sem ajuste).
   Limiar de "incerto" da expressão: o menor em que as caras do dev (10% do AffectNet train) acertam >= 70%.
B. Suavização temporal (multitarefa/temporal.py) em vídeos sintéticos: cada cara de teste é repetida 20 frames com
   pequenas perturbações (deslocamento ±3 px, rotação ±3°, escala ±3%, brilho ±15, ruído σ=4 e desfoque em 15% dos
   frames), como uma pessoa parada à frente da webcam. Mede-se o "flicker" (trocas de rótulo por frame), o desvio-
   padrão da idade dentro do vídeo e a exatidão, sem e com a média pesada. Num segundo teste a cara muda a meio
   (frame 10) e mede-se quantos frames a média demora a acompanhar.
C. Custo: ms por cara e tamanho do ONNX.
"""
import os
import sys
import time

import cv2 as cv
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from expressao.comum import CLASSES, K7, MODELS_DIR, ROOT, bootstrap, metrics, to7     # noqa: E402
from geral.dados import dev_split, load                                               # noqa: E402
from geral.modelo import GeralEstimator, save_config                                  # noqa: E402
from idade_genero.age_gender import AGE_GROUPS, age_group, group_label                # noqa: E402
from multitarefa.temporal import ALPHA, quality                                       # noqa: E402

RES = os.path.join(ROOT, "multitarefa", "resultados")
SPEC = {"MobileNet FT": dict(mae=6.03, f1=0.918), "FairFace": dict(mae=5.28, f1=0.942),
        "HSEmotion B0": dict(a8=0.584, a7=0.615, b7=0.749), "HSEmotion B2": dict(a8=0.598, a7=0.631, b7=0.745)}
FRAMES, SEQ = 20, 300


def attributes(name, U, E, B, A):
    est = GeralEstimator(name)
    te, nt, kp = np.where(U["split"] == 2)[0], np.where(~E["twin"])[0], np.where(B["keep"])[0]
    dv = np.where(dev_split(A["y"]))[0]
    _, g, a, _ = est.run(U["X"][te])
    xe, xb, xd = (est.run(D["X"][i])[3] for D, i in ((E, nt), (B, kp), (A, dv)))
    ya, yg, ye, yb, yd = U["age"][te].astype(float), U["gender"][te], E["y"][nt], B["y"][kp], A["y"][dv]
    m7, d7 = ye < K7, yd < K7
    err, grp = a - ya, np.array([age_group(v) for v in ya])
    r = dict(n=(len(te), len(nt), len(kp)), mae=np.abs(err).mean(), rmse=np.sqrt((err ** 2).mean()), bias=err.mean(),
             cs5=(np.abs(err) <= 5).mean(), mae_ci=bootstrap(lambda b: np.abs(err[b]).mean(), len(err)),
             groups=[np.abs(err[grp == k]).mean() for k in range(len(AGE_GROUPS))],
             gen=metrics(yg, g.argmax(1), 2), a8=metrics(ye, xe.argmax(1), 8),
             a7=metrics(ye[m7], to7(xe[m7]).argmax(1), 7), b7=metrics(yb, to7(xb).argmax(1), 7),
             a7_ci=bootstrap(lambda b: metrics(ye[m7][b], to7(xe[m7])[b].argmax(1), 7)["bal_acc"], int(m7.sum())),
             b7_ci=bootstrap(lambda b: metrics(yb[b], to7(xb)[b].argmax(1), 7)["bal_acc"], len(yb)))
    P = to7(xd[d7])
    hit, conf = P.argmax(1) == yd[d7], P.max(1)
    r["thr"] = next((t for t in np.arange(0.2, 0.96, 0.05) if (conf >= t).any() and hit[conf >= t].mean() >= 0.70), 0.95)
    c7 = to7(xe[m7]).max(1) >= r["thr"]
    r["cov"], r["acc_cov"] = c7.mean(), (to7(xe[m7]).argmax(1) == ye[m7])[c7].mean()
    t, one = [], U["X"][te[:1]]
    for _ in range(60):
        t0 = time.perf_counter(); est.run(one); t.append(time.perf_counter() - t0)
    r["ms"], r["mb"] = np.median(t[10:]) * 1000, os.path.getsize(os.path.join(MODELS_DIR, name + ".onnx")) / 1e6
    return r


def perturb(img, rng):
    a, s = rng.uniform(-3, 3), 1 + rng.uniform(-0.03, 0.03)
    M = cv.getRotationMatrix2D((56, 56), a, s)
    M[:, 2] += rng.uniform(-3, 3, 2)
    x = cv.warpAffine(img, M, (112, 112), borderMode=cv.BORDER_REFLECT_101).astype(np.float32)
    x = x + rng.uniform(-15, 15) + rng.normal(0, 4, x.shape)
    x = np.clip(x, 0, 255).astype(np.uint8)
    return cv.GaussianBlur(x, (0, 0), rng.uniform(1.0, 2.0)) if rng.rand() < 0.15 else x


def ema(seq, q):
    out, s = [], None
    for x, w in zip(seq, q):
        s = x.copy() if s is None else (1 - ALPHA * w) * s + ALPHA * w * x
        out.append(s)
    return np.array(out)


def temporal(est, U, E):
    """-> dict com flicker / desvio / exatidão (raw vs EMA) e atraso após troca de cara."""
    rng = np.random.RandomState(0)
    te = rng.permutation(np.where(U["split"] == 2)[0])[:SEQ]
    ex = np.where(~E["twin"] & (E["y"] < K7))[0]
    ex = rng.permutation(ex)[:SEQ]
    res = {}
    for kind, D, idx in (("utk", U, te), ("aff", E, ex)):
        frames = np.array([[perturb(D["X"][i], rng) for _ in range(FRAMES)] for i in idx])     # [S,F,112,112,3]
        q = np.array([[quality((0, 0, 112, 112), 1.0, c) for c in seq] for seq in frames])
        _, G, A, X = est.run(frames.reshape(-1, 112, 112, 3))
        G, A, X = G.reshape(len(idx), FRAMES, 2), A.reshape(len(idx), FRAMES), to7(X).reshape(len(idx), FRAMES, 7)
        Gs = np.array([ema(G[i], q[i]) for i in range(len(idx))])
        As = np.array([ema(A[i][:, None], q[i])[:, 0] for i in range(len(idx))])
        Xs = np.array([ema(X[i], q[i]) for i in range(len(idx))])
        flick = lambda L: float((L[:, 1:] != L[:, :-1]).mean())
        if kind == "utk":
            y, ya = U["gender"][idx], U["age"][idx].astype(float)
            res["gen"] = [(flick(P.argmax(2)), float((P.argmax(2) == y[:, None]).mean())) for P in (G, Gs)]
            res["age"] = [(float(Z[:, 5:].std(1).mean()), float(np.abs(Z - ya[:, None]).mean())) for Z in (A, As)]
        else:
            y = E["y"][idx]
            res["exp"] = [(flick(P.argmax(2)), float((P.argmax(2) == y[:, None]).mean())) for P in (X, Xs)]
            # troca de cara a meio: os 10 primeiros frames de uma cara e os 10 últimos de outra com outro rótulo previsto
            lab = X.mean(1).argmax(1)
            lags = []
            for i in range(len(idx) - 1):
                j = i + 1
                if lab[i] == lab[j]:
                    continue
                seq = np.concatenate([X[i, :10], X[j, 10:]])
                L = ema(seq, np.concatenate([q[i, :10], q[j, 10:]])).argmax(1)
                after = np.where(L[10:] == lab[j])[0]
                lags.append(after[0] if len(after) else 10)
            res["lag"] = (float(np.mean(lags)), float(np.median(lags)), len(lags))
    return res


def main():
    U, E, B, A = load("utk"), load("aff_eval"), load("btfer"), load("aff_train")
    names = [n for n in ("multitarefa", "geral_ramo", "geral_emb") if os.path.exists(os.path.join(MODELS_DIR, n + ".onnx"))]
    R = {n: attributes(n, U, E, B, A) for n in names}
    T = temporal(GeralEstimator("multitarefa"), U, E)
    m = R["multitarefa"]
    save_config(modelo="multitarefa", limiar_expressao=float(round(m["thr"], 2)))
    ci = lambda v: f"[{v[0]:.3f}, {v[1]:.3f}]"
    S = SPEC
    hdr = " | ".join(names)
    md = ["# Rede multitarefa: avaliação\n",
          "Backbone ArcFace (w600k_mbf) congelado + um ramo convolucional afinado por tarefa; identidade = ArcFace original.\n",
          "## A. Atributos (teste)\n", f"| Métrica | {hdr} | especialista leve | especialista preciso |", "|---|" + "---|" * (len(names) + 2),
          f"| Idade MAE, UTKFace ({m['n'][0]}) | " + " | ".join(f"{R[n]['mae']:.2f} {ci(R[n]['mae_ci'])}" for n in names) +
          f" | MobileNet FT {S['MobileNet FT']['mae']} | FairFace {S['FairFace']['mae']} |",
          "| Idade RMSE / viés / CS(5) | " + " | ".join(f"{R[n]['rmse']:.2f} / {R[n]['bias']:+.2f} / {R[n]['cs5']:.3f}" for n in names) + " | — | — |",
          "| Género F1 macro, UTKFace | " + " | ".join(f"{R[n]['gen']['F1_macro']:.3f}" for n in names) +
          f" | MobileNet FT {S['MobileNet FT']['f1']} | FairFace {S['FairFace']['f1']} |",
          f"| Expressão bal. acc., AffectNet 8 cl. ({m['n'][1]}) | " + " | ".join(f"{R[n]['a8']['bal_acc']:.3f}" for n in names) +
          f" | B0 {S['HSEmotion B0']['a8']} | B2 {S['HSEmotion B2']['a8']} |",
          "| Expressão bal. acc., AffectNet 7 cl. | " + " | ".join(f"{R[n]['a7']['bal_acc']:.3f} {ci(R[n]['a7_ci'])}" for n in names) +
          f" | B0 {S['HSEmotion B0']['a7']} | B2 {S['HSEmotion B2']['a7']} |",
          f"| Expressão bal. acc., BTFER 7 cl. ({m['n'][2]}) | " + " | ".join(f"{R[n]['b7']['bal_acc']:.3f} {ci(R[n]['b7_ci'])}" for n in names) +
          f" | B0 {S['HSEmotion B0']['b7']} | B2 {S['HSEmotion B2']['b7']} |",
          "| ms por cara (CPU, rede toda, inclui identidade) | " + " | ".join(f"{R[n]['ms']:.1f}" for n in names) + " | ArcFace 7.7 + FT 0.4 + B0 11.9 ≈ 20.0 | ArcFace 7.7 + FairFace 48.5 + B2 25.1 ≈ 81.3 |",
          "| Tamanho ONNX | " + " | ".join(f"{R[n]['mb']:.1f} MB" for n in names) + " | — | — |",
          f"\nLimiar de expressão incerta (multitarefa): {m['thr']:.2f} → no AffectNet de teste rotula {100 * m['cov']:.0f}% das caras "
          f"com {100 * m['acc_cov']:.1f}% de acerto. Gravado em geral/resultados/config.json (a app passa a usar a multitarefa).\n",
          "### Idade: MAE por faixa\n", "| Modelo | " + " | ".join(group_label(k) for k in range(len(AGE_GROUPS))) + " |",
          "|---|" + "---|" * len(AGE_GROUPS)]
    md += [f"| {n} | " + " | ".join(f"{v:.2f}" for v in R[n]["groups"]) + " |" for n in names]
    q = m["gen"]
    md += ["\n### Género (multitarefa)\n", f"F: P {q['P'][0]:.3f} R {q['R'][0]:.3f} F1 {q['F1'][0]:.3f}; M: P {q['P'][1]:.3f} R {q['R'][1]:.3f} "
           f"F1 {q['F1'][1]:.3f}; confusão [[F→F, F→M], [M→F, M→M]] = {q['cm'].tolist()}"]
    for title, key in (("AffectNet 7 classes", "a7"), ("BTFER", "b7")):
        q = m[key]
        md += [f"\n### Expressão, {title} (multitarefa): P / R / F1 por classe\n", "| Classe | P | R | F1 |", "|---|---|---|---|"]
        md += [f"| {CLASSES[c]} | {q['P'][c]:.2f} | {q['R'][c]:.2f} | {q['F1'][c]:.2f} |" for c in range(7)]
        md.append(f"\naccuracy {q['acc']:.3f}; confusão (linhas = real): {q['cm'].tolist()}")
    md += ["\n## B. Suavização temporal (vídeos sintéticos, 300 caras x 20 frames por tarefa)\n",
           f"EMA com α = {ALPHA} x qualidade do frame (nitidez).\n",
           "| Tarefa | flicker sem → com EMA (trocas/frame) | exatidão por frame sem → com | outra métrica sem → com |", "|---|---|---|---|",
           f"| Género | {T['gen'][0][0]:.3f} → {T['gen'][1][0]:.3f} | {T['gen'][0][1]:.3f} → {T['gen'][1][1]:.3f} | — |",
           f"| Idade | — | MAE {T['age'][0][1]:.2f} → {T['age'][1][1]:.2f} | desvio da idade no vídeo {T['age'][0][0]:.2f} → {T['age'][1][0]:.2f} anos |",
           f"| Expressão (7 cl.) | {T['exp'][0][0]:.3f} → {T['exp'][1][0]:.3f} | {T['exp'][0][1]:.3f} → {T['exp'][1][1]:.3f} | — |",
           f"\nAtraso quando a cara/expressão muda a meio: média {T['lag'][0]:.1f} frames, mediana {T['lag'][1]:.0f} "
           f"({T['lag'][2]} trocas). É o preço da estabilidade: a ~10 atualizações/s são ~{T['lag'][0] / 10:.1f} s."]
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, "avaliacao.md"), "w") as fp:
        fp.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
