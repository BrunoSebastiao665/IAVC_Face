"""Registo, limiar e teste da identidade, a partir de fotos.

Pastas (cada uma com uma subpasta por pessoa; '_desconhecidos' = pessoas NÃO registadas):
    data/identidades/registo/<nome>/*.jpg           fotos para registar
    data/identidades/validacao/<nome>/*.jpg         escolher o limiar     + validacao/_desconhecidos/*.jpg
    data/identidades/teste/<nome>/*.jpg             números do relatório  + teste/_desconhecidos/*.jpg
Usa fotos diferentes nas três (de preferência de dias diferentes) e desconhecidos diferentes em validação e teste.

    python geral/identidade.py registar             lê registo/ e cria a galeria (só descritores)
    python geral/identidade.py listar | remover NOME | limpar
    python geral/identidade.py provisorio           limiar provisório só com desconhecidos (UTKFace), sem as tuas fotos
    python geral/identidade.py calibrar             limiar na validacao/ (FAR <= 1%)
    python geral/identidade.py testar               rank-1, FAR, FRR e rejeição de Unknown em teste/ -> resultados/identidade.md
"""
import glob
import os
import sys

import cv2 as cv
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from expressao.comum import ROOT                                     # noqa: E402
from deteccao.face_alignment import align_face                                # noqa: E402
from deteccao.face_detector import create_detector                            # noqa: E402
from geral.galeria import Galeria                                    # noqa: E402
from geral.modelo import GeralEstimator, config, save_config         # noqa: E402

DIR = os.path.join(ROOT, "data", "identidades")
RES = os.path.join(ROOT, "geral", "resultados")
UNK = "_desconhecidos"
EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
FAR_TARGET = 0.01


def read_dir(est, det, part):
    """-> {pessoa: descritores [N,512]}; em cada foto usa a maior cara."""
    out = {}
    for d in sorted(glob.glob(os.path.join(DIR, part, "*", ""))):
        name, crops = os.path.basename(d.rstrip("/")), []
        for p in sorted(glob.glob(os.path.join(d, "**", "*"), recursive=True)):
            im = cv.imread(p) if p.lower().endswith(EXT) else None
            faces = [f for f in det.detect(im) if len(f["landmarks"]) == 5] if im is not None else []
            if faces:
                f = max(faces, key=lambda f: (f["box"][2] - f["box"][0]) * (f["box"][3] - f["box"][1]))
                crops.append(align_face(im, f["landmarks"])[0])
            elif im is not None:
                print(f"  sem cara: {p}")
            elif os.path.isfile(p) and not os.path.basename(p).startswith("."):
                print(f"  não consegui ler (converte HEIC para JPG): {p}")
        if crops:
            out[name] = est.run(np.stack(crops))[0]
    return out


def stats(gal, data, thr):
    """Contagens no limiar thr. FRR = conhecidos que não saem com o seu nome; FAR = desconhecidos aceites como alguém."""
    n = rank1 = right = wrong = 0
    for name, E in data.items():
        if name == UNK:
            continue
        if name not in gal.people:
            print(f"  aviso: {name} não está registado; ignorado")
            continue
        S = gal.scores(E)
        ok, acc = S.argmax(1) == gal.names.index(name), S.max(1) >= thr
        n += len(E); rank1 += ok.sum(); right += (ok & acc).sum(); wrong += (~ok & acc).sum()
    u = gal.scores(data[UNK]).max(1) if UNK in data else np.zeros(0)
    return dict(n=n, rank1=rank1 / max(n, 1), frr=1 - right / max(n, 1), troca=wrong / max(n, 1), nu=len(u),
                far=float((u >= thr).mean()) if len(u) else float("nan"))


def pick_threshold(gal, data):
    """Menor limiar com FAR <= alvo: mantém o máximo de reconhecimentos sem passar do alvo."""
    for t in np.arange(0.0, 1.0, 0.01):
        if (gal.scores(data[UNK]).max(1) >= t).mean() <= FAR_TARGET:
            return float(round(t, 2))
    return 0.99


def provisorio(est):
    from geral.dados import load
    d = load("utk")
    rng = np.random.RandomState(0)
    emb = {}
    for s, part in ((1, "val"), (2, "teste")):
        i = np.sort(rng.permutation(np.where((d["split"] == s) & (d["age"] >= 16))[0])[:1500])
        emb[part] = est.run(np.asarray(d["X"][i]))[0]
    rows = []
    for rep in range(20):                          # 19 "registados" ao acaso vs. todos os outros
        P = {p: rng.permutation(len(E)) for p, E in emb.items()}
        rows.append({p: (E[P[p][19:]] @ E[P[p][:19]].T).max(1) for p, E in emb.items()})
    V, T = np.concatenate([r["val"] for r in rows]), np.concatenate([r["teste"] for r in rows])
    thr = float(np.ceil(np.quantile(V, 1 - FAR_TARGET) * 100) / 100)
    print(f"limiar provisório {thr:.2f}: FAR val {(V >= thr).mean():.4f}, FAR teste {(T >= thr).mean():.4f} "
          f"(galeria de 19 caras do UTKFace, {len(emb['val'])} caras por split)")
    print("Só mede desconhecidos e com 1 foto por pessoa; corre 'calibrar' quando tiveres as fotos dos voluntários.")
    save_config(limiar_identidade=thr, origem_limiar="provisorio (UTKFace)")


def testar(est, det):
    reg, val, tst = (read_dir(est, det, p) for p in ("registo", "validacao", "teste"))
    if not reg or UNK not in val or UNK not in tst:
        raise SystemExit("faltam registo/, validacao/_desconhecidos/ ou teste/_desconhecidos/ em data/identidades/")
    pct = lambda v: f"{100 * v:.1f}%"
    md = ["# Identidade: resultados no teste\n", "Limiar escolhido na validação (FAR ≤ 1%) para cada linha; nunca no teste.\n",
          "| Estratégia | fotos de registo | limiar | rank-1 | FRR | nome trocado | FAR (desconhecidos aceites) | Unknown rejeitados |",
          "|---|---|---|---|---|---|---|---|"]
    rng = np.random.RandomState(0)
    sizes = [k for k in (1, 3, 5, 10) if k < max(map(len, reg.values()))] + [0]
    for mode in ("media", "clusters", "vizinho"):
        for k in sizes:
            R = []
            for _ in range(1 if k == 0 else 10):         # k fotos ao acaso, média de 10 sorteios
                g = Galeria({n: E if k == 0 else E[rng.permutation(len(E))[:k]] for n, E in reg.items()}, mode)
                t = pick_threshold(g, val)
                R.append((t, stats(g, tst, t)))
            m = {q: np.mean([r[q] for _, r in R]) for q in ("rank1", "frr", "troca", "far")}
            md.append(f"| {mode} | {k or 'todas'} | {np.mean([t for t, _ in R]):.2f} | {pct(m['rank1'])} | {pct(m['frr'])} | "
                      f"{pct(m['troca'])} | {pct(m['far'])} | {pct(1 - m['far'])} |")
    s = R[-1][1]
    md.append(f"\n{len(reg)} pessoas registadas; teste com {s['n']} fotos de conhecidos e {s['nu']} de desconhecidos.")
    open(os.path.join(RES, "identidade.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    gal = Galeria.load()
    if cmd == "listar":
        for n, E in gal.people.items():
            print(f"{n}: {len(E)} descritores")
        return
    if cmd == "remover":
        print("removido" if gal.remove(sys.argv[2]) else "não existe"); return gal.save()
    if cmd == "limpar":
        gal.clear(); gal.save(); print("galeria vazia"); return
    if cmd not in ("registar", "provisorio", "calibrar", "testar"):
        print(__doc__); return
    est = GeralEstimator()
    if cmd == "provisorio":
        return provisorio(est)
    det = create_detector("scrfd", threshold=0.5)
    if cmd == "registar":
        for n, E in read_dir(est, det, "registo").items():
            gal.remove(n)                                # voltar a registar substitui
            gal.add(n, E)
            print(f"{n}: {len(E)} fotos")
        return gal.save()
    if cmd == "testar":
        return testar(est, det)
    val = read_dir(est, det, "validacao")
    if not gal.people or UNK not in val:
        raise SystemExit("precisas de uma galeria registada e de validacao/_desconhecidos/")
    print("limiar  FAR    FRR")
    for t in np.arange(0.1, 0.71, 0.05):
        s = stats(gal, val, t)
        print(f"{t:.2f}   {s['far']:.3f}  {s['frr']:.3f}")
    thr = pick_threshold(gal, val)
    save_config(limiar_identidade=thr, origem_limiar="validacao")
    print(f"escolhido: {thr:.2f} (era {est.cfg['limiar_identidade']:.2f})", stats(gal, val, thr))


if __name__ == "__main__":
    main()
