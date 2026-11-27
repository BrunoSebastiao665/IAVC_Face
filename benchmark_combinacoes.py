"""Custo real de cada combinação de modelos na app: deteção (SCRFD) + idade/género + expressão.

Cada combinação corre num processo próprio (memória limpa). Mede a "atualização" completa, o mesmo trabalho que
a thread AsyncDetector faz por frame, em frames 1280x720 com 1, 2 e 4 caras (fotos do UTKFace coladas numa grelha):
  mediana e p95 da latência, atualizações/s, tempo por componente, núcleos de CPU usados e memória de pico.
Não mede a câmara nem o desenho da janela (correm noutra thread).

Saída: resultados_combinacoes/combinacoes.md e .json
Uso: fechar a app e outros programas pesados; python benchmark_combinacoes.py
"""
import glob
import json
import os
import resource
import subprocess
import sys
import time

import cv2 as cv
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "resultados_combinacoes")
AG = ["off", "mobilenet", "mobilenet_ft", "fairface"]
EX = ["off", "b0", "b2"]
NFACES = (1, 2, 4)
ITERS, TARGET = 40, 5.0           # iterações por medição; atualizações/s pretendidas


def make_frames(det, per_n=4, seed=0):
    """{n: [frames com exatamente n caras detetadas]}"""
    files = sorted(glob.glob(os.path.join(ROOT, "data", "UTKFace", "part1", "*.jpg")))
    rng = np.random.RandomState(seed)
    pos = {1: [(390, 110)], 2: [(80, 110), (700, 110)], 4: [(150, 5), (780, 5), (150, 365), (780, 365)]}
    side = {1: 500, 2: 500, 4: 350}
    ok = []
    for i in rng.permutation(len(files)):
        im = cv.imread(files[i])
        h, w = im.shape[:2]
        s = min(h, w)
        im = im[(h - s) // 2:(h + s) // 2, (w - s) // 2:(w + s) // 2]
        f = det.detect(cv.resize(im, (400, 400)))
        if len(f) == 1 and f[0]["score"] > 0.75 and f[0]["box"][2] - f[0]["box"][0] > 120:
            ok.append(im)
        if len(ok) >= 60:
            break
    out = {n: [] for n in NFACES}
    for n in NFACES:
        while len(out[n]) < per_n:
            fr = np.full((720, 1280, 3), 90, np.uint8)
            for (x, y), k in zip(pos[n], rng.permutation(len(ok))[:n]):
                fr[y:y + side[n], x:x + side[n]] = cv.resize(ok[k], (side[n], side[n]))
            if len(det.detect(fr)) == n:
                out[n].append(fr)
    return out


def child(ag, ex):
    from combinacoes import ModelPool
    from deteccao.face_detector import create_detector
    det, pool = create_detector("scrfd"), ModelPool()
    z = np.load(os.path.join(OUT, "frames.npz"))
    a, e = pool.age_gender(ag), pool.expression(ex)
    res = {}
    for n in NFACES:
        fr = z[f"n{n}"]
        T, cpu = [], 0.0
        for i in range(ITERS + 8):
            if i == 8:                                   # 8 de aquecimento
                cpu, wall = time.process_time(), time.perf_counter()
            f = fr[i % len(fr)]
            t0 = time.perf_counter()
            faces = det.detect(f)
            t1 = time.perf_counter()
            if a is not None:
                a.predict(f, faces)
            t2 = time.perf_counter()
            if e is not None:
                e.predict(f, faces)
            t3 = time.perf_counter()
            assert len(faces) == n
            if i >= 8:
                T.append((t1 - t0, t2 - t1, t3 - t2, t3 - t0))
        wall = time.perf_counter() - wall
        T = np.array(T) * 1000
        res[n] = dict(det=np.median(T[:, 0]), ag=np.median(T[:, 1]), ex=np.median(T[:, 2]), total=np.median(T[:, 3]),
                      p95=np.percentile(T[:, 3], 95), ups=1000 / np.median(T[:, 3]), cores=(time.process_time() - cpu) / wall)
    res["mem_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20      # macOS: bytes
    print("RES" + json.dumps(res))


def main():
    from deteccao.face_detector import create_detector
    os.makedirs(OUT, exist_ok=True)
    fr = make_frames(create_detector("scrfd"))
    np.savez(os.path.join(OUT, "frames.npz"), **{f"n{n}": np.stack(v) for n, v in fr.items()})
    pairs = [(a, e) for a in AG for e in EX]
    R = {}
    for k in np.random.RandomState(1).permutation(len(pairs)):        # ordem aleatória: a carga da máquina não favorece ninguém
        a, e = pairs[k]
        p = subprocess.run([sys.executable, __file__, "--child", a, e], capture_output=True, text=True, cwd=ROOT)
        line = [l for l in p.stdout.splitlines() if l.startswith("RES")]
        if not line:
            raise SystemExit(p.stderr)
        R[f"{a}+{e}"] = json.loads(line[0][3:])
        r = R[f"{a}+{e}"]
        print(f"{a:>13} + {e:<3}", {n: round(r[str(n)]["total"]) for n in NFACES}, "ms | det", round(r["1"]["det"]), f"| {r['mem_mb']:.0f} MB", flush=True)
    os.remove(os.path.join(OUT, "frames.npz"))
    json.dump(R, open(os.path.join(OUT, "combinacoes.json"), "w"), indent=1)
    dets = [R[c]["1"]["det"] for c in R]
    md = ["# Custo das combinações de modelos (atualização completa: SCRFD + idade/género + expressão)\n",
          f"- Apple M1 (8 GB), só CPU, Python x86_64 via Rosetta; frames 1280×720; mediana de {ITERS} medições, um processo por combinação, ordem aleatória",
          f"- Controlo da carga da máquina: a deteção (trabalho igual em todas) demorou {min(dets):.0f}–{max(dets):.0f} ms entre combinações",
          f"- **negrito** = pelo menos {TARGET:.0f} atualizações dos rótulos por segundo\n",
          "| Idade/género | Expressão | " + " | ".join(f"{n} cara{'s' if n > 1 else ''}: ms (p95) / atual. por s" for n in NFACES) + " | Núcleos CPU (4 caras) | Memória (MB) |",
          "|---|---|" + "---|" * (len(NFACES) + 2)]
    for a, e in pairs:
        r = R[f"{a}+{e}"]
        c = []
        for n in NFACES:
            q = r[str(n)]
            t = f"{q['total']:.0f} ({q['p95']:.0f}) / {q['ups']:.1f}"
            c.append(f"**{t}**" if q["ups"] >= TARGET else t)
        md.append(f"| {a} | {e} | " + " | ".join(c) + f" | {r['4']['cores']:.1f} | {r['mem_mb']:.0f} |")
    md += ["\n## Tempo por componente (ms, mediana)\n", "| Componente | " + " | ".join(f"{n} cara{'s' if n > 1 else ''}" for n in NFACES) + " |", "|" + "---|" * (len(NFACES) + 1)]
    med = lambda key, sel, n: np.median([R[c][str(n)][key] for c in R if sel(c)])
    md.append("| Deteção SCRFD | " + " | ".join(f"{med('det', lambda c: True, n):.1f}" for n in NFACES) + " |")
    for a in AG[1:]:
        md.append(f"| idade/género: {a} | " + " | ".join(f"{med('ag', lambda c: c.startswith(a + '+'), n):.1f}" for n in NFACES) + " |")
    for e in EX[1:]:
        md.append(f"| expressão: {e} | " + " | ".join(f"{med('ex', lambda c: c.endswith('+' + e), n):.1f}" for n in NFACES) + " |")
    open(os.path.join(OUT, "combinacoes.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    if sys.argv[1:2] == ["--child"]:
        child(*sys.argv[2:4])
    else:
        main()
