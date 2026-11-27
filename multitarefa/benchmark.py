"""Custo da atualização completa (deteção SCRFD + modelos) do modelo final vs especialistas, nas MESMAS condições.

Mesmo método de benchmark_combinacoes.py: frames 1280x720 com 1, 2 e 4 caras do UTKFace, 8 iterações de
aquecimento + 40 medidas, mediana; tudo no mesmo processo e em ordem alternada, para a carga da máquina não
favorecer ninguém. Os especialistas não tinham identidade, por isso mede-se também "+ ArcFace" (o w600k_mbf com
alinhamento), que é o que faz a comparação justa com a multitarefa (que já inclui a identidade).

Uso: python multitarefa/benchmark.py  -> multitarefa/resultados/benchmark.md
"""
import os
import resource
import sys
import time

import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmark_combinacoes import make_frames                 # noqa: E402
from combinacoes import ModelPool                             # noqa: E402
from deteccao.face_alignment import align_face                # noqa: E402
from deteccao.face_detector import create_detector            # noqa: E402
from expressao.comum import MODELS_DIR, ROOT                  # noqa: E402
from geral.galeria import Galeria                             # noqa: E402
from geral.modelo import GeralEstimator                       # noqa: E402
from multitarefa.temporal import Smoother                     # noqa: E402

NF, ITERS, WARM = (1, 2, 4), 40, 8
RES = os.path.join(ROOT, "multitarefa", "resultados")


class ArcFace:
    def __init__(self):
        so = ort.SessionOptions(); so.log_severity_level = 3
        self.s = ort.InferenceSession(os.path.join(MODELS_DIR, "w600k_mbf.onnx"), so, providers=["CPUExecutionProvider"])
        self.inp = self.s.get_inputs()[0].name

    def predict(self, frame, faces):
        x = np.stack([align_face(frame, f["landmarks"])[0] for f in faces])[..., ::-1].transpose(0, 3, 1, 2)
        return self.s.run(None, {self.inp: (x.astype(np.float32) - 127.5) / 127.5})[0]


def main():
    det = create_detector("scrfd")
    frames = make_frames(det)
    pool, arc, est = ModelPool(), ArcFace(), GeralEstimator("multitarefa")
    rng = np.random.RandomState(0)
    gal = Galeria({f"p{i}": rng.randn(20, 512).astype(np.float32) for i in range(5)}, "clusters", path="/dev/null")
    sm = Smoother()
    ag = {k: pool.age_gender(k) for k in ("mobilenet_ft", "fairface")}
    ex = {k: pool.expression(k) for k in ("b0", "b2")}
    systems = {
        "multitarefa (tudo, com galeria de 5 pessoas e média temporal)": lambda f, F: est.predict(f, F, gal, sm),
        "multitarefa sem média temporal": lambda f, F: est.predict(f, F, gal),
        "rapido: MobileNet FT + B0 (sem identidade)": lambda f, F: (ag["mobilenet_ft"].predict(f, F), ex["b0"].predict(f, F)),
        "rapido + ArcFace": lambda f, F: (ag["mobilenet_ft"].predict(f, F), ex["b0"].predict(f, F), arc.predict(f, F)),
        "preciso: FairFace + B2 (sem identidade)": lambda f, F: (ag["fairface"].predict(f, F), ex["b2"].predict(f, F)),
        "preciso + ArcFace": lambda f, F: (ag["fairface"].predict(f, F), ex["b2"].predict(f, F), arc.predict(f, F)),
    }
    T = {s: {n: [] for n in NF} for s in systems}
    D = {n: [] for n in NF}
    for n in NF:
        fr = frames[n]
        for i in range(WARM + ITERS):
            f = fr[i % len(fr)]
            for s, run in systems.items():                    # alternado: todos veem a mesma carga
                t0 = time.perf_counter()
                faces = det.detect(f)
                t1 = time.perf_counter()
                run(f, faces)
                t2 = time.perf_counter()
                assert len(faces) == n
                if i >= WARM:
                    T[s][n].append(((t1 - t0) * 1000, (t2 - t1) * 1000))
                    D[n].append((t1 - t0) * 1000)
    md = ["# Custo da atualização completa: modelo final vs especialistas\n",
          f"Frames 1280×720; mediana de {ITERS} medições após {WARM} de aquecimento; mesmo processo, sistemas alternados; só CPU, "
          "Python x86_64 por Rosetta. Cada célula: **ms totais (deteção + modelos) / atualizações por segundo**; "
          "entre parênteses, só os modelos.\n",
          "| Sistema | " + " | ".join(f"{n} cara{'s' if n > 1 else ''}" for n in NF) + " |", "|---|" + "---|" * len(NF)]
    for s in systems:
        c = []
        for n in NF:
            a = np.array(T[s][n])
            tot, mod = np.median(a.sum(1)), np.median(a[:, 1])
            c.append(f"{'**' if 1000 / tot >= 5 else ''}{tot:.0f} / {1000 / tot:.1f}{'**' if 1000 / tot >= 5 else ''} ({mod:.0f})")
        md.append(f"| {s} | " + " | ".join(c) + " |")
    md.append(f"\nDeteção SCRFD (igual para todos): {np.median(D[1]):.0f} ms. Memória máxima do processo (todos os modelos "
              f"carregados): {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20:.0f} MB. Negrito = ≥ 5 atualizações/s.")
    os.makedirs(RES, exist_ok=True)
    open(os.path.join(RES, "benchmark.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
