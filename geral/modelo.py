"""Inferência do modelo geral (só onnxruntime): uma passagem por cara dá identidade, idade, género e expressão.

    est = GeralEstimator(); gal = Galeria.load()
    for face, r in zip(faces, est.predict(frame, faces, gal)):
        r["name"]                       # 'Ana' ou 'Unknown'
        r["age"], r["age_range"], r["gender"], r["expr"]

As notas éticas das fases 3 e 4 mantêm-se: idade aparente, rótulo visual de género e configuração facial visível.
"""
import json
import os
import sys

import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from expressao.comum import CLASSES, MODELS_DIR, ROOT, to7                          # noqa: E402
from deteccao.face_alignment import align_face                                               # noqa: E402
from idade_genero.age_gender import GENDER_LABELS, age_group, group_label           # noqa: E402

CONFIG = os.path.join(ROOT, "geral", "resultados", "config.json")
UNKNOWN = "Unknown"


def config():
    c = {"modelo": "geral_ramo", "limiar_identidade": 0.40, "limiar_genero": 0.75, "limiar_expressao": 0.50}
    try:
        with open(CONFIG) as fp:
            c.update(json.load(fp))
    except (OSError, ValueError):
        pass
    return c


def save_config(**kw):
    c = config()
    c.update(kw)
    with open(CONFIG, "w") as fp:
        json.dump(c, fp, indent=2)


class GeralEstimator:
    def __init__(self, model=None):
        self.cfg = config()
        self.name = model or self.cfg["modelo"]
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.s = ort.InferenceSession(os.path.join(MODELS_DIR, self.name + ".onnx"), so, providers=["CPUExecutionProvider"])

    def run(self, crops, bs=64):
        """crops BGR uint8 [N,112,112,3] -> (embedding [N,512], género [N,2], idade [N], expressão [N,8])."""
        out = [self.s.run(None, {"data": np.asarray(crops[i:i + bs])[..., ::-1].transpose(0, 3, 1, 2).astype(np.float32)})
               for i in range(0, len(crops), bs)]
        e, g, a, x = (np.concatenate(o) for o in list(zip(*out))[:4])   # multitarefa.onnx tem ainda age_dist
        return e, g, a[:, 0], x

    def predict(self, frame, faces, gallery=None, smoother=None):
        """smoother: multitarefa.temporal.Smoother -> as saídas passam a ser médias pesadas dos frames anteriores."""
        res = [None] * len(faces)
        idx = [i for i, f in enumerate(faces) if len(f.get("landmarks", [])) == 5]
        if not idx:
            if smoother is not None:
                smoother.step(faces, res, [0.0] * len(faces))
            return res
        crops = np.stack([align_face(frame, faces[i]["landmarks"])[0] for i in idx])
        E, G, A, X = self.run(crops)
        S = gallery.scores(E) if gallery is not None and gallery.people else None
        tracks = [None] * len(faces)
        if smoother is not None:
            from multitarefa.temporal import quality
            obs, q = [None] * len(faces), [0.0] * len(faces)
            for j, i in enumerate(idx):
                obs[i] = {"g": G[j], "a": [A[j]], "x": X[j]}
                if S is not None:
                    obs[i]["sim"] = S[j]
                q[i] = quality(faces[i]["box"], faces[i].get("score", 1.0), crops[j])
            tracks = smoother.step(faces, obs, q)
        for j, i in enumerate(idx):
            e, g, a, x, sim, t = E[j], G[j], A[j], X[j], (S[j] if S is not None else None), tracks[i]
            if t is not None:
                g, a, x = t.s["g"], t.s["a"][0], t.s["x"]
                from multitarefa.temporal import identity
                name, sim = identity(t, gallery.names if S is not None else [], self.cfg["limiar_identidade"], UNKNOWN)
            elif sim is not None:
                p = int(sim.argmax())
                name, sim = (gallery.names[p] if sim[p] >= self.cfg["limiar_identidade"] else UNKNOWN), float(sim[p])
            else:
                name, sim = UNKNOWN, 0.0
            x = to7(np.asarray(x)[None])[0]
            k, o, a = int(np.argmax(g)), int(x.argmax()), float(np.clip(a, 0, 116))
            res[i] = {"emb": e, "name": name, "sim": sim, "age": a, "age_range": group_label(age_group(round(a))),
                      "gender": GENDER_LABELS[k], "gender_conf": float(g[k]),
                      "gender_uncertain": bool(g[k] < self.cfg["limiar_genero"]),
                      "expr": CLASSES[o], "expr_conf": float(x[o]), "expr_uncertain": bool(x[o] < self.cfg["limiar_expressao"]),
                      "track": t.id if t is not None else None}
        return res
