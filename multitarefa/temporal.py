"""Estimativas com base na média pesada dos frames anteriores (notas da aula).

1. Tracking por IoU: cada cara do frame atual herda o id da caixa do frame anterior com que mais se sobrepõe
   (IoU >= 0.3, emparelhamento guloso pela maior IoU). Uma track morre após MAX_MISSED frames sem cara.
2. Por track, média exponencial (EMA) das SAÍDAS CONTÍNUAS da rede, não dos rótulos:
       s_t = (1 - w_t) * s_{t-1} + w_t * x_t,      w_t = ALPHA * q_t
   - probabilidades de género (2) e de expressão (8), idade esperada, similaridade a cada pessoa da galeria;
   - q_t em [0, 1] é a qualidade do frame: tamanho da cara x score do detetor x nitidez (variância do Laplaciano).
     Um frame pequeno, desfocado ou duvidoso quase não mexe na média.
3. Identidade com histerese: o nome atual só se perde quando a similaridade média desce LIMIAR - HIST; um nome novo
   só entra quando passa o LIMIAR. Evita que o nome pisque quando a pessoa está perto do limiar.

Com ALPHA = 0.35 e qualidade 1, um frame novo pesa 35%; ~90% do peso está nos últimos 5 frames processados.
"""
import cv2 as cv
import numpy as np

ALPHA, MAX_MISSED, IOU_MIN, HIST = 0.35, 8, 0.3, 0.05


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x2 - x1) * max(0, y2 - y1)
    return i / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i, 1e-9)


def sharpness(crop):
    """variância do Laplaciano da cara alinhada 112x112 (cinzento)."""
    return float(cv.Laplacian(cv.cvtColor(crop, cv.COLOR_BGR2GRAY), cv.CV_32F).var())


def quality(box, score, crop=None, size_full=80.0, sharp_full=150.0):
    w = min(box[2] - box[0], box[3] - box[1])
    q = np.clip(w / size_full, 0.1, 1.0) * np.clip(score, 0.0, 1.0)
    if crop is not None:
        q *= np.clip(sharpness(crop) / sharp_full, 0.2, 1.0)
    return float(q)


class Track:
    def __init__(self, tid, box, obs):
        self.id, self.box, self.missed, self.n = tid, box, 0, 1
        self.s = {k: np.array(v, np.float64) for k, v in obs.items()}
        self.name = None

    def update(self, box, obs, w):
        self.box, self.missed, self.n = box, 0, self.n + 1
        for k, v in obs.items():
            v = np.array(v, np.float64)
            if k not in self.s or self.s[k].shape != v.shape:     # p.ex. alguém entrou na galeria a meio
                self.s[k] = v
            else:
                self.s[k] = (1 - w) * self.s[k] + w * v


class Smoother:
    def __init__(self, alpha=ALPHA):
        self.alpha, self.tracks, self.next_id = alpha, [], 0

    def reset(self):
        self.tracks = []

    def step(self, faces, obs, q):
        """faces: deteções do frame (com 'box'); obs[i]: dict de vetores da rede (ou None); q[i]: qualidade.
        -> lista (track ou None) alinhada com faces; as médias estão em track.s."""
        pairs = sorted(((iou(t.box, f["box"]), ti, fi) for ti, t in enumerate(self.tracks)
                        for fi, f in enumerate(faces) if obs[fi] is not None), reverse=True)
        used_t, out = set(), [None] * len(faces)
        for v, ti, fi in pairs:
            if v < IOU_MIN:
                break
            if ti in used_t or out[fi] is not None:
                continue
            used_t.add(ti)
            t = self.tracks[ti]
            t.update(f_box(faces[fi]), obs[fi], self.alpha * q[fi])
            out[fi] = t
        for t_i, t in enumerate(self.tracks):
            if t_i not in used_t:
                t.missed += 1
        for fi, f in enumerate(faces):
            if out[fi] is None and obs[fi] is not None:
                t = Track(self.next_id, f_box(f), obs[fi])
                self.next_id += 1
                self.tracks.append(t)
                out[fi] = t
        self.tracks = [t for t in self.tracks if t.missed <= MAX_MISSED]
        return out


def f_box(f):
    return [float(v) for v in f["box"]]


def identity(track, names, thr, unknown):
    """nome da track com histerese a partir da similaridade média (track.s['sim'])."""
    s = track.s.get("sim")
    if s is None or not len(names):
        track.name = None
        return unknown, 0.0
    p = int(s.argmax())
    cur = names.index(track.name) if track.name in names else None
    if cur is not None and s[cur] >= thr - HIST and (p == cur or s[p] < thr):
        return track.name, float(s[cur])
    track.name = names[p] if s[p] >= thr else None
    return (track.name or unknown), float(s[p])
