"""Prepara e audita os dados de expressão (AffectNet alinhado do Kaggle + BTFER).

AffectNet (data/AffectNet/AffectNetCustom/{train,val,test}/0..7, 112x112):
  - val+test (~500 por classe) = a validação oficial do AffectNet, dividida em duas pelo autor do Kaggle. É a
    nossa avaliação intra-dataset: o HSEmotion foi treinado pelos autores só no train oficial;
  - as imagens de val/test com uma (quase) igual no train ficam marcadas ('twin') e fora da avaliação principal.
    "quase igual" = correlação >= 0.92 entre miniaturas 24x24 em cinzento. O valor foi escolhido a olho com pares
    por faixa de correlação: >= 0.96 são todas a mesma foto, 0.92-0.94 já mistura fotos diferentes.
BTFER (data/BTFER/Batch_Ready 7/<classe>, 256x256, 300 por classe):
  - sai o que for (quase) igual a uma imagem do AffectNet (o teste cruzado tem de ser independente);
  - repetições internas: fica uma; se têm classes diferentes, saem todas;
  - guarda 2 versões: 'asis' (a imagem como é distribuída) e 'aligned' (SCRFD + alinhamento da app).

Saídas: affectnet_avaliacao.npz, btfer.npz, resultados/auditoria_dados.md
Uso (venv da app): python expressao/preparar_dados.py
"""
import glob
import hashlib
import os
import sys
from collections import defaultdict

import cv2 as cv
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from expressao.comum import (AFF_DIR, AFF_NPZ, BTFER_DIR, BTFER_FOLDERS, BTFER_NPZ, CLASSES, RES)  # noqa: E402
from expressao.expression import TEMPLATE112, align_face_expr      # noqa: E402
from deteccao.face_detector import create_detector                          # noqa: E402

THR = 0.92
THR_BT = 0.90         # BTFER vs AffectNet: mais baixo, porque os recortes/filtros das cópias são diferentes
SPLITS = ("train", "val", "test")


def thumb(im):
    g = cv.resize(cv.cvtColor(im, cv.COLOR_BGR2GRAY), (24, 24), interpolation=cv.INTER_AREA).astype(np.float32).ravel()
    g -= g.mean()
    return g / (np.linalg.norm(g) + 1e-6)


def best_match(A, B, block=2000):
    m, j = [], []
    for i in range(0, len(A), block):
        S = A[i:i + block] @ B.T
        m.append(S.max(1)); j.append(S.argmax(1))
    return np.concatenate(m), np.concatenate(j)


def drop_repeats(T, y, idx, keep, thr=0.98):
    """Grupos de imagens praticamente idênticas em idx: fica uma; com rótulos contraditórios saem todas.
    Altera keep; -> (nº retiradas com o mesmo rótulo, nº retiradas por contradição)."""
    parent = list(range(len(idx)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for i in range(0, len(idx), 2000):
        S = T[idx[i:i + 2000]] @ T[idx].T
        for a, b in zip(*np.where(S >= thr)):
            a += i
            if a < b:
                parent[find(b)] = find(a)
    groups = defaultdict(list)
    for a in range(len(idx)):
        groups[find(a)].append(idx[a])
    same = conf = 0
    for g in groups.values():
        if len(g) > 1:
            if len(set(y[g])) == 1:
                keep[g[1:]] = False; same += len(g) - 1
            else:
                keep[g] = False;     conf += len(g)
    return same, conf


def landmarks(det, im, pad):
    q = cv.copyMakeBorder(im, pad, pad, pad, pad, cv.BORDER_CONSTANT)
    f = det.detect(q)
    if not f:
        return None
    f = max(f, key=lambda f: (f["box"][2] - f["box"][0]) * (f["box"][3] - f["box"][1]))
    return np.array(f["landmarks"], np.float32).reshape(5, 2) - pad


def affectnet(md, det):
    """-> miniaturas de todas as imagens do AffectNet; grava a validação oficial (val+test) com a marca 'twin'."""
    paths = [(s, p) for s in SPLITS for p in sorted(glob.glob(os.path.join(AFF_DIR, s, "*", "*.jpg")))]
    split = np.array([SPLITS.index(s) for s, _ in paths], np.int8)
    y = np.array([int(p.split(os.sep)[-2]) for _, p in paths], np.int8)
    T, h, ev_imgs, L = np.zeros((len(paths), 576), np.float32), [], [], []
    sample = set(np.random.RandomState(0).permutation(np.where(split == 0)[0])[:1500].tolist())
    for i, (s, p) in enumerate(paths):
        with open(p, "rb") as fp:
            raw = fp.read()
        h.append(hashlib.md5(raw).hexdigest())
        im = cv.imdecode(np.frombuffer(raw, np.uint8), cv.IMREAD_COLOR)
        T[i] = thumb(im)
        if s != "train":
            ev_imgs.append(im)
        if i in sample:                                   # geometria dos crops do Kaggle
            k = landmarks(det, im, 40)
            if k is not None:
                L.append(k)
    tr, ev = np.where(split == 0)[0], np.where(split > 0)[0]
    htr = {h[i] for i in tr}
    exact = sum(h[i] in htr for i in ev)
    m, j = best_match(T[ev], T[tr])
    twin = m >= THR
    same = int((y[ev[twin]] == y[tr[j[twin]]]).sum())
    np.savez(AFF_NPZ, X=np.stack(ev_imgs), y=y[ev], part=np.array(SPLITS)[split[ev]], twin=twin,
             names=np.array([os.path.relpath(paths[i][1], AFF_DIR) for i in ev]))
    med = np.median(L, 0)
    row = lambda lab, idx: f"| {lab} | {len(idx)} | " + " | ".join(str(int((y[idx] == c).sum())) for c in range(8)) + " |"
    md += ["## AffectNet alinhado (Kaggle)\n",
           "| Partição | total | " + " | ".join(CLASSES) + " |", "|---|---|" + "---|" * 8,
           row("train (não usado: o modelo já vem treinado)", tr),
           row("val", np.where(split == 1)[0]), row("test", np.where(split == 2)[0]),
           row("**val + test = validação oficial**", ev),
           row("**validação oficial sem fuga (avaliação principal)**", ev[~twin]),
           f"\n- imagens de val/test bit a bit iguais a uma do train: **{exact}**",
           f"- imagens de val/test com uma quase igual no train (correlação ≥ {THR}): **{int(twin.sum())}** "
           f"({same} com o mesmo rótulo); ficam fora da avaliação principal",
           "- o train do Kaggle é só uma parte do train oficial com que os autores treinaram, por isso podem existir "
           "outras repetições que não conseguimos detetar",
           f"- landmarks medianos no train ({len(L)} caras): {np.round(med, 1).tolist()} "
           f"(TEMPLATE112 usado: {TEMPLATE112.tolist()})\n"]
    return T


def btfer(md, det, T):
    ps = [(BTFER_FOLDERS[d], p) for d in BTFER_FOLDERS for p in sorted(glob.glob(os.path.join(BTFER_DIR, d, "*")))]
    y = np.array([c for c, _ in ps], np.int8)
    A, B, h = [], [], []
    miss = 0
    for i, (_, p) in enumerate(ps):
        raw = open(p, "rb").read()
        h.append(hashlib.md5(raw).hexdigest())
        im = cv.imdecode(np.frombuffer(raw, np.uint8), cv.IMREAD_COLOR)
        k = landmarks(det, im, 96)
        miss += k is None
        A.append(cv.resize(im, (224, 224), interpolation=cv.INTER_AREA))
        B.append(align_face_expr(im, k, 224) if k is not None else A[-1])
    keep = np.ones(len(ps), bool)
    Tb = np.stack([thumb(cv.resize(b, (112, 112), interpolation=cv.INTER_AREA)) for b in B])
    m, _ = best_match(Tb, T)
    keep[m >= THR_BT] = False
    n_aff = int((m >= THR_BT).sum())
    same, conf = drop_repeats(Tb, y, np.where(keep)[0], keep)
    np.savez(BTFER_NPZ, asis=np.stack(A), aligned=np.stack(B), y=y, keep=keep,
             names=np.array([os.path.relpath(p, BTFER_DIR) for _, p in ps]))
    names7 = CLASSES[:7]
    md += ["## BTFER\n", "| | total | " + " | ".join(names7) + " |", "|---|---|" + "---|" * 7,
           f"| original | {len(y)} | " + " | ".join(str(int((y == c).sum())) for c in range(7)) + " |",
           f"| **usado** | {int(keep.sum())} | " + " | ".join(str(int(((y == c) & keep).sum())) for c in range(7)) + " |",
           f"\n- cópias bit a bit dentro do BTFER: {len(h) - len(set(h))}",
           f"- retiradas por serem (quase) iguais a uma imagem do AffectNet (correlação ≥ {THR_BT}): {n_aff} "
           f"(correlação máxima {m.max():.3f}; as de cima foram vistas a olho e incluem fotos realmente partilhadas)",
           f"- repetições internas retiradas (correlação ≥ 0.98): {same} com o mesmo rótulo, {conf} em grupos "
           "com rótulos contraditórios",
           f"- imagens sem cara detetada pelo SCRFD: {miss}\n"]


def main():
    os.makedirs(RES, exist_ok=True)
    det = create_detector("scrfd", threshold=0.4)
    md = ["# Auditoria dos dados de expressão\n"]
    btfer(md, det, affectnet(md, det))
    with open(os.path.join(RES, "auditoria_dados.md"), "w") as fp:
        fp.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
