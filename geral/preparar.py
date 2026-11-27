"""Crops 112x112 na geometria do ArcFace para todos os datasets do modelo geral (a mesma que a app usa).

UTKFace   (idade, género)  usa o split e as caras de data/UTKFace/utkface_96.npz; no treino deteta os landmarks,
                           em val/test usa os que já lá estão. Fica de fora a imagem cuja cara não seja a mesma.
AffectNet (expressão)      train do Kaggle + validação oficial (val+test, com a marca 'twin' de expressao/preparar_dados)
BTFER     (expressão)      só teste cross-dataset

O AffectNet do Kaggle já vem recortado mais justo do que o crop do ArcFace, por isso falta a margem da cara. Guarda-se
a imagem com a margem refletida e uma máscara dos píxeis reais: no treino a margem é variada (ver dados.py) para a
rede não aprender a depender dela.

Saída: data/geral/{utk,aff_train,aff_eval,btfer}.npz        Uso: python geral/preparar.py
"""
import glob
import os
import sys
import time

import cv2 as cv
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from expressao.comum import AFF_DIR, AFF_NPZ, BTFER_DIR, BTFER_FOLDERS, BTFER_NPZ, ROOT   # noqa: E402
from expressao.expression import TEMPLATE112                                              # noqa: E402
from expressao.preparar_dados import landmarks                                            # noqa: E402
from deteccao.face_alignment import ARCFACE_TEMPLATE, align_face, similarity_transform              # noqa: E402
from deteccao.face_detector import create_detector                                                  # noqa: E402

OUT = os.path.join(ROOT, "data", "geral")
UTK = os.path.join(ROOT, "data", "UTKFace")


def save(out, X, **meta):
    """X vai para <nome>_X.npy (lido por memmap em dados.py); o resto para <nome>.npz. Evita ter X duas vezes no disco."""
    np.save(out[:-4] + "_X.npy", X)
    np.savez(out, **meta)


def crop_tight(im, k):
    """Imagem já recortada -> (crop com margem refletida, máscara dos píxeis reais em bits)."""
    M = similarity_transform(k, ARCFACE_TEMPLATE)
    c = cv.warpAffine(im, M, (112, 112), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REFLECT_101)
    m = cv.warpAffine(np.ones(im.shape[:2], np.uint8), M, (112, 112), flags=cv.INTER_NEAREST, borderValue=0)
    return c, np.packbits(m)


def tight_set(det, items, pad, size, out, **extra):
    """items = lista de (imagem ou caminho)."""
    X, K, miss, t0 = np.zeros((len(items), 112, 112, 3), np.uint8), np.zeros((len(items), 1568), np.uint8), 0, time.time()
    for i, p in enumerate(items):
        im = cv.imread(p) if isinstance(p, str) else p
        k = landmarks(det, im, pad)
        if k is None:                                     # sem deteção: geometria mediana do dataset
            miss += 1
            k = TEMPLATE112 * (im.shape[0] / 112.0)
        X[i], K[i] = crop_tight(im, k)
        if (i + 1) % 2000 == 0:
            print(f"  {os.path.basename(out)} {i + 1}/{len(items)} ({(time.time() - t0) / (i + 1) * 1000:.0f} ms/img)", flush=True)
    save(out, X, mask=K, **extra)
    print(f"{out}: {len(items)} imagens, {miss} sem deteção (geometria mediana)", flush=True)


def utk(det):
    d = np.load(os.path.join(UTK, "utkface_96.npz"))
    names, box, kps, split = d["names"], d["box"], d["kps"].copy(), d["split"]
    X, ok, t0 = np.zeros((len(names), 112, 112, 3), np.uint8), np.ones(len(names), bool), time.time()
    for i, n in enumerate(names):
        im = cv.imread(os.path.join(UTK, str(n)))
        if not np.isfinite(kps[i]).all():
            f = det.detect(im)
            f = max(f, key=lambda f: (f["box"][2] - f["box"][0]) * (f["box"][3] - f["box"][1])) if f else None
            if f is None or np.abs(np.array(f["box"], np.float32) - box[i]).max() > 2:
                ok[i] = False
                continue
            kps[i] = np.array(f["landmarks"], np.float32)
        X[i] = align_face(im, kps[i])[0]
        if (i + 1) % 2000 == 0:
            print(f"  utk {i + 1}/{len(names)} ({(time.time() - t0) / (i + 1) * 1000:.0f} ms/img)", flush=True)
    save(os.path.join(OUT, "utk.npz"), X[ok], **{k: d[k][ok] for k in ("age", "gender", "race", "split", "names")})
    print(f"utk: {ok.sum()} caras, {(~ok).sum()} fora; split {np.bincount(split[ok])}", flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    which = sys.argv[1:] or ["btfer", "eval", "utk", "train"]
    if "btfer" in which:
        ps = [(c, p) for d, c in BTFER_FOLDERS.items() for p in sorted(glob.glob(os.path.join(BTFER_DIR, d, "*")))]
        keep = np.load(BTFER_NPZ)
        assert [os.path.relpath(p, BTFER_DIR) for _, p in ps] == keep["names"].tolist()
        tight_set(create_detector("scrfd", threshold=0.4), [p for _, p in ps], 96, 256, os.path.join(OUT, "btfer.npz"),
                  y=keep["y"], keep=keep["keep"])
    if "eval" in which:
        e = np.load(AFF_NPZ)
        tight_set(create_detector("scrfd", threshold=0.4), list(e["X"]), 40, 112, os.path.join(OUT, "aff_eval.npz"),
                  y=e["y"], part=e["part"], twin=e["twin"])
    if "utk" in which:
        utk(create_detector("scrfd", threshold=0.5))
    if "train" in which:
        ps = sorted(glob.glob(os.path.join(AFF_DIR, "train", "*", "*.jpg")))
        tight_set(create_detector("scrfd", threshold=0.4), ps, 40, 112, os.path.join(OUT, "aff_train.npz"),
                  y=np.array([int(p.split(os.sep)[-2]) for p in ps], np.int8))


if __name__ == "__main__":
    main()
