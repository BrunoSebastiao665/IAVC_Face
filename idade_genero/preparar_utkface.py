"""Prepara o UTKFace (in-the-wild) para treino/avaliação de idade e género.

Para cada imagem data/UTKFace/part*/<idade>_<género>_<etnia>_<data>.jpg:
  1. deteta caras com o SCRFD (o mesmo detetor da aplicação);
  2. escolhe a cara anotada = a maior deteção (o UTKFace tem 1 cara anotada por imagem;
     quando há outras pessoas, a anotada é quase sempre a principal);
  3. guarda o crop 96x96 igual ao da aplicação (age_gender.crop_face).

Rótulos UTKFace (do nome do ficheiro):
  género: 0 = male, 1 = female  -> convertido para a convenção do genderage.onnx (0 = F, 1 = M)
  etnia:  0 White, 1 Black, 2 Asian, 3 Indian, 4 Others  (-1 se faltar no nome)

Split fixo (seed 0): 70% treino / 10% validação / 20% teste.

Saída: data/UTKFace/utkface_96.npz
    X uint8 [N,96,96,3] BGR, age, gender, race, split, names, box, score,
    kps [N,5,2] (5 landmarks do SCRFD; usados pelos modelos com crop alinhado, ex. FairFace)

Uso:
    python idade_genero/preparar_utkface.py               # tudo (~12 min em CPU)
    python idade_genero/preparar_utkface.py --limit 500   # teste rápido
    python idade_genero/preparar_utkface.py --add-landmarks val test
        # acrescenta os landmarks a um .npz antigo (sem kps) SEM mudar o split: volta a detetar
        # só nessas imagens e confirma que a cara escolhida é a mesma (mesma caixa)
"""
import argparse
import glob
import os
import sys
import time

import cv2 as cv
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deteccao.face_detector import create_detector                              # noqa: E402
from idade_genero.age_gender import ROOT, crop_face                    # noqa: E402

UTK = os.path.join(ROOT, "data", "UTKFace")
OUT = os.path.join(UTK, "utkface_96.npz")
RACES = ["White", "Black", "Asian", "Indian", "Others"]
SPLITS = {"train": 0, "val": 1, "test": 2}


def parse_name(path):
    parts = os.path.basename(path).split("_")
    age, g = int(parts[0]), int(parts[1])
    if g not in (0, 1) or not 0 <= age <= 116:
        raise ValueError(f"rótulo inválido: {path}")      # ex.: 61_3_<data>.jpg (género 3)
    race = int(parts[2]) if len(parts) >= 4 and parts[2].isdigit() else -1
    return age, 1 - g, race                       # UTK 0=male -> 1 (M); UTK 1=female -> 0 (F)


def add_landmarks(splits):
    """Junta 'kps' a um .npz já existente, só para os splits pedidos (o resto fica NaN)."""
    d = dict(np.load(OUT))
    kps = d.get("kps", np.full((len(d["X"]), 5, 2), np.nan, np.float32))
    det = create_detector("scrfd", threshold=0.5)
    idx = np.where(np.isin(d["split"], [SPLITS[s] for s in splits]))[0]
    print(f"a detetar landmarks em {len(idx)} imagens ({', '.join(splits)})")
    mismatch, t0 = 0, time.time()
    for k, i in enumerate(idx):
        img = cv.imread(os.path.join(UTK, str(d["names"][i])))
        faces = det.detect(img)
        f = max(faces, key=lambda f: (f["box"][2] - f["box"][0]) * (f["box"][3] - f["box"][1]))
        if np.abs(np.array(f["box"], np.float32) - d["box"][i]).max() > 2:
            mismatch += 1
        kps[i] = np.array(f["landmarks"], np.float32)
        if (k + 1) % 1000 == 0:
            print(f"  {k + 1}/{len(idx)}  ({(time.time() - t0) / (k + 1) * 1000:.0f} ms/img)")
    d["kps"] = kps
    np.savez(OUT, **d)
    print(f"landmarks guardados em {OUT}; caixas diferentes da preparação original: {mismatch}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="só as primeiras N imagens (teste)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--add-landmarks", nargs="+", choices=list(SPLITS), metavar="SPLIT",
                    help="só acrescentar landmarks ao .npz existente para estes splits")
    a = ap.parse_args()
    if a.add_landmarks:
        return add_landmarks(a.add_landmarks)

    paths = sorted(glob.glob(os.path.join(UTK, "part*", "*.jpg")))
    rng = np.random.RandomState(a.seed)
    rng.shuffle(paths)
    if a.limit:
        paths = paths[:a.limit]
    print(f"{len(paths)} imagens")

    det = create_detector("scrfd", threshold=0.5)
    X, meta, failed, bad_name = [], [], [], []
    t0 = time.time()
    for k, p in enumerate(paths):
        try:
            age, gender, race = parse_name(p)
        except (ValueError, IndexError):
            bad_name.append(p)
            continue
        img = cv.imread(p)
        faces = det.detect(img) if img is not None else []
        if not faces:
            failed.append(p)
            continue
        f = max(faces, key=lambda f: (f["box"][2] - f["box"][0]) * (f["box"][3] - f["box"][1]))
        X.append(crop_face(img, f["box"]))
        meta.append((age, gender, race, os.path.relpath(p, UTK), f["box"], f["score"], f["landmarks"]))
        if (k + 1) % 500 == 0:
            el = time.time() - t0
            print(f"  {k + 1}/{len(paths)}  ({el / (k + 1) * 1000:.0f} ms/img, faltam ~{el / (k + 1) * (len(paths) - k - 1) / 60:.0f} min)")

    n = len(X)
    split = np.zeros(n, np.int8)
    idx = rng.permutation(n)
    split[idx[int(0.7 * n):int(0.8 * n)]] = SPLITS["val"]
    split[idx[int(0.8 * n):]] = SPLITS["test"]
    age, gender, race, names, box, score, kps = zip(*meta)
    np.savez(OUT, X=np.stack(X), age=np.array(age, np.int16), gender=np.array(gender, np.int8),
             race=np.array(race, np.int8), split=split, names=np.array(names),
             box=np.array(box, np.float32), score=np.array(score, np.float32),
             kps=np.array(kps, np.float32).reshape(-1, 5, 2))

    print(f"\n{n} caras guardadas em {OUT}")
    print(f"sem deteção: {len(failed)} | nome inválido: {len(bad_name)}")
    print("split:", {s: int((split == v).sum()) for s, v in SPLITS.items()})
    with open(os.path.join(UTK, "preparacao_log.txt"), "w") as fp:
        fp.write(f"imagens: {len(paths)}\nguardadas: {n}\nsem deteção: {len(failed)}\n"
                 f"nome inválido: {len(bad_name)}\n\n[sem deteção]\n" + "\n".join(failed) +
                 "\n\n[nome inválido]\n" + "\n".join(bad_name) + "\n")


if __name__ == "__main__":
    main()
