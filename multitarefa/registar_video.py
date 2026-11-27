"""Registo de uma pessoa a partir de um vídeo (notas da aula: "vídeo de frente e de lado, expressões, longe").

Guião (~20 s): de frente -> rodar devagar para a esquerda e para a direita -> sorrir / sério / surpreso
-> afastar-se um passo da câmara. Regista só quem deu consentimento.

"Que frames?" — em cada frame usa a maior cara e só a guarda se:
  1. tiver lado >= 60 px e score do detetor >= 0.70;
  2. estiver nítida: variância do Laplaciano do crop alinhado >= 40;
  3. não for quase repetida: cosseno < 0.95 com todos os descritores já guardados;
  4. for coerente com a pessoa: cosseno >= 0.35 com a média do que já foi guardado (evita outra pessoa a passar).
Cada frame guardado é etiquetado com a pose (frente / lado) e a distância (perto / longe), pelos 5 landmarks.
Guarda APENAS descritores de 512-D na galeria (data/galeria.npz), nunca imagens.

    python multitarefa/registar_video.py NOME video.mp4
    python multitarefa/registar_video.py NOME 0                 webcam (q termina)
    python multitarefa/registar_video.py NOME video.mp4 --max 60 --passo 2
"""
import argparse
import os
import sys
from collections import Counter

import cv2 as cv
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deteccao.face_alignment import align_face                  # noqa: E402
from deteccao.face_detector import create_detector              # noqa: E402
from geral.galeria import Galeria, clusters                     # noqa: E402
from geral.modelo import GeralEstimator                         # noqa: E402
from multitarefa.temporal import sharpness                      # noqa: E402

MIN_SIZE, MIN_SCORE, MIN_SHARP, DUP, SAME = 60, 0.70, 40.0, 0.95, 0.35


def pose(lm):
    """yaw aproximado: desvio do nariz em relação ao meio dos olhos, em unidades da distância entre olhos."""
    le, re, nose = np.array(lm[0]), np.array(lm[1]), np.array(lm[2])
    d = np.linalg.norm(re - le) + 1e-9
    return float((nose[0] - (le[0] + re[0]) / 2) / d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("nome")
    ap.add_argument("fonte", help="ficheiro de vídeo ou índice da câmara")
    ap.add_argument("--max", type=int, default=60, help="máximo de descritores guardados")
    ap.add_argument("--passo", type=int, default=2, help="processa 1 em cada N frames")
    ap.add_argument("--modelo", default=None)
    a = ap.parse_args()
    cap = cv.VideoCapture(int(a.fonte) if a.fonte.isdigit() else a.fonte)
    if not cap.isOpened():
        raise SystemExit(f"não consegui abrir {a.fonte}")
    det, est = create_detector("scrfd"), GeralEstimator(a.modelo)
    live = a.fonte.isdigit()
    E, tags, why, n = [], [], Counter(), 0
    while len(E) < a.max:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        if n % a.passo:
            continue
        faces = det.detect(frame)
        f = max(faces, key=lambda f: (f["box"][2] - f["box"][0]) * (f["box"][3] - f["box"][1])) if faces else None
        if f is None:
            why["sem cara"] += 1
        else:
            size = min(f["box"][2] - f["box"][0], f["box"][3] - f["box"][1])
            crop = align_face(frame, f["landmarks"])[0]
            if size < MIN_SIZE or f["score"] < MIN_SCORE:
                why["pequena / score baixo"] += 1
            elif sharpness(crop) < MIN_SHARP:
                why["desfocada"] += 1
            else:
                e = est.run(crop[None])[0][0]
                if E and float(np.max(np.array(E) @ e)) >= DUP:
                    why["repetida"] += 1
                elif E and float(np.mean(E, 0) @ e / np.linalg.norm(np.mean(E, 0))) < SAME:
                    why["outra pessoa?"] += 1
                else:
                    E.append(e)
                    y = pose(f["landmarks"])
                    tags.append(("frente" if abs(y) < 0.15 else "lado") + "/" + ("perto" if size >= 120 else "longe"))
        if live:
            view = frame.copy()
            if f is not None:
                cv.rectangle(view, tuple(map(int, f["box"][:2])), tuple(map(int, f["box"][2:])), (0, 255, 0), 2)
            cv.putText(view, f"{a.nome}: {len(E)}/{a.max}  (roda a cabeca, faz expressoes, afasta-te; q termina)",
                       (10, 26), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv.imshow("registo", view)
            if cv.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    cv.destroyAllWindows()
    if len(E) < 3:
        raise SystemExit(f"só {len(E)} frames bons; grava com mais luz / mais perto. Rejeitados: {dict(why)}")
    gal = Galeria.load("clusters")
    gal.add(a.nome, np.array(E))
    gal.save()
    k = len(clusters(gal.people[a.nome]))
    print(f"{a.nome}: {len(E)} descritores guardados de {n} frames -> {k} protótipos (clusters)")
    print("  poses/distâncias:", dict(Counter(tags)))
    print("  frames rejeitados:", dict(why))


if __name__ == "__main__":
    main()
