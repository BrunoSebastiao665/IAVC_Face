"""Compara os detetores de caras numa imagem (sem câmara).

Uso:
    python deteccao/test_retinaface.py                          # imagem de exemplo, todos os modelos
    python deteccao/test_retinaface.py img.jpg                  # a tua imagem, todos os modelos
    python deteccao/test_retinaface.py img.jpg --model scrfd    # só um modelo
"""
import argparse
import os
import sys
import time
import urllib.request

import cv2 as cv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # raiz do projeto
from deteccao.face_detector import DETECTORS, create_detector  # noqa: E402

SAMPLE_URL = "https://raw.githubusercontent.com/serengil/retinaface/master/tests/dataset/img3.jpg"
SAMPLE_PATH = "sample.jpg"


def load_image(path):
    if path is None:
        path = SAMPLE_PATH
        if not os.path.exists(path):
            urllib.request.urlretrieve(SAMPLE_URL, path)
    img = cv.imread(path)
    if img is None:
        raise SystemExit(f"Não foi possível ler {path}")
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?", help="imagem a testar (por omissão: exemplo)")
    parser.add_argument("--model", choices=list(DETECTORS), help="testar só este modelo")
    parser.add_argument("--runs", type=int, default=5, help="repetições para medir o tempo")
    args = parser.parse_args()

    img = load_image(args.image)
    print(f"Imagem: {img.shape[1]}x{img.shape[0]}\n")
    print(f"{'modelo':<20}{'caras':>7}{'ms/frame':>11}{'FPS':>7}")

    for name in ([args.model] if args.model else DETECTORS):
        detector = create_detector(name)
        faces = detector.detect(img)  # aquecimento (1ª chamada é mais lenta)

        start = time.time()
        for _ in range(args.runs):
            faces = detector.detect(img)
        ms = (time.time() - start) / args.runs * 1000
        print(f"{name:<20}{len(faces):>7}{ms:>11.0f}{1000 / ms:>7.1f}")

        out = detector.draw(img.copy(), faces)
        cv.imwrite(f"result_{name}.jpg", out)

    print("\nResultados guardados em result_<modelo>.jpg")


if __name__ == "__main__":
    main()
