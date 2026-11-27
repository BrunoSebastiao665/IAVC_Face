"""Descarrega o HSEmotion / EmotiEffLib EfficientNet-B0 e B2 (Savchenko; Apache-2.0) e junta os metadados da app.

Sem qualquer treino nosso. Os autores pré-treinaram em VGGFace2 (identidade de caras) e treinaram no AffectNet.
  b0 = enet_b0_8_va_mtl  224x224, multi-tarefa: 8 expressões + valência/ativação (usamos só as 8 expressões)
  b2 = enet_b2_8         260x260, 8 expressões
Não se usam as variantes *_best_afew / *_best_vgaf, afinadas noutros datasets.
Pré-processamento dos autores: resize bilinear, RGB, normalização ImageNet.

Uso (venv da app): python expressao/obter_hsemotion.py
"""
import hashlib
import json
import os
import sys
import urllib.request

import onnx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from expressao.comum import MODELS, MODELS_DIR      # noqa: E402

URL = "https://github.com/sb-ai-lab/EmotiEffLib/raw/main/models/affectnet_emotions/onnx/{}.onnx"
SRC = {"b0": ("enet_b0_8_va_mtl", 224, "c43e056ad388d4a8dc911832b8291435b2af537f967e5870ebd731574ec7e812"),
       "b2": ("enet_b2_8", 260, "180a9d4845b59393de4511598a0d1d34b705034691ea32959ce5009db7cf52b7")}
CLASSES_HS = "Anger,Contempt,Disgust,Fear,Happy,Neutral,Sad,Surprise"     # ordem dos autores (idx_to_emotion_class)


def main():
    for name, (src, size, sha) in SRC.items():
        raw = urllib.request.urlopen(URL.format(src), timeout=120).read()
        got = hashlib.sha256(raw).hexdigest()
        if got != sha:
            raise SystemExit(f"{src}: sha256 inesperado: {got}")
        m = onnx.load_from_string(raw)
        meta = dict(size=str(size), mean=json.dumps([0.485, 0.456, 0.406]), std=json.dumps([0.229, 0.224, 0.225]),
                    classes=CLASSES_HS, source=f"HSEmotion {src} (Apache-2.0)")
        for k, v in meta.items():
            m.metadata_props.add(key=k, value=v)
        out = os.path.join(MODELS_DIR, MODELS[name][0])
        onnx.save(m, out)
        print(f"{src} -> {out} ({len(raw) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
