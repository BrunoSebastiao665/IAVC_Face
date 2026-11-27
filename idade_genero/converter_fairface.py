"""Converte o modelo pré-treinado FairFace (ResNet-34) para ONNX, no formato da aplicação.

FairFace (Kärkkäinen & Joo, WACV 2021, https://github.com/dchen236/FairFace, licença CC BY 4.0):
  - ResNet-34 (ImageNet) com fine-tuning no dataset FairFace: 108k caras EQUILIBRADAS por etnia
    (7 grupos), género e idade; pensado para reduzir diferenças de desempenho entre grupos.
  - Pesos usados: res34_fair_align_multi_7_20190809.pt (o link do Drive oficial está morto;
    cópia do ficheiro original em https://huggingface.co/anning01/fairface).
  - Saída original: 18 logits = 7 etnia | 2 género (0 = Male, 1 = Female) | 9 faixas etárias
    (0-2, 3-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70+).
  - Entrada no treino: cara alinhada com dlib.get_face_chip(size=300, padding=0.25) -> 224x224,
    normalização ImageNet.

O que este script faz:
  1. lê os pesos com torch.load(weights_only=True) (só tensores, nunca executa código do pickle);
  2. embrulha a rede: RGB 0..255 -> normalização ImageNet -> ResNet-34 -> saídas da aplicação
       gender_prob [N,2] (F, M)   — reordenado: o FairFace usa (M, F)
       age_group_prob [N,9]       — saída nativa
       age [N,1]                  — idade em anos = sum_g p_g * centro_g  (centros FIXOS, ver AGE_CENTERS;
                                    não são ajustados ao UTKFace: o modelo continua só pré-treinado)
     (a etnia prevista é descartada: não é usada na aplicação);
  3. exporta para models/fairface_res34.onnx com metadados crop=align, size=224, align_zoom;
  4. --tune: escolhe o align_zoom (margem do crop) no split de VALIDAÇÃO do UTKFace — o template do
     dlib não coincide com os 5 landmarks do SCRFD, por isso a margem é um hiperparâmetro.

Uso:
    python idade_genero/converter_fairface.py PESOS.pt --tune
"""
import argparse
import os
import sys
import time

import numpy as np
import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from idade_genero.age_gender import (MODELS_DIR, ROOT, AgeGenderEstimator,   # noqa: E402
                                     align_crop)

OUT = os.path.join(MODELS_DIR, "fairface_res34.onnx")
DATA = os.path.join(ROOT, "data", "UTKFace", "utkface_96.npz")
AGE_CENTERS = [1.0, 6.0, 14.5, 24.5, 34.5, 44.5, 54.5, 64.5, 77.0]   # pontos médios; 70+ -> 77
ZOOMS = (0.75, 0.85, 0.95, 1.05, 1.15, 1.25, 1.35)


class FairFaceWrapper(nn.Module):
    def __init__(self, state_dict):
        super().__init__()
        self.net = torchvision.models.resnet34()
        self.net.fc = nn.Linear(512, 18)
        self.net.load_state_dict(state_dict)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255)
        self.register_buffer("centers", torch.tensor(AGE_CENTERS))

    def forward(self, x):
        out = self.net((x - self.mean) / self.std)
        gender = F.softmax(out[:, 7:9], 1)[:, [1, 0]]         # (M, F) -> (F, M)
        groups = F.softmax(out[:, 9:18], 1)
        return gender, groups @ self.centers[:, None], groups


def export(weights, zoom):
    sd = torch.load(weights, map_location="cpu", weights_only=True)
    model = FairFaceWrapper(sd).eval()
    torch.onnx.export(model, torch.zeros(1, 3, 224, 224), OUT, opset_version=13, input_names=["data"],
                      output_names=["gender_prob", "age", "age_group_prob"],
                      dynamic_axes={k: {0: "N"} for k in ("data", "gender_prob", "age", "age_group_prob")})
    set_zoom(zoom)
    return model


def set_zoom(zoom):
    m = onnx.load(OUT)
    meta = {"crop": "align", "size": "224", "align_zoom": f"{zoom:.2f}",
            "source": "FairFace res34_fair_align_multi_7_20190809 (CC BY 4.0)"}
    del m.metadata_props[:]
    for k, v in meta.items():
        m.metadata_props.add(key=k, value=v)
    onnx.save(m, OUT)


def tune(n=1000):
    """MAE e exatidão do género no split de validação para cada zoom -> devolve o melhor (menor MAE)."""
    import cv2 as cv
    d = np.load(DATA)
    idx = np.where(d["split"] == 1)[0]
    idx = idx[np.random.RandomState(0).permutation(len(idx))[:n]]
    assert not np.isnan(d["kps"][idx]).any(), "faltam landmarks: corre preparar_utkface.py --add-landmarks val test"
    imgs = [cv.imread(os.path.join(ROOT, "data", "UTKFace", str(d["names"][i]))) for i in idx]
    est = AgeGenderEstimator(OUT)
    print(f"\nescolha do zoom do crop alinhado ({n} caras de validação):")
    print(f"{'zoom':>6}{'MAE':>8}{'acc género':>12}")
    res = {}
    for z in ZOOMS:
        crops = np.stack([align_crop(im, d["kps"][i], 224, z) for im, i in zip(imgs, idx)])
        p, a, _ = est.predict_crops(crops)
        res[z] = (np.abs(a - d["age"][idx]).mean(), (p.argmax(1) == d["gender"][idx]).mean())
        print(f"{z:>6.2f}{res[z][0]:>8.2f}{res[z][1]:>12.3f}")
    best = min(res, key=lambda z: res[z][0])
    print(f"-> zoom escolhido: {best}")
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights", help="res34_fair_align_multi_7_20190809.pt")
    ap.add_argument("--zoom", type=float, default=1.15, help="margem do crop (se não usar --tune); 1.15 = escolhido na validação")
    ap.add_argument("--tune", action="store_true", help="escolher o zoom no split de validação")
    a = ap.parse_args()

    model = export(a.weights, a.zoom)
    # verificação torch vs ONNX com a mesma imagem aleatória (RGB NCHW para a torch, BGR NHWC para o ONNX)
    x = (torch.rand(4, 3, 224, 224) * 255).round()
    with torch.no_grad():
        g_t, a_t, _ = model(x)
    bgr = x.flip(1).permute(0, 2, 3, 1).contiguous().byte()
    crops = np.frombuffer(bytes(bgr.flatten().tolist()), np.uint8).reshape(4, 224, 224, 3)
    est = AgeGenderEstimator(OUT)
    g_o, a_o, _ = est.predict_crops(crops)
    print(f"ONNX vs torch: max |dif| género {np.abs(g_o - np.array(g_t.tolist())).max():.2e}, "
          f"idade {np.abs(a_o - np.array(a_t[:, 0].tolist())).max():.2e}")
    t0 = time.perf_counter()
    for _ in range(20):
        est.predict_crops(crops[:1])
    print(f"tempo: {(time.perf_counter() - t0) / 20 * 1000:.1f} ms/cara (CPU)")

    if a.tune:
        set_zoom(tune())
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
