"""Classificação da expressão facial visível (inferência, só onnxruntime; é o único ficheiro que a app usa).

Um rótulo é uma estimativa da configuração facial visível nas categorias do AffectNet, NÃO da emoção
que a pessoa sente.

Modelos: HSEmotion EfficientNet-B0 (enet_b0_8_va_mtl) e B2 (enet_b2_8), pré-treinados no AffectNet pelos autores,
sem treino nosso. Cada ONNX traz nos metadados o pré-processamento dos autores: size, mean, std (escala 0..1, RGB) e classes.
Crop: cara alinhada pelos 5 landmarks do SCRFD para a geometria do AffectNet alinhado (TEMPLATE112, mediana
medida no train por preparar_dados.py), para a app ver o mesmo tipo de imagem que o treino/teste.

Uso:
    from expressao.expression import ExpressionEstimator
    est = ExpressionEstimator()
    attrs = est.predict(frame, faces)        # [{expr, conf, uncertain, top}]
"""
import json
import os
import sys

import cv2 as cv
import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from expressao.comum import CLASSES, K7, MODELS, MODELS_DIR, RES, to7     # noqa: E402
from deteccao.face_alignment import similarity_transform                           # noqa: E402

# olho esq., olho dir., nariz, boca esq., boca dir. (imagem 112x112)
TEMPLATE112 = np.array([[34.0, 41.0], [79.0, 42.0], [56.0, 67.0], [38.0, 89.0], [73.0, 90.0]], np.float32)
CONFIG = os.path.join(RES, "config_app.json")


def align_face_expr(image, landmarks, size=112):
    """Crop BGR size x size alinhado pelos landmarks para a geometria do AffectNet."""
    M = similarity_transform(np.asarray(landmarks, np.float32).reshape(5, 2), TEMPLATE112 * (size / 112.0))
    return cv.warpAffine(image, M, (size, size), flags=cv.INTER_LINEAR, borderValue=0)


def app_config():
    """Modelo e limiares escolhidos no AffectNet (avaliar.py), sem usar o BTFER."""
    try:
        with open(CONFIG) as fp:
            return json.load(fp)
    except (OSError, ValueError):
        return {"modelo": "b0", "limiares": {}}


class ExpressionEstimator:
    def __init__(self, model=None, threshold=None, n_classes=K7):
        """model: 'b0' ou 'b2' (None = o escolhido em config_app.json).
        threshold: prob. mínima da melhor classe para mostrar o rótulo. n_classes: 7 (sem Contempt) ou 8."""
        cfg = app_config()
        self.name = model or cfg["modelo"]
        path = os.path.join(MODELS_DIR, MODELS[self.name][0])
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.session = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        meta = self.session.get_modelmeta().custom_metadata_map
        self.size = int(meta["size"])
        self.mean = np.array(json.loads(meta["mean"]), np.float32)
        self.std = np.array(json.loads(meta["std"]), np.float32)
        order = meta["classes"].split(",")
        self.perm = [order.index(c) for c in CLASSES]      # ordem do modelo -> ordem do projeto
        self.n_classes = n_classes
        self.threshold = cfg["limiares"].get(self.name, 0.5) if threshold is None else threshold

    def preprocess(self, crops_bgr):
        x = np.stack([cv.resize(c, (self.size, self.size), interpolation=cv.INTER_LINEAR) for c in crops_bgr])
        x = (x[..., ::-1].astype(np.float32) / 255.0 - self.mean) / self.std
        return x.transpose(0, 3, 1, 2)

    def predict_crops(self, crops_bgr):
        """[N,h,w,3] BGR uint8 -> probabilidades [N,8] na ordem CLASSES."""
        if len(crops_bgr) == 0:
            return np.zeros((0, len(CLASSES)), np.float32)
        z = self.session.run(None, {self.input_name: self.preprocess(crops_bgr)})[0][:, self.perm]
        z = z - z.max(1, keepdims=True)                     # (o HSEmotion *_va_mtl tem +2 saídas; perm ignora-as)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)

    def predict(self, frame, faces):
        """-> uma entrada por cara: expr, conf, uncertain, top = [(classe, prob)] por ordem decrescente."""
        out = [None] * len(faces)
        idx = [i for i, f in enumerate(faces) if len(f.get("landmarks", [])) == 5]
        if idx:
            probs = self.predict_crops([align_face_expr(frame, faces[i]["landmarks"]) for i in idx])
            probs = to7(probs) if self.n_classes == K7 else probs
            for i, p in zip(idx, probs):
                o = np.argsort(-p)
                out[i] = {"expr": CLASSES[o[0]], "conf": float(p[o[0]]),
                          "uncertain": bool(p[o[0]] < self.threshold),
                          "top": [(CLASSES[k], float(p[k])) for k in o]}
        return out

    @staticmethod
    def label(attr, k=1):
        """'Happy 0.91' | '? (Sad 0.35)' se incerto; k > 1 mostra as k classes mais prováveis."""
        if attr is None:
            return ""
        if k > 1:
            return ("? " if attr["uncertain"] else "") + "  ".join(f"{c} {p:.2f}" for c, p in attr["top"][:k])
        return f"? ({attr['expr']} {attr['conf']:.2f})" if attr["uncertain"] else f"{attr['expr']} {attr['conf']:.2f}"
