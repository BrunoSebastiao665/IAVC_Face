"""Estimativa de idade aparente e de género (rótulo visual definido pelo dataset).

Modelos suportados (todos ONNX, mesma interface; nomes curtos em MODEL_ALIASES):
  mobilenet     models/genderage.onnx             InsightFace original (MobileNet, 96x96)
                                                  saída "fc1" [N,3] = (logit F, logit M, idade/100)
  mobilenet_ft  models/genderage_ft_ordinal.onnx  a mesma MobileNet com fine-tuning no UTKFace
                (ou genderage_ft_{reg,cls,dldl})  (idade_genero/finetune.py)
  fairface      models/fairface_res34.onnx        FairFace ResNet-34 pré-treinado (224x224, alinhado),
                                                  convertido por idade_genero/converter_fairface.py
Os modelos convertidos/fine-tuned têm as saídas "gender_prob" [N,2] (F, M) e "age" [N,1] (anos);
o FairFace tem ainda "age_group_prob" [N,9] (faixas AGE_GROUPS), porque prevê faixas e não anos.

Pré-processamento — cada ONNX diz qual quer nos metadados (custom_metadata_map):
  crop=box    (por omissão; genderage) quadrado 1.5 x max(w,h) centrado na caixa -> 96x96, sem rotação
  crop=align  (FairFace) semelhança dos 5 landmarks para o template ArcFace com zoom `align_zoom`
              -> 224x224 (aproxima o get_face_chip do dlib usado no treino do FairFace)
A entrada é sempre RGB 0..255 [N,3,S,S]; a normalização está dentro de cada rede.

NOTA ÉTICA — o que a saída significa:
  * idade  = estimativa de idade APARENTE a partir da aparência, não idade cronológica;
  * género = classificação visual nas 2 categorias anotadas nos datasets (male/female),
             i.e. uma estimativa de apresentação de género aparente, NÃO a identidade de género.
  Saídas com confiança baixa são marcadas como "incerto" e não mostram o rótulo.

Uso:
    from idade_genero.age_gender import AgeGenderEstimator
    est = AgeGenderEstimator("mobilenet_ft")       # ou "fairface", "mobilenet", ou um caminho .onnx
    attrs = est.predict(frame, faces)              # faces = saída de FaceDetector.detect()
"""
import os
import sys

import cv2 as cv
import numpy as np
import onnxruntime as ort

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from deteccao.face_alignment import ARCFACE_TEMPLATE, similarity_transform   # noqa: E402

MODELS_DIR = os.path.join(ROOT, "models")
INPUT_SIZE = 96
CROP_FACTOR = 1.5
MODEL_ALIASES = {"mobilenet": "genderage", "mobilenet_ft": "genderage_ft_ordinal", "fairface": "fairface_res34"}

# faixas etárias usadas em todo o módulo (treino por classes, métricas por grupo e intervalo mostrado);
# são as mesmas 9 faixas do FairFace
AGE_GROUPS = [(0, 2), (3, 9), (10, 19), (20, 29), (30, 39), (40, 49), (50, 59), (60, 69), (70, 116)]
GENDER_LABELS = ("F", "M")          # índice 0 = female, 1 = male (convenção do genderage.onnx)


def age_group(age):
    """Índice da faixa etária de uma idade (anos)."""
    for i, (lo, hi) in enumerate(AGE_GROUPS):
        if age <= hi:
            return i
    return len(AGE_GROUPS) - 1


def group_label(i):
    lo, hi = AGE_GROUPS[i]
    return f"{lo}+" if i == len(AGE_GROUPS) - 1 else f"{lo}-{hi}"


def resolve_model(name):
    """'fairface' / 'genderage_ft_reg' / caminho .onnx -> caminho do ficheiro."""
    if name.endswith(".onnx"):
        return name
    return os.path.join(MODELS_DIR, MODEL_ALIASES.get(name, name) + ".onnx")


def crop_matrix(box, size=INPUT_SIZE, factor=CROP_FACTOR):
    """Matriz afim 2x3 que leva o quadrado (factor x max(w,h)) centrado na caixa para size x size."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    s = size / (max(x2 - x1, y2 - y1) * factor)
    return np.float32([[s, 0, size / 2 - cx * s], [0, s, size / 2 - cy * s]])


def crop_face(image, box, size=INPUT_SIZE, factor=CROP_FACTOR):
    """Crop BGR size x size pela caixa (pad a preto fora da imagem)."""
    return cv.warpAffine(image, crop_matrix(box, size, factor), (size, size), borderValue=0)


def align_crop(image, landmarks, size=224, zoom=1.0):
    """Crop BGR size x size alinhado pelos 5 landmarks (template ArcFace; zoom < 1 = mais margem)."""
    template = (ARCFACE_TEMPLATE - 56.0) * zoom * (size / 112.0) + size / 2.0
    M = similarity_transform(np.asarray(landmarks, np.float32).reshape(5, 2), template)
    return cv.warpAffine(image, M, (size, size), flags=cv.INTER_LINEAR, borderValue=0)


def _softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


class AgeGenderEstimator:
    def __init__(self, model="mobilenet_ft", gender_threshold=0.75):
        """model: nome curto (MODEL_ALIASES), nome em models/ sem .onnx, ou caminho.
        gender_threshold: prob. mínima da classe vencedora para mostrar o rótulo de género."""
        path = resolve_model(model)
        so = ort.SessionOptions()
        so.log_severity_level = 3          # o genderage.onnx declara batch 1; esconde o aviso com batch N
        self.session = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.outputs = [o.name for o in self.session.get_outputs()]
        self.original = self.outputs == ["fc1"]
        self.gender_threshold = gender_threshold
        self.name = os.path.splitext(os.path.basename(path))[0]
        meta = self.session.get_modelmeta().custom_metadata_map
        self.crop = meta.get("crop", "box")
        self.size = int(meta.get("size", INPUT_SIZE))
        self.box_factor = float(meta.get("box_factor", CROP_FACTOR))
        self.align_zoom = float(meta.get("align_zoom", 1.0))

    def make_crop(self, image, face):
        """Crop no formato que este modelo espera (BGR uint8 size x size)."""
        if self.crop == "align" and len(face.get("landmarks", [])) == 5:
            return align_crop(image, face["landmarks"], self.size, self.align_zoom)
        return crop_face(image, face["box"], self.size, self.box_factor)

    def predict_crops(self, crops_bgr):
        """crops_bgr: uint8 [N,S,S,3] BGR -> (prob género [N,2] (F, M), idade [N] em anos,
        prob. das faixas [N,9] ou None se o modelo não prevê faixas)."""
        crops_bgr = np.asarray(crops_bgr)
        if len(crops_bgr) == 0:
            return np.zeros((0, 2), np.float32), np.zeros(0, np.float32), None
        blob = crops_bgr[..., ::-1].transpose(0, 3, 1, 2).astype(np.float32)      # BGR -> RGB, NCHW
        out = self.session.run(None, {self.input_name: blob})
        if self.original:
            return _softmax(out[0][:, :2]), out[0][:, 2] * 100, None
        res = dict(zip(self.outputs, out))
        return res["gender_prob"], res["age"][:, 0], res.get("age_group_prob")

    def predict(self, frame, faces):
        """-> lista (uma por cara) de dicts com idade, intervalo, género, confiança e 'uncertain'."""
        if not faces:
            return []
        crops = np.stack([self.make_crop(frame, f) for f in faces])
        probs, ages, groups = self.predict_crops(crops)
        out = []
        for k, (p, a) in enumerate(zip(probs, ages)):
            g = int(np.argmax(p))
            a = float(np.clip(a, 0, 116))
            grp = int(np.argmax(groups[k])) if groups is not None else age_group(round(a))
            out.append({
                "age": a,
                "age_range": group_label(grp),
                "gender": GENDER_LABELS[g],
                "gender_conf": float(p[g]),
                "gender_uncertain": bool(p[g] < self.gender_threshold),
            })
        return out

    @staticmethod
    def label(attr):
        """Texto curto para mostrar: '~32 (30-39) M 0.93'  ou  '~32 (30-39) ?' se incerto."""
        g = "?" if attr["gender_uncertain"] else f"{attr['gender']} {attr['gender_conf']:.2f}"
        return f"~{attr['age']:.0f} ({attr['age_range']}) {g}"
