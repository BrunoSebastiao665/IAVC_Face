"""Detetores de caras com uma interface comum.

Modelos disponíveis (ver DETECTORS):
  - "retinaface"        RetinaFace ResNet50 (serengil/retinaface) — preciso mas pesado.
  - "retinaface_mobile" RetinaFace MobileNet0.25 (yakhyo/retinaface-pytorch, ONNX) — leve e rápido.
  - "scrfd"             SCRFD-500MF (deepinsight/insightface, ONNX) — leve e rápido.

Todos devolvem a mesma estrutura em detect():
    [{"score": float, "box": (x1, y1, x2, y2), "landmarks": [(x, y) * 5]}, ...]
com os 5 pontos: olho, olho, nariz, canto da boca, canto da boca.
"""
import os
from itertools import product

import cv2 as cv
import numpy as np
import onnxruntime as ort

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def _onnx_session(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Modelo não encontrado: {path}")
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


class FaceDetector:
    """Classe base: cada detetor implementa _detect(frame) no frame já redimensionado."""

    name = "base"

    def __init__(self, threshold=0.9, scale=1.0):
        self.threshold = threshold
        self.scale = scale  # fator de redução do frame antes da deteção

    def _detect(self, img):
        raise NotImplementedError

    def detect(self, frame):
        small = cv.resize(frame, None, fx=self.scale, fy=self.scale) if self.scale != 1 else frame
        faces = self._detect(small)
        s = 1 / self.scale
        for f in faces:
            f["box"] = tuple(int(v * s) for v in f["box"])
            f["landmarks"] = [(int(x * s), int(y * s)) for x, y in f["landmarks"]]
        return faces

    @staticmethod
    def draw(frame, faces, color=(0, 255, 0)):
        """Desenha caixas, score e landmarks sobre o frame (in-place)."""
        for face in faces:
            x1, y1, x2, y2 = face["box"]
            cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv.putText(frame, f"{face['score']:.2f}", (x1, max(y1 - 8, 0)),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            for point in face["landmarks"]:
                cv.circle(frame, point, 3, (0, 0, 255), -1)
        return frame


# --------------------------------------------------------------------------- #
# RetinaFace ResNet50 (biblioteca serengil/retinaface)
# --------------------------------------------------------------------------- #
class RetinaFaceResNet(FaceDetector):
    name = "retinaface"

    def __init__(self, threshold=0.9, scale=0.5):
        super().__init__(threshold, scale)
        from retinaface import RetinaFace  # import local: só carrega se for usado
        self._rf = RetinaFace
        self.model = RetinaFace.build_model()

    def _detect(self, img):
        resp = self._rf.detect_faces(img, threshold=self.threshold, model=self.model,
                                     allow_upscaling=False)
        if not isinstance(resp, dict):  # sem caras a biblioteca pode devolver uma tupla
            return []
        return [{
            "score": float(f["score"]),
            "box": tuple(f["facial_area"]),
            "landmarks": [tuple(f["landmarks"][k]) for k in
                          ("right_eye", "left_eye", "nose", "mouth_right", "mouth_left")],
        } for f in resp.values()]


# --------------------------------------------------------------------------- #
# RetinaFace MobileNet0.25 (ONNX de yakhyo/retinaface-pytorch)
# --------------------------------------------------------------------------- #
class RetinaFaceMobileNet(FaceDetector):
    name = "retinaface_mobile"
    MIN_SIZES = [[16, 32], [64, 128], [256, 512]]
    STEPS = [8, 16, 32]
    VARIANCE = (0.1, 0.2)
    MEAN = np.array([104, 117, 123], dtype=np.float32)  # BGR
    NMS_THRESHOLD = 0.4

    def __init__(self, threshold=0.9, scale=1.0,
                 model_path=os.path.join(MODELS_DIR, "retinaface_mv1_0.25.onnx")):
        super().__init__(threshold, scale)
        self.session = _onnx_session(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self._priors_cache = {}

    def _priors(self, h, w):
        """Anchors (cx, cy, w, h) normalizados, iguais ao PriorBox do treino."""
        if (h, w) not in self._priors_cache:
            anchors = []
            for step, min_sizes in zip(self.STEPS, self.MIN_SIZES):
                fh, fw = int(np.ceil(h / step)), int(np.ceil(w / step))
                for i, j in product(range(fh), range(fw)):
                    for m in min_sizes:
                        anchors.append([(j + 0.5) * step / w, (i + 0.5) * step / h, m / w, m / h])
            self._priors_cache[(h, w)] = np.array(anchors, dtype=np.float32)
        return self._priors_cache[(h, w)]

    def _detect(self, img):
        h, w = img.shape[:2]
        blob = (img.astype(np.float32) - self.MEAN).transpose(2, 0, 1)[None]
        loc, conf, landms = self.session.run(None, {self.input_name: blob})
        loc, scores, landms = loc[0], conf[0][:, 1], landms[0]

        keep = scores > self.threshold
        if not keep.any():
            return []
        priors, loc, scores, landms = self._priors(h, w)[keep], loc[keep], scores[keep], landms[keep]

        v0, v1 = self.VARIANCE
        centers = priors[:, :2] + loc[:, :2] * v0 * priors[:, 2:]
        sizes = priors[:, 2:] * np.exp(loc[:, 2:] * v1)
        boxes = np.hstack([centers - sizes / 2, centers + sizes / 2]) * [w, h, w, h]
        pts = landms.reshape(-1, 5, 2)
        pts = (priors[:, None, :2] + pts * v0 * priors[:, None, 2:]) * [w, h]

        xywh = np.hstack([boxes[:, :2], boxes[:, 2:] - boxes[:, :2]]).tolist()
        idx = cv.dnn.NMSBoxes(xywh, scores.tolist(), self.threshold, self.NMS_THRESHOLD)
        return [{
            "score": float(scores[i]),
            "box": tuple(boxes[i]),
            "landmarks": [tuple(p) for p in pts[i]],
        } for i in np.array(idx).flatten()]


# --------------------------------------------------------------------------- #
# SCRFD (insightface)
# --------------------------------------------------------------------------- #
class SCRFDDetector(FaceDetector):
    name = "scrfd"

    def __init__(self, threshold=0.5, scale=1.0, det_size=(640, 640),
                 model_path=os.path.join(MODELS_DIR, "det_500m.onnx")):
        """det_size: resolução de entrada da rede (a imagem é ajustada com letterbox)."""
        super().__init__(threshold, scale)
        from insightface.model_zoo.scrfd import SCRFD
        self.det_size = det_size
        self.model = SCRFD(model_file=model_path, session=_onnx_session(model_path))
        self.model.prepare(-1, det_thresh=threshold, input_size=det_size)

    def _detect(self, img):
        bboxes, kpss = self.model.detect(img, input_size=self.det_size)
        return [{
            "score": float(b[4]),
            "box": tuple(b[:4]),
            "landmarks": [tuple(p) for p in kpss[i]] if kpss is not None else [],
        } for i, b in enumerate(bboxes)]


DETECTORS = {
    RetinaFaceResNet.name: RetinaFaceResNet,
    RetinaFaceMobileNet.name: RetinaFaceMobileNet,
    SCRFDDetector.name: SCRFDDetector,
}


def create_detector(name, **kwargs):
    if name not in DETECTORS:
        raise ValueError(f"Modelo desconhecido '{name}'. Opções: {list(DETECTORS)}")
    return DETECTORS[name](**kwargs)
