"""Constantes e métricas da expressão facial. Só numpy.

NOTA — o que a saída significa: cada rótulo é uma estimativa da CONFIGURAÇÃO FACIAL VISÍVEL, segundo as
categorias anotadas no AffectNet. Não é evidência de que a pessoa esteja a sentir essa emoção.
"""
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")
RES = os.path.join(ROOT, "expressao", "resultados")
AFF_DIR = os.path.join(ROOT, "data", "AffectNet", "AffectNetCustom")
AFF_NPZ = os.path.join(ROOT, "data", "AffectNet", "affectnet_avaliacao.npz")   # validação oficial, 112x112 BGR
BTFER_DIR = os.path.join(ROOT, "data", "BTFER", "Batch_Ready 7")
BTFER_NPZ = os.path.join(ROOT, "data", "BTFER", "btfer.npz")

# ordem usada em todo o projeto; índice 0..7 = nome da pasta no AffectNet do Kaggle
# (confirmado em preparar_dados.py: a previsão mais frequente do HSEmotion em cada pasta é esta classe)
CLASSES = ("Anger", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral", "Contempt")
K7 = 7                                  # as 7 primeiras são as comuns ao BTFER (todas menos Contempt)
BTFER_FOLDERS = {"Angry": 0, "Disgust": 1, "Fear": 2, "Happy": 3, "Sad": 4, "Surprise": 5, "Neutral": 6}
MODELS = {                              # nome curto -> (ficheiro em models/, nome bonito)
    "b0": ("expr_hsemotion_b0.onnx", "HSEmotion EfficientNet-B0"),
    "b2": ("expr_hsemotion_b2.onnx", "HSEmotion EfficientNet-B2"),
}


def to7(probs):
    """Probabilidades de 8 classes -> 7: ignora Contempt e renormaliza."""
    p = probs[:, :K7]
    return p / np.maximum(p.sum(1, keepdims=True), 1e-12)


def confusion(y, p, k):
    cm = np.zeros((k, k), int)
    np.add.at(cm, (y, p), 1)            # [verdadeiro, previsto]
    return cm


def metrics(y, p, k):
    """-> dict: acc, bal_acc, P/R/F1 macro, e por classe. Macro = média simples das k classes."""
    cm = confusion(y, p, k)
    tp = np.diag(cm).astype(float)
    P = tp / np.maximum(cm.sum(0), 1)
    R = tp / np.maximum(cm.sum(1), 1)
    F1 = 2 * P * R / np.maximum(P + R, 1e-9)
    return dict(acc=tp.sum() / cm.sum(), bal_acc=R.mean(), P_macro=P.mean(), R_macro=R.mean(),
                F1_macro=F1.mean(), P=P, R=R, F1=F1, cm=cm)


def bootstrap(fn, n, reps=1000, seed=0):
    """IC 95% de fn(idx) reamostrando n exemplos com reposição."""
    rng = np.random.RandomState(seed)
    v = [fn(rng.randint(0, n, n)) for _ in range(reps)]
    return np.percentile(v, [2.5, 97.5])
