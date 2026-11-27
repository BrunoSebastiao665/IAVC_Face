"""Modelos que se podem escolher na app e combinações prontas (usado por main.py e benchmark_combinacoes.py)."""
import os

from expressao.comum import MODELS as EXPR_MODELS
from expressao.expression import ExpressionEstimator, app_config
from idade_genero.age_gender import AgeGenderEstimator, resolve_model

OFF = "off"
# nome -> descrição curta (menu). Os números são do teste do UTKFace e do AffectNet/BTFER (PROGRESSO_E_RESULTADOS.md)
AGE_GENDER = {
    "fairface": "FairFace ResNet-34, pré-treinado   MAE 5.28  F1 género 0.942  (mais exato, lento)",
    "mobilenet_ft": "MobileNet fine-tuned (ordinal)     MAE 6.03  F1 género 0.918  (muito rápida)",
    "genderage_ft_reg": "MobileNet fine-tuned (regressão)   MAE 6.29  F1 género 0.921",
    "genderage_ft_cls": "MobileNet fine-tuned (classes)     MAE 6.26  F1 género 0.924",
    "genderage_ft_dldl": "MobileNet fine-tuned (DLDL)        MAE 6.22  F1 género 0.918",
    "mobilenet": "MobileNet original                 MAE 9.45  F1 género 0.901",
    OFF: "desligado",
}
EXPRESSION = {
    "b2": "HSEmotion EfficientNet-B2          bal. acc. AffectNet 0.631 / BTFER 0.745",
    "b0": "HSEmotion EfficientNet-B0          bal. acc. AffectNet 0.615 / BTFER 0.749  (mais rápido)",
    OFF: "desligada",
}
# combinações prontas: nome -> (idade/género, expressão, quando usar); ver benchmark_combinacoes.py
COMBOS = {
    "preciso": ("fairface", "b2", "melhor exatidão; até 2 caras (9.4 / 5.6 atualizações/s com 1 / 2)"),
    "equilibrado": ("fairface", "b0", "como o preciso, com mais folga; até 2 caras (11.1 / 6.7)"),
    "leve": ("mobilenet_ft", "b2", "várias caras com a expressão b2 (18.0 / 12.7; 7.7 com 4 caras)"),
    "rapido": ("mobilenet_ft", "b0", "o mais rápido com tudo ligado (23.0 / 18.6; 12.9 com 4 caras)"),
}


class ModelPool:
    """Carrega cada modelo só na primeira vez que é pedido e guarda-o."""

    def __init__(self):
        self.ag, self.ex = {OFF: None}, {OFF: None}

    def age_gender(self, name):
        if name not in self.ag:
            path = resolve_model(name)
            if not os.path.exists(path):
                raise FileNotFoundError(f"não existe {path}")
            self.ag[name] = AgeGenderEstimator(path)
        return self.ag[name]

    def expression(self, name):
        if name == "auto":
            name = app_config()["modelo"]
        if name not in self.ex:
            if name not in EXPR_MODELS:
                raise FileNotFoundError(f"modelo de expressão desconhecido: {name}")
            self.ex[name] = ExpressionEstimator(name)
        return self.ex[name]


def ask(title, options, default):
    """Menu numerado no terminal -> chave escolhida (Enter = default)."""
    keys = list(options)
    print(f"\n{title}")
    for i, k in enumerate(keys, 1):
        print(f"  {i}. {k:<18} {options[k]}{'   [Enter]' if k == default else ''}")
    while True:
        r = input("> ").strip().lower()
        if not r:
            return default
        if r in keys:
            return r
        if r.isdigit() and 1 <= int(r) <= len(keys):
            return keys[int(r) - 1]
        print(f"  escolhe 1-{len(keys)} ou um nome")


def menu():
    """-> (idade/género, expressão)"""
    opts = {k: f"{a} + {e}  —  {d}" for k, (a, e, d) in COMBOS.items()}
    opts["manual"] = "escolher cada modelo"
    c = ask("Combinação de modelos", opts, "preciso")
    if c != "manual":
        return COMBOS[c][:2]
    return ask("Idade e género", AGE_GENDER, "fairface"), ask("Expressão facial", EXPRESSION, "b2")
