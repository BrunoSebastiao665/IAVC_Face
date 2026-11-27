"""Port do genderage.onnx (InsightFace) para PyTorch + cabeças de idade alternativas.

Arquitetura original (lida do ONNX): MobileNet-v1 pequena
    tronco partilhado conv_1 ... conv_12  (96x96 -> 6x6x128)
    ramo t0: dw/pw conv_13, conv_14 (256 canais) -> GAP -> FC 256->2   (género: F, M)
    ramo t1: dw/pw conv_13, conv_14 (256 canais) -> GAP -> FC 256->1   (idade / 100)
Os pesos são carregados do ONNX e o modelo é depois ajustado (fine-tuning) no UTKFace.

Formulações de idade (argumento `variant`), todas sobre o ramo t1:
  reg      regressão direta: FC 256->1 (inicializada com a FC original), perda L1 em anos.
  cls      classificação em faixas etárias (AGE_GROUPS): FC 256->G, cross-entropy;
           idade = sum_g p_g * idade média da faixa g no treino.
  dldl     distribuição (DLDL / DEX): FC 256->117 (0..116 anos), alvo = gaussiana centrada na
           idade (sigma=2), perda KL + L1 da esperança; idade = sum_k p_k * k.
  ordinal  ordinal (CORAL): K-1 classificadores binários "idade > k" com pesos partilhados e
           biases próprios, perda BCE; idade = sum_k P(idade > k).

Nota: a torch 2.2.2 (última com builds para Mac x86_64) foi compilada para numpy 1.x e o venv tem
numpy 2 (exigido pelo OpenCV 5), por isso os arrays passam para tensores com torch.frombuffer.
"""
import numpy as np
import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
from onnx import numpy_helper

MAX_AGE = 116
VARIANTS = ("reg", "cls", "dldl", "ordinal")


def np2t(a, dtype=np.float32):
    a = np.ascontiguousarray(a, dtype=dtype)
    t = torch.frombuffer(bytearray(a.tobytes()), dtype={np.float32: torch.float32, np.uint8: torch.uint8,
                                                         np.int64: torch.int64}[np.dtype(dtype).type])
    return t.reshape(a.shape)


class ConvBN(nn.Sequential):
    def __init__(self, cin, cout, k, stride, groups):
        super().__init__(nn.Conv2d(cin, cout, k, stride, k // 2, groups=groups, bias=False),
                         nn.BatchNorm2d(cout, eps=1e-3), nn.ReLU(inplace=True))


def _parse_onnx(path):
    """-> (pesos {nome: np.array}, convs [(prefixo, cin, cout, k, stride, groups)] por ordem)."""
    m = onnx.load(path)
    w = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
    convs = []
    for n in m.graph.node:
        if n.op_type != "Conv":
            continue
        at = {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}
        cout, cin_g, k, _ = w[n.input[1]].shape
        convs.append((n.output[0][:-len("_conv2d")], cin_g * at["group"], cout, k, at["strides"][0], at["group"]))
    return w, convs


class GenderAgeNet(nn.Module):
    def __init__(self, onnx_path, variant="reg", group_centers=None, n_groups=9):
        super().__init__()
        assert variant in VARIANTS
        self.variant = variant
        w, convs = _parse_onnx(onnx_path)
        mk = lambda sel: [(p, ConvBN(*c)) for p, *c in convs if sel(p)]
        trunk, b_g, b_a = mk(lambda p: "_t" not in p), mk(lambda p: "_t0" in p), mk(lambda p: "_t1" in p)
        self.trunk = nn.Sequential(*[m for _, m in trunk])
        self.branch_gender = nn.Sequential(*[m for _, m in b_g])
        self.branch_age = nn.Sequential(*[m for _, m in b_a])
        for prefix, mod in trunk + b_g + b_a:                     # pesos do ONNX
            mod[0].weight.data = np2t(w[prefix + "_conv2d_weight"])
            bn = mod[1]
            bn.weight.data, bn.bias.data = np2t(w[prefix + "_batchnorm_gamma"]), np2t(w[prefix + "_batchnorm_beta"])
            bn.running_mean.data = np2t(w[prefix + "_batchnorm_moving_mean"])
            bn.running_var.data = np2t(w[prefix + "_batchnorm_moving_var"])

        self.fc_gender = nn.Linear(256, 2)
        self.fc_gender.weight.data, self.fc_gender.bias.data = np2t(w["fullyconnected0_weight"]), np2t(w["fullyconnected0_bias"])

        if variant == "reg":
            self.fc_age = nn.Linear(256, 1)
            self.fc_age.weight.data, self.fc_age.bias.data = np2t(w["fullyconnected1_weight"]), np2t(w["fullyconnected1_bias"])
        elif variant == "cls":
            self.fc_age = nn.Linear(256, n_groups)
            centers = group_centers if group_centers is not None else np.linspace(1, 80, n_groups)
            self.register_buffer("centers", np2t(centers))
        elif variant == "dldl":
            self.fc_age = nn.Linear(256, MAX_AGE + 1)
            self.register_buffer("ages", torch.arange(MAX_AGE + 1, dtype=torch.float32))
        else:                                                      # CORAL
            self.fc_age = nn.Linear(256, 1, bias=False)
            self.ord_bias = nn.Parameter(torch.linspace(2, -2, MAX_AGE))   # thresholds k = 0..115
        self.register_buffer("mean", torch.tensor(127.5))
        self.register_buffer("scale", torch.tensor(0.0078125))

    def forward(self, x):
        """x: float RGB 0..255 [N,3,96,96] -> (logits género [N,2], saída bruta da cabeça de idade)."""
        f = self.trunk((x - self.mean) * self.scale)
        g = self.fc_gender(self.branch_gender(f).mean((2, 3)))
        a = self.fc_age(self.branch_age(f).mean((2, 3)))
        if self.variant == "ordinal":
            a = a + self.ord_bias
        return g, a

    def age_from_output(self, a):
        """Saída bruta -> idade estimada em anos [N]."""
        if self.variant == "reg":
            return a[:, 0] * 100
        if self.variant == "cls":
            return F.softmax(a, 1) @ self.centers
        if self.variant == "dldl":
            return F.softmax(a, 1) @ self.ages
        return torch.sigmoid(a).sum(1)

    def age_loss(self, a, age, group):
        """age: float [N] anos; group: long [N] índice da faixa."""
        if self.variant == "reg":
            return F.l1_loss(a[:, 0] * 100, age) / 10
        if self.variant == "cls":
            return F.cross_entropy(a, group)
        if self.variant == "dldl":
            target = torch.exp(-0.5 * ((self.ages[None] - age[:, None]) / 2.0) ** 2)
            target = target / target.sum(1, keepdim=True)
            kl = F.kl_div(F.log_softmax(a, 1), target, reduction="batchmean")
            return kl + F.l1_loss(self.age_from_output(a), age) / 10
        levels = (age[:, None] > torch.arange(MAX_AGE, dtype=torch.float32)[None]).float()
        return F.binary_cross_entropy_with_logits(a, levels, reduction="sum") / len(a) / 10


class ExportWrapper(nn.Module):
    """Versão para a aplicação: saídas "gender_prob" [N,2] (F, M) e "age" [N,1] (anos)."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        g, a = self.net(x)
        return F.softmax(g, 1), self.net.age_from_output(a)[:, None]


def export_onnx(net, path):
    net.eval()
    torch.onnx.export(ExportWrapper(net), torch.zeros(1, 3, 96, 96), path, opset_version=13,
                      input_names=["data"], output_names=["gender_prob", "age"],
                      dynamic_axes={"data": {0: "N"}, "gender_prob": {0: "N"}, "age": {0: "N"}})
