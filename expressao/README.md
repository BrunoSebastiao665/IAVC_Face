# Fase 4 — Expressão facial

Comparação de dois modelos **pré-treinados** (sem treino nosso): HSEmotion EfficientNet-**B0** e **B2**.
Resultados completos: [resultados/avaliacao.md](resultados/avaliacao.md) · Dados: [resultados/auditoria_dados.md](resultados/auditoria_dados.md)

> Um rótulo é uma estimativa da **configuração facial visível**, não evidência da emoção que a pessoa sente.

| Ficheiro | Papel |
|---|---|
| `comum.py` | Classes, caminhos e métricas (accuracy, balanced accuracy, P/R/F1 macro, bootstrap) |
| `obter_hsemotion.py` | Descarrega os ONNX dos autores, confirma o sha256 e junta metadados (tamanho, normalização, classes) |
| `preparar_dados.py` | Audita AffectNet/BTFER: imagens repetidas entre partições e entre datasets; alinha o BTFER |
| `expression.py` | Inferência usada pela app: alinhar → prever → 7 classes → rótulo, confiança, top-k, incerto |
| `avaliar.py` | Intra (AffectNet) + cross (BTFER) + escolha do modelo e do limiar da app |

```bash
source faceregognition/bin/activate
python expressao/obter_hsemotion.py
python expressao/preparar_dados.py      # ~3 min
python expressao/avaliar.py             # ~3 min
python main.py --expression b2          # ou b0; ver também --menu e --combo; teclas: e = trocar, t = top-3
```

Dados (em `data/`): AffectNet alinhado (kaggle `yakhyokhuja/affectnetaligned`) e BTFER (kaggle `gipinze/btfer7`).
Licença dos modelos: Apache-2.0 (Savchenko, EmotiEffLib/HSEmotion) — citar no relatório.
