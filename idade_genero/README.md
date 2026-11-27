# Fase 3 — Idade aparente + Género

Modelo base: `genderage.onnx` do InsightFace (pack `buffalo_s`), uma MobileNet com 2 ramos
(género: F/M · idade/100), com fine-tuning no **UTKFace**. Comparado com um segundo modelo só
pré-treinado, o **FairFace ResNet-34** (224×224, alinhado). Resultados: `resultados/idade_genero.md`.

## Ficheiros

| Ficheiro | Papel |
|---|---|
| `age_gender.py` | Módulo de inferência usado pela app: crop 96×96 da caixa (MobileNet) ou 224×224 alinhado (FairFace), conforme os metadados do ONNX → idade, faixa etária, género, confiança, "incerto" |
| `preparar_utkface.py` | UTKFace in-the-wild → deteção SCRFD → crops 96×96 + rótulos + split 70/10/20 → `data/UTKFace/utkface_96.npz` |
| `modelo_torch.py` | Port do `genderage.onnx` para PyTorch (pesos carregados do ONNX) + 4 formulações da idade + export ONNX |
| `finetune.py` | Fine-tuning multi-tarefa (idade + género) → `models/genderage_ft_<variante>.onnx` |
| `converter_fairface.py` | Pesos FairFace (PyTorch) → `models/fairface_res34.onnx`; `--tune` escolhe a margem do crop na validação (`fairface_zoom_log.txt`) |
| `avaliar.py` | Métricas no split de teste → `resultados/idade_genero.md` + figuras |

## Formulações da idade estudadas

| Variante | Cabeça | Perda | Idade prevista |
|---|---|---|---|
| `reg` | FC → 1 | L1 | saída × 100 |
| `cls` | FC → 9 faixas | cross-entropy | Σ p(faixa) · idade média da faixa |
| `dldl` (distribuição) | FC → 117 (0..116) | KL p/ gaussiana σ=2 + L1 | Σ p(k) · k |
| `ordinal` (CORAL) | 116 × "idade > k" | BCE | Σ P(idade > k) |

## Ordem de execução (a partir da raiz P3, com o venv ativo)

```bash
source faceregognition/bin/activate
python idade_genero/preparar_utkface.py          # ~12 min
python idade_genero/finetune.py --variant all    # 4 variantes x 12 épocas, ~55 min em CPU
python idade_genero/converter_fairface.py PESOS.pt --tune   # ~15 min (só para refazer o ONNX do FairFace)
python idade_genero/avaliar.py                   # tabelas + figuras em idade_genero/resultados/
python main.py                                   # webcam, FairFace (por omissão)
python main.py --age-gender mobilenet_ft         # webcam, MobileNet FT ordinal; tecla 'g' troca de modelo; ver também --menu e --combo
```

## Dados

- UTKFace in-the-wild (https://susanqq.github.io/UTKFace/) em `data/UTKFace/part1..3`.
  (O link "Aligned & Cropped" do site está morto; o in-the-wild permite usar o mesmo detetor da app.)
- Nome do ficheiro: `idade_género_etnia_data.jpg`; género UTK 0 = male, 1 = female
  (convertido para a convenção do genderage: 0 = F, 1 = M).

## Notas técnicas

- `torch==2.2.2` é a última versão com builds para Mac x86_64 e foi compilada para numpy 1.x;
  o venv tem numpy 2 (exigido pelo OpenCV 5). Por isso a mensagem "A module that was compiled
  using NumPy 1.x…" aparece ao importar a torch — é inofensiva: os dados passam para tensores com
  `torch.frombuffer`, sem usar a ponte numpy↔torch.
- A aplicação não precisa de torch: os modelos fine-tuned são exportados para ONNX.
