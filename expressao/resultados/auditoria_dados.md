# Auditoria dos dados de expressão

## AffectNet alinhado (Kaggle)

| Partição | total | Anger | Disgust | Fear | Happy | Sad | Surprise | Neutral | Contempt |
|---|---|---|---|---|---|---|---|---|---|
| train (não usado: o modelo já vem treinado) | 37533 | 4999 | 3801 | 4999 | 4996 | 4998 | 4996 | 4999 | 3745 |
| val | 798 | 100 | 99 | 100 | 100 | 100 | 100 | 99 | 100 |
| test | 3199 | 400 | 400 | 400 | 400 | 400 | 399 | 400 | 400 |
| **val + test = validação oficial** | 3997 | 500 | 499 | 500 | 500 | 500 | 499 | 499 | 500 |
| **validação oficial sem fuga (avaliação principal)** | 3445 | 427 | 391 | 371 | 473 | 424 | 447 | 460 | 452 |

- imagens de val/test bit a bit iguais a uma do train: **80**
- imagens de val/test com uma quase igual no train (correlação ≥ 0.92): **552** (367 com o mesmo rótulo); ficam fora da avaliação principal
- o train do Kaggle é só uma parte do train oficial com que os autores treinaram, por isso podem existir outras repetições que não conseguimos detetar
- landmarks medianos no train (1494 caras): [[33.0, 41.0], [79.0, 42.0], [56.0, 67.0], [38.0, 89.0], [73.0, 90.0]] (TEMPLATE112 usado: [[34.0, 41.0], [79.0, 42.0], [56.0, 67.0], [38.0, 89.0], [73.0, 90.0]])

## BTFER

| | total | Anger | Disgust | Fear | Happy | Sad | Surprise | Neutral |
|---|---|---|---|---|---|---|---|---|
| original | 2100 | 300 | 300 | 300 | 300 | 300 | 300 | 300 |
| **usado** | 1847 | 263 | 250 | 243 | 291 | 267 | 283 | 250 |

- cópias bit a bit dentro do BTFER: 179
- retiradas por serem (quase) iguais a uma imagem do AffectNet (correlação ≥ 0.9): 38 (correlação máxima 0.960; as de cima foram vistas a olho e incluem fotos realmente partilhadas)
- repetições internas retiradas (correlação ≥ 0.98): 203 com o mesmo rótulo, 12 em grupos com rótulos contraditórios
- imagens sem cara detetada pelo SCRFD: 0

