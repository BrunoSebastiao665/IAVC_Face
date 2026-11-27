# Comparação de detetores — WIDER FACE validação

- 100 imagens (amostra aleatória, seed 0), 563 caras válidas (GT `invalid` ignoradas)
- Matching IoU ≥ 0.5; AP com o detetor a threshold 0.02; restantes métricas no **threshold da aplicação**: SCRFD-500MF 0.5, RetinaFace MobileNet0.25 0.9, RetinaFace R50 0.9
- Só CPU (Apple M1, Python x86_64 via Rosetta)

## 1. Exatidão e tempo

| | SCRFD-500MF | RetinaFace MobileNet0.25 | RetinaFace R50 |
|---|---|---|---|
| **AP** (IoU 0.5) | **0.762** | **0.761** | **0.720** |
| Precisão @ threshold app | 0.978 | 0.985 | 0.972 |
| Recall @ threshold app | 0.625 | 0.567 | 0.552 |
| ms/imagem (média) | 29.4 | 62.3 | 395.0 |
| ms/imagem (p95) | 53.5 | 127.2 | 718.4 |
| FPS | 34.0 | 16.1 | 2.5 |

Curva P-R: `deteccao_curva_PR.png`

## 2. Robustez — recall por condição (threshold da app)

Recall = fração das caras GT dessa condição que foram detetadas; a coluna "caras" indica quantas caras GT há nessa condição.

### Tamanho (√(w·h) px)

| Tamanho | caras | SCRFD-500MF | RetinaFace MobileNet0.25 | RetinaFace R50 |
|---|---|---|---|---|
| < 16 | 85 | 0.024 | 0.000 | 0.000 |
| 16–32 | 140 | 0.557 | 0.414 | 0.350 |
| 32–64 | 203 | 0.754 | 0.700 | 0.704 |
| 64–128 | 89 | 0.854 | 0.843 | 0.876 |
| ≥ 128 | 46 | 0.935 | 0.957 | 0.891 |

### Pose

| Pose | caras | SCRFD-500MF | RetinaFace MobileNet0.25 | RetinaFace R50 |
|---|---|---|---|---|
| típica | 501 | 0.661 | 0.601 | 0.587 |
| atípica | 62 | 0.339 | 0.290 | 0.274 |

### Oclusão

| Oclusão | caras | SCRFD-500MF | RetinaFace MobileNet0.25 | RetinaFace R50 |
|---|---|---|---|---|
| nenhuma | 387 | 0.762 | 0.724 | 0.693 |
| parcial | 76 | 0.434 | 0.329 | 0.329 |
| forte | 100 | 0.240 | 0.140 | 0.180 |

### Blur

| Blur | caras | SCRFD-500MF | RetinaFace MobileNet0.25 | RetinaFace R50 |
|---|---|---|---|---|
| nítida | 197 | 0.914 | 0.888 | 0.878 |
| normal | 202 | 0.683 | 0.619 | 0.589 |
| forte | 164 | 0.207 | 0.116 | 0.116 |

### Iluminação

| Iluminação | caras | SCRFD-500MF | RetinaFace MobileNet0.25 | RetinaFace R50 |
|---|---|---|---|---|
| normal | 526 | 0.612 | 0.557 | 0.538 |
| extrema | 37 | 0.811 | 0.703 | 0.757 |

### Expressão

| Expressão | caras | SCRFD-500MF | RetinaFace MobileNet0.25 | RetinaFace R50 |
|---|---|---|---|---|
| típica | 556 | 0.621 | 0.563 | 0.549 |
| exagerada | 7 | 1.000 | 0.857 | 0.857 |

## 3. Robustez — várias caras na mesma imagem (threshold da app)

| Caras por imagem | imagens | SCRFD-500MF R / P | RetinaFace MobileNet0.25 R / P | RetinaFace R50 R / P |
|---|---|---|---|---|
| 1 | 33 | 0.909 / 1.000 | 0.909 / 1.000 | 0.909 / 1.000 |
| 2–5 | 31 | 0.783 / 0.935 | 0.750 / 0.958 | 0.750 / 0.986 |
| 6–20 | 33 | 0.584 / 0.986 | 0.531 / 0.990 | 0.520 / 0.974 |
| > 20 | 3 | 0.512 / 1.000 | 0.378 / 1.000 | 0.329 / 0.900 |
