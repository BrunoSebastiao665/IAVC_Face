# Rede multitarefa: avaliação

Backbone ArcFace (w600k_mbf) congelado + um ramo convolucional afinado por tarefa; identidade = ArcFace original.

## A. Atributos (teste)

| Métrica | multitarefa | especialista leve | especialista preciso |
|---|---|---|---|
| Idade MAE, UTKFace (4682) | 4.14 [4.017, 4.267] | MobileNet FT 6.03 | FairFace 5.28 |
| Idade RMSE / viés / CS(5) | 6.10 / +0.07 / 0.706 | — | — |
| Género F1 macro, UTKFace | 0.952 | MobileNet FT 0.918 | FairFace 0.942 |
| Expressão bal. acc., AffectNet 8 cl. (3445) | 0.573 | B0 0.584 | B2 0.598 |
| Expressão bal. acc., AffectNet 7 cl. | 0.600 [0.583, 0.617] | B0 0.615 | B2 0.631 |
| Expressão bal. acc., BTFER 7 cl. (1847) | 0.725 [0.704, 0.745] | B0 0.749 | B2 0.745 |
| ms por cara (CPU, rede toda, inclui identidade) | 11.4 | ArcFace 7.7 + FT 0.4 + B0 11.9 ≈ 20.0 | ArcFace 7.7 + FairFace 48.5 + B2 25.1 ≈ 81.3 |
| Tamanho ONNX | 25.0 MB | — | — |

Limiar de expressão incerta (multitarefa): 0.45 → no AffectNet de teste rotula 73% das caras com 68.7% de acerto. Gravado em geral/resultados/config.json (a app passa a usar a multitarefa).

### Idade: MAE por faixa

| Modelo | 0-2 | 3-9 | 10-19 | 20-29 | 30-39 | 40-49 | 50-59 | 60-69 | 70+ |
|---|---|---|---|---|---|---|---|---|---|
| multitarefa | 0.93 | 2.01 | 3.26 | 3.01 | 4.77 | 5.95 | 5.82 | 6.02 | 6.98 |

### Género (multitarefa)

F: P 0.942 R 0.958 F1 0.950; M: P 0.962 R 0.947 F1 0.955; confusão [[F→F, F→M], [M→F, M→M]] = [[2123, 92], [130, 2337]]

### Expressão, AffectNet 7 classes (multitarefa): P / R / F1 por classe

| Classe | P | R | F1 |
|---|---|---|---|
| Anger | 0.58 | 0.51 | 0.54 |
| Disgust | 0.59 | 0.50 | 0.54 |
| Fear | 0.61 | 0.48 | 0.54 |
| Happy | 0.77 | 0.89 | 0.82 |
| Sad | 0.59 | 0.62 | 0.60 |
| Surprise | 0.57 | 0.60 | 0.58 |
| Neutral | 0.53 | 0.61 | 0.57 |

accuracy 0.608; confusão (linhas = real): [[218, 54, 20, 12, 31, 28, 64], [70, 195, 17, 28, 38, 14, 29], [16, 20, 178, 10, 41, 85, 21], [2, 8, 2, 420, 5, 15, 21], [27, 25, 17, 8, 261, 15, 71], [9, 15, 55, 38, 16, 266, 48], [36, 11, 5, 32, 50, 44, 282]]

### Expressão, BTFER (multitarefa): P / R / F1 por classe

| Classe | P | R | F1 |
|---|---|---|---|
| Anger | 0.72 | 0.69 | 0.71 |
| Disgust | 0.71 | 0.67 | 0.69 |
| Fear | 0.67 | 0.65 | 0.66 |
| Happy | 0.79 | 0.96 | 0.87 |
| Sad | 0.69 | 0.61 | 0.65 |
| Surprise | 0.82 | 0.71 | 0.76 |
| Neutral | 0.67 | 0.77 | 0.72 |

accuracy 0.729; confusão (linhas = real): [[182, 48, 16, 5, 8, 2, 2], [29, 168, 12, 9, 27, 1, 4], [9, 6, 157, 6, 25, 33, 7], [0, 2, 0, 280, 2, 6, 1], [17, 9, 1, 3, 164, 0, 73], [3, 1, 46, 24, 1, 202, 6], [12, 1, 1, 29, 11, 3, 193]]

## B. Suavização temporal (vídeos sintéticos, 300 caras x 20 frames por tarefa)

EMA com α = 0.35 x qualidade do frame (nitidez).

| Tarefa | flicker sem → com EMA (trocas/frame) | exatidão por frame sem → com | outra métrica sem → com |
|---|---|---|---|
| Género | 0.022 → 0.006 | 0.952 → 0.955 | — |
| Idade | — | MAE 4.13 → 3.98 | desvio da idade no vídeo 1.25 → 0.35 anos |
| Expressão (7 cl.) | 0.119 → 0.024 | 0.607 → 0.619 | — |

Atraso quando a cara/expressão muda a meio: média 1.7 frames, mediana 1 (239 trocas). É o preço da estabilidade: a ~10 atualizações/s são ~0.2 s.
