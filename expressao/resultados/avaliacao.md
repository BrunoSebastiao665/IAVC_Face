# Expressão facial — HSEmotion B0 e B2 (pré-treinados)

- Os rótulos são **estimativas da configuração facial visível**, não evidência da emoção sentida.
- Modelos treinados pelos autores no train oficial do AffectNet: B0 = `enet_b0_8_va_mtl`, B2 = `enet_b2_8`. **Sem treino nosso.**
- Métricas com [IC 95%] por bootstrap (1 000 reamostragens das imagens).

| | HSEmotion B0 | HSEmotion B2 |
|---|---|---|
| Entrada | 224×224 | 260×260 |
| Ficheiro (MB) | 16.1 | 30.8 |
| ms por cara, CPU, com resize (mediana / p95) | 11.9 / 13.2 | 25.1 / 31.9 |

## 1. Intra-dataset — validação oficial do AffectNet

Principal: sem as 552 imagens com uma (quase) igual no train ('sem fuga').

### 8 classes, sem fuga (principal)

| Métrica | HSEmotion B0 | HSEmotion B2 |
|---|---|---|
| imagens | 3445 | 3445 |
| Accuracy | 0.590 [0.575, 0.606] | 0.600 [0.584, 0.617] |
| **Balanced accuracy** | 0.584 [0.569, 0.599] | 0.598 [0.581, 0.615] |
| Precisão macro | 0.601 | 0.600 |
| Recall macro | 0.584 | 0.598 |
| **F1 macro** | 0.584 [0.568, 0.599] | 0.598 [0.581, 0.615] |
| Top-2 accuracy | 0.799 | 0.802 |

### 7 classes (sem Contempt), sem fuga

| Métrica | HSEmotion B0 | HSEmotion B2 |
|---|---|---|
| imagens | 2993 | 2993 |
| Accuracy | 0.624 [0.608, 0.641] | 0.636 [0.619, 0.653] |
| **Balanced accuracy** | 0.615 [0.599, 0.632] | 0.631 [0.614, 0.647] |
| Precisão macro | 0.630 | 0.633 |
| Recall macro | 0.615 | 0.631 |
| **F1 macro** | 0.614 [0.597, 0.632] | 0.630 [0.613, 0.646] |
| Top-2 accuracy | 0.830 | 0.818 |

### Validação oficial completa (para comparar com os autores)

| Accuracy | HSEmotion B0 | HSEmotion B2 |
|---|---|---|
| 8 classes: nós / autores | 0.6112 / 0.6193 | 0.6207 / 0.6303 |
| 7 classes: nós / autores | 0.6448 / 0.6494 | 0.6566 / 0.6629 |

### Por classe (8 classes, sem fuga): precisão / recall / F1

| Classe | imagens | HSEmotion B0 | HSEmotion B2 |
|---|---|---|---|
| Anger | 427 | 0.520 / 0.637 / 0.573 | 0.593 / 0.499 / 0.542 |
| Disgust | 391 | 0.669 / 0.409 / 0.508 | 0.529 / 0.583 / 0.555 |
| Fear | 371 | 0.646 / 0.488 / 0.556 | 0.606 / 0.580 / 0.592 |
| Happy | 473 | 0.679 / 0.822 / 0.744 | 0.725 / 0.759 / 0.742 |
| Sad | 424 | 0.608 / 0.625 / 0.616 | 0.626 / 0.620 / 0.623 |
| Surprise | 447 | 0.586 / 0.600 / 0.593 | 0.635 / 0.591 / 0.612 |
| Neutral | 460 | 0.469 / 0.591 / 0.523 | 0.510 / 0.561 / 0.534 |
| Contempt | 452 | 0.630 / 0.498 / 0.556 | 0.577 / 0.591 / 0.584 |

### Matrizes de confusão (8 classes, sem fuga) — figura `confusao_affectnet.png`

**HSEmotion B0**

| real \ prevista | Anger | Disgust | Fear | Happy | Sad | Surprise | Neutral | Contempt | recall |
|---|---|---|---|---|---|---|---|---|---|
| **Anger** | **272** | 25 | 16 | 2 | 21 | 26 | 53 | 12 | 0.64 |
| **Disgust** | 96 | **160** | 18 | 22 | 38 | 13 | 26 | 18 | 0.41 |
| **Fear** | 30 | 10 | **181** | 9 | 38 | 82 | 21 | 0 | 0.49 |
| **Happy** | 1 | 5 | 0 | **389** | 3 | 17 | 18 | 40 | 0.82 |
| **Sad** | 42 | 16 | 14 | 10 | **265** | 10 | 58 | 9 | 0.62 |
| **Surprise** | 13 | 6 | 40 | 36 | 22 | **268** | 56 | 6 | 0.60 |
| **Neutral** | 42 | 4 | 8 | 13 | 40 | 34 | **272** | 47 | 0.59 |
| **Contempt** | 27 | 13 | 3 | 92 | 9 | 7 | 76 | **225** | 0.50 |

Erros mais frequentes (% da classe real): Disgust → Anger (96, 25%); Fear → Surprise (82, 22%); Contempt → Happy (92, 20%); Contempt → Neutral (76, 17%); Sad → Neutral (58, 14%); Surprise → Neutral (56, 13%)

**HSEmotion B2**

| real \ prevista | Anger | Disgust | Fear | Happy | Sad | Surprise | Neutral | Contempt | recall |
|---|---|---|---|---|---|---|---|---|---|
| **Anger** | **213** | 65 | 20 | 5 | 29 | 22 | 50 | 23 | 0.50 |
| **Disgust** | 51 | **228** | 21 | 17 | 27 | 6 | 24 | 17 | 0.58 |
| **Fear** | 14 | 24 | **215** | 8 | 33 | 57 | 16 | 4 | 0.58 |
| **Happy** | 2 | 11 | 1 | **359** | 0 | 17 | 14 | 69 | 0.76 |
| **Sad** | 26 | 28 | 19 | 6 | **263** | 10 | 59 | 13 | 0.62 |
| **Surprise** | 5 | 19 | 63 | 28 | 17 | **264** | 39 | 12 | 0.59 |
| **Neutral** | 27 | 25 | 14 | 9 | 35 | 34 | **258** | 58 | 0.56 |
| **Contempt** | 21 | 31 | 2 | 63 | 16 | 6 | 46 | **267** | 0.59 |

Erros mais frequentes (% da classe real): Fear → Surprise (57, 15%); Anger → Disgust (65, 15%); Happy → Contempt (69, 15%); Surprise → Fear (63, 14%); Contempt → Happy (63, 14%); Sad → Neutral (59, 14%)

## 2. Cross-dataset — BTFER (7 classes comuns, sem ajuste)

### Limpo, alinhado (principal)

| Métrica | HSEmotion B0 | HSEmotion B2 |
|---|---|---|
| imagens | 1847 | 1847 |
| Accuracy | 0.754 [0.734, 0.774] | 0.748 [0.729, 0.768] |
| **Balanced accuracy** | 0.749 [0.729, 0.768] | 0.745 [0.725, 0.766] |
| Precisão macro | 0.755 | 0.746 |
| Recall macro | 0.749 | 0.745 |
| **F1 macro** | 0.749 [0.728, 0.768] | 0.744 [0.724, 0.765] |
| Top-2 accuracy | 0.927 | 0.907 |

### Sensibilidade (balanced accuracy / F1 macro)

| | HSEmotion B0 | HSEmotion B2 |
|---|---|---|
| limpo, alinhado (principal) | 0.749 / 0.749 | 0.745 / 0.744 |
| limpo, imagem tal como distribuída | 0.742 / 0.741 | 0.746 / 0.744 |
| completo (2100, com repetições), alinhado | 0.757 / 0.755 | 0.754 / 0.753 |

### Por classe: precisão / recall / F1

| Classe | imagens | HSEmotion B0 | HSEmotion B2 |
|---|---|---|---|
| Anger | 263 | 0.670 / 0.810 / 0.733 | 0.781 / 0.719 / 0.749 |
| Disgust | 250 | 0.825 / 0.696 / 0.755 | 0.682 / 0.764 / 0.721 |
| Fear | 243 | 0.696 / 0.658 / 0.677 | 0.714 / 0.679 / 0.696 |
| Happy | 291 | 0.802 / 0.962 / 0.875 | 0.802 / 0.931 / 0.862 |
| Sad | 267 | 0.741 / 0.697 / 0.718 | 0.734 / 0.693 / 0.713 |
| Surprise | 283 | 0.817 / 0.724 / 0.768 | 0.798 / 0.724 / 0.759 |
| Neutral | 250 | 0.734 / 0.696 / 0.715 | 0.713 / 0.704 / 0.708 |

### Matrizes de confusão — figura `confusao_btfer.png`

**HSEmotion B0**

| real \ prevista | Anger | Disgust | Fear | Happy | Sad | Surprise | Neutral | recall |
|---|---|---|---|---|---|---|---|---|
| **Anger** | **213** | 28 | 11 | 2 | 6 | 1 | 2 | 0.81 |
| **Disgust** | 38 | **174** | 10 | 5 | 21 | 1 | 1 | 0.70 |
| **Fear** | 12 | 5 | **160** | 3 | 25 | 32 | 6 | 0.66 |
| **Happy** | 1 | 0 | 0 | **280** | 0 | 8 | 2 | 0.96 |
| **Sad** | 28 | 2 | 1 | 2 | **186** | 0 | 48 | 0.70 |
| **Surprise** | 5 | 0 | 48 | 19 | 2 | **205** | 4 | 0.72 |
| **Neutral** | 21 | 2 | 0 | 38 | 11 | 4 | **174** | 0.70 |

Erros mais frequentes (% da classe real): Sad → Neutral (48, 18%); Surprise → Fear (48, 17%); Neutral → Happy (38, 15%); Disgust → Anger (38, 15%); Fear → Surprise (32, 13%); Anger → Disgust (28, 11%)

**HSEmotion B2**

| real \ prevista | Anger | Disgust | Fear | Happy | Sad | Surprise | Neutral | recall |
|---|---|---|---|---|---|---|---|---|
| **Anger** | **189** | 53 | 9 | 1 | 6 | 3 | 2 | 0.72 |
| **Disgust** | 21 | **191** | 7 | 5 | 22 | 1 | 3 | 0.76 |
| **Fear** | 7 | 8 | **165** | 2 | 21 | 35 | 5 | 0.68 |
| **Happy** | 0 | 9 | 1 | **271** | 1 | 7 | 2 | 0.93 |
| **Sad** | 10 | 13 | 2 | 2 | **185** | 0 | 55 | 0.69 |
| **Surprise** | 3 | 1 | 47 | 22 | 1 | **205** | 4 | 0.72 |
| **Neutral** | 12 | 5 | 0 | 35 | 16 | 6 | **176** | 0.70 |

Erros mais frequentes (% da classe real): Sad → Neutral (55, 21%); Anger → Disgust (53, 20%); Surprise → Fear (47, 17%); Fear → Surprise (35, 14%); Neutral → Happy (35, 14%); Disgust → Sad (22, 9%)

## 3. Generalização: AffectNet (7 classes, sem fuga) → BTFER

| | HSEmotion B0 | HSEmotion B2 |
|---|---|---|
| Accuracy | 0.624 → 0.754 (+0.130) | 0.636 → 0.748 (+0.112) |
| **Balanced accuracy** | 0.615 → 0.749 (+0.134) | 0.631 → 0.745 (+0.114) |
| Precisão macro | 0.630 → 0.755 (+0.125) | 0.633 → 0.746 (+0.113) |
| Recall macro | 0.615 → 0.749 (+0.134) | 0.631 → 0.745 (+0.114) |
| **F1 macro** | 0.614 → 0.749 (+0.135) | 0.630 → 0.744 (+0.114) |
| Recall Anger | 0.644 → 0.810 | 0.506 → 0.719 |
| Recall Disgust | 0.414 → 0.696 | 0.604 → 0.764 |
| Recall Fear | 0.488 → 0.658 | 0.580 → 0.679 |
| Recall Happy | 0.890 → 0.962 | 0.882 → 0.931 |
| Recall Sad | 0.627 → 0.697 | 0.632 → 0.693 |
| Recall Surprise | 0.602 → 0.724 | 0.595 → 0.724 |
| Recall Neutral | 0.639 → 0.696 | 0.622 → 0.704 |

## 4. HSEmotion B2 − HSEmotion B0: a diferença é real? (bootstrap emparelhado, IC 95%)

| Dados | Balanced accuracy | F1 macro |
|---|---|---|
| AffectNet, 8 classes, sem fuga | +0.014 [+0.000, +0.027] | +0.014 [-0.000, +0.028] |
| AffectNet, 7 classes, sem fuga | +0.016 [+0.002, +0.030] | +0.016 [+0.002, +0.030] |
| BTFER, 7 classes | -0.004 [-0.020, +0.011] | -0.005 [-0.021, +0.011] |

## 5. Confiança e rótulo "incerto" (7 classes)

Limiar de cada modelo = o menor com exatidão ≥ 70% no `val` do AffectNet (625 imagens). A coluna AffectNet é só o `test`, que não foi usado na escolha.

**HSEmotion B0**

| Limiar | AffectNet test: cobertura / exatidão / bal. acc. | BTFER: cobertura / exatidão / bal. acc. |
|---|---|---|
| 0.0 | 1.00 / 0.622 / 0.612 | 1.00 / 0.754 / 0.749 |
| 0.3 | 0.99 / 0.629 / 0.617 | 1.00 / 0.756 / 0.751 |
| 0.4 | 0.92 / 0.651 / 0.637 | 0.98 / 0.763 / 0.757 |
| 0.5 | 0.80 / 0.696 / 0.676 | 0.90 / 0.790 / 0.781 |
| 0.6 ← app | 0.64 / 0.733 / 0.705 | 0.80 / 0.822 / 0.810 |
| 0.7 | 0.48 / 0.797 / 0.763 | 0.67 / 0.847 / 0.827 |
| 0.8 | 0.34 / 0.862 / 0.811 | 0.54 / 0.882 / 0.845 |
| 0.9 | 0.20 / 0.924 / 0.816 | 0.38 / 0.921 / 0.833 |

**HSEmotion B2**

| Limiar | AffectNet test: cobertura / exatidão / bal. acc. | BTFER: cobertura / exatidão / bal. acc. |
|---|---|---|
| 0.0 | 1.00 / 0.634 / 0.629 | 1.00 / 0.748 / 0.745 |
| 0.3 | 0.91 / 0.666 / 0.658 | 0.97 / 0.759 / 0.755 |
| 0.4 ← app | 0.65 / 0.735 / 0.724 | 0.81 / 0.804 / 0.802 |
| 0.5 | 0.35 / 0.761 / 0.719 | 0.56 / 0.821 / 0.825 |
| 0.6 | 0.22 / 0.812 / 0.595 | 0.41 / 0.862 / 0.693 |
| 0.7 | 0.14 / 0.860 / 0.619 | 0.31 / 0.893 / 0.642 |
| 0.8 | 0.06 / 0.864 / 0.627 | 0.17 / 0.913 / 0.657 |
| 0.9 | 0.01 / 0.900 / 0.286 | 0.05 / 0.966 / 0.440 |

## 6. Modelo da app (escolhido só com AffectNet e latência)

Regra: entre os modelos com latência <= 40 ms, o de maior balanced accuracy a 7 classes no AffectNet sem fuga; se outro mais rápido ficar a menos de 1 p.p., o mais rápido.

**Escolhido: HSEmotion EfficientNet-B2** (limiar 0.4). Erros mais confiantes deste modelo: `erros_affectnet.jpg`, `erros_btfer.jpg`
