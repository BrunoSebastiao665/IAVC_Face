# Alinhamento de caras com landmarks

Detetor: SCRFD-500MF (5 landmarks) · alinhamento: semelhança (4 GL) para o template ArcFace 112×112 · descritor: ArcFace MobileFaceNet (`w600k_mbf.onnx`)

## 1. Erro de alinhamento (todas as caras detetadas nas 5 imagens)

| Caras | Inclinação dos olhos antes (média \|°\|) | Erro residual landmarks↔template (px, em 112×112) |
|---|---|---|
| 40 | 5.1° | 4.90 ± 3.69 |

Figura: `alinhamento_exemplos.jpg`

## 2. Robustez à rotação no plano (10 caras grandes e frontais do WIDER val)

Cada cara é rodada de -45° a +45°. Similaridade de cosseno entre o descritor ArcFace da cara rodada e o da mesma cara a 0° (média das caras detetadas). Perto de 1 = o descritor não muda com a rotação.

| Rotação | Detetadas | \|inclinação olhos\| detetada | \|inclinação\| após alinhar | Similaridade SEM alinhamento | Similaridade COM alinhamento |
|---|---|---|---|---|---|
| -45° | 9/10 | 44.9° | 4.2° | 0.093 | **0.850** |
| -30° | 10/10 | 32.2° | 1.5° | 0.369 | **0.936** |
| -15° | 10/10 | 16.8° | 0.3° | 0.752 | **0.979** |
| +0° | 10/10 | 4.4° | 0.6° | 1.000 | **1.000** |
| +15° | 10/10 | 14.6° | 0.9° | 0.737 | **0.977** |
| +30° | 10/10 | 30.9° | 0.5° | 0.367 | **0.952** |
| +45° | 8/10 | 46.2° | 2.0° | 0.108 | **0.930** |

Figura: `alinhamento_rotacao.jpg`
