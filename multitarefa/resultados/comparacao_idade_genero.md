# Multitarefa vs especialistas — idade e género (UTKFace teste, 4 682 caras, as mesmas para todos)

| Modelo | MAE | RMSE | viés | CS(5) | F1 género |
|---|---|---|---|---|---|
| Multitarefa | 4.14 | 6.10 | +0.07 | 0.706 | 0.952 |
| FairFace | 5.28 | 7.76 | -2.25 | 0.621 | 0.942 |
| MobileNet FT ordinal | 6.03 | 8.44 | -0.12 | 0.554 | 0.918 |

## Bootstrap emparelhado (1 000 reamostragens das mesmas caras)

| Diferença | valor | IC 95% |
|---|---|---|
| MAE multitarefa − FairFace (anos) | -1.14 | [-1.27, -1.01] |
| F1 género multitarefa − FairFace | +0.010 | [+0.004, +0.017] |
| MAE multitarefa − MobileNet FT ordinal (anos) | -1.89 | [-2.04, -1.75] |
| F1 género multitarefa − MobileNet FT ordinal | +0.034 | [+0.027, +0.042] |

## MAE / viés por faixa etária

| Faixa | caras | Multitarefa | FairFace | MobileNet FT ordinal |
|---|---|---|---|---|
| 0-2 | 332 | 0.93 / +0.89 | 1.00 / +0.82 | 1.39 / +1.29 |
| 3-9 | 267 | 2.01 / +1.17 | 2.18 / +1.44 | 3.91 / +2.72 |
| 10-19 | 280 | 3.26 / +1.46 | 3.29 / +1.13 | 7.62 / +5.08 |
| 20-29 | 1412 | 3.01 / +1.74 | 3.35 / +0.74 | 4.81 / +2.36 |
| 30-39 | 934 | 4.77 / -0.13 | 5.13 / -1.82 | 5.70 / -0.44 |
| 40-49 | 463 | 5.95 / -1.37 | 7.01 / -4.32 | 7.87 / -2.73 |
| 50-59 | 439 | 5.82 / -1.13 | 7.77 / -5.56 | 8.74 / -3.41 |
| 60-69 | 263 | 6.02 / -1.79 | 10.00 / -8.92 | 8.65 / -4.53 |
| 70+ | 292 | 6.98 / -4.80 | 13.99 / -13.89 | 9.33 / -7.19 |

## Por etnia anotada: MAE bruto / MAE ajustado à idade / F1 género

(ajustado = MAE de cada faixa dentro da etnia, pesado pela distribuição global das faixas; só faixas com ≥ 5 caras, como na fase 3)

| Etnia | caras | Multitarefa | FairFace | MobileNet FT ordinal |
|---|---|---|---|---|
| White | 1993 | 4.41 / 4.13 / 0.960 | 6.06 / 5.19 / 0.948 | 6.49 / 6.07 / 0.927 |
| Black | 904 | 4.60 / 4.82 / 0.960 | 5.92 / 6.15 / 0.960 | 6.90 / 7.37 / 0.930 |
| Asian | 668 | 3.04 / 3.91 / 0.931 | 3.66 / 4.54 / 0.902 | 4.72 / 5.95 / 0.879 |
| Indian | 784 | 4.07 / 4.40 / 0.957 | 4.62 / 5.36 / 0.958 | 5.49 / 6.29 / 0.918 |
| Others | 332 | 3.64 / 3.78 / 0.915 | 3.72 / 3.86 / 0.903 | 4.86 / 5.35 / 0.900 |
