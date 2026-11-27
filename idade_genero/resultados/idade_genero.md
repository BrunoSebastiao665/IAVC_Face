# Idade aparente e género — UTKFace (split de teste)

- 4682 caras de teste (split fixo 70/10/20, seed 0), caras detetadas pelo SCRFD
- **Pré-treinados** (nunca viram o UTKFace): MobileNet genderage (InsightFace) e FairFace ResNet-34. **Fine-tuned**: a MobileNet ajustada no treino do UTKFace com 4 formulações da idade
- **Idade** = estimativa de idade *aparente*; **género** = classificação visual nas 2 categorias anotadas no UTKFace (female/male), não identidade de género
- Limiar de confiança do género para a aplicação: **0.75** (abaixo → "incerto")

## 1. Idade — resumo

| Métrica | MobileNet pré-treinada | FairFace ResNet-34 pré-treinado | MobileNet FT regressão | MobileNet FT classes | MobileNet FT DLDL | MobileNet FT ordinal |
|---|---|---|---|---|---|---|
| **MAE** (anos) | **9.45** | **5.28** | **6.29** | **6.26** | **6.22** | **6.03** |
| RMSE (anos) | 12.56 | 7.76 | 9.09 | 8.87 | 8.95 | 8.44 |
| Erro médio com sinal (viés, anos) | +5.90 | -2.25 | +1.06 | +1.61 | +1.73 | -0.12 |
| CS(5) — erro ≤ 5 anos | 0.382 | 0.621 | 0.559 | 0.544 | 0.551 | 0.554 |
| Faixa etária exata | 0.346 | 0.579 | 0.517 | 0.509 | 0.504 | 0.510 |
| Faixa etária ±1 | 0.779 | 0.937 | 0.903 | 0.909 | 0.911 | 0.916 |
| ms / cara (CPU, batch 1) | 0.40 | 48.50 | 0.35 | 0.35 | 0.39 | 0.37 |
| Entrada (crop) | box 96×96 | align 224×224 | box 96×96 | box 96×96 | box 96×96 | box 96×96 |

Dispersão prevista vs real: `idade_dispersao.png`

## 2. Idade — MAE por faixa etária

| Faixa | caras | MobileNet pré-treinada | FairFace ResNet-34 pré-treinado | MobileNet FT regressão | MobileNet FT classes | MobileNet FT DLDL | MobileNet FT ordinal |
|---|---|---|---|---|---|---|---|
| 0-2 | 332 | 8.33 | 1.00 | 1.69 | 1.63 | 1.53 | 1.39 |
| 3-9 | 267 | 19.48 | 2.18 | 4.29 | 4.38 | 4.34 | 3.91 |
| 10-19 | 280 | 14.61 | 3.29 | 7.92 | 7.55 | 7.69 | 7.62 |
| 20-29 | 1412 | 7.35 | 3.35 | 4.23 | 5.20 | 5.26 | 4.81 |
| 30-39 | 934 | 7.98 | 5.13 | 6.91 | 6.35 | 6.18 | 5.70 |
| 40-49 | 463 | 9.74 | 7.01 | 8.80 | 8.04 | 7.59 | 7.87 |
| 50-59 | 439 | 9.70 | 7.77 | 9.04 | 8.57 | 8.87 | 8.74 |
| 60-69 | 263 | 8.17 | 10.00 | 8.85 | 7.78 | 8.83 | 8.65 |
| 70+ | 292 | 11.81 | 13.99 | 9.31 | 9.25 | 8.13 | 9.33 |

### Matriz de confusão das faixas etárias — FairFace ResNet-34 pré-treinado (linhas = real, colunas = prevista)

| real \ prev | 0-2 | 3-9 | 10-19 | 20-29 | 30-39 | 40-49 | 50-59 | 60-69 | 70+ | recall |
|---|---|---|---|---|---|---|---|---|---|---|
| **0-2** | **296** | 33 | 0 | 2 | 1 | 0 | 0 | 0 | 0 | 0.89 |
| **3-9** | 37 | **190** | 36 | 3 | 1 | 0 | 0 | 0 | 0 | 0.71 |
| **10-19** | 0 | 46 | **158** | 76 | 0 | 0 | 0 | 0 | 0 | 0.56 |
| **20-29** | 0 | 3 | 88 | **1110** | 190 | 17 | 4 | 0 | 0 | 0.79 |
| **30-39** | 0 | 1 | 15 | 403 | **370** | 134 | 9 | 2 | 0 | 0.40 |
| **40-49** | 0 | 0 | 3 | 55 | 159 | **190** | 54 | 2 | 0 | 0.41 |
| **50-59** | 0 | 0 | 1 | 17 | 42 | 163 | **177** | 37 | 2 | 0.40 |
| **60-69** | 0 | 0 | 2 | 0 | 17 | 41 | 109 | **87** | 7 | 0.33 |
| **70+** | 0 | 0 | 0 | 2 | 3 | 9 | 41 | 102 | **135** | 0.46 |

### Matriz de confusão das faixas etárias — MobileNet FT ordinal (linhas = real, colunas = prevista)

| real \ prev | 0-2 | 3-9 | 10-19 | 20-29 | 30-39 | 40-49 | 50-59 | 60-69 | 70+ | recall |
|---|---|---|---|---|---|---|---|---|---|---|
| **0-2** | **218** | 107 | 4 | 2 | 1 | 0 | 0 | 0 | 0 | 0.66 |
| **3-9** | 27 | **163** | 55 | 17 | 3 | 0 | 1 | 1 | 0 | 0.61 |
| **10-19** | 1 | 43 | **88** | 115 | 25 | 4 | 2 | 2 | 0 | 0.31 |
| **20-29** | 0 | 8 | 91 | **844** | 412 | 43 | 13 | 1 | 0 | 0.60 |
| **30-39** | 0 | 1 | 18 | 250 | **478** | 145 | 38 | 3 | 1 | 0.51 |
| **40-49** | 0 | 0 | 3 | 28 | 189 | **158** | 67 | 17 | 1 | 0.34 |
| **50-59** | 0 | 0 | 0 | 7 | 67 | 128 | **140** | 84 | 13 | 0.32 |
| **60-69** | 0 | 0 | 0 | 3 | 12 | 30 | 71 | **106** | 41 | 0.40 |
| **70+** | 0 | 0 | 0 | 0 | 4 | 2 | 16 | 76 | **194** | 0.66 |

## 3. Género — por classe

| Métrica | MobileNet pré-treinada | FairFace ResNet-34 pré-treinado | MobileNet FT regressão | MobileNet FT classes | MobileNet FT DLDL | MobileNet FT ordinal |
|---|---|---|---|---|---|---|
| Precisão female | 0.896 | 0.948 | 0.909 | 0.917 | 0.918 | 0.902 |
| Recall female | 0.894 | 0.930 | 0.926 | 0.924 | 0.909 | 0.927 |
| F1 female | 0.895 | 0.939 | 0.917 | 0.921 | 0.913 | 0.915 |
| Precisão male | 0.905 | 0.938 | 0.932 | 0.931 | 0.919 | 0.933 |
| Recall male | 0.907 | 0.954 | 0.917 | 0.925 | 0.927 | 0.910 |
| F1 male | 0.906 | 0.946 | 0.925 | 0.928 | 0.923 | 0.921 |
| **F1 macro** | **0.901** | **0.942** | **0.921** | **0.924** | **0.918** | **0.918** |
| Exatidão (só referência) | 0.901 | 0.943 | 0.921 | 0.925 | 0.918 | 0.918 |

### Matrizes de confusão (linhas = real, colunas = previsto; F, M)

| Modelo | real F → F | real F → M | real M → F | real M → M |
|---|---|---|---|---|
| MobileNet pré-treinada | 1980 | 235 | 229 | 2238 |
| FairFace ResNet-34 pré-treinado | 2059 | 156 | 113 | 2354 |
| MobileNet FT regressão | 2050 | 165 | 204 | 2263 |
| MobileNet FT classes | 2046 | 169 | 184 | 2283 |
| MobileNet FT DLDL | 2013 | 202 | 180 | 2287 |
| MobileNet FT ordinal | 2054 | 161 | 222 | 2245 |

### Limiar de confiança ("incerto") — cobertura / F1 macro

Cobertura = fração de caras com rótulo de género mostrado; F1 macro calculado só nessas caras.

| Limiar | MobileNet pré-treinada | FairFace ResNet-34 pré-treinado | MobileNet FT regressão | MobileNet FT classes | MobileNet FT DLDL | MobileNet FT ordinal |
|---|---|---|---|---|---|---|
| 0.50 | 1.000 / 0.901 | 1.000 / 0.942 | 1.000 / 0.921 | 1.000 / 0.924 | 1.000 / 0.918 | 1.000 / 0.918 |
| 0.60 | 0.981 / 0.907 | 0.979 / 0.950 | 0.960 / 0.936 | 0.953 / 0.939 | 0.955 / 0.935 | 0.965 / 0.933 |
| 0.70 | 0.960 / 0.917 | 0.953 / 0.959 | 0.916 / 0.948 | 0.908 / 0.953 | 0.903 / 0.953 | 0.927 / 0.944 |
| 0.75 ← app | 0.952 / 0.919 | 0.937 / 0.963 | 0.894 / 0.956 | 0.882 / 0.962 | 0.881 / 0.960 | 0.907 / 0.951 |
| 0.80 | 0.935 / 0.924 | 0.924 / 0.967 | 0.868 / 0.964 | 0.859 / 0.967 | 0.857 / 0.965 | 0.881 / 0.958 |
| 0.90 | 0.897 / 0.934 | 0.882 / 0.974 | 0.802 / 0.978 | 0.794 / 0.979 | 0.784 / 0.980 | 0.820 / 0.974 |
| 0.95 | 0.860 / 0.943 | 0.838 / 0.980 | 0.739 / 0.984 | 0.727 / 0.986 | 0.723 / 0.986 | 0.761 / 0.982 |

## 4. Diferenças demográficas (etnia anotada no UTKFace)

| Etnia | caras | MobileNet pré-treinada MAE / F1 | FairFace ResNet-34 pré-treinado MAE / F1 | MobileNet FT regressão MAE / F1 | MobileNet FT classes MAE / F1 | MobileNet FT DLDL MAE / F1 | MobileNet FT ordinal MAE / F1 |
|---|---|---|---|---|---|---|---|
| White | 1993 | 10.90 / 0.890 | 6.06 / 0.948 | 6.95 / 0.930 | 6.86 / 0.931 | 6.73 / 0.926 | 6.48 / 0.927 |
| Black | 904 | 8.59 / 0.927 | 5.92 / 0.960 | 7.11 / 0.931 | 7.21 / 0.927 | 6.93 / 0.927 | 6.90 / 0.930 |
| Asian | 668 | 7.05 / 0.882 | 3.66 / 0.902 | 4.59 / 0.896 | 4.36 / 0.887 | 4.73 / 0.882 | 4.72 / 0.879 |
| Indian | 784 | 8.79 / 0.920 | 4.62 / 0.958 | 5.64 / 0.920 | 5.73 / 0.940 | 5.74 / 0.930 | 5.49 / 0.918 |
| Others | 332 | 9.50 / 0.876 | 3.72 / 0.903 | 5.01 / 0.878 | 5.20 / 0.909 | 5.34 / 0.885 | 4.86 / 0.900 |

**MAE por etnia ajustado à idade** (MAE de cada faixa etária dentro da etnia, pesado pela distribuição global das faixas; remove o efeito de cada etnia ter idades diferentes no teste):

| Etnia | MobileNet pré-treinada | FairFace ResNet-34 pré-treinado | MobileNet FT regressão | MobileNet FT classes | MobileNet FT DLDL | MobileNet FT ordinal |
|---|---|---|---|---|---|---|
| White | 10.49 | 5.19 | 6.52 | 6.58 | 6.47 | 6.06 |
| Black | 9.99 | 6.15 | 7.50 | 7.67 | 7.34 | 7.37 |
| Asian | 8.03 | 4.54 | 6.06 | 5.48 | 5.96 | 5.95 |
| Indian | 9.70 | 5.36 | 6.15 | 6.42 | 6.41 | 6.29 |
| Others | 9.04 | 3.86 | 5.43 | 5.68 | 5.60 | 5.35 |

| Género anotado | caras | MobileNet pré-treinada MAE | FairFace ResNet-34 pré-treinado MAE | MobileNet FT regressão MAE | MobileNet FT classes MAE | MobileNet FT DLDL MAE | MobileNet FT ordinal MAE |
|---|---|---|---|---|---|---|---|
| female | 2215 | 9.39 | 5.52 | 5.99 | 5.95 | 6.14 | 6.03 |
| male | 2467 | 9.51 | 5.07 | 6.56 | 6.55 | 6.28 | 6.04 |

## 5. FairFace − MobileNet FT ordinal: a diferença é real?

Bootstrap emparelhado (1000 reamostragens das caras de teste). Mede a incerteza da amostra de teste, não a variação entre treinos.

| | Diferença | IC 95% |
|---|---|---|
| MAE (anos; negativo = FairFace melhor) | -0.75 | [-0.913, -0.574] |
| F1 macro do género (positivo = FairFace melhor) | +0.024 | [+0.015, +0.032] |

**Viés por faixa etária** (erro médio com sinal, anos):

| Faixa | MobileNet pré-treinada | FairFace ResNet-34 pré-treinado | MobileNet FT regressão | MobileNet FT classes | MobileNet FT DLDL | MobileNet FT ordinal |
|---|---|---|---|---|---|---|
| 0-2 | +8.33 | +0.82 | +1.66 | +1.58 | +1.50 | +1.30 |
| 3-9 | +19.42 | +1.44 | +3.16 | +3.87 | +3.52 | +2.72 |
| 10-19 | +14.61 | +1.13 | +5.05 | +5.50 | +5.40 | +5.08 |
| 20-29 | +6.71 | +0.74 | +2.45 | +3.75 | +3.62 | +2.36 |
| 30-39 | +5.23 | -1.82 | +0.38 | +1.74 | +1.54 | -0.44 |
| 40-49 | +4.86 | -4.32 | -0.47 | +0.40 | +0.17 | -2.73 |
| 50-59 | +3.76 | -5.56 | -0.63 | -0.05 | -0.39 | -3.41 |
| 60-69 | +0.59 | -8.92 | -0.86 | -1.36 | -0.34 | -4.53 |
| 70+ | -9.75 | -13.89 | -3.21 | -7.89 | -4.20 | -7.19 |

**Género "incerto" por faixa etária** (fração de caras abaixo do limiar da app) e F1 macro sem limiar:

| Faixa | MobileNet pré-treinada | FairFace ResNet-34 pré-treinado | MobileNet FT regressão | MobileNet FT classes | MobileNet FT DLDL | MobileNet FT ordinal |
|---|---|---|---|---|---|---|
| 0-2 | 0.120 / 0.581 | 0.377 / 0.657 | 0.584 / 0.630 | 0.684 / 0.669 | 0.699 / 0.641 | 0.464 / 0.652 |
| 3-9 | 0.097 / 0.646 | 0.157 / 0.855 | 0.277 / 0.789 | 0.315 / 0.813 | 0.356 / 0.783 | 0.307 / 0.779 |
| 10-19 | 0.107 / 0.821 | 0.096 / 0.917 | 0.146 / 0.867 | 0.161 / 0.892 | 0.143 / 0.885 | 0.150 / 0.882 |
| 20-29 | 0.034 / 0.961 | 0.025 / 0.969 | 0.042 / 0.965 | 0.050 / 0.965 | 0.042 / 0.959 | 0.038 / 0.958 |
| 30-39 | 0.020 / 0.959 | 0.015 / 0.982 | 0.034 / 0.969 | 0.033 / 0.967 | 0.031 / 0.963 | 0.019 / 0.963 |
| 40-49 | 0.039 / 0.951 | 0.022 / 0.987 | 0.028 / 0.953 | 0.026 / 0.950 | 0.030 / 0.948 | 0.024 / 0.953 |
| 50-59 | 0.039 / 0.940 | 0.025 / 0.959 | 0.052 / 0.934 | 0.046 / 0.952 | 0.055 / 0.943 | 0.055 / 0.955 |
| 60-69 | 0.023 / 0.947 | 0.023 / 0.976 | 0.042 / 0.952 | 0.053 / 0.952 | 0.072 / 0.956 | 0.046 / 0.921 |
| 70+ | 0.079 / 0.816 | 0.082 / 0.949 | 0.164 / 0.897 | 0.171 / 0.856 | 0.144 / 0.876 | 0.140 / 0.870 |

Exemplos: `exemplos_teste.jpg`
