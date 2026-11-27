# Guia do projeto: o que foi feito e como ler os resultados

> **Para quem é:** para quem quer perceber o trabalho todo sem ter acompanhado o desenvolvimento: colegas de
> grupo, quem escreve o relatório, quem prepara a defesa.
> **O que tem:** o sistema numa figura, os conceitos necessários, um resumo de cada fase e, em detalhe, o
> **modelo final**: uma rede com backbone partilhado, comparada com os modelos "especialistas" das fases 3 e 4,
> que não partilham backbone.
>
> Todos os números vêm de ficheiros de resultados do projeto, indicados em cada secção. O histórico completo das
> fases 1–4 está em [PROGRESSO_E_RESULTADOS.md](PROGRESSO_E_RESULTADOS.md).

## Índice
1. [Resumo em 30 segundos](#1-resumo-em-30-segundos)
2. [O sistema numa figura](#2-o-sistema-numa-figura)
3. [Conceitos necessários](#3-conceitos-necessários)
4. [Fases 1–4 em resumo (a abordagem sem backbone partilhado)](#4-fases-14-em-resumo-a-abordagem-sem-backbone-partilhado)
5. [O modelo final: rede multitarefa com backbone](#5-o-modelo-final-rede-multitarefa-com-backbone)
6. [Resultados do modelo final vs especialistas](#6-resultados-do-modelo-final-vs-especialistas)
7. [Tempos e desempenho em tempo real](#7-tempos-e-desempenho-em-tempo-real)
8. [Estabilidade no tempo (média pesada dos frames)](#8-estabilidade-no-tempo-média-pesada-dos-frames)
9. [Identidade](#9-identidade)
10. [Interpretação: porque é que o modelo final ganha e perde onde ganha e perde](#10-interpretação)
11. [Limitações e cuidados ao reportar](#11-limitações-e-cuidados-ao-reportar)
12. [O que falta](#12-o-que-falta)
13. [Onde está cada coisa](#13-onde-está-cada-coisa)

---

## 1. Resumo em 30 segundos

- O sistema deteta caras numa webcam e, para cada uma, diz **quem é** (ou *Unknown*), a **idade aparente**, o
  **género aparente** e a **expressão**.
- Houve duas abordagens:
  - **Especialistas (fases 3–4):** uma rede diferente para cada tarefa: FairFace ou MobileNet para idade e
    género, HSEmotion para expressão, ArcFace para identidade. Cada rede tem o seu backbone.
  - **Modelo final (multitarefa):** **uma só rede**. O backbone do ArcFace calcula as features da cara uma vez,
    e cada tarefa tem o seu ramo convolucional.
- **Resultado:** o modelo final é **o melhor em idade e género** e **o mais rápido**. Na expressão fica **ligeiramente
  abaixo** dos especialistas.

| | Modelo final | Melhor especialista |
|---|---|---|
| Idade, MAE (anos) | **4,14** | 5,28 (FairFace) |
| Género, F1 | **0,952** | 0,942 (FairFace) |
| Expressão, AffectNet 7 classes (bal. acc.) | 0,600 | **0,631** (HSEmotion B2) |
| Expressão, BTFER (bal. acc.) | 0,725 | **0,749** (HSEmotion B0) |
| Atualização completa com 4 caras, **com identidade** | **70 ms (14,3/s)** | 115 ms (rápido) / 364 ms (preciso) |

---

## 2. O sistema numa figura

```
 câmara / vídeo
      │ frame 1280×720
      ▼
 ┌──────────────┐  caixa + 5 landmarks   ┌──────────────┐   cara 112×112
 │ DETEÇÃO      │ ─────────────────────► │ ALINHAMENTO  │ ───────────────┐
 │ SCRFD-500MF  │   (olhos, nariz,       │ semelhança → │                │
 │ ~24 ms/frame │    cantos da boca)     │ template     │                ▼
 └──────────────┘                        │ ArcFace      │   ┌──────────────────────────┐
                                         └──────────────┘   │ MODELO FINAL (1 rede)    │
                                                            │ identidade, idade,       │
                                                            │ género, expressão        │
                                                            └────────────┬─────────────┘
                                                                         ▼
                                         ┌───────────────────────────────────────────────┐
                                         │ GALERIA (cosseno, clusters) → nome / Unknown  │
                                         │ MÉDIA PESADA dos frames anteriores (por cara) │
                                         └───────────────────────┬───────────────────────┘
                                                                 ▼
                                                    desenho no ecrã + estatísticas
```

- A deteção e os modelos correm **numa thread à parte**: o vídeo nunca bloqueia, e os rótulos atualizam-se ao
  ritmo que o processador permitir.
- **Especialistas:** o bloco "modelo final" é substituído por 3–4 redes independentes, cada uma com o seu
  próprio crop (96×96, 224×224, 260×260 ou 112×112).

---

## 3. Conceitos necessários

| Termo | O que é, em linguagem simples |
|---|---|
| **Backbone (tronco)** | A parte inicial de uma rede convolucional, que transforma a imagem num "mapa de features": manchas, contornos, olhos, formas. É a parte mais cara de calcular. |
| **Ramo / cabeça** | As camadas finais que pegam nas features e dão a resposta a uma tarefa (por exemplo, "idade = 32"). |
| **Backbone partilhado** | Várias tarefas leem o mesmo mapa de features. O backbone calcula-se **uma vez** por cara, em vez de uma vez por tarefa. |
| **Embedding / descritor** | Um vetor (aqui com 512 números) que resume a cara. Caras da mesma pessoa dão vetores parecidos. |
| **ArcFace** | Rede de reconhecimento facial treinada com uma *loss* de **margem angular aditiva**: obriga as caras da mesma pessoa a ficar juntas na esfera de 512 dimensões, e as de pessoas diferentes a ficar afastadas por uma margem. Por isso **não classifica pessoas**: compara-se o embedding com uma galeria pelo **cosseno**. |
| **Congelar** | Não alterar os pesos de uma parte da rede durante o treino. |
| **Fine-tuning (afinar)** | Partir de pesos pré-treinados e ajustá-los com poucos dados para uma nova tarefa. |
| **DLDL** | Forma de prever a idade como uma **distribuição** sobre 0–116 anos (alvo = gaussiana à volta da idade real). Lida bem com rótulos ruidosos. |
| **Galeria** | Os embeddings guardados das pessoas registadas, sem imagens. **Media** = um vetor por pessoa; **clusters** = 1 a 4 vetores por pessoa (frente/lado/longe); **vizinho** = todas as amostras. |
| **EMA (média exponencial pesada)** | `média = (1−w)·média_anterior + w·valor_novo`. O passado vai perdendo peso aos poucos. |
| **MAE** | Erro médio absoluto da idade, em anos. Menor é melhor. |
| **RMSE** | Como o MAE, mas penaliza mais os erros grandes. |
| **Viés** | Erro médio **com sinal**: + sobrestima, − subestima. |
| **CS(5)** | Percentagem de caras com erro de idade ≤ 5 anos. |
| **F1 macro** | Média do F1 das classes (F e M), sem favorecer a classe maior. |
| **Balanced accuracy** | Média do recall de cada classe. É a métrica principal da expressão, porque as classes têm tamanhos diferentes. |
| **IC 95% (bootstrap)** | Intervalo onde o valor provavelmente estaria com outra amostra de teste. Se o intervalo de uma diferença não inclui 0, a diferença não é sorte da amostra. |
| **FAR / FRR** | Identidade. FAR = desconhecidos aceites como alguém. FRR = conhecidos rejeitados. |
| **Atualizações/s** | Quantas vezes por segundo os rótulos no ecrã são recalculados (≠ FPS do vídeo, que é sempre fluido). Critério do projeto: ≥ 5/s. |

**Regras usadas em todas as avaliações:**
- **Treino** ajusta os pesos.
- **Validação** escolhe épocas, limiares e modelos.
- **Teste** só é usado no fim, para os números do relatório.

Nenhum número de teste deste guia influenciou escolhas.

---

## 4. Fases 1–4 em resumo (a abordagem sem backbone partilhado)

### Fase 1 — Deteção ([deteccao/resultados/](../deteccao/resultados/parte1_comparacao_detetores.md))
WIDER FACE val, amostra de 100 imagens (563 caras), IoU ≥ 0,5:

| | SCRFD-500MF | RetinaFace Mob0.25 | RetinaFace R50 |
|---|---|---|---|
| AP | **0,762** | 0,761 | 0,720 |
| ms/imagem | **29** | 62 | 395 |

**SCRFD escolhido**: a mesma AP, 2× mais rápido. O ponto fraco de todos são as caras < 16 px (recall 0,02),
que na webcam quase não existem.

### Fase 2 — Alinhamento ([deteccao/resultados/alinhamento.md](../deteccao/resultados/alinhamento.md))
Transformação de semelhança dos 5 landmarks para o template do ArcFace (112×112). Com a cara rodada 30°, a
similaridade ArcFace com a original passa de **0,37 sem alinhamento** (seria "outra pessoa") para **0,95 com
alinhamento**. É indispensável para a identidade e é usado por todos os modelos do modelo final.

### Fase 3 — Idade e género ([idade_genero/resultados/idade_genero.md](../idade_genero/resultados/idade_genero.md))
UTKFace: 23 407 caras, split fixo 16 384 / 2 341 / **4 682 de teste**.

| | MobileNet pré-treinada | MobileNet fine-tuned (4 formulações; ordinal na app) | FairFace ResNet-34 |
|---|---|---|---|
| MAE | 9,45 | 6,03–6,29 (ordinal 6,03) | **5,28** |
| F1 género | 0,901 | 0,918 | **0,942** |
| ms/cara | 0,4 | 0,4 | 48,5 |
| Backbone | próprio (MobileNet-v1, 96×96) | próprio | próprio (ResNet-34, 224×224) |

- O fine-tuning reduziu o MAE da MobileNet em 36%.
- O FairFace, maior e sem treino nosso, foi o melhor.

### Fase 4 — Expressão ([expressao/resultados/avaliacao.md](../expressao/resultados/avaliacao.md))

| bal. acc. | HSEmotion B0 (EfficientNet-B0) | HSEmotion B2 (EfficientNet-B2) |
|---|---|---|
| AffectNet, 8 classes (3 445, sem imagens partilhadas) | 0,584 | **0,598** |
| AffectNet, 7 classes | 0,615 | **0,631** |
| BTFER, 7 classes (cross-dataset, 1 847) | **0,749** | 0,745 |
| ms/cara | 11,9 | 25,1 |

São modelos pré-treinados pelos autores. Reproduzimos os números publicados a menos de 1 p.p.

### O problema da abordagem com especialistas
Cada rede repete o trabalho de "olhar para a cara" com o seu próprio backbone. Para ter tudo (identidade + idade/género
+ expressão) é preciso correr 3 redes por cara, com 3 crops diferentes. O FairFace sozinho custa ~50 ms por cara, o
que torna as combinações precisas lentas com várias caras (secção 7). **Foi isto que motivou o modelo final.**

---

## 5. O modelo final: rede multitarefa com backbone

Código: [multitarefa/rede.py](../multitarefa/rede.py) · Ficheiro: `models/multitarefa.onnx` (25 MB, 6,2 M parâmetros)

### 5.1 Ideia (notas da aula)
> "Backbone cria embedding e a partir disso temos um modelo para tudo… convoluções para cada propósito… ter as
> redes para tudo menos identificação (identificação ser invariante às emoções)."

1. Usa-se como backbone o **ArcFace MobileFaceNet** (`w600k_mbf`, treinado pela InsightFace em ~600 mil
   identidades). As suas features já descrevem muito bem a estrutura de uma cara.
2. Corta-se a rede a meio, no mapa **256×14×14**. Daí para a frente há **um ramo por tarefa**.
3. O **ramo da identidade é o original e fica congelado**: o embedding continua invariante à expressão, à idade e à
   pose. Se fosse treinado com expressões, aprenderia precisamente a ver o que deve ignorar.
4. Cada ramo de atributos é uma **cópia das camadas finais do ArcFace** (arranca com esses pesos) e é **afinado só
   para a sua tarefa**.

### 5.2 Camadas (lidas do grafo ONNX)

```
entrada: cara alinhada 112×112 RGB, normalizada (x − 127.5)/127.5
│
├── TRONCO (partilhado, CONGELADO) — 1,09 M parâmetros, 38 convoluções
│     conv 3×3 s2 → blocos "bottleneck" MobileFaceNet (conv 1×1 expande → conv 3×3 depthwise → conv 1×1 projeta,
│     ligações residuais, ativação PReLU); resolução 112 → 56 → 28 → 14
│     saída: mapa 256×14×14
│
├── RAMO IDENTIDADE (original, CONGELADO) — 2,31 M
│     bottleneck de redução (1×1 256→512, 3×3 dw s2 → 7×7, 1×1 512→256)
│     2 bottlenecks residuais 256 (7×7) → conv 1×1 256→512
│     conv 1×1 512→64 → flatten (3 136) → FC 3 136→512 → BatchNorm → normalizar
│     saída: embedding 512-D (igual ao ArcFace original)
│
├── RAMO GÉNERO    ┐  cada um: as mesmas camadas do ramo de identidade até ao mapa 512×7×7
├── RAMO IDADE     ├  (bottleneck de redução + 2 residuais + 1×1 256→512) = 0,67 M parâmetros, AFINADO
└── RAMO EXPRESSÃO ┘  → pooling média + máximo espaciais (1 024)
                      → cabeça MLP: Dropout 0,3 → FC 1024→256 → BatchNorm → ReLU → Dropout 0,3 → FC 256→saída
                         género: 2 classes (F, M)                                 0,26 M
                         idade: 117 classes (0..116 anos, DLDL) → idade esperada  0,29 M
                         expressão: 8 classes (7 + Contempt; a app usa 7)         0,26 M
```

| Parte | Parâmetros | Treinado? |
|---|---|---|
| Tronco | 1,09 M | não |
| Ramo identidade | 2,31 M | não |
| 3 ramos de atributos | 3 × 0,67 M = 2,01 M | sim |
| 3 cabeças | 0,81 M | sim |
| **Total** | **~6,2 M** (2,8 M treinados) | |

Comparação (contado nos ONNX): FairFace ResNet-34 21,3 M parâmetros (85 MB); HSEmotion B2 7,7 M (31 MB); HSEmotion B0 4,0 M; ArcFace sozinho 13,6 MB. **A multitarefa faz as 4 tarefas com menos parâmetros do que o FairFace faz 2.**

### 5.3 Treino por blocos ([multitarefa/treinar.py](../multitarefa/treinar.py))
"Diferentes blocos de treino" e "branch específico com finetune específico" (notas):

| Ramo | Dados | Loss | Época escolhida por |
|---|---|---|---|
| Género | UTKFace train (16 384) | cross-entropy | balanced acc. no **val** do UTKFace |
| Idade | UTKFace train | DLDL (gaussiana σ=2) + L1/10 | MAE no **val** do UTKFace |
| Expressão | AffectNet train do Kaggle (33 785) | cross-entropy com pesos de classe + label smoothing 0,1 | balanced acc. no **dev** (10% do AffectNet train) |

- **Cada ramo tem o seu otimizador** (AdamW, lr 1e-3 nas cabeças e 2e-4 no ramo, ou seja, afinação 5× mais
  suave), o seu *one-cycle schedule* e a sua melhor época. Os ramos não partilham parâmetros nem gradientes. Correm
  na mesma execução só para não repetir a passagem pelo tronco congelado; o resultado é igual a treiná-los à parte.
- **Aumento de dados:** flip, rotação ±8°, escala e translação ±8%, brilho e contraste, 10% em cinzento. No
  AffectNet, a margem em falta à volta da cara é preenchida de várias formas, para a expressão não aprender a
  depender dela.
- **Dados:** crops 112×112 na geometria do ArcFace, os mesmos que a app usa ([geral/preparar.py](../geral/preparar.py)).
- **Custo:** 6 épocas × 263 passos, ~24 min por época, **~2 h 30 min** no CPU (Python x86 por Rosetta). Preparar os
  dados demorou ~25 min.

**Curvas de treino** ([multitarefa/resultados/treino.csv](../multitarefa/resultados/treino.csv)), sempre em validação/dev:

| Época | Género (bal. acc.) | Idade (MAE) | Expressão (bal. acc. dev) |
|---|---|---|---|
| 1 | 0,934 | 5,18 | 0,553 |
| 2 | 0,928 | 6,61 | 0,591 |
| 3 | 0,944 | 4,54 | 0,609 |
| 4 | 0,945 | 4,48 | 0,611 |
| 5 | **0,949** | 4,26 | 0,614 |
| 6 | 0,944 | **4,25** | **0,614** |

- **Leitura:** logo na 1.ª época a idade já está melhor do que o FairFace (5,18 vs 5,28 no teste), o que mostra a
  qualidade das features do ArcFace.
- A expressão estabiliza a partir da época 4. Mais épocas dariam pouco ganho.
- O pico da época 2 na idade (6,61) é o lr no máximo do *one-cycle*.

---

## 6. Resultados do modelo final vs especialistas

Ficheiros: [multitarefa/resultados/avaliacao.md](../multitarefa/resultados/avaliacao.md) ·
[comparacao_idade_genero.md](../multitarefa/resultados/comparacao_idade_genero.md).
Os testes são **exatamente os mesmos** das fases 3 e 4. Os números dos especialistas foram **recalculados nas mesmas
caras** e coincidem com os da fase 3, o que confirma que a comparação é justa.

### 6.1 Idade e género (UTKFace teste, 4 682 caras)

| Modelo | MAE | RMSE | Viés | CS(5) | F1 género |
|---|---|---|---|---|---|
| **Multitarefa** | **4,14** | **6,10** | **+0,07** | **0,706** | **0,952** |
| FairFace ResNet-34 | 5,28 | 7,76 | −2,25 | 0,621 | 0,942 |
| MobileNet FT ordinal | 6,03 | 8,44 | −0,12 | 0,554 | 0,918 |

**As diferenças são reais** (bootstrap emparelhado, 1 000 reamostragens das mesmas caras):

| Diferença | Valor | IC 95% |
|---|---|---|
| MAE multitarefa − FairFace | **−1,14 anos** | [−1,27, −1,01] |
| F1 género multitarefa − FairFace | **+0,010** | [+0,004, +0,017] |
| MAE multitarefa − MobileNet FT | **−1,89 anos** | [−2,04, −1,75] |
| F1 género multitarefa − MobileNet FT | **+0,034** | [+0,027, +0,042] |

**MAE / viés por faixa etária:**

| Faixa | caras | Multitarefa | FairFace | MobileNet FT |
|---|---|---|---|---|
| 0-2 | 332 | **0,93** / +0,89 | 1,00 / +0,82 | 1,39 / +1,29 |
| 3-9 | 267 | **2,01** / +1,17 | 2,18 / +1,44 | 3,91 / +2,72 |
| 10-19 | 280 | **3,26** / +1,46 | 3,29 / +1,13 | 7,62 / +5,08 |
| 20-29 | 1 412 | **3,01** / +1,74 | 3,35 / +0,74 | 4,81 / +2,36 |
| 30-39 | 934 | **4,77** / −0,13 | 5,13 / −1,82 | 5,70 / −0,44 |
| 40-49 | 463 | **5,95** / −1,37 | 7,01 / −4,32 | 7,87 / −2,73 |
| 50-59 | 439 | **5,82** / −1,13 | 7,77 / −5,56 | 8,74 / −3,41 |
| 60-69 | 263 | **6,02** / −1,79 | 10,00 / −8,92 | 8,65 / −4,53 |
| 70+ | 292 | **6,98** / −4,80 | 13,99 / −13,89 | 9,33 / −7,19 |

- O modelo final **ganha em todas as faixas**.
- A maior vantagem está nos **mais velhos**: o FairFace subestima os 70+ em 14 anos (tem um teto de 77 anos, porque
  prevê faixas) e o modelo final em 4,8. Na fase 3 nenhum modelo era bom em todas as idades; este é.
- Continua a haver uma tendência ligeira para "puxar para o centro": sobrestima os jovens (+1 a +1,7) e subestima os
  idosos. É comum em regressão de idade com poucos exemplos nos extremos.

**Por etnia anotada** (MAE bruto / MAE ajustado à idade / F1 género):

| Etnia | caras | Multitarefa | FairFace | MobileNet FT |
|---|---|---|---|---|
| White | 1 993 | **4,41 / 4,13 / 0,960** | 6,06 / 5,19 / 0,948 | 6,49 / 6,07 / 0,927 |
| Black | 904 | **4,60 / 4,82** / 0,960 | 5,92 / 6,15 / 0,960 | 6,90 / 7,37 / 0,930 |
| Asian | 668 | **3,04 / 3,91 / 0,931** | 3,66 / 4,54 / 0,902 | 4,72 / 5,95 / 0,879 |
| Indian | 784 | **4,07 / 4,40** / 0,957 | 4,62 / 5,36 / 0,958 | 5,49 / 6,29 / 0,918 |
| Others | 332 | **3,64 / 3,78 / 0,915** | 3,72 / 3,86 / 0,903 | 4,86 / 5,35 / 0,900 |

- Melhora todos os grupos. **A desigualdade mantém-se:** ajustado à idade, o grupo Black continua com o maior erro
  (4,82 vs 4,13 no White).
- O género é pior nos grupos Asian e Others (F1 0,93 e 0,92 vs 0,96), o mesmo padrão dos especialistas.
- Estes pontos devem entrar na discussão ética do relatório.

**Género por classe (multitarefa):** F (P 0,942, R 0,958) e M (P 0,962, R 0,947): equilibrado. Erros: 92 F→M e
130 M→F em 4 682.

### 6.2 Expressão

| bal. acc. | Multitarefa | HSEmotion B0 | HSEmotion B2 |
|---|---|---|---|
| AffectNet, 8 classes (3 445) | 0,573 | 0,584 | **0,598** |
| AffectNet, 7 classes | 0,600 [0,583, 0,617] | 0,615 | **0,631** |
| BTFER, 7 classes (1 847) | 0,725 [0,704, 0,745] | **0,749** | 0,745 |

- **Fica 1,5–3 p.p. abaixo** dos especialistas. No AffectNet a 7 classes, o IC do modelo final (até 0,617)
  sobrepõe-se ao B0 (0,615), mas não ao B2 (0,631).
- **Por classe (AffectNet, 7 cl., F1):**
  - Happy 0,82 (a mais fácil);
  - Sad 0,60, Surprise 0,58, Neutral 0,57;
  - Anger, Disgust e Fear 0,54.
  - Confusões típicas, as mesmas dos especialistas: Fear↔Surprise (olhos abertos), Anger↔Disgust (sobrancelhas
    franzidas), Sad→Neutral.
- **Generalização:** sobe 12,5 p.p. no BTFER (0,600 → 0,725), como os especialistas (+11 a +13). O BTFER é mais
  fácil (expressões encenadas), por isso isto não prova "melhor generalização".
- **Rótulo "incerto":** o limiar 0,45 foi escolhido no dev (exatidão ≥ 70%). No teste, o modelo rotula 73% das caras,
  com 68,7% de acerto.

---

## 7. Tempos e desempenho em tempo real

Ficheiro: [multitarefa/resultados/benchmark.md](../multitarefa/resultados/benchmark.md) · script:
[multitarefa/benchmark.py](../multitarefa/benchmark.py)

**Método:**
- frames 1280×720 com 1, 2 e 4 caras (os mesmos do benchmark das combinações);
- 8 iterações de aquecimento + 40 medidas, mediana;
- todos os sistemas no **mesmo processo, alternados**, para a carga da máquina ser igual para todos;
- Apple M1 de 8 GB, só CPU, Python x86 por Rosetta.

Os especialistas **não tinham identidade**. Mede-se também "+ ArcFace" porque o modelo final já a inclui: só assim a
comparação é justa.

**Atualização completa = deteção + modelos.** Cada célula: ms totais / **atualizações por segundo** (só modelos entre
parênteses). Negrito = ≥ 5/s.

| Sistema | 1 cara | 2 caras | 4 caras |
|---|---|---|---|
| **Multitarefa: tudo** (identidade + idade + género + expressão, galeria, média temporal) | **44 / 22,7** (18) | **52 / 19,1** (28) | **70 / 14,3** (47) |
| Multitarefa sem média temporal | **42 / 23,8** (18) | **51 / 19,7** (28) | **69 / 14,6** (46) |
| "rapido" MobileNet FT + B0, **sem identidade** | **46 / 21,5** (23) | **58 / 17,3** (34) | **78 / 12,7** (56) |
| "rapido" + ArcFace | **64 / 15,5** (39) | **82 / 12,2** (57) | **115 / 8,7** (92) |
| "preciso" FairFace + B2, **sem identidade** | **111 / 9,0** (88) | **191 / 5,2** (163) | 324 / 3,1 (301) |
| "preciso" + ArcFace | **127 / 7,9** (104) | 208 / 4,8 (185) | 364 / 2,7 (340) |

Deteção SCRFD: 24 ms em todos (não depende do nº de caras).

**Como ler:**
- **Com o mesmo trabalho (tudo, incluindo identidade), o modelo final é 1,6× mais rápido que a combinação mais leve e
  5× mais rápido que a mais precisa** com 4 caras (70 vs 115 e 364 ms). Mesmo **sem** identidade, os especialistas mais leves são mais
  lentos do que o modelo final **com** ela.
- **Melhor qualidade e mais rápido ao mesmo tempo:** antes era preciso escolher entre "preciso" (FairFace, lento) e
  "rápido" (MobileNet, MAE 6). Agora a melhor idade/género vem com o menor custo.
- **Por cara**, a rede sozinha custa ~11 ms isolada (batch 1) e ~10 ms por cara extra na app (inclui alinhamento,
  nitidez e galeria). O custo cresce linearmente com o nº de caras, mas devagar: 4 caras ainda dão 14/s.
- **A média temporal custa ~1 ms** por atualização (tracking + EMA): é desprezável.
- **Porque é mais rápido:** um só backbone (em vez de 3), uma só entrada 112×112 (em vez de 96/224/260) e um só
  alinhamento. A parte por tarefa (os ramos) trabalha em mapas de 14×14 e 7×7, que são baratos.
- **Memória:** 25 MB de modelo, contra 85 (FairFace) + 31 (B2) + 14 (ArcFace) MB. O processo do benchmark tinha
  todos os modelos carregados (671 MB) e por isso não é comparável. Na fase 3/4, "rapido" usava 323 MB e "preciso"
  549 MB.
- **Ressalva:** com Python arm64 nativo tudo seria mais rápido (não medido). As proporções devem manter-se.

---

## 8. Estabilidade no tempo (média pesada dos frames)

Código: [multitarefa/temporal.py](../multitarefa/temporal.py) · notas: *"estimativas com base em média pesada dos frames
anteriores"* · enunciado, Stage 2: *"stabilise outputs over time"*.

**Como funciona:**
1. Cada cara é associada à caixa mais sobreposta do frame anterior (IoU ≥ 0,3).
2. Para essa cara, faz-se a média exponencial das **probabilidades** de género e de expressão, da idade e das
   similaridades com a galeria: `s = (1 − w)·s + w·novo`, com `w = 0,35 × qualidade`.
3. **Qualidade** = tamanho da cara × score do detetor × nitidez. Um frame desfocado ou pequeno quase não conta.
4. O nome tem **histerese**: só muda quando a similaridade média passa o limiar, e só se perde quando desce 0,05
   abaixo dele.

**Teste** (vídeos sintéticos: 300 caras × 20 frames por tarefa, com pequenas perturbações de posição, rotação, brilho,
ruído e desfoque, como uma pessoa parada à frente da webcam):

| | Sem média | Com média |
|---|---|---|
| Expressão: trocas de rótulo por frame | 0,119 | **0,024** (5× menos) |
| Género: trocas por frame | 0,022 | **0,006** |
| Idade: variação dentro do vídeo (desvio-padrão) | 1,25 anos | **0,35 anos** |
| Exatidão da expressão por frame | 0,607 | **0,619** |
| MAE da idade | 4,13 | **3,98** |
| Atraso quando a expressão muda | — | mediana **1 frame** (média 1,7) |

- A média torna os rótulos estáveis **e** ligeiramente mais exatos, porque um frame mau é diluído pelos bons.
- O preço é cerca de 1 a 2 frames de atraso (~0,1 s), impercetível.
- Na app, a tecla `s` liga e desliga a média, o que dá para mostrar a diferença no vídeo de demonstração.
- ⚠️ O enunciado diz que *"face tracking and temporal identity association are explicitly excluded"*. A associação
  por IoU aqui serve **apenas para suavizar**: não é um tracker avaliado nem um requisito. A histerese do nome é
  um extra. Convém dizê-lo assim no relatório.

---

## 9. Identidade

Código: [geral/galeria.py](../geral/galeria.py), [geral/identidade.py](../geral/identidade.py),
[multitarefa/registar_video.py](../multitarefa/registar_video.py)

- **Embedding:** o do ArcFace original. Confirmado: no modelo final, o cosseno mínimo com o ArcFace é 0,9999996 em
  256 caras, ou seja, igual. A identidade **não piora** por se treinar o resto.
- **Galeria em modo clusters** (notas: *"cada pessoa tem as suas representações específicas"*): k-means esférico dos
  embeddings de cada pessoa → 1 a 4 protótipos (por exemplo frente, lado, longe). A similaridade com a pessoa é a do
  protótipo mais parecido. Guardam-se apenas vetores, nunca imagens. Máximo de 19 pessoas.
- **Registo por vídeo** (notas: *"vídeo de frente e de lado, expressões, longe"*): de cada frame só se guarda a cara
  se tiver ≥ 60 px, score ≥ 0,70, estiver nítida, não for quase repetida (cos < 0,95) e for coerente com a mesma
  pessoa (cos ≥ 0,35). Responde à pergunta *"que frames?"*.
- **Limiar provisório:** 0,27, escolhido com desconhecidos do UTKFace (FAR 0,87% na validação e 0,76% no teste).
- ⚠️ **Ainda não há resultados de reconhecimento** (rank-1, FRR, comparação media/clusters/vizinho). É preciso
  gravar os voluntários (secção 12).

---

## 10. Interpretação

**Porque ganha na idade e no género?**
1. **Features melhores à partida.** O ArcFace aprendeu a descrever a forma da cara (ossos, proporções, rugas,
   textura) a partir de milhões de imagens de 600 mil pessoas. A idade e o género estão muito ligados a essa
   estrutura: logo na 1.ª época o MAE de validação já era 5,18.
2. **Treino nos mesmos dados e no mesmo crop que a app usa.** O FairFace nunca viu o UTKFace e prevê faixas (teto de
   77 anos). Os ramos da multitarefa foram afinados no UTKFace com DLDL, que prevê qualquer idade de 0 a 116.
3. **Mais capacidade do que a MobileNet FT:** 0,67 M parâmetros por ramo sobre um tronco forte, contra uma MobileNet-v1
   96×96 sem alinhamento.

**Porque perde (pouco) na expressão?**
1. **O tronco foi treinado para ignorar a expressão.** A *loss* do ArcFace junta a mesma pessoa a sorrir e séria. Por
   isso, as features de 14×14 já perderam parte do sinal da expressão, e o ramo tem de o recuperar com o que resta.
   Este é **o compromisso fundamental** de partilhar o backbone da identidade: a mesma invariância que torna a
   identidade boa prejudica a expressão.
2. **Os especialistas têm uma rede inteira para a expressão**, maior (EfficientNet-B0/B2), com mais resolução
   (224/260 vs 112) e treinada pelos autores com o AffectNet completo. Nós usámos só o subconjunto do Kaggle.
3. **O tronco está congelado.** Afinar também o tronco ajudaria a expressão, mas estragaria a identidade, que é
   exatamente o que as notas pedem para evitar.

**Conclusão:** para uma aplicação em tempo real que precisa das 4 tarefas, o modelo final é a melhor escolha. Ganha
claramente em 2 tarefas e no custo, perde pouco numa e é igual na identidade. Se a expressão fosse prioritária, havia
um híbrido possível: multitarefa para identidade, idade e género + HSEmotion B0 só para a expressão. Custaria mais
~12 ms por cara (estimativa, não medida).

---

## 11. Limitações e cuidados ao reportar

1. **Uma só execução de treino (seed 0), 6 épocas.** Não se mediu a variação entre treinos. Na fase 3 essa variação era
   ~±0,5 anos; a vantagem de 1,1 anos sobre o FairFace está acima disso, mas a da expressão (1,5–3 p.p.) pode ser
   sensível ao seed.
2. **Split do UTKFace por imagem, não por pessoa.** A mesma pessoa pode estar no treino e no teste. Isto favorece os
   modelos treinados no UTKFace (multitarefa e MobileNet FT), não o FairFace. Parte da vantagem pode vir daqui.
3. **Os modelos diferem em várias coisas ao mesmo tempo** (backbone, resolução, dados, formulação da idade). Não se pode
   atribuir o ganho a um único fator.
4. **O teste temporal é sintético.** Mostra o efeito do método, mas não substitui um vídeo real anotado.
5. **Os números vêm de datasets de fotos.** Na webcam (pouca luz, perfil, compressão), todos os modelos devem piorar.
   Falta o teste qualitativo.
6. **Identidade sem avaliação com pessoas reais**, e o limiar ainda é provisório.
7. **Rótulos e categorias:**
   - idade aparente ≠ idade real;
   - género binário atribuído por anotadores ≠ identidade de género;
   - expressão ≠ emoção.
   A app mostra estimativas e esconde as incertas (secções 5.7 e 6.6 do PROGRESSO).
8. **Expressão:** treinada no subconjunto do AffectNet do Kaggle; as imagens partilhadas com a validação foram
   retiradas da avaliação (como na fase 4).

---

## 12. O que falta

| Tarefa | Como |
|---|---|
| Testar a identidade com voluntários | gravar ~20 s por pessoa → `python multitarefa/registar_video.py NOME video.mp4`; fotos em `data/identidades/{registo,validacao,teste}/` → `python geral/identidade.py calibrar` e `testar` |
| Teste qualitativo na webcam | `python app_geral.py`: luz fraca, perfil, distância, várias pessoas; capturas para o relatório |
| (Opcional) comparar ramo partilhado vs um por tarefa | `python geral/treinar.py --mode ramo` (~2,5 h) e `python multitarefa/avaliar.py` (põe as duas colunas lado a lado) |
| (Opcional) WIDER val completa | `python deteccao/eval_wider.py --detectors scrfd retinaface_mobile` |
| Relatório e vídeo | este guia + PROGRESSO têm as tabelas e a discussão |

---

## 13. Onde está cada coisa

| O quê | Onde |
|---|---|
| Como correr tudo | [../README.md](../README.md) |
| Plano das notas da aula | [../multitarefa/README.md](../multitarefa/README.md), [notas_aula.md](notas_aula.md) |
| Resultados do modelo final | [../multitarefa/resultados/](../multitarefa/resultados/): `avaliacao.md`, `comparacao_idade_genero.md`, `benchmark.md`, `treino.csv` |
| Deteção / alinhamento | [../deteccao/resultados/](../deteccao/resultados/) |
| Idade e género (especialistas) | [../idade_genero/resultados/idade_genero.md](../idade_genero/resultados/idade_genero.md) |
| Expressão (especialistas) | [../expressao/resultados/avaliacao.md](../expressao/resultados/avaliacao.md) |
| Custo das combinações de especialistas | [../resultados_combinacoes/combinacoes.md](../resultados_combinacoes/combinacoes.md) |
| Histórico completo das fases 1–4 | [PROGRESSO_E_RESULTADOS.md](PROGRESSO_E_RESULTADOS.md) |
