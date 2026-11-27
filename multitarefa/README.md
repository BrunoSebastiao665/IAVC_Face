# multitarefa/ — aplicar as notas da aula

Notas originais: [../docs/notas_aula.md](../docs/notas_aula.md).
Esta pasta é a **evolução de [../geral/](../geral/)**. Reutiliza o leitor do grafo ONNX, os dados e a galeria do
`geral/`; o ONNX final tem as mesmas saídas, por isso a app e a galeria servem para os dois modelos.
Resultados: [resultados/avaliacao.md](resultados/avaliacao.md).

---

## 1. O que as notas pedem, por palavras simples

| Nota | O que significa | Onde já estamos (`geral/`) | O que muda |
|---|---|---|---|
| *"Backbone cria embedding… um modelo para tudo"* | Uma única rede por cara. O tronco extrai as features e cada tarefa lê dessas features | ✅ tronco do ArcFace `w600k_mbf` congelado, cortado no mapa 256×14×14 | nada |
| *"Convoluções para cada propósito / branch específico com finetune específico"* | Cada tarefa tem **o seu próprio bloco convolucional**, que parte do tronco e é afinado só para essa tarefa | ⚠️ modo `ramo`: há 2 ramos (idade+género juntos, expressão) | **3 ramos de atributos: género, idade e expressão**, mais a identidade |
| *"Diferentes blocos de treino"* | Cada ramo é treinado à parte, com os seus dados, a sua loss e a sua melhor época | ⚠️ treino conjunto (um batch UTK + um batch AffectNet por passo) | treinar cada ramo **sozinho**: `--ramo genero/idade/expressao` |
| *"Redes para tudo menos identificação (invariante às emoções)"* | O ramo da identidade **não se treina**: o ArcFace já foi treinado para ignorar a expressão, a idade e a pose. Afinar o tronco com expressões estragava isso | ✅ ramo da identidade congelado, embedding igual bit a bit ao original | manter |
| *"ArcFace funciona com base na loss"* | O ArcFace é um embedding aprendido com uma *additive angular margin loss*: as caras da mesma pessoa ficam juntas na hiperesfera e as de pessoas diferentes ficam afastadas. Por isso **não há classificador de pessoas**: compara-se por cosseno com uma galeria | ✅ galeria + limiar de cosseno | manter. Treinar o ArcFace exigiria milhares de identidades, e temos <20 |
| *"Identificar as pessoas exatas (fotos de cada pessoa)"* | Registo das pessoas conhecidas | ✅ `geral/identidade.py` com fotos (registo/validação/teste) | registo também **a partir de vídeo** |
| *"Vídeo de frente e de lado… expressões, longe"* | Para cada pessoa, um vídeo curto que cubra as condições reais: pose, expressão, distância e luz | ❌ | `registar_video.py` |
| *"Cluster com cada tipo de imagem… cada pessoa tem as suas representações"* | Em vez de **uma** média por pessoa (modo `media`), **K protótipos** por pessoa (por exemplo frente, lado e longe), obtidos com k-means dos embeddings do vídeo | ⚠️ `media` (1 template) ou `vizinho` (todas as amostras) | modo `clusters`: um meio-termo entre os dois |
| *"Que frames?"* | Nem todos os frames do vídeo servem: escolher os nítidos, com cara grande e score alto, e evitar quase-duplicados | ❌ | filtro de qualidade |
| *"Média pesada dos frames anteriores"* | Na app, suavizar no tempo por cara seguida: as previsões deixam de saltar entre frames | ❌ (já estava em "O que falta", fase 3) | tracking + EMA |

---

## 2. Arquitetura alvo

```
                         ┌─► ramo IDENTIDADE (original ArcFace, congelado) ─► embedding 512-D ─► galeria (clusters) ─► nome / Unknown
cara 112×112 ─► TRONCO ──┤
 (alinhada)   (w600k_mbf,├─► ramo GÉNERO    (conv, afinado) ─► 2 classes
              congelado, ├─► ramo IDADE     (conv, afinado) ─► DLDL 0..116 → idade esperada
              até 256×14×14)
                         └─► ramo EXPRESSÃO (conv, afinado) ─► 8 classes (7 + contempt)

por cara seguida no vídeo:  previsão_t ──► média pesada (EMA) com os frames anteriores ──► desenhar
```

- Cada ramo de atributos é uma **cópia das últimas camadas do ArcFace** (arranca com esses pesos), como o
  `ramo` do `geral/`. A diferença é que passa a haver um ramo por tarefa.
- Custo: o tronco corre uma vez por cara e cada ramo acrescenta pouco. Comparar ms/cara com o `geral_ramo`.
- Pergunta para o relatório: o género e a idade beneficiam de ter ramos separados, ou partilhar o ramo
  ajudava (multi-task)? Responde-se com a comparação `geral_ramo` (partilhado) vs `multitarefa` (separado).

---

## 3. Plano por passos (cada passo fecha com teste + tabela)

### Passo 0 — Dados *(reutilizar, não refazer)*
- Usa os crops 112×112 de `geral/preparar.py` → `data/geral/{utk,aff_train,aff_eval,btfer}.npz`.
- ⚠️ **Estado atual:** só existe `btfer.npz`. Os logs (`geral/prep1.log`, `prep2.log`) mostram que a
  preparação do `aff_eval` e do `aff_train` foi interrompida, e o `utk` nem chegou a começar. É preciso correr
  `python geral/preparar.py` até ao fim (este passo bloqueia todos os outros).

### Passo 1 — Rede com um ramo por tarefa
- **Implementar:** `multitarefa/rede.py`. Reutiliza `Graph` e `split_backbone` de `geral/rede.py`. Faz 3
  cópias do ramo de atributos (`genero`, `idade`, `expressao`) com uma cabeça cada.
- **Teste:** o embedding da identidade é igual bit a bit ao do `w600k_mbf.onnx` em 10 caras, e congelar
  um ramo não altera os outros.

### Passo 2 — Treino por blocos
- **Implementar:** `multitarefa/treinar.py --ramo {genero,idade,expressao}`, em que cada execução treina só
  um ramo e guarda os pesos desse ramo.
  - género: CE no UTKFace; escolher a época pelo F1 no val
  - idade: DLDL no UTKFace; escolher a época pelo MAE no val
  - expressão: CE com pesos de classe no AffectNet train; escolher a época pela balanced accuracy no dev (10%)
- No fim, `multitarefa/exportar.py` junta tronco + 4 ramos num único `models/multitarefa.onnx`.
- **Tabela:** MAE, F1 do género e acc 8/7 + BTFER 7, lado a lado com os especialistas (fases 3–4) e com
  `geral_emb` / `geral_ramo`, mais ms/cara e MB do modelo.

### Passo 3 — Registo por vídeo + seleção de frames
- **Implementar:** `multitarefa/registar_video.py NOME video.mp4` (ou webcam com `--webcam`):
  1. detetar e alinhar em cada frame (com `deteccao/`);
  2. **filtrar frames** ("que frames?"):
     - cara ≥ 80 px, score do detetor ≥ 0.7;
     - nitidez: variância do Laplaciano acima de um limiar;
     - pose estimada pelos 5 landmarks (assimetria olhos–nariz), para etiquetar frente / lado;
     - não guardar quase-duplicados (cosseno > 0.95 com um embedding já guardado);
  3. guardar só **embeddings** (nunca imagens, pela questão da privacidade).
- Guião do vídeo (~20 s por pessoa): de frente → rodar para os dois lados → sorrir/sério/surpreso →
  afastar-se da câmara.

### Passo 4 — Galeria com clusters
- **Implementar:** modo `clusters` na galeria, baseado em `geral/galeria.py`:
  - por pessoa, k-means (K = 3–5, com cosseno nos embeddings normalizados) → K protótipos normalizados;
  - score da pessoa = **máximo** do cosseno aos seus protótipos.
- **Teste/tabela:** `media` vs `vizinho` vs `clusters` nas mesmas fotos de `data/identidades/teste/`, com
  rank-1, FAR, FRR, rejeição de Unknown e o limiar calibrado na validação (FAR ≤ 1%), separado por pose
  (frente/lado) e distância. Hipótese: `clusters` ≈ `vizinho` em exatidão, com menos memória e menos
  ruído.

### Passo 5 — Suavização temporal na app
- **Implementar:** `multitarefa/temporal.py`:
  - **tracking** simples por IoU entre frames consecutivos (com id por cara; o track morre após N frames
    sem match);
  - por track, **média exponencial pesada** (EMA, por exemplo α = 0.3) de:
    - probabilidades de género e de expressão (suavizar as probabilidades, não o rótulo);
    - distribuição DLDL da idade (ou a idade esperada);
    - similaridades com a galeria → o nome só muda quando a média passa o limiar (histerese);
  - pesar cada frame pela **qualidade** (tamanho, nitidez, score), para um frame mau quase não contar.
- **Implementar:** `app_multitarefa.py` na raiz, baseado em `app_geral.py` + `temporal.py`.
- **Teste/tabela:** num vídeo gravado, comparar com e sem EMA: nº de trocas de rótulo por segundo
  (flicker), atraso até mudar de expressão (em frames) e FPS.

---

## 4. Ficheiros (implementados)

```
multitarefa/
├── rede.py              MultiNet: tronco + identidade (congelados) + 3 ramos/cabeças; export ONNX
├── treinar.py           treino por blocos (um otimizador/loss/melhor época por ramo)
├── avaliar.py           testes das fases 3–4 + teste temporal sintético + custo
├── temporal.py          tracking por IoU + EMA pesada pela qualidade + histerese do nome
├── registar_video.py    registo por vídeo com seleção de frames → galeria
└── resultados/          pesos.pt, treino.csv/.log, avaliacao.md
geral/galeria.py         + modo 'clusters' (k-means esférico, até 4 protótipos por pessoa)
geral/modelo.py          predict(..., smoother=) — média pesada opcional
app_geral.py             --modelo multitarefa, tecla [s] liga/desliga a suavização, galeria 'clusters' por omissão
models/multitarefa.onnx  a rede toda num ficheiro
```

Comandos:
```
python geral/preparar.py                       # crops 112x112 (feito)
python multitarefa/treinar.py --epochs 6       # ~3 h no CPU desta máquina
python multitarefa/avaliar.py                  # -> resultados/avaliacao.md e config da app
python multitarefa/registar_video.py Ana video_ana.mp4
python geral/identidade.py testar              # media vs clusters vs vizinho (precisa de data/identidades/)
python app_geral.py --modelo multitarefa
```

## 5. Ordem sugerida, tendo em conta a entrega a 29/11

1. Passo 0 (correr a preparação, que demora horas) → enquanto corre, fazer os **passos 3 e 4**, que só
   precisam do ArcFace original e não do treino.
2. Passo 5 (temporal): melhora a app de imediato, mesmo com o `geral_ramo` ou com os especialistas.
3. Passos 1–2 (ramos separados) por último: são o que tem menos impacto visível e o maior custo de treino.
