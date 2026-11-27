# Plano de implementação — Assignment #3 (Real-Time Facial Analysis)

Entrega: **29/11/2026**

## Ideia central

O trabalho é **integrar** modelos pré-treinados numa aplicação e **avaliá-los** — não treinar redes
de raiz. Cada frame percorre:

```
frame → detetar caras → alinhar cada cara → (idade/género, expressão, identidade ArcFace) → desenhar
```

Cada módulo deve receber/devolver estruturas bem definidas (frame, deteção, crop alinhado,
descritor, atributos), para se poder trocar um modelo sem mexer no resto.

**Regra:** uma fase só está concluída com **o teste de sanidade a passar** e **a métrica
obrigatória numa tabela** (essas tabelas são o esqueleto do relatório).

---

## Fase 1 — Aquisição + Deteção *(a maior parte já está feita: main.py, face_detector.py)*

**Implementar**
- Fonte de vídeo: webcam **ou** ficheiro de vídeo, com a mesma interface (frame + timestamp).
- Estrutura de deteção comum a todos os detetores: caixa, score, 5 landmarks.
- Tamanho mínimo de cara (caras pequenas não seguem para as fases seguintes).
- Avaliação no **WIDER FACE — partição de validação completa vs ground truth** (só avaliação, sem treino):
  - leitor de `wider_face_val_bbx_gt.txt` (caixas + atributos blur/oclusão/pose/invalid);
  - matching deteção ↔ GT por **IoU ≥ 0.5** (cada GT só casa com uma deteção, por ordem de score);
  - varrimento do threshold → curva precisão-recall → **AP**.

**Testes**
| Teste | O que esperar / reportar |
|---|---|
| Sanidade webcam e vídeo | mesmo comportamento nas duas fontes |
| Visual | landmarks nos olhos, nariz e cantos da boca |
| Validar o matching | numa imagem contada à mão, TP/FP/FN corretos |
| **WIDER val completa** | curva P-R + AP; P e R no threshold usado na app |
| Recall por condição | tamanho (pequena/média/grande), oclusão, pose, multidão |
| Comparação | SCRFD vs RetinaFace (AP e ms/imagem) |
| (Opcional) protocolo oficial | easy/medium/hard com o toolbox do WIDER |
| Robustez qualitativa | rodar a cabeça, tapar meia cara, afastar, várias pessoas |

---

## Fase 2 — Alinhamento

**Implementar**
- `align(frame, landmarks)`: transformação de semelhança dos 5 pontos para o template canónico do
  ArcFace (112×112) + `warpAffine`.
- Crop quadrado simples (caixa + margem) para modelos que não precisem de alinhamento.

**Testes**
| Teste | O que esperar |
|---|---|
| Visual | olhos sempre à mesma altura |
| Inclinar a cabeça ±30° | cara alinhada fica "direita" |
| Cara junto à borda | não rebenta (pad com preto) |
| Impacto (feito na Fase 5) | similaridade da mesma pessoa maior **com** alinhamento |

---

## Fase 3 — Idade + Género  ✅ *(feito: pasta `idade_genero/`, ver o README lá dentro)*

**Enquadramento exigido pelo enunciado (4.2)**
- Idade = **idade aparente** estimada pela aparência, não idade cronológica → mostrar valor
  arredondado + intervalo (ex.: "~32 (30-39)").
- Género = **classificação visual nas categorias do dataset** (UTKFace: female/male), i.e.
  apresentação de género aparente — **não** identidade de género.
- Confiança baixa → "incerto" (rótulo de género escondido, texto a cinzento).

**Implementado**
- Modelo base: `genderage.onnx` (InsightFace, MobileNet com ramo de género e ramo de idade).
- Dados: UTKFace in-the-wild → deteção SCRFD (a mesma da app) → crops 96×96;
  23 407 caras (697 sem deteção), split fixo 70/10/20 (16 384 / 2 341 / 4 682).
- Port do ONNX para PyTorch + **fine-tuning multi-tarefa** (idade + género) com 4 formulações
  da idade: **regressão**, **classes etárias**, **distribuição (DLDL)** e **ordinal (CORAL)**.
- Modelos exportados para ONNX (`models/genderage_ft_*.onnx`) → a app não precisa de torch.
- Segundo modelo, **só pré-treinado**: FairFace ResNet-34 (224×224, alinhado pelos landmarks).
- App: FairFace por omissão; `--menu`, `--combo` ou `--age-gender` / `--expression` escolhem os modelos (teclas `g`, `e`, `c`).

**Resultados (teste UTKFace, 4 682 caras)** — tabela completa em
`idade_genero/resultados/idade_genero.md`

| | MobileNet original | FairFace (sem FT) | FT reg | FT cls | FT DLDL | FT ordinal |
|---|---|---|---|---|---|---|
| MAE (anos) | 9.45 | **5.28** | 6.29 | 6.26 | 6.22 | 6.03 |
| RMSE | 12.56 | **7.76** | 9.09 | 8.87 | 8.95 | 8.44 |
| Viés (anos) | +5.90 | −2.25 | +1.06 | +1.61 | +1.73 | **−0.12** |
| F1 macro género | 0.901 | **0.942** | 0.921 | 0.924 | 0.918 | 0.918 |
| ms / cara (CPU) | 0.4 | 48.5 | 0.4 | 0.4 | 0.4 | 0.4 |

- Limiar 0.75 no género (cobertura / F1 nas caras com rótulo): FT ordinal 90.7% / 0.951; FairFace 93.7% / 0.963.
- FairFace − FT ordinal: MAE −0.75 anos (IC 95% [−0.91, −0.57]); perde nos 60+ (subestima) e é ~130× mais lento.

**Pontos para a discussão no relatório**
- O modelo original prevê ~20–30 anos para crianças ("chão" de idade; dados de treino com
  poucos menores) → MAE 19.5 nos 3–9 anos; o fine-tuning baixa para ~4.
- A variante `cls` satura na idade média da última faixa (~80): a granularidade das faixas limita
  a estimativa. As formulações ordinal/distribuição exploram a ordem das idades e têm melhor RMSE.
- Piores faixas: 10–19 e 40–69 (menos exemplos e maior ambiguidade da idade aparente).
- Diferenças demográficas: ajustando à idade, Black tem o maior MAE (FT ordinal 7.4, FairFace 6.2);
  a diferença White/Asian da MobileNet FT era sobretudo efeito da idade. F1 do género mais baixo em
  Asian/Others. O FairFace baixa o erro de todos, mas não fecha a diferença entre grupos.
- Limitações das anotações: idades do UTKFace estimadas/rotuladas de forma ruidosa, etnia
  em 5 categorias grosseiras, género binário; possíveis pessoas repetidas entre splits.
- Riscos: inferir atributos sensíveis pela aparência pode errar de forma sistemática para
  certos grupos e não deve ser usado para decisões sobre pessoas.
- A validação oscila ±0.5 de MAE entre épocas → escolha da época pela validação, teste separado.

**Ainda por fazer (manual)**
| Teste | Reportar |
|---|---|
| Sanidade webcam | idades plausíveis e estáveis; "incerto" aparece em perfil/pouca luz |
| Condições | pouca luz, perfil, distância (qualitativo, capturas para o relatório) |

---

## Fase 4 — Expressão facial  ✅ *(feito: pasta `expressao/`; resultados em `PROGRESSO_E_RESULTADOS.md`, secção 6)*

**Implementar**
- Dois modelos **pré-treinados no AffectNet pelos autores**, sem treino nosso: HSEmotion B0 e B2. O enunciado
  aceita *"pre-trained … or fine-tune"*, mas o ponto 1 da avaliação fala em *"train"*: dizê-lo no relatório.
- Classe mais provável + confiança (+ top-3 opcional) + "uncertain".
- Mapeamento de classes entre datasets (só classes comuns).

**Testes**
| Teste | Reportar |
|---|---|
| Sanidade webcam | happy/surprise/neutral funcionam; fear/disgust falham mais |
| **Intra-dataset** | AffectNet → validação oficial: accuracy, **balanced accuracy**, macro P/R/F1, matriz de confusão |
| **Cross-dataset** | mesmo modelo, sem re-treino, no BTFER: mesmas métricas |
| Fuga de dados | nenhuma imagem de treino na avaliação |
| Comparação de modelos | HSEmotion B0 vs B2, com IC por bootstrap |

---

## Fase 5 — Identidade com ArcFace (galeria < 20 pessoas)

```
ENROLLMENT:      N imagens → detetar → alinhar 112×112 → backbone ArcFace → N descritores → guardar
RECONHECIMENTO:  cara      → detetar → alinhar 112×112 → backbone ArcFace → descritor
                 → cosseno vs descritores da galeria → nome | "Unknown"
```

**Implementar**
- Extrator de descritores: backbone ArcFace (`w600k_mbf.onnx`, já tens) → vetor 512-d normalizado (L2).
- Enrollment com consentimento: guardar **só descritores**, nunca imagens; remover pessoa; limpar galeria.
- Comparação de descritores com duas estratégias:
  1. **template médio** — média dos descritores da pessoa;
  2. **vizinho mais próximo** — comparar com todos os descritores e ficar com o máximo.
- Limiar → "Unknown".
- Protocolo: pessoas registadas (imagens em enrollment / validação / teste) + pessoas desconhecidas
  (nunca registadas).

**Testes**
| Teste | Reportar |
|---|---|
| Sanidade | tu vs tu ≈ 0.5–0.8; tu vs outro ≈ 0–0.3 |
| **Calibração do limiar** | na **validação**: FAR/FRR vs limiar, escolher p/ FAR ≈ 1% — **nunca** no teste |
| **Teste** | rank-1, FAR, FRR, rejeição de Unknown |
| Template médio vs vizinho mais próximo | mesmas métricas |
| Nº de imagens de enrollment | 1, 3, 5, 10 |
| (Opcional) backbone | MobileFaceNet (`w600k_mbf`) vs ResNet50 (`w600k_r50`, pacote buffalo_l) |
| Com/sem alinhamento | mostra porque o ArcFace precisa da cara alinhada |
| Condições | pose, distância, luz, óculos/máscara |
| Desconhecido | pessoa não registada → **"Unknown"** |

---

## Fase 6 — Aplicação integrada + Tempo real

**Implementar**
- Pipeline que liga tudo, com módulos que se ligam/desligam.
- Overlay: caixa + identidade/Unknown em destaque; idade, género, expressão como secundários;
  incertos a cinzento.
- Interface: start/stop câmara, escolher fonte, registar/remover/limpar galeria, ligar/desligar módulos.
- Monitorização: FPS, ms por etapa, CPU, RAM.
- Otimização: análise numa thread (como o teu `AsyncDetector`), resolução de deteção, detetor leve.

**Testes**
| Teste | Reportar |
|---|---|
| **Benchmark** | vídeo fixo, detetor × módulos ligados: FPS, latência média/p95 por etapa, CPU, RAM |
| Escalabilidade | 1, 2, 3+ caras |
| **Testes 6.2** (gravar vídeos) | 1 pessoa · várias · entrar/sair · pose/distância/luz · 2 registados em simultâneo · 1 não registado |
| Interface | cada ação funciona e a galeria persiste após reiniciar |

---

## Fase 7 — Entrega
- README: instalação, obtenção dos modelos, execução, limitações.
- Licenças dos modelos (não incluir pesos na submissão).
- Relatório: arquitetura, todas as tabelas, limitações, **ética** (idade aparente, género como
  rótulo do dataset, expressão ≠ emoção, consentimento, eliminação de dados).
- Vídeo demo: várias caras, overlays, identidades reconhecidas e um "Unknown".

---

## Calendário sugerido
| Dias | O quê |
|---|---|
| 17–18 nov | Fase 2 (alinhamento) + descarregar WIDER val e UTKFace |
| 19–21 nov | Fase 1 (avaliação WIDER) + Fase 3 |
| 22–23 nov | Fase 5 (enrollment ArcFace, limiar, avaliação) |
| 24–25 nov | Integrar modelo de expressão + avaliação intra/cross |
| 26–27 nov | Fase 6 (app, benchmark, testes 6.2) |
| 28–29 nov | Relatório + vídeo |
