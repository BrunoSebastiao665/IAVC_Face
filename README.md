# Assignment #3 — Real-Time Facial Analysis

Deteção de caras em tempo real e, para cada cara: **identidade** (ou *Unknown*), **idade aparente**, **género aparente**
e **expressão**.

- **Modelo final:** uma rede multitarefa (backbone ArcFace partilhado + um ramo por tarefa) → [multitarefa/](multitarefa/README.md)
- **Para perceber o projeto e os resultados:** [docs/GUIA_RESULTADOS.md](docs/GUIA_RESULTADOS.md)
- **Histórico das fases 1–4:** [docs/PROGRESSO_E_RESULTADOS.md](docs/PROGRESSO_E_RESULTADOS.md)

---

## Instalação

```bash
cd P3
source faceregognition/bin/activate       # venv já criado (Python 3.9 x86_64)
# num ambiente novo: python3 -m venv faceregognition && pip install -r requirements.txt
```

- Correr **sempre a partir da pasta `P3`** (raiz).
- O aviso *"A module that was compiled using NumPy 1.x…"* ao importar a torch é inofensivo: o código não usa a ponte
  NumPy↔torch.
- No macOS, a primeira vez pede autorização para a câmara: *Definições do Sistema → Privacidade → Câmara*.

---

## 1. Aplicação principal: modelo final (`app_geral.py`)

```bash
python app_geral.py                               # webcam, modelo de geral/resultados/config.json (= multitarefa)
python app_geral.py --modelo multitarefa          # escolher o modelo explicitamente (nome em models/ sem .onnx)
python app_geral.py --source 1                    # outra câmara
python app_geral.py --source video.mp4            # ficheiro de vídeo
python app_geral.py --detector retinaface_mobile  # outro detetor: scrfd (omissão) | retinaface_mobile | retinaface
python app_geral.py --galeria media               # galeria: clusters (omissão) | media | vizinho
python app_geral.py --sem-suavizacao              # arranca sem a média pesada dos frames anteriores
```

| Tecla | Ação |
|---|---|
| `n` | registar a cara maior (escreve o nome + Enter; recolhe 10 amostras: mexe a cabeça); `x` cancela |
| `r` / `L` | remover uma pessoa / limpar a galeria (escrever `SIM`) |
| `1` `2` `3` | ligar/desligar identidade / idade+género / expressão |
| `s` | ligar/desligar a média temporal |
| `p` | pausa |
| `q` | sair |

Regista só quem deu consentimento. A galeria (`data/galeria.npz`) guarda **apenas descritores**, nunca imagens.

## 2. Aplicação com os modelos especialistas (`main.py`, fases 3–4)

Uma rede por tarefa, sem identidade. Serve de comparação.

```bash
python main.py                                    # FairFace, sem expressão
python main.py --menu                             # escolher num menu
python main.py --combo preciso                    # preciso | equilibrado | leve | rapido
python main.py --age-gender mobilenet_ft --expression b0    # cada um à mão; "off" desliga
python main.py --model retinaface_mobile          # outro detetor
python main.py --camera 1                         # outra câmara
python main.py --list                             # listar modelos e combinações
```
Teclas: `g` troca idade/género, `e` troca expressão, `c` troca combinação, `t` top-3 das expressões, `q` sai.

---

## 3. Identidade

```bash
# registo por vídeo (recomendado: ~20 s de frente, de lado, a fazer expressões e ao longe)
python multitarefa/registar_video.py Ana video_ana.mp4
python multitarefa/registar_video.py Ana 0                    # ao vivo pela webcam (q termina)
python multitarefa/registar_video.py Ana video.mp4 --max 60 --passo 2   # máx. descritores; 1 em cada N frames

# registo por fotos e avaliação
#   data/identidades/registo/<nome>/*.jpg
#   data/identidades/validacao/<nome>/*.jpg  + validacao/_desconhecidos/*.jpg
#   data/identidades/teste/<nome>/*.jpg      + teste/_desconhecidos/*.jpg
python geral/identidade.py registar        # cria a galeria a partir de registo/
python geral/identidade.py listar          # quem está registado
python geral/identidade.py remover NOME
python geral/identidade.py limpar
python geral/identidade.py provisorio      # limiar só com desconhecidos do UTKFace (já feito: 0,27)
python geral/identidade.py calibrar        # limiar na validacao/ (FAR ≤ 1%)
python geral/identidade.py testar          # rank-1, FAR, FRR; media vs clusters vs vizinho -> geral/resultados/identidade.md
```

---

## 4. Modelo final: reproduzir treino e resultados

```bash
python geral/preparar.py                    # crops 112×112 de todos os datasets (~25 min) -> data/geral/
python geral/preparar.py utk eval           # só alguns: btfer | eval | utk | train

python multitarefa/treinar.py --epochs 6    # os 3 ramos (~2 h 30 no CPU) -> models/multitarefa.onnx
python multitarefa/treinar.py --ramos expressao          # treinar só um bloco (os outros vêm de pesos.pt)
python multitarefa/treinar.py --epochs 1 --steps 2       # teste rápido do pipeline
#   outras opções: --lr (cabeças, 1e-3)  --lr-ramo (ramos, 2e-4)

python multitarefa/avaliar.py               # testes das fases 3–4 + teste temporal (~5 min) -> resultados/avaliacao.md
python multitarefa/comparar.py              # vs FairFace / MobileNet nas mesmas caras (~5 min) -> comparacao_idade_genero.md
python multitarefa/benchmark.py             # tempos da atualização completa vs especialistas (~3 min) -> benchmark.md
```
Resultados em [multitarefa/resultados/](multitarefa/resultados/). O `avaliar.py` também grava o modelo e o limiar da
expressão em `geral/resultados/config.json`, que é o que a app usa.

**Versão anterior (`geral/`, opcional):** um ramo partilhado por idade e género.
```bash
python geral/treinar.py --mode ramo         # ou --mode emb (cabeças sobre o embedding); --epochs, --steps
python geral/avaliar.py                     # precisa de geral_emb e geral_ramo treinados
```

---

## 5. Fases 1–4 (especialistas)

```bash
# Fase 1 — deteção
python deteccao/test_retinaface.py                         # imagem de exemplo, todos os detetores
python deteccao/test_retinaface.py foto.jpg --model scrfd  # a tua imagem, um detetor
python deteccao/eval_wider.py --max-images 100             # WIDER val, amostra (usa cache)
python deteccao/eval_wider.py --detectors scrfd retinaface_mobile   # validação completa (sem o R50, lento)

# Fase 2 — alinhamento
python deteccao/face_alignment.py                          # figuras e tabela -> deteccao/resultados/

# Fase 3 — idade + género
python idade_genero/preparar_utkface.py                    # ~12 min
python idade_genero/finetune.py --variant all              # ~55 min (reg, cls, dldl, ordinal)
python idade_genero/converter_fairface.py PESOS.pt --tune  # só para refazer o ONNX do FairFace
python idade_genero/avaliar.py                             # ~4 min -> idade_genero/resultados/
python idade_genero/avaliar.py --gender-threshold 0.8

# Fase 4 — expressão
python expressao/obter_hsemotion.py                        # descarrega B0 e B2
python expressao/preparar_dados.py                         # ~3 min (auditoria de imagens partilhadas)
python expressao/avaliar.py                                # ~3 min -> expressao/resultados/

# Custo das combinações de especialistas (fechar a app antes)
python benchmark_combinacoes.py                            # ~2 min -> resultados_combinacoes/
```

---

## Estrutura

```
P3/
├── app_geral.py             app com o modelo final (uma rede por cara)
├── main.py                  app com os especialistas (fases 3–4)
├── combinacoes.py           combinações de especialistas usadas pelo main.py
├── benchmark_combinacoes.py custo das combinações → resultados_combinacoes/
├── requirements.txt
├── docs/
│   ├── GUIA_RESULTADOS.md   ★ como perceber o projeto e todos os resultados
│   ├── PROGRESSO_E_RESULTADOS.md   histórico detalhado das fases 1–4
│   ├── PLANO_PIPELINE.md    plano inicial por fases
│   ├── notas_aula.md        notas da aula que deram origem ao modelo final
│   ├── enunciado/           PDF do assignment
│   └── relatorios/          PDFs de análise (v2 = a mais recente)
├── deteccao/                fases 1–2: detetores, alinhamento, avaliação WIDER (+ resultados/)
├── idade_genero/            fase 3: MobileNet fine-tuned, FairFace
├── expressao/               fase 4: HSEmotion B0/B2
├── geral/                   preparação de dados, galeria, identidade, inferência (usados pelo modelo final)
├── multitarefa/             ★ modelo final: rede, treino, avaliação, média temporal, registo por vídeo
├── resultados_combinacoes/
├── models/                  pesos ONNX (multitarefa.onnx = modelo final)
├── data/                    datasets (WIDER, UTKFace, AffectNet, BTFER, geral) e galeria
└── faceregognition/         venv Python (não mover)
```
