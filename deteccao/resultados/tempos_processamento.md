# Tempos de processamento — deteção de caras (5 imagens WIDER FACE val)

- CPU: Apple M1 (x86_64), só CPU (onnxruntime CPUExecutionProvider / TF CPU)
- 20 repetições por imagem após 1 de aquecimento; tempo = só `detect()` (inclui pré e pós-processamento)

## Tabela 1 — Resumo por modelo (média das 5 imagens)

| Modelo | Configuração | ms/imagem (média ± dp) | mediana (ms) | FPS | vs SCRFD |
|---|---|---|---|---|---|
| RetinaFace R50 | scale 0.5, thr 0.9 | 399.2 ± 75.2 | 354.8 | 2.5 | 12.4× |
| RetinaFace R50 | scale 1.0, thr 0.9 | 1646.4 ± 221.5 | 1448.1 | 0.6 | 51.1× |
| RetinaFace MobileNet0.25 | scale 1.0, thr 0.9 | 48.8 ± 13.2 | 38.6 | 20.5 | 1.5× |
| SCRFD-500MF | 640x640, thr 0.5 | 32.2 ± 15.4 | 21.5 | 31.1 | 1.0× |

## Tabela 2 — ms por imagem (média ± dp)

| Modelo | Configuração | 0--Parade (1 caras, 1024x1024) | 1--Handshaking (4 caras, 1024x736) | 11--Meeting (12 caras, 1024x732) | 12--Group (30 caras, 1024x768) | 2--Demonstration (81 caras, 1024x754) |
|---|---|---|---|---|---|---|
| RetinaFace R50 | scale 0.5, thr 0.9 | 470.9 ± 28.3 | 432.1 ± 113.6 | 374.5 ± 97.4 | 339.5 ± 17.5 | 379.2 ± 69.0 |
| RetinaFace R50 | scale 1.0, thr 0.9 | 2150.5 ± 236.4 | 1552.4 ± 278.5 | 1446.4 ± 150.7 | 1596.4 ± 276.7 | 1486.2 ± 112.2 |
| RetinaFace MobileNet0.25 | scale 1.0, thr 0.9 | 53.8 ± 5.3 | 38.0 ± 3.2 | 38.3 ± 3.6 | 49.2 ± 15.1 | 64.9 ± 24.4 |
| SCRFD-500MF | 640x640, thr 0.5 | 48.1 ± 16.9 | 32.7 ± 16.1 | 22.7 ± 7.6 | 25.5 ± 12.4 | 32.0 ± 20.7 |
