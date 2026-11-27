# Custo das combinações de modelos (atualização completa: SCRFD + idade/género + expressão)

- Apple M1 (8 GB), só CPU, Python x86_64 via Rosetta; frames 1280×720; mediana de 40 medições, um processo por combinação, ordem aleatória
- Controlo da carga da máquina: a deteção (trabalho igual em todas) demorou 17–24 ms entre combinações
- **negrito** = pelo menos 5 atualizações dos rótulos por segundo

| Idade/género | Expressão | 1 cara: ms (p95) / atual. por s | 2 caras: ms (p95) / atual. por s | 4 caras: ms (p95) / atual. por s | Núcleos CPU (4 caras) | Memória (MB) |
|---|---|---|---|---|---|---|
| off | off | **17 (18) / 59.2** | **17 (42) / 58.7** | **17 (18) / 59.1** | 4.0 | 216 |
| off | b0 | **38 (40) / 26.4** | **49 (51) / 20.6** | **72 (77) / 13.9** | 4.2 | 314 |
| off | b2 | **49 (55) / 20.4** | **72 (97) / 13.9** | **120 (174) / 8.3** | 4.0 | 338 |
| mobilenet | off | **24 (26) / 41.1** | **25 (29) / 39.4** | **27 (29) / 37.3** | 5.2 | 240 |
| mobilenet | b0 | **42 (63) / 23.7** | **53 (62) / 18.8** | **74 (76) / 13.5** | 4.4 | 301 |
| mobilenet | b2 | **55 (58) / 18.1** | **77 (81) / 12.9** | **128 (139) / 7.8** | 4.2 | 423 |
| mobilenet_ft | off | **25 (28) / 40.0** | **26 (33) / 39.0** | **27 (30) / 37.0** | 4.9 | 251 |
| mobilenet_ft | b0 | **43 (46) / 23.0** | **54 (68) / 18.6** | **78 (93) / 12.9** | 4.4 | 323 |
| mobilenet_ft | b2 | **55 (63) / 18.0** | **79 (98) / 12.7** | **129 (140) / 7.7** | 4.2 | 410 |
| fairface | off | **75 (80) / 13.3** | **122 (138) / 8.2** | 215 (221) / 4.7 | 4.1 | 419 |
| fairface | b0 | **90 (112) / 11.1** | **150 (270) / 6.7** | 273 (372) / 3.7 | 4.0 | 385 |
| fairface | b2 | **107 (121) / 9.4** | **178 (187) / 5.6** | 326 (342) / 3.1 | 4.0 | 549 |

## Tempo por componente (ms, mediana)

| Componente | 1 cara | 2 caras | 4 caras |
|---|---|---|---|
| Deteção SCRFD | 21.5 | 21.8 | 21.9 |
| idade/género: mobilenet | 1.2 | 2.0 | 3.3 |
| idade/género: mobilenet_ft | 1.1 | 1.9 | 3.4 |
| idade/género: fairface | 53.1 | 100.5 | 196.7 |
| expressão: b0 | 18.4 | 28.8 | 49.7 |
| expressão: b2 | 31.5 | 54.1 | 103.3 |
