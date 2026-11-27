# Custo da atualização completa: modelo final vs especialistas

Frames 1280×720; mediana de 40 medições após 8 de aquecimento; mesmo processo, sistemas alternados; só CPU, Python x86_64 por Rosetta. Cada célula: **ms totais (deteção + modelos) / atualizações por segundo**; entre parênteses, só os modelos.

| Sistema | 1 cara | 2 caras | 4 caras |
|---|---|---|---|
| multitarefa (tudo, com galeria de 5 pessoas e média temporal) | **44 / 22.7** (18) | **52 / 19.1** (28) | **70 / 14.3** (47) |
| multitarefa sem média temporal | **42 / 23.8** (18) | **51 / 19.7** (28) | **69 / 14.6** (46) |
| rapido: MobileNet FT + B0 (sem identidade) | **46 / 21.5** (23) | **58 / 17.3** (34) | **78 / 12.7** (56) |
| rapido + ArcFace | **64 / 15.5** (39) | **82 / 12.2** (57) | **115 / 8.7** (92) |
| preciso: FairFace + B2 (sem identidade) | **111 / 9.0** (88) | **191 / 5.2** (163) | 324 / 3.1 (301) |
| preciso + ArcFace | **127 / 7.9** (104) | 208 / 4.8 (185) | 364 / 2.7 (340) |

Deteção SCRFD (igual para todos): 24 ms. Memória máxima do processo (todos os modelos carregados): 671 MB. Negrito = ≥ 5 atualizações/s.
