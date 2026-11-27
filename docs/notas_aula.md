# Notas da aula (tal como foram tiradas)

> A interpretação e o plano para as aplicar estão em [../multitarefa/README.md](../multitarefa/README.md).

- Backbone cria embedding e com esse embedding e a partir disso temos um modelo para tudo
- Identificar as pessoas exatas (ter fotos de cada pessoa)
- Modelo para criar o embedding da imagem e depois os diferentes modelos a fazer as classificações
- Diferentes blocos de treino
- Descritor que vai buscar convoluções do backbone, última layer para género, depois outra para idades,
  expressões e outra para identificação com ArcFace
- Backbone e depois especializa em cada feature
- Ter uma parte de rede pré-treinada, ter as redes para tudo menos identificação
  (identificação ser invariante às emoções etc.)
- Branch específico vai ter um finetune específico
- Convoluções para cada propósito
- Vídeo de frente e de lado (para cada pessoa, expressões diferentes, longe, etc.)
- Cluster com cada tipo de imagem: embeddings e média (cada pessoa tem as suas representações específicas)
- ArcFace funciona com base na loss
- Backbone —— convolução para cada
- Que frames?
- Estimativas com base em média pesada dos frames anteriores
