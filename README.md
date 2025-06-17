# Pigeon Detector - YOLO12n

This repository contains a lightweight and efficient pigeon detector using YOLO12n.  
The solution was developed for a hackathon organized by Creative Pack and Porto do Itaqui, focused on automating bird detection in port environments.

## Contents

- English
- Português

---

## English

## Project Overview

This project uses the YOLO12n object detection model to identify pigeons in images and videos.  
The model was trained on a public dataset available on Roboflow Universe and implemented using the Ultralytics YOLO interface.

## Dataset

We used the publicly available "pigeon" dataset:

- Dataset: https://universe.roboflow.com/wiings/pigeon-mg46t
- License: CC BY 4.0
- Citation (BibTeX):

@misc{
pigeon-mg46t_dataset,
title = { pigeon Dataset },
type = { Open Source Dataset },
author = { Wiings },
howpublished = { \url{ https://universe.roboflow.com/wiings/pigeon-mg46t } },
url = { https://universe.roboflow.com/wiings/pigeon-mg46t },
journal = { Roboflow Universe },
publisher = { Roboflow },
year = { 2022 },
month = { may },
note = { visited on 2025-06-17 },
}

## Training Pipeline

The model was trained using Ultralytics' YOLO12n with the following steps:

```python

pip install ultralytics

from ultralytics import YOLO

model = YOLO("yolo12n.pt")

model.train(
    data="/content/dataset/data.yaml",
    epochs=50,
    imgsz=640,
    project="treinamento",
    name="pombos_yolo12n",
    exist_ok=True
)

```

## Notes

- Model: YOLO12n
- Trained for: 50 epochs
- Image size: 640x640
- Dataset: Roboflow (see above)
- Developed entirely by the author during the hackathon

---

## Português

## Visão Geral do Projeto

Este projeto utiliza o modelo de detecção de objetos YOLO12n para identificar pombos em imagens e vídeos.  
O modelo foi treinado com um dataset público disponível no Roboflow Universe e implementado usando a interface Ultralytics YOLO.

## Dataset

Foi utilizado o dataset público "pigeon-mg46t":

- Dataset: https://universe.roboflow.com/wiings/pigeon-mg46t
- Licença: CC BY 4.0
- Citação (BibTeX): veja acima

## Pipeline de Treinamento

O modelo foi treinado com o YOLO12n da Ultralytics utilizando os seguintes passos:

```python
pip install ultralytics

from ultralytics import YOLO

model = YOLO("yolo12n.pt")

model.train(
    data="/content/dataset/data.yaml",
    epochs=50,
    imgsz=640,
    project="treinamento",
    name="pombos_yolo12n",
    exist_ok=True
)

```

## Notas

- Modelo: YOLO12n
- Treinamento: 50 épocas
- Tamanho da imagem: 640x640
- Dataset: Roboflow (veja acima)
- Solução desenvolvida inteiramente pelo autor durante o hackathon

## License

The dataset used in this project is under the Creative Commons Attribution 4.0 (CC BY 4.0) license.  
Please refer to the dataset's page for more information: https://universe.roboflow.com/wiings/pigeon-mg46t
