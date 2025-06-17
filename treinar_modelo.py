import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Corrige conflito OpenMP

from ultralytics import YOLO
import torch

# Verifica se a GPU está disponível
if torch.cuda.is_available():
    device = "cuda"
    print(f'GPU detectada: {torch.cuda.get_device_name(0)}')
else:
    device = "cpu"
    print("GPU não detectada. Usando CPU.")

# Carrega o modelo YOLOv8n pré-treinado
model = YOLO("D:/Porto Itaqui/Yolo/Aplicação/pesos/yolov8n.pt")

# Treina com seu dataset usando a GPU
model.train(
    data="D:/Porto Itaqui/Yolo/Aplicação/dataset/data.yaml",
    epochs=50,
    imgsz=640,
    device=device,  # Usa CUDA explicitamente se disponível
    project="treinamento",
    name="pombos_yolov8n",
    exist_ok=True
)
