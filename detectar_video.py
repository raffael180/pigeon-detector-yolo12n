import cv2
from ultralytics import YOLO

# Carrega o modelo treinado
model = YOLO("D:/Porto Itaqui/Yolo/Aplicação/treinamento/pombos_yolov12n/weights/best.pt")

# Caminho do vídeo
video_path = "videos/exemplo.mp4"
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Roda o modelo na imagem
    results = model(frame)

    # Desenha as caixas
    annotated_frame = results[0].plot()

    cv2.imshow("Pombos Detectados", annotated_frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC para sair
        break

cap.release()
cv2.destroyAllWindows()
