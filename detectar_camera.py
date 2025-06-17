import cv2
from ultralytics import YOLO

model = YOLO("D:/Porto Itaqui/Yolo/Aplicação/treinamento/pombos_yolov12n/weights/best.pt")

cap = cv2.VideoCapture(0)  # mude para 1, 2 se tiver várias webcams

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    annotated_frame = results[0].plot()

    cv2.imshow("Pombos ao Vivo", annotated_frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
