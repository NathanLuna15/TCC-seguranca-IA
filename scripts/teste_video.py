from ultralytics import YOLO

# Carrega o modelo treinado
model = YOLO("runs/detect/train-ppe-50/weights/best.pt")

# Analisa o vídeo
model.predict(
    source="machoMen.mp4",
    show=True,
    save=True
)