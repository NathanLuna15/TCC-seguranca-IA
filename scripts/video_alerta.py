from ultralytics import YOLO
import cv2
import os
import json

# ==============================
# CONFIGURAÇÕES
# ==============================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Modelo treinado
modelo_path = os.path.join(
    BASE_DIR,
    "runs",
    "detect",
    "train-ppe-50",
    "weights",
    "best.pt"
)

# Vídeo
video_path = os.path.join(
    BASE_DIR,
    "video",
    "machoMen.mp4"
)

# Arquivo do alerta
alerta_path = os.path.join(
    BASE_DIR,
    "resultados",
    "alerta.json"
)

# ==============================
# CARREGA MODELO
# ==============================

model = YOLO(modelo_path)

# ==============================
# ABRE VÍDEO
# ==============================

video = cv2.VideoCapture(video_path)

if not video.isOpened():
    print("❌ Não foi possível abrir o vídeo.")
    exit()

print("🎥 Analisando vídeo...")

# Evita vários alertas
alerta_gerado = False

# ==============================
# PROCESSAMENTO
# ==============================

while True:

    sucesso, frame = video.read()

    if not sucesso:
        break


    # YOLO analisa o frame
    resultado = model(frame, verbose=False)[0]

    for i in range(len(resultado.boxes)):
        classe = int(resultado.boxes.cls[i])
        confidence = float(resultado.boxes.conf[i])

        print(
            f"Detectado: {model.names[classe]} "
            f"| confiança: {confidence:.2f}"
        )

    # Analisa todas as detecções
    for i in range(len(resultado.boxes)):

        classe = int(resultado.boxes.cls[i])
        confidence = float(resultado.boxes.conf[i])

        # Classe 7 = no_helmet
        if classe == 7:

            # Gera apenas um alerta
            if not alerta_gerado:

                box = resultado.boxes.xyxy[i].tolist()

                alerta = {
                    "alerta": True,
                    "tipo": "SEM_CAPACETE",
                    "confidence_percent": round(
                        confidence * 100,
                        1
                    ),
                    "box": {
                        "x1": box[0],
                        "y1": box[1],
                        "x2": box[2],
                        "y2": box[3]
                    }
                }

                # Salva JSON
                with open(
                    alerta_path,
                    "w",
                    encoding="utf-8"
                ) as arquivo:

                    json.dump(
                        alerta,
                        arquivo,
                        indent=4,
                        ensure_ascii=False
                    )

                print("🚨 ALERTA: Pessoa sem capacete!")
                print(
                    f"📊 Confiança: "
                    f"{alerta['confidence_percent']}%"
                )
                print(
                    f"📄 Arquivo: {alerta_path}"
                )

                alerta_gerado = True

    # Mostra vídeo
    frame_analisado = resultado.plot()

    cv2.imshow(
        "Sentinel Control - IA",
        frame_analisado
    )

    # Aperte Q para sair
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ==============================
# FINALIZA
# ==============================

video.release()
cv2.destroyAllWindows()

print("✅ Análise do vídeo finalizada.")