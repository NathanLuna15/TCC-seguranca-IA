from ultralytics import YOLO
import cv2
import os
import json
import time
import requests

# ==============================
# CONFIGURAÇÕES (mexa só aqui)
# ==============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELO = os.path.join(BASE_DIR, "runs", "detect", "train", "weights", "best.pt")

# Vídeo: coloque o caminho. Câmera ao vivo: coloque 0
FONTE = 0

ALERTA_PATH = os.path.join(BASE_DIR, "resultados", "alerta.json")

API_URL = "http://localhost:8080/v1/oroscorp/ia/alertas"   # troque pela rota do seu back-end

CONF_MIN = 0.25            # confiança mínima das detecções
CONF_PESSOA = 0.50         # só considera pessoa se tiver certeza
FRAMES_PARA_ALERTA = 10    # quantos frames seguidos sem capacete para alertar
SEGUNDOS_ENTRE_ALERTAS = 30  # evita mandar alerta repetido

# ==============================
# FUNÇÕES
# ==============================
def tem_capacete(pessoa, capacetes):
    """Capacete conta se o centro dele está na parte de cima (cabeça) da pessoa."""
    x1, y1, x2, y2 = pessoa
    limite_cabeca = y1 + (y2 - y1) * 0.40

    for c in capacetes:
        cx = (c[0] + c[2]) / 2
        cy = (c[1] + c[3]) / 2
        if x1 <= cx <= x2 and y1 <= cy <= limite_cabeca:
            return True
    return False


def enviar_alerta(alerta):
    # Salva o JSON no arquivo
    with open(ALERTA_PATH, "w", encoding="utf-8") as f:
        json.dump(alerta, f, indent=4, ensure_ascii=False)
    print("ALERTA salvo em", ALERTA_PATH)

    # Envia para o back-end
    try:
        resposta = requests.post(API_URL, json=alerta, timeout=5)
        print("Back-end respondeu:", resposta.status_code)
    except requests.exceptions.RequestException as erro:
        print("Não consegui enviar o alerta:", erro)


# ==============================
# EXECUÇÃO
# ==============================
model = YOLO(MODELO)
video = cv2.VideoCapture(FONTE)

if not video.isOpened():
    print(" Não foi possível abrir a fonte de vídeo.")
    exit()

frames_seguidos = 0
ultimo_alerta = 0

print("Analisando o video...")

while True:
    ok, frame = video.read()
    if not ok:
        break

    resultado = model(frame, conf=CONF_MIN, verbose=False)[0]

    pessoas, capacetes, sem_capacete_modelo = [], [], []

    for i in range(len(resultado.boxes)):
        nome = model.names[int(resultado.boxes.cls[i])]
        conf = float(resultado.boxes.conf[i])
        box = resultado.boxes.xyxy[i].tolist()

        if nome == "Person" and conf >= CONF_PESSOA:
            pessoas.append((box, conf))
        elif nome == "helmet":
            capacetes.append(box)
        elif nome == "no_helmet":
            sem_capacete_modelo.append((box, conf))

    # Pessoas sem capacete (regra) + no_helmet (se o modelo detectar)
    infracoes = []
    for box, conf in pessoas:
        if not tem_capacete(box, capacetes):
            infracoes.append({"origem": "regra_pessoa_sem_capacete",
                              "confidence_percent": round(conf * 100, 1),
                              "box": box})
    for box, conf in sem_capacete_modelo:
        infracoes.append({"origem": "classe_no_helmet",
                          "confidence_percent": round(conf * 100, 1),
                          "box": box})

    # Conta frames seguidos com infração
    if infracoes:
        frames_seguidos += 1
    else:
        frames_seguidos = 0

    # Dispara o alerta
    agora = time.time()
    if (frames_seguidos >= FRAMES_PARA_ALERTA
            and agora - ultimo_alerta > SEGUNDOS_ENTRE_ALERTAS):
        alerta = {
            "alerta": True,
            "tipo": "SEM_CAPACETE",
            "quantidade": len(infracoes),
            "horario": time.strftime("%Y-%m-%d %H:%M:%S"),
            "ocorrencias": [
                {"origem": i["origem"],
                 "confidence_percent": i["confidence_percent"],
                 "box": {"x1": i["box"][0], "y1": i["box"][1],
                         "x2": i["box"][2], "y2": i["box"][3]}}
                for i in infracoes
            ],
        }
        enviar_alerta(alerta)
        ultimo_alerta = agora
        frames_seguidos = 0

    # Mostra o vídeo
    cv2.imshow("Sentinel Control - IA", resultado.plot())
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
print(" Análise finalizada.")