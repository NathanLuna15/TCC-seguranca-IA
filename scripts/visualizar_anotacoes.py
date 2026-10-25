from pathlib import Path
import cv2

imagem = Path("datasets/construction-ppe/images/val/image1010.jpg")
rotulos = Path("datasets/construction-ppe/labels/val/image1010.txt")
saida = Path("resultados/image1010_anotacoes.jpg")

classes = [
    "helmet", "gloves", "vest", "boots", "goggles",
    "none", "Person", "no_helmet", "no_goggle",
    "no_gloves", "no_boots"
]

img = cv2.imread(str(imagem))

if img is None:
    raise FileNotFoundError(f"Imagem nao encontrada: {imagem}")
if not rotulos.exists():
    raise FileNotFoundError(f"Rotulos nao encontrados: {rotulos}")

h, w = img.shape[:2]

for linha in rotulos.read_text().splitlines():
    dados = linha.split()
    classe = int(dados[0])
    xc, yc, bw, bh = map(float, dados[1:5])

    x1 = int((xc - bw / 2) * w)
    y1 = int((yc - bh / 2) * h)
    x2 = int((xc + bw / 2) * w)
    y2 = int((yc + bh / 2) * h)

    nome = classes[classe]
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        img, nome, (x1, max(y1 - 5, 20)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2
    )

saida.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(saida), img)
print(f"Imagem salva em: {saida.resolve()}")
