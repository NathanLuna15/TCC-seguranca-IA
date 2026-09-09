from ultralytics import YOLO
import json
import sys

# Verifica se foi informada uma imagem
if len(sys.argv) < 2:
    print("Uso: python teste_json.py nome_da_imagem.jpg")
    exit()

# Pega o nome da imagem
imagem = sys.argv[1]

#carregar o modelo trinado 
model = YOLO("runs/detect/train-ppe-50/weights/best.pt")

#analiza a imagem
results = model("barbudoSemCapacete.png")

#pega o resultado
resultado = results[0]

#converte as detecção para JSON
json_resultado = resultado.to_json()

#mostra o JSON no terminal 
print(json_resultado)      

# Converte o JSON para objeto Python
dados = json.loads(json_resultado)

# Salva em um arquivo JSON
with open("resultado.json", "w", encoding="utf-8") as arquivo:
    json.dump(dados, arquivo, indent=4, ensure_ascii=False)

print(f"\nImagem analisada: {imagem}")
print("JSON salvo em: resultado.json")