from ultralytics import YOLO

#carregar o modelo trinado 
model = YOLO("runs\detect\train-ppe-50\weights")

#analiza a imagem
results = model("operador.jpg")

#pega o resultado
resultado = results[0]

#converte as detecção para JSON
resultadoJSON = resultado.to_json()

#mostra o JSON no terminal 
print(resultadoJSON)