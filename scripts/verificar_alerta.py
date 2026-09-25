import json
import os

# Localiza a pasta principal do projeto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Caminho do resultado da IA
resultado_path = os.path.join(BASE_DIR, "resultados", "resultado.json")

# Caminho onde o alerta será salvo
alerta_path = os.path.join(BASE_DIR, "resultados", "alerta.json")

# Abre o JSON da IA
with open(resultado_path, "r", encoding="utf-8") as arquivo:
    deteccoes = json.load(arquivo)

# Separa pessoas e capacetes
pessoas = [
    deteccao for deteccao in deteccoes
    if deteccao["name"] == "Person"
]

capacetes = [
    deteccao for deteccao in deteccoes
    if deteccao["name"] == "helmet"
]

# Lista das pessoas sem capacete
pessoas_sem_capacete = []

# Analisa cada pessoa
for pessoa in pessoas:

    pessoa_x1 = pessoa["box"]["x1"]
    pessoa_y1 = pessoa["box"]["y1"]
    pessoa_x2 = pessoa["box"]["x2"]
    pessoa_y2 = pessoa["box"]["y2"]

    tem_capacete = False

    # Procura um capacete dentro da pessoa
    for capacete in capacetes:

        capacete_x1 = capacete["box"]["x1"]
        capacete_y1 = capacete["box"]["y1"]
        capacete_x2 = capacete["box"]["x2"]
        capacete_y2 = capacete["box"]["y2"]

        # Centro do capacete
        centro_x = (capacete_x1 + capacete_x2) / 2
        centro_y = (capacete_y1 + capacete_y2) / 2

        # Verifica se o capacete está dentro da pessoa
        if (
            pessoa_x1 <= centro_x <= pessoa_x2
            and pessoa_y1 <= centro_y <= pessoa_y2
        ):
            tem_capacete = True
            break

    # Se não encontrou capacete
    if not tem_capacete:
        pessoas_sem_capacete.append({
            "confidence_percent": round(pessoa["confidence"] * 100, 1),            
            "box": pessoa["box"]
        })


# Monta o JSON do alerta
alerta = {
    "alerta": len(pessoas_sem_capacete) > 0,
    "tipo": "SEM_CAPACETE" if pessoas_sem_capacete else None,
    "quantidade": len(pessoas_sem_capacete),
    "pessoas": pessoas_sem_capacete
}

# Salva o alerta.json
with open(alerta_path, "w", encoding="utf-8") as arquivo:
    json.dump(alerta, arquivo, indent=4, ensure_ascii=False)

# Mostra no terminal
if pessoas_sem_capacete:
    print("🚨 Alerta gerado!")
    print(f"👤 Pessoas sem capacete: {len(pessoas_sem_capacete)}")
    print(f"📄 Arquivo: {alerta_path}")
else:
    print("✅ Nenhuma pessoa sem capacete.")