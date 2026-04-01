import numpy as np
import re

# =========================
# CONFIGURAÇÃO DE DESLOCAMENTO
# =========================
offset_x = 60590.0
offset_y = 102008.0
offset_z = 113013.0

# =========================
# LER ARQUIVO DE COORDENADAS
# =========================
input_file = "Paciente7.txt"
output_file = "output_electrodes/P7_lead_Manual_formatado.txt"

# Regex para extrair os dados
pattern = r"Eletrodo #[0-9]+ \((.*?)\): X=([-0-9.]+), Y=([-0-9.]+), Z=([-0-9.]+)"

# Lista para armazenar os resultados
coords = []
labels = []

with open(input_file, "r") as f:
    for line in f:
        match = re.search(pattern, line)
        if match:
            label = match.group(1)
            x = float(match.group(2)) * 1000 + offset_x
            y = float(match.group(3)) * 1000 + offset_y
            z = float(match.group(4)) * 1000 + offset_z
            coords.append([x, y, z])
            labels.append(label)

# =========================
# IMPRIMIR EM FORMATO DE CÓDIGO
# =========================
print("LEADS_REAL = np.array(")
print("    [")
for coord, label in zip(coords, labels):
    print(f"        [{coord[0]:.1f}, {coord[1]:.1f}, {coord[2]:.1f}],  # {label}")
print("    ],")
print("    dtype=np.float64,")
print(")")

