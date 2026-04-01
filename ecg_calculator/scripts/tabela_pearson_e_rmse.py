#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =========================================================
# Saída
# =========================================================
PNG_SAIDA = Path(r"S:\Projeto\ecg_multicanal\tabelas_metricas_pacientes.png")

# =========================================================
# Canais (ordem fixa)
# =========================================================
CANAIS = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']

# =========================================================
# ENTRADA: métricas já calculadas (como você enviou)
#   - Cada paciente tem:
#       'titulo'          -> texto do cabeçalho (ex.: "Comparando 'real' x 'automatico'")
#       'pearson'         -> lista com 12 valores na ordem de CANAIS
#       'pearson_global'  -> float
#       'rrmse'           -> lista com 12 valores na ordem de CANAIS (em %)
#       'rrmse_global'    -> float (em %)
# =========================================================
PACIENTES = [
    {   # Paciente 1  (par=1)
        "paciente": "Paciente 1",
        "titulo":   "Comparando 'real' x 'automatico'",
        "pearson":  [0.5541, 0.9994, -0.3888, 0.9810, -0.1012, -0.5151, 0.9377, 0.9802, 0.9835, 0.9805, 0.9975, 0.9932],
        "pearson_global": 0.8828,
        "rrmse":    [112.87, 90.46, 134.40, 33.59, 144.90, 117.46, 40.39, 19.92, 19.08, 20.06, 35.18, 49.83],
        "rrmse_global": 47.20,
    },
    {   # Paciente 2  (par=2)
        "paciente": "Paciente 2",
        "titulo":   "Comparando 'real' x 'automatico'",
        "pearson":  [0.9996, 0.9991, 0.9962, 0.9999, 0.9989, -0.6579, -0.7131, 0.9818, 0.9539, 0.8992, 0.9989, 0.9983],
        "pearson_global": 0.2593,
        "rrmse":    [46.86, 92.05, 9.41, 62.39, 30.99, 621.30, 100.45, 95.11, 93.63, 92.08, 51.34, 35.77],
        "rrmse_global": 97.99,
    },
    {   # Paciente 3  (par=3)  -> nota: aqui o título veio como "AUTO x REAL"
        "paciente": "Paciente 3",
        "titulo":   "Comparando 'real' x 'automatico'",
        "pearson":  [0.9636, 0.9949, 0.1647, 0.9868, 0.7917, -0.5762, 0.9983, 0.9906, 0.8476, 0.8482, 0.9436, 0.9981],
        "pearson_global": 0.8193,
        "rrmse":    [27.43, 869.08, 107.67, 60.40, 63.25, 212.34, 56.03, 53.11, 64.35, 69.26, 34.89, 35.06],
        "rrmse_global": 60.87,
    },
    {   # Paciente 4  (par=4)
        "paciente": "Paciente 4",
        "titulo":   "Comparando 'real' x 'automatico'",
        "pearson":  [0.9997, 0.9996, 0.9899, 0.9999, 0.9990, -0.9876, -0.6710, 0.9995, 0.9994, 0.9981, 0.9999, 0.9998],
        "pearson_global": 0.8584,
        "rrmse":    [88.98, 1149.04, 42.26, 193.38, 28.37, 208.17, 157.72, 27.68, 56.07, 48.95, 9.97, 19.64],
        "rrmse_global": 80.55,
    },
]

# =========================================================
# Função p/ desenhar uma tabela em um eixo
# =========================================================
def desenhar_tabela(ax, cabecalho: str, paciente: str,
                    pears: list, pears_global: float,
                    rmses: list, rmses_global: float):
    ax.axis('off')

    # Título (cabecalho + paciente)
    ax.set_title(f"{cabecalho}\n{paciente}", fontsize=12, pad=10)

    # Linhas: 12 canais + Global
    linhas = []
    for i, canal in enumerate(CANAIS):
        linhas.append([canal, f"{pears[i]:.4f}", f"{rmses[i]:.2f} %"])
    linhas.append(["Global", f"{pears_global:.4f}", f"{rmses_global:.2f} %"])

    # Monta tabela
    tabela = ax.table(
        cellText=linhas,
        colLabels=["Canal", "Pearson", "rRMSE (%)"],
        cellLoc='center',
        loc='center'
    )
    tabela.auto_set_font_size(False)
    tabela.set_fontsize(11)
    tabela.scale(1.2, 1.25)

# =========================================================
# Principal: monta um grid 2x2 e salva PNG
# =========================================================
def main():
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=150)
    axes = axes.ravel()

    for i, met in enumerate(PACIENTES):
        desenhar_tabela(
            axes[i],
            cabecalho=met["titulo"],
            paciente=met["paciente"],
            pears=met["pearson"],
            pears_global=met["pearson_global"],
            rmses=met["rrmse"],
            rmses_global=met["rrmse_global"],
        )

    plt.tight_layout()
    PNG_SAIDA.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(PNG_SAIDA, bbox_inches="tight")
    print(f"✅ PNG gerado em: {PNG_SAIDA}")

if __name__ == "__main__":
    main()
