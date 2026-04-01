#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Dados dos pacientes
# ------------------------------------------------------------
pacientes = [
    ["P1", "Homem", 71, "76.4 kg"],
    ["P2", "Mulher", 79, "54.0 kg"],
    ["P3", "Homem", 71, "Desconhecido"],
    ["P4", "Homem", 72, "119 kg"],
]

colunas = ["Paciente", "Sexo", "Idade (anos)", "Peso"]

# ------------------------------------------------------------
# Função para salvar tabela como PNG
# ------------------------------------------------------------
def save_table(data, columns, out_path, title=None):
    fig, ax = plt.subplots(figsize=(6, 2.5), dpi=200)
    ax.axis("off")

    tabela = ax.table(
        cellText=data,
        colLabels=columns,
        cellLoc="center",
        loc="center"
    )

    tabela.auto_set_font_size(False)
    tabela.set_fontsize(10)
    tabela.scale(1.2, 1.4)

    for (r, c), cell in tabela.get_celld().items():
        if r == 0:  # cabeçalho
            cell.set_text_props(fontweight="bold")

    if title:
        ax.set_title(title, fontsize=12, pad=10)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------
# Salvar tabela
# ------------------------------------------------------------
saida = Path("tabela_pacientes.png")
save_table(
    pacientes,
    colunas,
    saida,
    title="Dados dos Pacientes"
)

print(f"✅ Tabela salva em: {saida.resolve()}")
