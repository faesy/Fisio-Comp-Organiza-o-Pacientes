#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Colunas = 12 derivações clínicas
COLS = ["V1","V2","V3","V4","V5","V6"]

# Dados rRMSE(%) por canal para cada eletrodo substituído
DATA = {
    "V1" :  [0,0,0,0,0,0,34.79,0,0,0,0,0],
    "V2" :  [0,0,0,0,0,0,0,12.36,0,0,0,0],
    "V3" :  [0,0,0,0,0,0,0,0,17.94,0,0,0],
    "V4" :  [0,0,0,0,0,0,0,0,0,24.58,0,0],
    "V5" :  [0,0,0,0,0,0,0,0,0,0,33.70,0],
    "V6" :  [0,0,0,0,0,0,0,0,0,0,0,43.49],
}

# Ordem de linhas (0..10)
ROW_ORDER = ["V1","V2","V3","V4","V5","V6"]

# Monta matriz e arredonda p/ inteiro
M = np.array([DATA[row] for row in ROW_ORDER], dtype=float)
M_int = np.rint(M).astype(int)

# ------------------------------------------------------------
# Função p/ salvar tabela
# ------------------------------------------------------------
def save_table_png(matrix, row_labels, col_labels, out_path, title=None):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5.2), dpi=200)
    ax.axis("off")

    table = ax.table(
        cellText=matrix,
        rowLabels=row_labels,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.1, 1.4)

    for (r, c), cell in table.get_celld().items():
        if r == 0 or c == -1:
            cell.set_text_props(fontweight='bold')

    if title:
        ax.set_title(title, fontsize=12, pad=12)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

# ------------------------------------------------------------
# Salvar resultado
# ------------------------------------------------------------
saida = Path("tabela_rRMSE_ordem_V1aRA.png")
save_table_png(
    matrix=M_int,
    row_labels=ROW_ORDER,
    col_labels=COLS,
    out_path=saida,
)

print(f"✅ Tabela salva em: {saida.resolve()}")
