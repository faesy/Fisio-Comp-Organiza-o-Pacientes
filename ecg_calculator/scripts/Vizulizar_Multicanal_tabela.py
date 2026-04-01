import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
pio.renderers.default = "browser"
from tkinter import Tk, filedialog, simpledialog

nomes_canais = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6']

def carregar_ecg(filepath):
    data = np.loadtxt(filepath, skiprows=1)
    tempo = data[:, 0]
    sinais = data[:, 1:]
    return tempo, sinais

root = Tk()
root.withdraw()

file1 = filedialog.askopenfilename(
    title="Selecione o primeiro arquivo ECG (.txt)",
    filetypes=[("Text Files", "*.txt")]
)
file2 = filedialog.askopenfilename(
    title="Selecione o segundo arquivo ECG (.txt)",
    filetypes=[("Text Files", "*.txt")]
)

nome1 = simpledialog.askstring("Nome do sinal", "Digite o nome do primeiro sinal (ex: Automático):")
nome2 = simpledialog.askstring("Nome do sinal", "Digite o nome do segundo sinal (ex: Manual):")

tempo1, sinais1 = carregar_ecg(file1)
tempo2, sinais2 = carregar_ecg(file2)

fig = make_subplots(
    rows=2, cols=3,
    horizontal_spacing=0.08,
    vertical_spacing=0.20,   # espaço entre 1ª e 2ª linha
)

for i, canal in enumerate(nomes_canais):
    row = i // 3 + 1
    col = i % 3 + 1

    # Curva 1
    fig.add_trace(
        go.Scatter(
            x=tempo1,
            y=sinais1[:, i],
            mode='lines',
            line=dict(color='blue'),
            name=nome1,
            showlegend=(i == 0),  # legenda só no primeiro
        ),
        row=row,
        col=col
    )

    # Curva 2
    fig.add_trace(
        go.Scatter(
            x=tempo2,
            y=sinais2[:, i],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name=nome2,
            showlegend=(i == 0),
        ),
        row=row,
        col=col
    )

    # Índice do eixo desse subplot
    axis_index = col + (row - 1) * 3

    # Mapeia para nomes de eixo válidos na tua versão:
    # eixo 1 -> "x domain" / "y domain"
    # eixo >=2 -> "x2 domain", "x3 domain", ...
    if axis_index == 1:
        xref = "x domain"
        yref = "y domain"
    else:
        xref = f"x{axis_index} domain"
        yref = f"y{axis_index} domain"

    # Título do canal acima do subplot
    fig.add_annotation(
        text=canal,
        xref=xref,
        yref=yref,
        x=0.5,
        y=1.2,           # um pouco acima do gráfico
        showarrow=False,
        font=dict(size=26),
        xanchor='center'
    )

    # Remove ticks
    fig.update_xaxes(showticklabels=False, title_text="", row=row, col=col)
    fig.update_yaxes(showticklabels=False, title_text="", row=row, col=col)

fig.update_layout(
    height=900,
    width=1200,
    margin=dict(t=120, b=80, l=20, r=20),
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.12,
        xanchor="center",
        x=0.5,
        font=dict(size=28)
    ),
    plot_bgcolor="white"
)

fig.show()
