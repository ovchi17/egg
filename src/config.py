# config.py
# Здесь лежат константы (FS, диапазоны λ/α), цвета/шрифты интерфейса и функции, которые задают единый стиль графиков и сохраняют matplotlib-фигуры в PNG.

from __future__ import annotations
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

FS_HZ_DEFAULT = 250.0
LAMBDA_BAND_HZ = (4.0, 6.0)
ALPHA_BAND_HZ = (7.0, 13.0)

UI_TEXT = {
    "lambda_hint": "Рекомендуемый диапазон λ-ритма: 4–6 Гц (по методичке)",
    "alpha_hint": "Альфа-ритм: 8–13 Гц (справочно)",
    "export_progress": "Экспорт…",
}
UI = {
    "bg":     "#FFF7ED",
    "panel":  "#FFFFFF",
    "panel2": "#FFF1E6",
    "text":   "#1F2937",
    "muted":  "#6B7280",
    "accent": "#F97316",
    "accent2":"#FDBA74",
    "danger": "#EF4444",
    "good":   "#22C55E",
    "border": "#FED7AA",
    "hover":  "#FFEDD5",
}

FONT_MAIN  = ("SF Pro Text", 12)
FONT_TITLE = ("SF Pro Display", 18, "bold")
FONT_H2    = ("SF Pro Text", 13, "bold")
FONT_SMALL = ("SF Pro Text", 11)

def apply_mpl_style():
    plt.rcParams.update({
        "figure.facecolor": UI["panel"],
        "axes.facecolor": UI["panel2"],
        "axes.edgecolor": UI["border"],
        "axes.labelcolor": UI["muted"],
        "xtick.color": UI["muted"],
        "ytick.color": UI["muted"],
        "text.color": UI["text"],
        "grid.color": UI["border"],
        "grid.alpha": 0.7,
        "axes.grid": True,
        "legend.frameon": True,
        "legend.facecolor": UI["panel"],
        "legend.edgecolor": UI["border"],
        "font.size": 10,
        "savefig.facecolor": UI["panel"],
        "savefig.edgecolor": UI["panel"],
    })

def style_axes(ax):
    ax.set_facecolor(UI["panel2"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(UI["border"])
    ax.spines["bottom"].set_color(UI["border"])
    ax.grid(True, alpha=0.65, linewidth=0.8)
    ax.tick_params(colors=UI["muted"])
    ax.title.set_color(UI["text"])
    ax.xaxis.label.set_color(UI["muted"])
    ax.yaxis.label.set_color(UI["muted"])

def _hex_to_rgb(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def _blend(hex_a: str, hex_b: str, t: float) -> str:
    a = _hex_to_rgb(hex_a); b = _hex_to_rgb(hex_b)
    c = tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))
    return "#{:02X}{:02X}{:02X}".format(*c)

def save_figure_png_threadsafe(fig: Figure, path: str, dpi: int = 160):
    FigureCanvasAgg(fig)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
