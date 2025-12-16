from __future__ import annotations
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from config import style_axes, LAMBDA_BAND_HZ, ALPHA_BAND_HZ
from signal_analysis import compute_psd, extract_lambda_signal, sliding_window_power

def make_raw_figure(t: np.ndarray, x: np.ndarray, fs_hz: float, title: str, for_pdf: bool = False) -> Figure:
    fig = Figure(figsize=(10, 4), dpi=110 if not for_pdf else 120)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    style_axes(ax1); style_axes(ax2)

    max_over = min(len(x), int(10 * fs_hz))
    max_zoom = min(len(x), int(4 * fs_hz))

    ax1.plot(t[:max_over], x[:max_over], linewidth=1.5)
    ax1.axhline(0, linestyle="--", linewidth=0.9, alpha=0.7)
    ax1.set_title(f"{title}\nПервые {max_over / fs_hz:.1f} с", fontsize=10)
    ax1.set_xlabel("Время, с")
    ax1.set_ylabel("Амплитуда")

    ax2.plot(t[:max_zoom], x[:max_zoom], linewidth=1.6)
    ax2.axhline(0, linestyle="--", linewidth=0.9, alpha=0.7)
    ax2.set_title(f"{title}\nЗум 4.0 с", fontsize=10)
    ax2.set_xlabel("Время, с")
    ax2.set_ylabel("Амплитуда")

    fig.tight_layout()
    return fig

def make_psd_figure(x: np.ndarray, fs_hz: float, title: str, for_pdf: bool = False) -> Figure:
    fig = Figure(figsize=(10, 3.5), dpi=110 if not for_pdf else 120)
    ax = fig.add_subplot(111)
    style_axes(ax)

    freqs, psd = compute_psd(x, fs_hz=fs_hz, nperseg=1024)
    ax.semilogy(freqs, psd, linewidth=1.4, label="PSD")
    ax.axvspan(LAMBDA_BAND_HZ[0], LAMBDA_BAND_HZ[1], alpha=0.18, label="λ (4–6 Гц)")
    ax.axvspan(ALPHA_BAND_HZ[0], ALPHA_BAND_HZ[1], alpha=0.18, label="α (7–13 Гц)")
    ax.set_xlim(0, 30)
    ax.set_xlabel("Частота, Гц")
    ax.set_ylabel("PSD, у.е.")
    ax.set_title(f"{title} — спектр мощности", fontsize=11)
    ax.legend(fontsize=9)

    fig.tight_layout()
    return fig

def make_lambda_figure(t: np.ndarray, x: np.ndarray, fs_hz: float, title: str, for_pdf: bool = False) -> Figure:
    fig = Figure(figsize=(10, 4), dpi=110 if not for_pdf else 120)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    style_axes(ax1); style_axes(ax2)

    lam = extract_lambda_signal(x, fs_hz=fs_hz)
    max_samp = min(len(x), int(8 * fs_hz))

    ax1.plot(t[:max_samp], x[:max_samp], linewidth=1.2, label="ЭЭГ")
    ax1.plot(t[:max_samp], lam[:max_samp], linewidth=1.7, alpha=0.95, label="λ (4–6 Гц)")
    ax1.axhline(0, linestyle="--", linewidth=0.9, alpha=0.7)
    ax1.set_title(f"{title}\nЭЭГ и λ-ритм", fontsize=10)
    ax1.set_xlabel("Время, с")
    ax1.set_ylabel("Амплитуда")
    ax1.legend(fontsize=9)

    t_win, p_win = sliding_window_power(lam, fs_hz=fs_hz, window_sec=2.0, overlap=0.5)
    ax2.plot(t_win, p_win, linewidth=1.6)
    ax2.set_title(f"{title}\nМощность λ(t)", fontsize=10)
    ax2.set_xlabel("Время, с")
    ax2.set_ylabel("mean(x²)")

    fig.tight_layout()
    return fig

def make_bars_figure(summary_df: pd.DataFrame, metric: str = "Средняя мощность λ(t)") -> Figure:
    df = summary_df.copy().sort_values(metric, ascending=True)
    labels = df["Файл"].astype(str).values
    vals = df[metric].values

    h = max(4.8, 0.55 * len(labels) + 2.0)
    fig = Figure(figsize=(10, h), dpi=120)
    ax = fig.add_subplot(111)
    style_axes(ax)

    y = np.arange(len(labels))
    ax.barh(y, vals)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Мощность, у.е.")
    ax.set_title(metric)

    ax.grid(True, axis="x", alpha=0.55)
    ax.grid(False, axis="y")

    fig.tight_layout()
    return fig
