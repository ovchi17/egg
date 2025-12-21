# plotting.py
# Построение графиков для приложения и PDF: сигнал во времени, PSD (Welch), выделение λ-ритма, сравнение столбиками.

from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from config import LAMBDA_BAND_HZ, ALPHA_BAND_HZ, style_axes
from signal_analysis import compute_psd, extract_lambda_signal, sliding_window_power


def _nice_title(file_name: str, channel_label: str, kind: str) -> str:
    base = file_name
    if channel_label:
        return f"{kind} — {base} | канал/электрод: {channel_label}"
    return f"{kind} — {base}"


def make_raw_figure(t, x, fs_hz, name, channel_label: str = "", for_pdf: bool = False):
    fig = Figure(figsize=(9.2, 3.6) if for_pdf else (9.0, 3.4), dpi=120)
    ax = fig.add_subplot(111)
    style_axes(ax)

    ax.plot(t, x, linewidth=1.6)
    ax.set_title(_nice_title(name, channel_label, "Сигнал во времени"), fontsize=12)
    ax.set_xlabel("Время, с")
    ax.set_ylabel("Амплитуда, усл. ед.")
    return fig


def make_psd_figure(
        x: np.ndarray,
        fs_hz: float,
        file_label: str,
        channel_label: Optional[str] = None,
        for_pdf: bool = False,
) -> Figure:
    fig = Figure(figsize=(10, 3.3) if for_pdf else (10, 3.6), dpi=110)
    ax = fig.add_subplot(111)
    style_axes(ax)

    freqs, psd = compute_psd(np.asarray(x, dtype=float), fs_hz=float(fs_hz), nperseg=1024)
    if freqs.size and psd.size:
        ax.plot(freqs, psd, linewidth=1.3)
        ax.axvspan(LAMBDA_BAND_HZ[0], LAMBDA_BAND_HZ[1], alpha=0.18,
                   label=f"λ: {LAMBDA_BAND_HZ[0]}–{LAMBDA_BAND_HZ[1]} Гц")
        ax.axvspan(ALPHA_BAND_HZ[0], ALPHA_BAND_HZ[1], alpha=0.10,
                   label=f"α: {ALPHA_BAND_HZ[0]}–{ALPHA_BAND_HZ[1]} Гц")
        ax.set_xlim(0, 40)
        ax.legend(fontsize=9, loc="upper right")

    ax.set_title(_nice_title("Спектр мощности (PSD, Welch)", file_label, channel_label), fontsize=12)
    ax.set_xlabel("Частота, Гц")
    ax.set_ylabel("PSD, усл. ед.")
    ax.grid(True, alpha=0.25)
    return fig


def make_lambda_figure(
        t: np.ndarray,
        x: np.ndarray,
        fs_hz: float,
        file_label: str,
        channel_label: Optional[str] = None,
        for_pdf: bool = False,
) -> Figure:
    fig = Figure(figsize=(10, 4.6) if for_pdf else (10, 5.0), dpi=110)
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    style_axes(ax1)
    style_axes(ax2)

    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)

    lam = extract_lambda_signal(x, fs_hz=float(fs_hz))
    ax1.plot(t, lam, linewidth=1.1)
    ax1.set_title(_nice_title(f"Полосовой фильтр λ ({LAMBDA_BAND_HZ[0]}–{LAMBDA_BAND_HZ[1]} Гц)",
                         file_label, channel_label), fontsize=12)
    ax1.set_xlabel("Время, с")
    ax1.set_ylabel("Амплитуда, усл. ед.")
    ax1.grid(True, alpha=0.25)

    tw, pw = sliding_window_power(lam, fs_hz=float(fs_hz), window_sec=2.0, overlap=0.5)
    if tw.size and pw.size:
        ax2.plot(tw, pw, linewidth=1.2)
    ax2.set_title("Динамика мощности λ(t) (скользящее окно)", fontsize=11)
    ax2.set_xlabel("Время, с")
    ax2.set_ylabel("Мощность, усл. ед.")
    ax2.grid(True, alpha=0.25)

    fig.tight_layout(pad=1.0)
    return fig


def make_bars_figure(summary_df: pd.DataFrame, metric: str = "Средняя мощность λ(t)") -> Figure:
    fig = Figure(figsize=(10, 3.8), dpi=110)
    ax = fig.add_subplot(111)
    style_axes(ax)

    if summary_df is None or summary_df.empty or metric not in summary_df.columns:
        ax.set_title("Нет данных для сравнения")
        return fig

    df = summary_df.copy()
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df = df.dropna(subset=[metric])

    labels = []
    values = []
    for _, r in df.iterrows():
        lab = f"{r.get('Файл', '')}\n{r.get('Электрод/канал', '')}"
        labels.append(lab)
        values.append(float(r[metric]))

    ax.bar(range(len(values)), values)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title(f"Сравнение: {metric}", fontsize=12)
    ax.set_ylabel(metric)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout(pad=1.0)
    return fig
