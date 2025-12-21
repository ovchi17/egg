# signal_analysis.py
# Функции анализа ЭЭГ: PSD (Welch), мощность в диапазонах, полосовой фильтр λ, мощность λ(t) во времени и текст выводов.

from __future__ import annotations

from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import pandas as pd
from scipy.signal import welch, butter, filtfilt

from config import LAMBDA_BAND_HZ, ALPHA_BAND_HZ


def compute_psd(x: np.ndarray, fs_hz: float, nperseg: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
    """
    Оценка спектральной плотности мощности (PSD) методом Уэлча.

    Параметры:
        x      — сигнал (1D)
        fs_hz  — частота дискретизации (Гц)
        nperseg — длина сегмента Уэлча

    Возвращает:
        freqs_hz, psd
    """
    x = np.asarray(x, dtype=float)
    if x.size < 8 or not np.isfinite(fs_hz) or fs_hz <= 0:
        return np.asarray([]), np.asarray([])

    nper = int(min(nperseg, max(64, x.size)))
    freqs_hz, psd = welch(x, fs=fs_hz, nperseg=nper)
    return freqs_hz, psd


def integrate_band_power(freqs_hz: np.ndarray, psd: np.ndarray, band_hz: Tuple[float, float]) -> float:
    """
    Интегральная мощность в частотном диапазоне band_hz (low..high).

    ВАЖНО:
        Используем интеграл (trapz), а не mean(psd), чтобы "мощность диапазона"
        была физически более корректной оценкой.

    Возвращает:
        float (NaN если данных нет)
    """
    if freqs_hz.size == 0 or psd.size == 0:
        return float("nan")

    low, high = float(band_hz[0]), float(band_hz[1])
    if high <= low:
        return float("nan")

    m = (freqs_hz >= low) & (freqs_hz <= high)
    if not np.any(m):
        return float("nan")

    return float(np.trapz(psd[m], freqs_hz[m]))


def butter_bandpass(x: np.ndarray, low_hz: float, high_hz: float, fs_hz: float, order: int = 4) -> np.ndarray:
    """
    Полосовой фильтр Баттерворта + filtfilt (нулевая фазовая задержка).

    Если параметры некорректны — возвращает исходный сигнал.
    """
    x = np.asarray(x, dtype=float)
    if x.size < 8 or not np.isfinite(fs_hz) or fs_hz <= 0:
        return x.copy()

    nyq = fs_hz / 2.0
    low = max(0.001, low_hz / nyq)
    high = min(0.999, high_hz / nyq)

    if not (0.0 < low < high < 1.0):
        return x.copy()

    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, x)


def extract_lambda_signal(x: np.ndarray, fs_hz: float) -> np.ndarray:
    """
    Выделение λ-составляющей (по методическим указаниям): 4–6 Гц.
    """
    return butter_bandpass(
        np.asarray(x, dtype=float),
        low_hz=float(LAMBDA_BAND_HZ[0]),
        high_hz=float(LAMBDA_BAND_HZ[1]),
        fs_hz=fs_hz,
        order=4,
    )


def sliding_window_power(
    x: np.ndarray,
    fs_hz: float,
    window_sec: float = 2.0,
    overlap: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Оценка мощности во времени: среднее квадрата сигнала в скользящем окне.

    Возвращает:
        t_vals (сек), p_vals

    Примечание:
        Это простая (и устойчивая) оценка "энергии" сигнала во времени.
    """
    x = np.asarray(x, dtype=float)
    if x.size < 8 or not np.isfinite(fs_hz) or fs_hz <= 0:
        return np.asarray([]), np.asarray([])

    win = int(max(2, round(window_sec * fs_hz)))
    step = int(max(1, round(win * (1.0 - overlap))))

    if win >= x.size:
        return np.asarray([0.0]), np.asarray([float(np.mean(x ** 2))])

    t_vals: List[float] = []
    p_vals: List[float] = []

    for start in range(0, x.size - win + 1, step):
        seg = x[start : start + win]
        p_vals.append(float(np.mean(seg ** 2)))
        t_vals.append(float(start / fs_hz))

    return np.asarray(t_vals), np.asarray(p_vals)


def _safe_num(v: Any) -> Optional[float]:
    try:
        x = float(v)
        if np.isfinite(x):
            return x
        return None
    except Exception:
        return None


def build_conclusions(
    summary_df: pd.DataFrame,
    records: List[Dict[str, Any]],
    fs_user: float,
    montage: str = "",
    channel_hint: str = "",
) -> str:
    """
    Формирует текст "Анализ и выводы" в более официальном стиле для курсовой.

    summary_df ожидается в формате твоего app.py (после merge):
        - "Файл"
        - "Электрод/канал"
        - "P_total", "P_λ", "P_α"
        - "P_λ / P_total", "P_α / P_total"
        - "Средняя мощность λ(t)", "Максимум λ(t)", "Минимум λ(t)"
    """
    if summary_df is None or summary_df.empty:
        return "Анализ не выполнен или отсутствуют данные для формирования выводов."

    df = summary_df.copy()

    file_col = "Файл" if "Файл" in df.columns else None
    ch_col = "Электрод/канал" if "Электрод/канал" in df.columns else None

    lam_t_mean_col = "Средняя мощность λ(t)" if "Средняя мощность λ(t)" in df.columns else None
    lam_t_max_col = "Максимум λ(t)" if "Максимум λ(t)" in df.columns else None
    lam_t_min_col = "Минимум λ(t)" if "Минимум λ(t)" in df.columns else None

    ratio_lam_col = "P_λ / P_total" if "P_λ / P_total" in df.columns else None
    ratio_alp_col = "P_α / P_total" if "P_α / P_total" in df.columns else None

    for c in ["P_total", "P_λ", "P_α", ratio_lam_col, ratio_alp_col, lam_t_mean_col, lam_t_max_col, lam_t_min_col]:
        if c and c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    n_files = df[file_col].nunique() if file_col else len(df)
    used_channels: List[str] = []
    if ch_col:
        used_channels = [str(x) for x in df[ch_col].dropna().unique().tolist()]

    lines: List[str] = []

    lines.append("АНАЛИЗ И ВЫВОДЫ ПО РЕЗУЛЬТАТАМ ИССЛЕДОВАНИЯ λ-РИТМА")
    lines.append("")
    lines.append(f"Частота дискретизации (FS): {fs_user:.3f} Гц.")
    if montage:
        lines.append(f"Указанные электроды/монтаж: {montage}.")
    if used_channels:
        lines.append(f"Каналы, учтённые в расчётах: {', '.join(used_channels)}.")
    elif channel_hint:
        lines.append(f"Каналы (по подсказке): {channel_hint}.")

    lines.append(f"Количество проанализированных файлов: {n_files}.")
    lines.append("Режим анализа: " + ("анализ одного файла." if n_files <= 1 else "сравнительный анализ нескольких файлов/условий."))
    lines.append("")

    lines.append("1. Теоретическая справка (кратко)")
    lines.append(
        f"λ-ритм рассматривается как активность ЭЭГ в диапазоне {LAMBDA_BAND_HZ[0]}–{LAMBDA_BAND_HZ[1]} Гц "
        "(по методическим указаниям). Для оценки использованы два подхода:"
    )
    lines.append("• Спектральный подход: расчёт PSD методом Уэлча и вычисление мощности в заданном диапазоне частот.")
    lines.append("• Временной подход: выделение λ-составляющей полосовой фильтрацией и оценка мощности λ(t) в скользящем окне.")
    lines.append("")

    lines.append("2. Интерпретация получаемых показателей")
    lines.append("• P_λ — мощность в диапазоне λ (4–6 Гц). Чем выше значение, тем сильнее выражены колебания в этом диапазоне.")
    lines.append("• P_λ / P_total — доля мощности λ относительно общей мощности (например, 0.5–40 Гц). Это удобно для сравнения условий.")
    lines.append("• Средняя/максимальная/минимальная мощность λ(t) отражает динамику выраженности λ-ритма во времени.")
    lines.append("")

    lines.append("3. Факторы, влияющие на достоверность результатов")
    lines.append(
        "На результаты могут существенно влиять артефакты (движения, моргания), качество контакта электродов, "
        "а также корректность частоты дискретизации. При наличии выраженного шума показатели мощности могут завышаться."
    )
    lines.append("")

    lines.append("4. Числовая сводка результатов")

    if n_files <= 1:
        row = df.dropna(subset=[lam_t_mean_col] if lam_t_mean_col else []).iloc[0] if not df.empty else df.iloc[0]

        f_name = str(row[file_col]) if file_col else "—"
        ch_name = str(row[ch_col]) if ch_col else "—"

        lines.append(f"Файл: {f_name}.")
        if ch_col:
            lines.append(f"Канал/электрод: {ch_name}.")

        # λ(t)
        if lam_t_mean_col and lam_t_mean_col in df.columns:
            v = _safe_num(row.get(lam_t_mean_col))
            if v is not None:
                lines.append(f"Средняя мощность λ(t): {v:.6f}.")
        if lam_t_max_col and lam_t_max_col in df.columns:
            v = _safe_num(row.get(lam_t_max_col))
            if v is not None:
                lines.append(f"Максимум λ(t): {v:.6f}.")
        if lam_t_min_col and lam_t_min_col in df.columns:
            v = _safe_num(row.get(lam_t_min_col))
            if v is not None:
                lines.append(f"Минимум λ(t): {v:.6f}.")

        # PSD/доли
        if ratio_lam_col and ratio_lam_col in df.columns:
            v = _safe_num(row.get(ratio_lam_col))
            if v is not None:
                lines.append(f"Доля λ в общей мощности (P_λ / P_total): {v:.4f}.")
        if ratio_alp_col and ratio_alp_col in df.columns:
            v = _safe_num(row.get(ratio_alp_col))
            if v is not None:
                lines.append(f"Доля α в общей мощности (P_α / P_total): {v:.4f}.")

        lines.append("")
        lines.append(
            "Вывод по одному файлу: значения отражают выраженность λ-компоненты в записи. "
            "Для корректного сравнения условий требуется анализ нескольких записей, выполненных в разных состояниях/воздействиях."
        )
        return "\n".join(lines)

    comp_df = df.copy()

    if ratio_lam_col and ratio_lam_col in comp_df.columns and comp_df[ratio_lam_col].notna().any():
        comp_df = comp_df.dropna(subset=[ratio_lam_col])
        best = comp_df.sort_values(ratio_lam_col, ascending=False).iloc[0]
        worst = comp_df.sort_values(ratio_lam_col, ascending=True).iloc[0]

        def _fmt(r: pd.Series) -> str:
            fn = str(r.get(file_col, "—")) if file_col else "—"
            ch = str(r.get(ch_col, "—")) if ch_col else "—"
            v_ratio = _safe_num(r.get(ratio_lam_col))
            v_lamt = _safe_num(r.get(lam_t_mean_col)) if lam_t_mean_col else None
            s = f"{fn}"
            if ch_col:
                s += f" | {ch}"
            if v_ratio is not None:
                s += f" | P_λ/P_total={v_ratio:.4f}"
            if v_lamt is not None:
                s += f" | mean λ(t)={v_lamt:.6f}"
            return s

        lines.append("Сравнение выполнено по доле λ в общей мощности (P_λ / P_total).")
        lines.append("Максимальная выраженность λ:")
        lines.append(f"— {_fmt(best)}")
        lines.append("Минимальная выраженность λ:")
        lines.append(f"— {_fmt(worst)}")
        lines.append("")
    elif lam_t_mean_col and lam_t_mean_col in comp_df.columns and comp_df[lam_t_mean_col].notna().any():
        comp_df = comp_df.dropna(subset=[lam_t_mean_col])
        best = comp_df.sort_values(lam_t_mean_col, ascending=False).iloc[0]
        worst = comp_df.sort_values(lam_t_mean_col, ascending=True).iloc[0]

        def _fmt(r: pd.Series) -> str:
            fn = str(r.get(file_col, "—")) if file_col else "—"
            ch = str(r.get(ch_col, "—")) if ch_col else "—"
            v = _safe_num(r.get(lam_t_mean_col))
            s = f"{fn}"
            if ch_col:
                s += f" | {ch}"
            if v is not None:
                s += f" | mean λ(t)={v:.6f}"
            return s

        lines.append("Сравнение выполнено по средней мощности λ(t) (так как доля P_λ/P_total недоступна или отсутствует).")
        lines.append("Максимальная выраженность λ:")
        lines.append(f"— {_fmt(best)}")
        lines.append("Минимальная выраженность λ:")
        lines.append(f"— {_fmt(worst)}")
        lines.append("")
    else:
        lines.append("Недостаточно данных для автоматического сравнения (не найдены ключевые числовые показатели).")
        lines.append("")

    lines.append("5. Итоговые выводы")
    lines.append(
        "При сравнительном анализе можно оценить, в каких записях/условиях λ-компонента выражена сильнее. "
        "Для корректной интерпретации важно соблюдать одинаковые условия записи и контролировать артефакты."
    )
    lines.append(
        "Рекомендуется фиксировать расположение электродов (особенно затылочные области) и не изменять их в процессе записи, "
        "так как это напрямую влияет на сопоставимость результатов."
    )

    return "\n".join(lines)