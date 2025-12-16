from __future__ import annotations
from typing import Tuple, List, Dict, Any
import numpy as np
import pandas as pd
from scipy.signal import welch, butter, filtfilt
from config import LAMBDA_BAND_HZ, ALPHA_BAND_HZ

def compute_psd(x: np.ndarray, fs_hz: float, nperseg: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    freqs_hz, psd = welch(x, fs=fs_hz, nperseg=min(nperseg, max(64, len(x))))
    return freqs_hz, psd

def integrate_band_power(freqs_hz: np.ndarray, psd: np.ndarray, band_hz: Tuple[float, float]) -> float:
    low, high = band_hz
    m = (freqs_hz >= low) & (freqs_hz <= high)
    if not np.any(m):
        return 0.0
    return float(np.trapezoid(psd[m], freqs_hz[m]))

def butter_bandpass(data: np.ndarray, low_hz: float, high_hz: float, fs_hz: float, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs_hz
    low = low_hz / nyq
    high = high_hz / nyq
    low = max(1e-6, min(low, 0.999))
    high = max(1e-6, min(high, 0.999))
    if low >= high:
        raise ValueError("Некорректные границы фильтра (low >= high). Проверь FS и диапазон.")
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, data)

def extract_lambda_signal(x: np.ndarray, fs_hz: float) -> np.ndarray:
    return butter_bandpass(np.asarray(x, dtype=float),
                           LAMBDA_BAND_HZ[0], LAMBDA_BAND_HZ[1],
                           fs_hz=fs_hz, order=4)

def sliding_window_power(x: np.ndarray, fs_hz: float, window_sec: float = 2.0, overlap: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    win = int(window_sec * fs_hz)
    if win < 2:
        raise ValueError("Слишком маленькое окно для λ(t). Увеличь FS или window_sec.")
    step = max(1, int(win * (1.0 - overlap)))

    t_vals, p_vals = [], []
    for start in range(0, len(x) - win, step):
        seg = x[start:start + win]
        p_vals.append(float(np.mean(seg ** 2)))
        t_vals.append(start / fs_hz)
    return np.asarray(t_vals), np.asarray(p_vals)

def build_conclusions(
    summary_df: pd.DataFrame,
    records: List[Dict[str, Any]],
    fs_user: float,
    montage: str = "",
    channel_hint: str = "",
) -> str:
    if summary_df is None or summary_df.empty or not records:
        return "Анализ не выполнен или недостаточно данных для выводов."

    lines = []
    lines.append("1. Общая информация")
    lines.append(f"• Частота дискретизации (FS): {fs_user:.2f} Гц")
    if montage:
        lines.append(f"• Расположение электродов (указано пользователем): {montage}")
    if channel_hint:
        lines.append(f"• Канал (подсказка пользователя): {channel_hint}")
    lines.append("")

    lines.append("2. Качество данных")
    warn = 0
    for r in records:
        msg = []
        if r.get("time_col") == "synthetic_time":
            msg.append("время отсутствует/некорректно → использовано FS")
        if r.get("duration_s", 0) < 10:
            msg.append("короткая запись")
        if r.get("nan_ratio", 0) > 0.05:
            msg.append("много NaN в сигнале")
        if msg:
            warn += 1
            lines.append(f"• {r.get('name','файл')}: " + "; ".join(msg))
    if warn == 0:
        lines.append("• Существенных проблем качества данных не обнаружено.")
    lines.append("")

    lines.append("3. Сравнение условий по выраженности λ-ритма")
    df = summary_df.copy()

    need_cols = ["Файл", "P_λ / P_total", "P_α / P_total", "Средняя мощность λ(t)", "Максимум λ(t)", "Минимум λ(t)"]
    for c in need_cols:
        if c not in df.columns:
            lines.append("• Недостаточно метрик в таблице для расширенных выводов.")
            return "\n".join(lines)

    df_rank = df.sort_values("P_λ / P_total", ascending=False).reset_index(drop=True)
    top = df_rank.iloc[0]
    bot = df_rank.iloc[-1]
    lines.append(f"• Наибольшая относительная мощность λ: {top['Файл']} (P_λ/P_total = {float(top['P_λ / P_total']):.6f})")
    lines.append(f"• Наименьшая относительная мощность λ: {bot['Файл']} (P_λ/P_total = {float(bot['P_λ / P_total']):.6f})")

    lines.append("")
    lines.append("4. Показатели λ(t) (динамика в окнах)")
    lines.append("• Чем выше средняя мощность λ(t), тем устойчивее выражен λ-ритм на протяжении записи.")
    lines.append(f"• Максимальная средняя мощность λ(t): {df.sort_values('Средняя мощность λ(t)', ascending=False).iloc[0]['Файл']}")
    lines.append(f"• Минимальная средняя мощность λ(t): {df.sort_values('Средняя мощность λ(t)', ascending=True).iloc[0]['Файл']}")

    lines.append("")
    lines.append("5. Сопоставление λ и α диапазонов")
    df_a = df.sort_values("P_α / P_total", ascending=False).reset_index(drop=True)
    lines.append(f"• Наибольшая относительная мощность α: {df_a.iloc[0]['Файл']} (P_α/P_total = {float(df_a.iloc[0]['P_α / P_total']):.6f})")
    lines.append("• Если α-доля растёт, это может отражать более выраженную альфа-активность по сравнению с λ при данном условии.")

    lines.append("")
    lines.append("6. Итоговый вывод")
    lines.append("• В рамках выбранных условий наблюдаются различия в относительной мощности λ-диапазона (4–6 Гц).")
    lines.append("• Различия можно использовать для сравнения режимов (покой/фиксация/поиск/нагрузка),")
    lines.append("  но корректность зависит от качества записи и согласованности FS/канала.")
    lines.append("• Рекомендуется одинаковая длина записи и одинаковые параметры фильтрации для честного сравнения условий.")

    return "\n".join(lines)
