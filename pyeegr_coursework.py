import os
import time
import csv
import threading
import queue
import tempfile
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Tuple, Any, cast
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib import font_manager
from scipy.signal import welch, butter, filtfilt
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
DND_OK = False
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    DND_OK = True
except (ImportError, ModuleNotFoundError):
    DND_OK = False
SERIAL_OK = False
try:
    import serial
    import serial.tools.list_ports
    SERIAL_OK = True
except (ImportError, ModuleNotFoundError):
    SERIAL_OK = False

REPORTLAB_OK = False
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
        Table, TableStyle, PageBreak
    )
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    REPORTLAB_OK = True
except Exception:
    REPORTLAB_OK = False

FS_HZ_DEFAULT = 250.0
LAMBDA_BAND_HZ = (4.0, 6.0)
ALPHA_BAND_HZ = (7.0, 13.0)

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

def _save_figure_png_threadsafe(fig: Figure, path: str, dpi: int = 160):
    FigureCanvasAgg(fig)  # attach Agg canvas
    fig.savefig(path, dpi=dpi, bbox_inches="tight")


# ---------------------------
# CSV чтение
# ---------------------------
def _try_read_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep=None, engine="python")
    except Exception:
        pass
    for sep in [",", ";", "\t"]:
        for dec in [".", ","]:
            try:
                return pd.read_csv(path, sep=sep, decimal=dec, engine="python")
            except Exception:
                continue
    return pd.read_csv(path, engine="python")


def load_time_and_signal(path: str) -> Tuple[np.ndarray, np.ndarray, str, str]:
    df = _try_read_csv(path)
    cols = list(df.columns)

    time_candidates = [c for c in cols if "время" in str(c).lower() or "time" in str(c).lower()]
    sig_candidates = [c for c in cols if "a0" in str(c).lower() or "eeg" in str(c).lower() or "amp" in str(c).lower()]

    def to_num(s: pd.Series) -> pd.Series:
        if s.dtype == object:
            s = s.astype(str).str.replace(",", ".", regex=False)
        return pd.to_numeric(s, errors="coerce")

    numeric_cols = []
    for c in cols:
        sn = to_num(df[c])
        if sn.notna().sum() >= max(5, int(0.05 * len(df))):
            numeric_cols.append(c)

    if not numeric_cols:
        raise ValueError(f"{os.path.basename(path)}: нет числовых столбцов")

    time_col = next((c for c in time_candidates if c in numeric_cols), None)
    sig_col = next((c for c in sig_candidates if c in numeric_cols), None)

    if time_col is None or sig_col is None:
        time_like, signal_like = [], []
        for c in numeric_cols:
            s = to_num(df[c]).dropna()
            if len(s) < 10:
                continue
            is_mono = s.is_monotonic_increasing
            unique_ratio = s.nunique() / max(1, len(s))
            if is_mono and unique_ratio > 0.9:
                time_like.append(c)
            else:
                signal_like.append(c)

        if time_col is None and time_like:
            time_col = time_like[0]
        if sig_col is None and signal_like:
            sig_col = signal_like[0]

        if time_col is None and len(numeric_cols) >= 2:
            time_col = numeric_cols[0]
        if sig_col is None and len(numeric_cols) >= 2:
            sig_col = numeric_cols[1]
        if sig_col is None:
            sig_col = numeric_cols[0]
        if time_col is None:
            x_tmp = to_num(df[sig_col]).dropna().to_numpy(dtype=float)
            t_tmp = np.arange(len(x_tmp)) / FS_HZ_DEFAULT
            return t_tmp, x_tmp, "synthetic_time", sig_col

    t = to_num(df[time_col]).to_numpy(dtype=float)
    x = to_num(df[sig_col]).to_numpy(dtype=float)

    mask = np.isfinite(t) & np.isfinite(x)
    t = t[mask]
    x = x[mask]

    if len(t) >= 3 and not np.all(np.diff(t) > 0):
        t = np.arange(len(x)) / FS_HZ_DEFAULT
        time_col = "synthetic_time"

    n = min(len(t), len(x))
    return t[:n], x[:n], time_col, sig_col


def estimate_fs_from_time(t: np.ndarray, fallback: float = FS_HZ_DEFAULT) -> float:
    if len(t) < 10:
        return fallback
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) < 5:
        return fallback
    med = float(np.median(dt))
    if med <= 0:
        return fallback
    fs = 1.0 / med
    if fs < 10 or fs > 2000:
        return fallback
    return fs


# ---------------------------
# ЛР5 вычисления
# ---------------------------
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
    # защита от неверных частот
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

# ---------------------------
# Дополнительная аналитика и выводы (курсовая часть)
# ---------------------------
def robust_stats(x: np.ndarray) -> dict:
    """Набор устойчивых статистик по сигналу (без NaN/inf)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan, "p05": np.nan, "p95": np.nan}
    p05, p95 = np.percentile(x, [5, 95])
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "p05": float(p05),
        "p95": float(p95),
    }


def estimate_peak_freq(freqs_hz: np.ndarray, psd: np.ndarray, band: Tuple[float, float]) -> float:
    """Частота пика PSD внутри заданного диапазона."""
    low, high = band
    m = (freqs_hz >= low) & (freqs_hz <= high)
    if not np.any(m):
        return np.nan
    i = int(np.argmax(psd[m]))
    return float(freqs_hz[m][i])


def compute_quality_metrics(t: np.ndarray, x: np.ndarray, fs_hz: float, time_col: str) -> dict:
    """Быстрая оценка качества записи (для отображения и отчёта)."""
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    dur = float(t[-1] - t[0]) if len(t) > 1 else 0.0

    # Признаки
    time_ok = (time_col != "synthetic_time")
    nan_ratio = float(np.mean(~np.isfinite(x))) if len(x) else 1.0

    xs = x[np.isfinite(x)]
    st = robust_stats(xs)

    # Нулевая «залипшая» линия (плохо, если много одинаковых значений подряд)
    if len(xs) > 5:
        repeats = np.mean(np.diff(xs) == 0.0)
    else:
        repeats = np.nan

    # Очень грубая оценка «обрезки» сигнала: много значений возле min/max
    if len(xs) > 50 and np.isfinite(st["min"]) and np.isfinite(st["max"]) and st["max"] > st["min"]:
        eps = 0.01 * (st["max"] - st["min"])
        clip_ratio = float(np.mean((xs <= st["min"] + eps) | (xs >= st["max"] - eps)))
    else:
        clip_ratio = np.nan

    return {
        "dur_s": dur,
        "fs_hz": float(fs_hz),
        "time_ok": bool(time_ok),
        "nan_ratio": nan_ratio,
        "repeat_ratio": float(repeats) if np.isfinite(repeats) else np.nan,
        "clip_ratio": float(clip_ratio) if np.isfinite(clip_ratio) else np.nan,
        **st,
    }


def build_conclusions(records: List[dict],
                     band_power_df: pd.DataFrame,
                     lambda_time_df: pd.DataFrame,
                     summary_df: pd.DataFrame) -> Tuple[str, pd.DataFrame]:
    """
    Формирует текст «Анализ и выводы» + таблицу расширенной оценки (quality_df).
    Это не «медицинский диагноз», а инженерная интерпретация измерений.
    """
    # quality_df
    q_rows = []
    for r in records:
        freqs, psd = compute_psd(r["x"], fs_hz=r["fs"], nperseg=1024)
        peak_lambda = estimate_peak_freq(freqs, psd, LAMBDA_BAND_HZ)
        peak_alpha = estimate_peak_freq(freqs, psd, ALPHA_BAND_HZ)

        q = compute_quality_metrics(r["t"], r["x"], r["fs"], r.get("time_col", ""))
        q_rows.append({
            "Файл": r["name"],
            "Длительность, с": q["dur_s"],
            "FS, Гц": q["fs_hz"],
            "Время в CSV": "да" if q["time_ok"] else "нет",
            "NaN, доля": q["nan_ratio"],
            "Повторы, доля": q["repeat_ratio"],
            "Обрезка, доля": q["clip_ratio"],
            "Среднее": q["mean"],
            "Std": q["std"],
            "P05": q["p05"],
            "P95": q["p95"],
            "Пик λ, Гц": peak_lambda,
            "Пик α, Гц": peak_alpha,
        })
    quality_df = pd.DataFrame(q_rows)

    # Собираем удобную сводку для ранжирования
    # (используем то, что уже рассчитано)
    # Важно: используем относительную долю λ (P_λ / P_total) и среднюю мощность λ(t)
    try:
        df = summary_df.copy()
        # На всякий случай: разные имена столбцов в разных версиях
        col_ratio = next((c for c in df.columns if "P_λ" in c and "total" in c), None)
        col_mean = next((c for c in df.columns if "Средняя мощность" in c), None)
        col_max = next((c for c in df.columns if "Максимум" in c), None)

        # Топы
        top_ratio = df.sort_values(col_ratio, ascending=False).head(1).iloc[0] if col_ratio else None
        top_mean = df.sort_values(col_mean, ascending=False).head(1).iloc[0] if col_mean else None
        top_max = df.sort_values(col_max, ascending=False).head(1).iloc[0] if col_max else None
    except Exception:
        top_ratio = top_mean = top_max = None

    # Текст выводов (официальный стиль)
    lines = []
    lines.append("**Аналитическая часть (кратко)**")
    lines.append("В рамках работы выполнены: спектральный анализ методом Уэлча, расчёт мощности в диапазонах λ (4–6 Гц) и α (7–13 Гц), "
                 "а также оценка динамики мощности λ(t) по скользящему окну. Ниже приведена инженерная интерпретация полученных метрик.")

    # Качество данных
    if not quality_df.empty:
        dur_min = float(np.nanmin(quality_df["Длительность, с"].values))
        dur_max = float(np.nanmax(quality_df["Длительность, с"].values))
        fs_med = float(np.nanmedian(quality_df["FS, Гц"].values))
        no_time = int(np.sum(quality_df["Время в CSV"].values == "нет"))
        lines.append(f"**Качество данных:** длительность записей {dur_min:.1f}–{dur_max:.1f} с; медиана FS ≈ {fs_med:.1f} Гц; "
                     f"файлов без корректного столбца времени: {no_time}.")

    # Лидеры по метрикам
    if top_ratio is not None:
        lines.append(f"**Максимальная относительная доля λ (P_λ/P_total):** файл «{top_ratio['Файл']}».")
    if top_mean is not None:
        lines.append(f"**Максимальная средняя мощность λ(t):** файл «{top_mean['Файл']}».")
    if top_max is not None:
        lines.append(f"**Наибольший максимум λ(t):** файл «{top_max['Файл']}» (характерно для кратковременных всплесков активности).")

    # Шаблонные аккуратные выводы по смыслу
    lines.append("**Интерпретация:** более высокая доля мощности λ в PSD и более высокая средняя λ(t) указывают на выраженность λ-активности в записи "
                 "(в рамках выбранной методики и параметров фильтрации). Важно учитывать, что это зависит от условий эксперимента, качества контакта электродов "
                 "и наличия артефактов (движения, моргания).")

    # Рекомендации по улучшению анализа
    lines.append("**Рекомендации:** для сравнения условий удобнее использовать относительные показатели (например, P_λ/P_total), "
                 "а также нормировать λ(t) на общую мощность/дисперсию, чтобы снизить влияние амплитудных масштабов разных записей.")

    text_md = "\n".join(lines)
    return text_md, quality_df


# ---------------------------
# Скролл-область для графиков
# ---------------------------
class ScrollablePlotArea(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)

        self.canvas = tk.Canvas(self, highlightthickness=0, bg=UI["bg"])
        self.vbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vbar.set)

        self.vbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.inner = ttk.Frame(self.canvas, style="Card2.TFrame")
        self.inner_id = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")

        self.inner.bind("<Configure>", self._on_frame_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel_linux)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel_linux)

    def _on_frame_configure(self, _event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self.canvas.itemconfigure(self.inner_id, width=event.width)

    def _on_mousewheel(self, event):
        delta = int(-1 * (event.delta / 120))
        self.canvas.yview_scroll(delta, "units")

    def _on_mousewheel_linux(self, event):
        if event.num == 4:
            self.canvas.yview_scroll(-2, "units")
        elif event.num == 5:
            self.canvas.yview_scroll(2, "units")

    def clear(self):
        for child in list(self.inner.winfo_children()):
            child.destroy()

class ScrollableFrame(ttk.Frame):
    """Обычный скроллимый контейнер для вкладок/панелей (Canvas + Frame)."""
    def __init__(self, parent, bg=None):
        super().__init__(parent)

        self.canvas = tk.Canvas(self, highlightthickness=0, bg=bg if bg else None)
        self.vbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vbar.set)

        self.vbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.inner = ttk.Frame(self.canvas)
        self.inner_id = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")

        self.inner.bind("<Configure>", self._on_frame_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        # колесо мыши
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel_linux)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel_linux)

    def _on_frame_configure(self, _event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self.canvas.itemconfigure(self.inner_id, width=event.width)

    def _on_mousewheel(self, event):
        delta = int(-1 * (event.delta / 120))
        self.canvas.yview_scroll(delta, "units")

    def _on_mousewheel_linux(self, event):
        if event.num == 4:
            self.canvas.yview_scroll(-2, "units")
        elif event.num == 5:
            self.canvas.yview_scroll(2, "units")
# ---------------------------
# Serial
# ---------------------------
@dataclass
class SerialConfig:
    port: str = ""
    baudrate: int = 115200
    delimiter: str = ","
    channels: int = 1


class ArduinoSerialStreamer(threading.Thread):
    def __init__(self, cfg: SerialConfig, out_queue: "queue.Queue[Tuple[float, float]]"):
        super().__init__(daemon=True)
        self.cfg = cfg
        self.q = out_queue
        self._stop = threading.Event()
        self.ser = None
        self.t0 = None

    def connect(self):
        if not SERIAL_OK:
            raise RuntimeError("pyserial не установлен")
        if not self.cfg.port:
            raise RuntimeError("Не выбран порт")
        self.ser = serial.Serial(self.cfg.port, self.cfg.baudrate, timeout=1)
        time.sleep(1.2)
        self.ser.reset_input_buffer()
        self.t0 = time.time()

    def stop(self):
        self._stop.set()

    def run(self):
        try:
            self.connect()
        except Exception as e:
            self.q.put(("__ERROR__", float("nan")))
            self.q.put((0.0, str(e)))
            return

        try:
            while not self._stop.is_set():
                line = self.ser.readline().decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                raw = line.split(self.cfg.delimiter)[0].strip().replace(",", ".")
                try:
                    val = float(raw)
                except Exception:
                    continue
                t = time.time() - self.t0
                self.q.put((t, val))
        finally:
            try:
                if self.ser and self.ser.is_open:
                    self.ser.close()
            except Exception:
                pass


_BaseTk = TkinterDnD.Tk if DND_OK else tk.Tk


# ---------------------------
# Приложение
# ---------------------------
class EEGApp(_BaseTk):
    def __init__(self):
        super().__init__()
        apply_mpl_style()

        self.title("# Лямбда-ритмы ЭЭГ при разных воздействиях")
        self.geometry("1240x820")
        self.configure(bg=UI["bg"])

        # Serial
        self.serial_queue: "queue.Queue[tuple[Any, Any]]" = queue.Queue()
        self.streamer: Optional[ArduinoSerialStreamer] = None
        self.live_t: List[float] = []
        self.live_x: List[float] = []
        self.live_max_sec = 10.0

        # Files / results
        self.loaded_files: List[str] = []
        self._last_records: Optional[list[dict[str, Any]]] = None
        self._last_fs_user = FS_HZ_DEFAULT
        self.eeg_montage = tk.StringVar(value="O1–Oz–O2 (затылочная область)")
        self.eeg_channel_hint = tk.StringVar(value="Авто (по CSV)")
        self.band_power_df: Optional[pd.DataFrame] = None
        self.lambda_time_df: Optional[pd.DataFrame] = None
        self.summary_df: Optional[pd.DataFrame] = None
        self.quality_df: Optional[pd.DataFrame] = None
        self.conclusions_md: str = ""

        # analysis thread
        self._analysis_thread: Optional[threading.Thread] = None
        self._analysis_busy = False
        self._ui_queue: "queue.Queue[Tuple[str, object]]" = queue.Queue()

        # PDF thread (FIX)
        self._pdf_thread: Optional[threading.Thread] = None
        self._pdf_busy = False

        self._setup_style()
        self._build_ui()

        self.after(60, self._poll_serial_queue)
        self.after(60, self._poll_ui_queue)

    def _setup_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure(".", background=UI["bg"], foreground=UI["text"], font=FONT_MAIN)
        style.configure("TFrame", background=UI["bg"])
        style.configure("Card.TFrame", background=UI["panel"])
        style.configure("Card2.TFrame", background=UI["panel2"])
        style.configure(
            "Drop.TLabel",
            background=UI["panel2"],
            foreground=_blend(UI["text"], UI["accent"], 0.35),  # тёплый читаемый текст
            font=(FONT_MAIN[0], 12, "bold"),
        )
        style.configure("TLabel", background=UI["bg"], foreground=UI["text"], font=FONT_MAIN)
        style.configure("Muted.TLabel", background=UI["bg"], foreground=UI["muted"], font=FONT_SMALL)
        style.configure("Title.TLabel", background=UI["bg"], foreground=UI["text"], font=FONT_TITLE)
        style.configure("H2.TLabel", background=UI["bg"], foreground=UI["text"], font=FONT_H2)

        style.configure("TButton", padding=(14, 10), relief="flat", font=FONT_MAIN)

        style.configure("Primary.TButton", background=UI["accent"], foreground="white")
        style.map("Primary.TButton",
                  background=[("active", UI["accent2"]), ("disabled", UI["border"])],
                  foreground=[("disabled", UI["muted"])])

        style.configure("Ghost.TButton", background=UI["panel2"], foreground=UI["text"])
        style.map("Ghost.TButton",
                  background=[("active", UI["hover"]), ("disabled", UI["panel2"])],
                  foreground=[("disabled", UI["muted"])])

        style.configure("Danger.TButton", background=UI["danger"], foreground="white")
        style.map("Danger.TButton",
                  background=[("active", "#FCA5A5"), ("disabled", UI["border"])],
                  foreground=[("disabled", UI["muted"])])

        style.configure("TEntry", fieldbackground=UI["panel2"], foreground=UI["text"])
        style.configure("TCombobox", fieldbackground=UI["panel2"], foreground=UI["text"])

        style.configure("TNotebook", background=UI["bg"], borderwidth=0)
        style.configure("TNotebook.Tab",
                        padding=(16, 10),
                        background=UI["panel2"],
                        foreground=UI["muted"])
        style.map("TNotebook.Tab",
                  background=[("selected", UI["panel"]), ("active", UI["hover"])],
                  foreground=[("selected", UI["text"]), ("active", UI["text"])])

        style.configure("TProgressbar", troughcolor=UI["panel2"], background=UI["accent"], bordercolor=UI["border"])

        style.configure("Treeview",
                        background=UI["panel2"],
                        fieldbackground=UI["panel2"],
                        foreground=UI["text"],
                        rowheight=28,
                        bordercolor=UI["border"])
        style.map("Treeview",
                  background=[("selected", UI["hover"])],
                  foreground=[("selected", UI["text"])])

        style.configure("Treeview.Heading",
                        background=UI["panel"],
                        foreground=UI["text"],
                        relief="flat")
        style.map("Treeview.Heading",
                  background=[("active", UI["hover"])])

        style.configure("Seg.TRadiobutton",
                        background=UI["panel2"],
                        foreground=UI["muted"],
                        padding=(12, 8))
        style.map("Seg.TRadiobutton",
                  background=[("selected", UI["hover"]), ("active", UI["hover"])],
                  foreground=[("selected", UI["text"]), ("active", UI["text"])])
        # --- Combobox dropdown (Listbox внутри popdown) ---
        # фиксит "чёрный" выпадающий список
        self.option_add("*TCombobox*Listbox.background", UI["panel2"])
        self.option_add("*TCombobox*Listbox.foreground", UI["text"])
        self.option_add("*TCombobox*Listbox.selectBackground", UI["hover"])
        self.option_add("*TCombobox*Listbox.selectForeground", UI["text"])
        self.option_add("*TCombobox*Listbox.font", FONT_MAIN)
        self.option_add("*TCombobox*Listbox.borderWidth", 0)
        self.option_add("*TCombobox*Listbox.highlightThickness", 1)
        self.option_add("*TCombobox*Listbox.highlightBackground", UI["border"])

    # -------- UI --------
    def _build_ui(self):
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=12, pady=12)

        self.tab_live = ttk.Frame(nb)
        self.tab_files = ttk.Frame(nb)
        self.tab_analysis = ttk.Frame(nb)

        nb.add(self.tab_live, text="Онлайн")
        nb.add(self.tab_files, text="Файлы")
        nb.add(self.tab_analysis, text="Анализ ЛР5")

        self._build_live_tab()
        self._build_files_tab()
        self._build_analysis_tab()

    # ---------------------------
    # Вкладка: Онлайн
    # ---------------------------
    def _build_live_tab(self):
        root = ttk.Frame(self.tab_live)
        root.pack(fill="both", expand=True)

        header = ttk.Frame(root)
        header.pack(fill="x", pady=(0, 10))
        ttk.Label(header, text="Онлайн запись (Arduino)", style="Title.TLabel").pack(side="left")

        card = ttk.Frame(root, padding=14, style="Card.TFrame")
        card.pack(fill="x")

        ttk.Label(card, text="Порт:", style="Muted.TLabel").pack(side="left", padx=(0, 6))
        self.cbo_port = ttk.Combobox(card, width=34, state="normal")
        self.cbo_port.pack(side="left", padx=(0, 10))

        self.btn_ports = ttk.Button(card, text="Обновить", command=self.refresh_ports, style="Ghost.TButton")
        self.btn_ports.pack(side="left", padx=(0, 16))

        ttk.Label(card, text="Скорость:", style="Muted.TLabel").pack(side="left", padx=(0, 6))
        self.ent_baud = ttk.Entry(card, width=10)
        self.ent_baud.insert(0, "115200")
        self.ent_baud.pack(side="left", padx=(0, 16))

        ttk.Label(card, text="Окно (с):", style="Muted.TLabel").pack(side="left", padx=(0, 6))
        self.ent_win = ttk.Entry(card, width=8)
        self.ent_win.insert(0, "10")
        self.ent_win.pack(side="left", padx=(0, 16))

        self.btn_start = ttk.Button(card, text="▶ Старт", command=self.start_stream, style="Primary.TButton")
        self.btn_start.pack(side="left", padx=(0, 8))

        self.btn_stop = ttk.Button(card, text="■ Стоп", command=self.stop_stream, state="disabled",
                                   style="Danger.TButton")
        self.btn_stop.pack(side="left", padx=(0, 8))

        self.btn_save = ttk.Button(card, text="💾 Сохранить CSV", command=self.save_live_csv, state="disabled",
                                   style="Ghost.TButton")
        self.btn_save.pack(side="left")

        plot_card = ttk.Frame(root, padding=14, style="Card.TFrame")
        plot_card.pack(fill="both", expand=True, pady=(12, 0))

        self.fig_live = Figure(figsize=(10, 4), dpi=110)
        self.ax_live = self.fig_live.add_subplot(111)
        style_axes(self.ax_live)
        self.ax_live.set_title("Сигнал в реальном времени")
        self.ax_live.set_ylabel("A0 (у.е.)")
        self.line_live, = self.ax_live.plot([], [], linewidth=2.0)

        self.canvas_live = FigureCanvasTkAgg(self.fig_live, master=plot_card)
        self.canvas_live.get_tk_widget().pack(fill="both", expand=True)

        self.lbl_live_status = ttk.Label(root, text="Статус: не запущено", style="Muted.TLabel")
        self.lbl_live_status.pack(anchor="w", pady=(10, 0))

        self.refresh_ports(silent=True)

    def refresh_ports(self, silent: bool = False):
        ports = []
        if SERIAL_OK:
            try:
                ports = [p.device for p in serial.tools.list_ports.comports()]
            except Exception:
                ports = []
        self.cbo_port["values"] = ports
        if ports and not self.cbo_port.get():
            self.cbo_port.set(ports[0])
        if not silent:
            self.lbl_live_status.config(text="Статус: список портов обновлён")

    def start_stream(self):
        if not SERIAL_OK:
            messagebox.showerror("Serial", "pyserial не установлен.\n\npip install pyserial")
            return

        port = (self.cbo_port.get() or "").strip()
        try:
            baud = int(self.ent_baud.get().strip())
        except Exception:
            messagebox.showerror("Ошибка", "Неверная скорость (baudrate).")
            return

        try:
            self.live_max_sec = float(self.ent_win.get().strip())
        except Exception:
            self.live_max_sec = 10.0

        self.live_t.clear()
        self.live_x.clear()

        cfg = SerialConfig(port=port, baudrate=baud, delimiter=",", channels=1)
        self.streamer = ArduinoSerialStreamer(cfg, self.serial_queue)
        self.streamer.start()

        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.btn_save.config(state="normal")
        self.lbl_live_status.config(text=f"Статус: подключение к {port} @ {baud}")

    def stop_stream(self):
        if self.streamer:
            self.streamer.stop()
            self.streamer = None
        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")
        self.lbl_live_status.config(text="Статус: остановлено")

    def _poll_serial_queue(self):
        updated = False
        while True:
            try:
                item = self.serial_queue.get_nowait()
            except queue.Empty:
                break

            if isinstance(item[0], str) and item[0] == "__ERROR__":
                try:
                    _, msg = self.serial_queue.get_nowait()
                    messagebox.showerror("Ошибка Serial", f"Не удалось подключиться:\n{msg}")
                except Exception:
                    messagebox.showerror("Ошибка Serial", "Не удалось подключиться.")
                self.stop_stream()
                break

            t, x = item
            if not isinstance(t, (int, float)) or not isinstance(x, (int, float)):
                continue

            self.live_t.append(float(t))
            self.live_x.append(float(x))
            updated = True

        if updated:
            self._update_live_plot()

        self.after(60, self._poll_serial_queue)

    def _update_live_plot(self):
        if not self.live_t:
            return

        t = np.asarray(self.live_t)
        x = np.asarray(self.live_x)

        tmax = t[-1]
        m = t >= max(0.0, tmax - self.live_max_sec)
        t2, x2 = t[m], x[m]

        self.line_live.set_data(t2, x2)
        self.ax_live.set_xlim(float(t2[0]), float(t2[-1]) if len(t2) > 1 else float(t2[0]) + 1e-3)

        ymin, ymax = float(np.min(x2)), float(np.max(x2))
        pad = 0.05 * (ymax - ymin) if ymax > ymin else 0.5
        self.ax_live.set_ylim(ymin - pad, ymax + pad)

        self.canvas_live.draw_idle()
        self.lbl_live_status.config(text=f"Статус: поток | точек: {len(self.live_t)} | t={t[-1]:.2f}с")

    def save_live_csv(self):
        if not self.live_t:
            messagebox.showwarning("Нет данных", "Сначала запусти запись и дождись данных.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile=f"eeg_live_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        )
        if not path:
            return

        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["время_с", "a0"])
            for t, x in zip(self.live_t, self.live_x):
                w.writerow([f"{t:.6f}", f"{x:.6f}"])

        messagebox.showinfo("Сохранено", f"CSV сохранён:\n{path}")

    def _build_files_tab(self):
        # Контейнер вкладки
        root = ttk.Frame(self.tab_files)
        root.pack(fill="both", expand=True)

        # Общий скролл всей вкладки
        sf = ScrollableFrame(root, bg=UI["bg"])
        sf.pack(fill="both", expand=True)

        # ВАЖНО: дальше работаем не с root, а с sf.inner
        page = sf.inner

        header = ttk.Frame(page)
        header.pack(fill="x", pady=(0, 10), padx=12)
        ttk.Label(header, text="Файлы CSV", style="Title.TLabel").pack(side="left")

        card = ttk.Frame(page, padding=14, style="Card.TFrame")
        card.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        # --- Верхняя панель кнопок ---
        top = ttk.Frame(card, style="Card.TFrame")
        top.pack(fill="x")

        self.btn_add_files = ttk.Button(
            top, text="➕ Добавить CSV", command=self.add_csv_files, style="Primary.TButton"
        )
        self.btn_add_files.pack(side="left")

        self.btn_remove_file = ttk.Button(
            top, text="🗑 Удалить", command=self.remove_selected_file, style="Ghost.TButton"
        )
        self.btn_remove_file.pack(side="left", padx=8)

        self.btn_clear_files = ttk.Button(
            top, text="Очистить", command=self.clear_file_list, style="Danger.TButton"
        )
        self.btn_clear_files.pack(side="left", padx=8)

        self.lbl_files_count = ttk.Label(top, text="Файлов: 0", style="Muted.TLabel")
        self.lbl_files_count.pack(side="right")

        # --- Зона drag&drop ---
        drop = ttk.Frame(card, padding=12, style="Card2.TFrame")
        drop.pack(fill="x", pady=(12, 10))

        drop_text = "Перетащите CSV сюда" if DND_OK else "Добавьте CSV кнопкой выше"
        self.drop_label = ttk.Label(drop, text=drop_text, style="Muted.TLabel",
                                    anchor="center", justify="center")
        self.drop_label.pack(fill="x")

        # --- Как пользоваться ---
        help_box = ttk.Frame(card, padding=12, style="Card2.TFrame")
        help_box.pack(fill="x", pady=(0, 12))

        ttk.Label(help_box, text="Как пользоваться", style="H2.TLabel").pack(anchor="w", pady=(0, 8))
        steps = [
            "1) Добавьте один или несколько CSV-файлов с записью ЭЭГ (каждый файл — отдельное условие).",
            "2) Убедитесь, что файл содержит столбец сигнала. Столбец времени желателен, но не обязателен.",
            "3) Откройте вкладку «Анализ ЛР5» и при необходимости задайте частоту дискретизации FS (Гц).",
            "4) Нажмите «Запустить анализ» и дождитесь окончания расчётов (появятся таблицы и графики).",
            "5) Нажмите «Экспорт PDF», чтобы сохранить отчёт с результатами (таблицы + рисунки).",
        ]
        for s in steps:
            ttk.Label(help_box, text=s, style="Muted.TLabel",
                      wraplength=980, justify="left").pack(anchor="w", pady=2)

        # --- Список файлов ---
        ttk.Label(card, text="Список добавленных файлов", style="H2.TLabel").pack(anchor="w", pady=(6, 6))

        list_frame = ttk.Frame(card, style="Card.TFrame")
        list_frame.pack(fill="x", pady=(0, 12))

        list_frame.configure(height=170)
        list_frame.pack_propagate(False)

        self.lst_files = tk.Listbox(
            list_frame,
            height=6,
            bg=UI["panel2"],
            fg=UI["text"],
            selectbackground=UI.get("hover", UI["accent"]),
            selectforeground=UI["text"],
            highlightthickness=0,
            bd=0,
        )
        self.lst_files.pack(side="left", fill="both", expand=True, padx=(0, 8))

        scr = ttk.Scrollbar(list_frame, orient="vertical", command=self.lst_files.yview)
        scr.pack(side="left", fill="y")
        self.lst_files.configure(yscrollcommand=scr.set)

        # --- Быстрая оценка ---
        ttk.Separator(card, orient="horizontal").pack(fill="x", pady=10)
        ttk.Label(card, text="Быстрая оценка", style="H2.TLabel").pack(anchor="w", pady=(0, 6))

        self.lbl_quick = ttk.Label(card, text="Пока файлов нет.", style="Muted.TLabel",
                                   wraplength=980, justify="left")
        self.lbl_quick.pack(anchor="w", pady=(0, 10))

        tbl_wrap = ttk.Frame(card, style="Card.TFrame")
        tbl_wrap.pack(fill="both", expand=True)

        self.quick_tbl = ttk.Treeview(tbl_wrap, show="headings", height=7)
        self.quick_tbl.pack(side="left", fill="both", expand=True)

        self.quick_tbl_scr_y = ttk.Scrollbar(tbl_wrap, orient="vertical", command=self.quick_tbl.yview)
        self.quick_tbl_scr_y.pack(side="left", fill="y")

        self.quick_tbl_scr_x = ttk.Scrollbar(card, orient="horizontal", command=self.quick_tbl.xview)
        self.quick_tbl_scr_x.pack(fill="x")

        self.quick_tbl.configure(
            yscrollcommand=self.quick_tbl_scr_y.set,
            xscrollcommand=self.quick_tbl_scr_x.set
        )

        # --- Пояснения ---
        ttk.Separator(card, orient="horizontal").pack(fill="x", pady=10)
        ttk.Label(card, text="Что означают показатели", style="H2.TLabel").pack(anchor="w", pady=(0, 6))

        desc = (
            "• Отсчётов (N) — количество значений сигнала.\n"
            "• Длительность, с — длительность записи (если есть корректное время; иначе N/FS).\n"
            "• FS, Гц — частота дискретизации.\n"
            "• Среднее — среднее значение амплитуды (оценка смещения).\n"
            "• σ (СКО) — стандартное отклонение (разброс амплитуды).\n"
            "• RMS — среднеквадратичное значение (характеризует «энергию/мощность» сигнала).\n"
            "• Размах — max-min (амплитудный диапазон).\n"
            "• Медиана — устойчивый центральный уровень (менее чувствительна к выбросам)."
        )

        desc_frame = ttk.Frame(card, style="Card2.TFrame", padding=10)
        desc_frame.pack(fill="x", pady=(0, 4))

        txt = tk.Text(desc_frame, height=6, wrap="word",
                      bg=UI["panel2"], fg=UI["muted"],
                      bd=0, highlightthickness=0)
        txt.pack(side="left", fill="x", expand=True)
        txt.insert("1.0", desc)
        txt.configure(state="disabled")

        # Drag&Drop bind
        if DND_OK:
            for widget in (self.drop_label, self.lst_files):
                widget.drop_target_register(DND_FILES)
                widget.dnd_bind("<<DropEnter>>", self._on_drop_enter)
                widget.dnd_bind("<<DropLeave>>", self._on_drop_leave)
                widget.dnd_bind("<<Drop>>", self._on_drop_files)

    def _on_drop_enter(self, _event=None):
        self.drop_label.config(text="Отпустите файлы, чтобы добавить")
        # подсветка
        self.drop_label.configure(foreground=UI["accent"])

    def _on_drop_leave(self, _event=None):
        self.drop_label.config(text="Перетащите CSV сюда")
        self.drop_label.configure(foreground=_blend(UI["text"], UI["accent"], 0.35))

    def _parse_dnd_files(self, data: str) -> List[str]:
        data = (data or "").strip()
        if not data:
            return []
        out, buf, in_brace = [], "", False
        for ch in data:
            if ch == "{":
                in_brace = True
                buf = ""
            elif ch == "}":
                in_brace = False
                out.append(buf)
                buf = ""
            elif ch == " " and not in_brace:
                if buf:
                    out.append(buf)
                    buf = ""
            else:
                buf += ch
        if buf:
            out.append(buf)
        return [p.strip().strip('"') for p in out if p.strip()]

    def _on_drop_files(self, event):
        paths = self._parse_dnd_files(event.data)
        self._on_drop_leave()
        self._add_paths(paths)

    def _refresh_files_count(self):
        self.lbl_files_count.config(text=f"Файлов: {len(self.loaded_files)}")

    def add_csv_files(self):
        paths = filedialog.askopenfilenames(filetypes=[("CSV", "*.csv"), ("Все файлы", "*.*")])
        self._add_paths(list(paths))

    def _add_paths(self, paths: List[str]):
        added = 0
        for p in paths:
            p = os.path.abspath(p)
            if os.path.isdir(p):
                for fn in os.listdir(p):
                    fp = os.path.join(p, fn)
                    if os.path.isfile(fp) and fn.lower().endswith(".csv") and fp not in self.loaded_files:
                        self.loaded_files.append(fp)
                        self.lst_files.insert("end", fp)
                        added += 1
                continue

            if not os.path.isfile(p):
                continue
            if not p.lower().endswith(".csv"):
                continue
            if p not in self.loaded_files:
                self.loaded_files.append(p)
                self.lst_files.insert("end", p)
                added += 1

        self._refresh_files_count()
        self._update_quick_check()
        self.lbl_quick.config(text=f"Добавлено файлов: {added}" if added else "Файлы не добавлены.")

    def remove_selected_file(self):
        sel = self.lst_files.curselection()
        if not sel:
            return
        idx = sel[0]
        path = self.lst_files.get(idx)
        self.lst_files.delete(idx)
        self.loaded_files = [p for p in self.loaded_files if p != path]
        self._refresh_files_count()
        self._update_quick_check()

    def clear_file_list(self):
        self.lst_files.delete(0, "end")
        self.loaded_files.clear()
        self._refresh_files_count()
        self._update_quick_check()
        self.lbl_quick.config(text="Пока файлов нет.")

    def _update_quick_check(self):
        if not hasattr(self, "quick_tbl"):
            return

        if not self.loaded_files:
            self.lbl_quick.config(text="Пока файлов нет.")
            self.quick_tbl.delete(*self.quick_tbl.get_children())
            self.quick_tbl["columns"] = []
            self._refresh_files_count()
            return

        rows = []
        total_dur = 0.0
        fs_vals = []

        for p in self.loaded_files[:50]:
            try:
                t, x, tcol, xcol = load_time_and_signal(p)
                fs = estimate_fs_from_time(t, fallback=np.nan)

                n = int(len(x))
                dur = float(t[-1] - t[0]) if len(t) > 1 else (n / FS_HZ_DEFAULT)
                total_dur += max(0.0, dur)

                if np.isfinite(fs):
                    fs_vals.append(float(fs))

                xs = x[np.isfinite(x)]
                mean = float(np.mean(xs)) if len(xs) else np.nan
                std = float(np.std(xs)) if len(xs) else np.nan
                rms = float(np.sqrt(np.mean(xs ** 2))) if len(xs) else np.nan
                vmin = float(np.min(xs)) if len(xs) else np.nan
                vmax = float(np.max(xs)) if len(xs) else np.nan
                ptp = float(vmax - vmin) if np.isfinite(vmax) and np.isfinite(vmin) else np.nan
                med = float(np.median(xs)) if len(xs) else np.nan

                def fmt(v, nd=3):
                    if v is None or (isinstance(v, float) and not np.isfinite(v)):
                        return "—"
                    return f"{v:.{nd}f}"

                rows.append([
                    os.path.basename(p),
                    f"{n:,}".replace(",", " "),
                    fmt(dur, 2),
                    fmt(fs, 2) if np.isfinite(fs) else "—",
                    fmt(mean, 3),
                    fmt(std, 3),
                    fmt(rms, 3),
                    fmt(ptp, 3),
                    fmt(med, 3),
                ])
            except Exception:
                rows.append([os.path.basename(p), "—", "—", "—", "—", "—", "—", "—", "—"])

        fs_text = f"{np.median(fs_vals):.2f} Гц" if fs_vals else "—"
        self.lbl_quick.config(
            text=f"Файлов: {len(self.loaded_files)} • Суммарная длительность: {total_dur / 60:.2f} мин • FS (медиана): {fs_text}"
        )

        cols = ["Файл", "Отсчётов (N)", "Длительность, с", "FS, Гц", "Среднее", "σ (СКО)", "RMS", "Размах", "Медиана"]
        self.quick_tbl["columns"] = cols

        # ширины + выравнивание
        col_cfg = {
            "Файл": dict(width=260, anchor="w", stretch=True),
            "Отсчётов (N)": dict(width=120, anchor="center", stretch=False),
            "Длительность, с": dict(width=130, anchor="center", stretch=False),
            "FS, Гц": dict(width=90, anchor="center", stretch=False),
            "Среднее": dict(width=90, anchor="center", stretch=False),
            "σ (СКО)": dict(width=90, anchor="center", stretch=False),
            "RMS": dict(width=90, anchor="center", stretch=False),
            "Размах": dict(width=110, anchor="center", stretch=False),
            "Медиана": dict(width=90, anchor="center", stretch=False),
        }

        for c in cols:
            self.quick_tbl.heading(c, text=c)
            self.quick_tbl.column(c, **col_cfg[c])

        self.quick_tbl.delete(*self.quick_tbl.get_children())
        for r in rows:
            self.quick_tbl.insert("", "end", values=r)

        self._refresh_files_count()
    def _build_analysis_tab(self):
        root = ttk.Frame(self.tab_analysis)
        root.pack(fill="both", expand=True)

        # Переменные (если вдруг у вас ещё нет)
        if not hasattr(self, "eeg_montage"):
            self.eeg_montage = tk.StringVar(value="O1–Oz–O2 (затылочная область)")
        if not hasattr(self, "eeg_channel_hint"):
            self.eeg_channel_hint = tk.StringVar(value="Авто (по CSV)")
        if not hasattr(self, "conclusions_text"):
            self.conclusions_text = ""

        header = ttk.Frame(root)
        header.pack(fill="x", pady=(0, 10))
        ttk.Label(header, text="Лямбда-ритм ЭЭГ при разных воздействиях", style="Title.TLabel").pack(side="left")

        controls = ttk.Frame(root, padding=14, style="Card.TFrame")
        controls.pack(fill="x")

        ttk.Label(controls, text="Частота дискретизации FS (Гц):", style="Muted.TLabel").pack(side="left", padx=(0, 8))
        self.ent_fs = ttk.Entry(controls, width=10)
        self.ent_fs.insert(0, "250")
        self.ent_fs.pack(side="left", padx=(0, 14))

        # --- НОВОЕ: выбор расположения электродов/канала ---
        ttk.Label(controls, text="Расположение электродов:", style="Muted.TLabel").pack(side="left", padx=(8, 8))
        self.cbo_montage = ttk.Combobox(
            controls,
            textvariable=self.eeg_montage,
            state="readonly",
            width=30,
            values=[
                "O1–Oz–O2 (затылочная область)",
                "Pz (теменно-затылочная область)",
                "T5/T6 (височно-затылочная область)",
                "Другое (указать в выводах)",
            ],
        )
        self.cbo_montage.pack(side="left", padx=(0, 14))

        ttk.Label(controls, text="Канал:", style="Muted.TLabel").pack(side="left", padx=(0, 8))
        self.cbo_channel = ttk.Combobox(
            controls,
            textvariable=self.eeg_channel_hint,
            state="readonly",
            width=18,
            values=["Авто (по CSV)", "A0", "EEG", "Ch1", "Ch2"],
        )
        self.cbo_channel.pack(side="left", padx=(0, 14))
        # --- /НОВОЕ ---

        self.btn_run = ttk.Button(controls, text="▶ Запустить анализ", command=self.run_lab5, style="Primary.TButton")
        self.btn_run.pack(side="left", padx=(0, 8))

        self.btn_report = ttk.Button(controls, text="📄 Экспорт PDF", command=self.export_report_pdf,
                                     style="Ghost.TButton")
        self.btn_report.pack(side="left", padx=(0, 12))

        self.pb = ttk.Progressbar(controls, mode="indeterminate", length=180)
        self.pb.pack(side="left", padx=(0, 10))

        self.lbl_an_status = ttk.Label(controls, text="Готово", style="Muted.TLabel")
        self.lbl_an_status.pack(side="left")

        body = ttk.PanedWindow(root, orient="horizontal")
        body.pack(fill="both", expand=True, pady=(12, 0))

        left = ttk.Frame(body)
        right = ttk.Frame(body)
        body.add(left, weight=2)
        body.add(right, weight=5)

        # Левая карточка: таблица
        left_card = ttk.Frame(left, padding=14, style="Card.TFrame")
        left_card.pack(fill="both", expand=True)

        ttk.Label(left_card, text="Таблица результатов", style="H2.TLabel").pack(anchor="w")
        ttk.Label(controls, text="Схема 10–20:", style="Muted.TLabel") \
            .pack(side="left", padx=(18, 8))

        self.ten20 = TenTwentySelector(
            controls,
            on_change=self._on_electrodes_changed,
            width=300,
            height=220
        )
        self.ten20.pack(side="left", padx=(0, 12))

        ttk.Label(controls, text="Канал:", style="Muted.TLabel") \
            .pack(side="left", padx=(0, 8))

        self.cbo_channel = ttk.Combobox(
            controls,
            state="readonly",
            width=16,
            values=["auto"]
        )
        self.cbo_channel.set("auto")
        self.cbo_channel.pack(side="left")
        self.cbo_table = ttk.Combobox(
            left_card,
            state="readonly",
            values=["Сводная", "Мощности по диапазонам", "Статистики λ(t)"]
        )
        self.cbo_table.current(0)
        self.cbo_table.pack(fill="x", pady=(10, 10))
        self.cbo_table.bind("<<ComboboxSelected>>", lambda e: self._render_current_table())

        self.tbl = ttk.Treeview(left_card, show="headings")
        self.tbl.pack(fill="both", expand=True)

        self.tbl_scr = ttk.Scrollbar(left_card, orient="vertical", command=self.tbl.yview)
        self.tbl_scr.pack(side="right", fill="y")
        self.tbl.configure(yscrollcommand=self.tbl_scr.set)

        # Правая карточка: графики + выводы
        right_card = ttk.Frame(right, padding=14, style="Card.TFrame")
        right_card.pack(fill="both", expand=True)

        seg = ttk.Frame(right_card, padding=10, style="Card2.TFrame")
        seg.pack(fill="x", pady=(0, 12))

        ttk.Label(seg, text="Режим отображения:", style="Muted.TLabel").pack(side="left", padx=(0, 10))

        self.plot_mode = tk.StringVar(value="RAW")
        for key, label in [("RAW", "Сигнал"), ("PSD", "Спектр (PSD)"), ("LAMBDA", "λ-ритм"), ("BARS", "Сравнение")]:
            ttk.Radiobutton(
                seg,
                text=label,
                value=key,
                variable=self.plot_mode,
                style="Seg.TRadiobutton",
                command=self._render_plots
            ).pack(side="left", padx=(0, 8))

        self.plot_host = ttk.Frame(right_card, style="Card2.TFrame")
        self.plot_host.pack(fill="both", expand=True)

        self.plot_area = ScrollablePlotArea(self.plot_host)
        self.plot_area.pack(fill="both", expand=True)

        # --- НОВОЕ: блок "Анализ и выводы" ---
        concl_card = ttk.Frame(root, padding=14, style="Card.TFrame")
        concl_card.pack(fill="x", pady=(12, 0))

        ttk.Label(concl_card, text="Анализ и выводы", style="H2.TLabel").pack(anchor="w", pady=(0, 8))

        self.txt_conclusions = tk.Text(
            concl_card,
            height=10,
            bg=UI["panel2"],
            fg=UI["text"],
            insertbackground=UI["text"],
            highlightthickness=1,
            highlightbackground=UI["border"],
            bd=0,
            wrap="word",
        )
        self.txt_conclusions.pack(fill="both", expand=True)
        self.txt_conclusions.insert("1.0", "Запустите анализ, чтобы сформировать выводы.")
        self.txt_conclusions.config(state="disabled")
        # --- /НОВОЕ ---

    def _set_status(self, text: str):
        self.lbl_an_status.config(text=text)
        self.update_idletasks()

    def _set_conclusions_text(self, md_text: str):
        """Показывает выводы в UI (простая очистка markdown)."""
        if not hasattr(self, "txt_conclusions"):
            return
        txt = md_text or ""
        # очень простая «очистка» markdown для Tk Text
        txt = txt.replace("**", "")
        txt = txt.replace("\r", "")
        self.txt_conclusions.configure(state="normal")
        self.txt_conclusions.delete("1.0", "end")
        self.txt_conclusions.insert("1.0", txt.strip())
        self.txt_conclusions.configure(state="disabled")

    def _busy(self, on: bool, status_text: str):
        self._analysis_busy = on
        if on:
            self.pb.start(10)
            self.btn_run.config(state="disabled")
            self.btn_report.config(state="disabled")
            self.ent_fs.config(state="disabled")
        else:
            self.pb.stop()
            self.btn_run.config(state="normal")
            self.btn_report.config(state="normal")
            self.ent_fs.config(state="normal")
        self._set_status(status_text)

    def run_lab5(self):
        if self._analysis_busy or self._pdf_busy:
            return
        if not self.loaded_files:
            messagebox.showwarning("Нет файлов", "Добавь CSV во вкладке «Файлы».")
            return

        try:
            fs_user = float(self.ent_fs.get().strip())
        except Exception:
            fs_user = FS_HZ_DEFAULT
        if fs_user <= 0:
            fs_user = FS_HZ_DEFAULT
        self._last_fs_user = fs_user

        self._busy(True, "Анализ: запуск…")
        self._analysis_thread = threading.Thread(target=self._run_lab5_worker, args=(fs_user,), daemon=True)
        self._analysis_thread.start()

    def _run_lab5_worker(self, fs_user: float):
        try:
            self._ui_queue.put(("status", "Чтение CSV…"))
            records = []
            for path in self.loaded_files:
                t, x, tcol, xcol = load_time_and_signal(path)
                fs_est = estimate_fs_from_time(t, fallback=fs_user)
                fs_hz = fs_user if fs_user > 0 else fs_est
                name = os.path.splitext(os.path.basename(path))[0]

                dur = float(t[-1] - t[0]) if len(t) > 1 else 0.0
                nan_ratio = float(np.mean(~np.isfinite(x))) if len(x) else 1.0

                records.append({
                    "name": name,
                    "path": path,
                    "t": t,
                    "x": x,
                    "fs": fs_hz,
                    "time_col": tcol,
                    "sig_col": xcol,
                    "duration_s": dur,
                    "nan_ratio": nan_ratio,
                    "montage": self.eeg_montage.get() if hasattr(self, "eeg_montage") else "",
                    "channel_hint": self.eeg_channel_hint.get() if hasattr(self, "eeg_channel_hint") else "",
                })

            self._ui_queue.put(("status", "PSD и мощности диапазонов…"))
            band_rows = []
            for r in records:
                freqs, psd = compute_psd(r["x"], fs_hz=r["fs"], nperseg=1024)
                p_total = integrate_band_power(freqs, psd, (0.5, 40.0))
                p_lambda = integrate_band_power(freqs, psd, LAMBDA_BAND_HZ)
                p_alpha = integrate_band_power(freqs, psd, ALPHA_BAND_HZ)

                band_rows.append({
                    "Файл": r["name"],
                    "Канал": r["sig_col"],
                    "FS (Гц)": r["fs"],
                    "Длительность (с)": r["duration_s"],
                    "Доля NaN": r["nan_ratio"],
                    "P_total": p_total,
                    "P_λ": p_lambda,
                    "P_α": p_alpha,
                    "P_λ / P_total": (p_lambda / p_total) if p_total > 0 else np.nan,
                    "P_α / P_total": (p_alpha / p_total) if p_total > 0 else np.nan,
                })

            band_power_df = pd.DataFrame(band_rows)

            self._ui_queue.put(("status", "λ(t) статистики…"))
            lambda_rows = []
            for r in records:
                lam = extract_lambda_signal(r["x"], fs_hz=r["fs"])
                t_win, p_win = sliding_window_power(lam, fs_hz=r["fs"], window_sec=2.0, overlap=0.5)

                lambda_rows.append({
                    "Файл": r["name"],
                    "Средняя мощность λ(t)": float(np.mean(p_win)) if len(p_win) else np.nan,
                    "Максимум λ(t)": float(np.max(p_win)) if len(p_win) else np.nan,
                    "Минимум λ(t)": float(np.min(p_win)) if len(p_win) else np.nan,
                })

            lambda_time_df = pd.DataFrame(lambda_rows)

            summary_df = (
                band_power_df
                .merge(lambda_time_df, on=["Файл"])
                .sort_values("P_λ / P_total", ascending=False)
                .reset_index(drop=True)
            )

            self._ui_queue.put(("done", {
                "records": records,
                "band_power_df": band_power_df,
                "lambda_time_df": lambda_time_df,
                "summary_df": summary_df,
            }))

        except Exception as e:
            self._ui_queue.put(("error", str(e)))

    def _poll_ui_queue(self):
        while True:
            try:
                kind, payload = self._ui_queue.get_nowait()
            except queue.Empty:
                break

            if kind == "status":
                self._set_status(str(payload))

            elif kind == "error":
                self._busy(False, "Ошибка")
                messagebox.showerror("Ошибка анализа", str(payload))

            elif kind == "done":
                self.band_power_df = payload["band_power_df"]
                self.lambda_time_df = payload["lambda_time_df"]
                self.summary_df = payload["summary_df"]
                self._last_records = payload["records"]

                self._render_current_table()
                self._render_plots()

                # --- НОВОЕ: сформировать выводы и показать ---
                self.conclusions_text = build_conclusions(
                    summary_df=self.summary_df,
                    records=self._last_records,
                    fs_user=self._last_fs_user,
                    montage=(self.eeg_montage.get() if hasattr(self, "eeg_montage") else ""),
                    channel_hint=(self.eeg_channel_hint.get() if hasattr(self, "eeg_channel_hint") else ""),
                )
                if hasattr(self, "txt_conclusions"):
                    self.txt_conclusions.config(state="normal")
                    self.txt_conclusions.delete("1.0", "end")
                    self.txt_conclusions.insert("1.0", self.conclusions_text)
                    self.txt_conclusions.config(state="disabled")
                # --- /НОВОЕ ---

                self._busy(False, "Готово ✅")

        self.after(60, self._poll_ui_queue)

    def _render_current_table(self):
        choice = self.cbo_table.get()
        if choice == "Мощности по диапазонам":
            df = self.band_power_df
        elif choice == "Статистики λ(t)":
            df = self.lambda_time_df
        elif choice == "Качество записи":
            df = self.quality_df
        else:
            df = self.summary_df

        if df is None or df.empty:
            self.tbl.delete(*self.tbl.get_children())
            self.tbl["columns"] = []
            return

        self.tbl.delete(*self.tbl.get_children())
        cols = list(df.columns)
        self.tbl["columns"] = cols

        for c in cols:
            self.tbl.heading(c, text=c)
            self.tbl.column(c, width=160, anchor="w")

        show_df = df.copy()
        for c in show_df.columns:
            if pd.api.types.is_numeric_dtype(show_df[c]):
                show_df[c] = show_df[c].round(6)

        for _, row in show_df.iterrows():
            self.tbl.insert("", "end", values=[row[c] for c in cols])

    def _render_plots(self):
        self.plot_area.clear()

        if self._last_records is None:
            ttk.Label(self.plot_area.inner, text="Запустите анализ, чтобы увидеть графики.", style="Muted.TLabel") \
                .pack(padx=12, pady=12, anchor="w")
            return

        mode = self.plot_mode.get()

        if mode == "BARS":
            for metric in ["Средняя мощность λ(t)", "Максимум λ(t)", "Минимум λ(t)"]:
                wrap = ttk.Frame(self.plot_area.inner, padding=14, style="Card.TFrame")
                wrap.pack(fill="x", expand=True, padx=12, pady=12)

                fig = make_bars_figure(self.summary_df, metric=metric)
                canv = FigureCanvasTkAgg(fig, master=wrap)
                canv.get_tk_widget().pack(fill="x", expand=True)
                canv.draw()
            return

        for r in self._last_records:
            wrap = ttk.Frame(self.plot_area.inner, padding=14, style="Card.TFrame")
            wrap.pack(fill="x", expand=True, padx=12, pady=12)

            ttk.Label(wrap, text=r["name"], style="H2.TLabel").pack(anchor="w", pady=(0, 10))

            if mode == "RAW":
                fig = make_raw_figure(r["t"], r["x"], r["fs"], r["name"])
            elif mode == "PSD":
                fig = make_psd_figure(r["x"], r["fs"], r["name"])
            else:
                fig = make_lambda_figure(r["t"], r["x"], r["fs"], r["name"])

            canv = FigureCanvasTkAgg(fig, master=wrap)
            canv.get_tk_widget().pack(fill="x", expand=True)
            canv.draw()

    # ---------------------------
    # Экспорт PDF (FIXED)
    # ---------------------------
    def export_report_pdf(self):
        if self._analysis_busy or self._pdf_busy:
            return

        if not REPORTLAB_OK:
            messagebox.showerror("PDF", "reportlab не установлен.\n\npip install reportlab")
            return
        if self.summary_df is None or self._last_records is None:
            messagebox.showwarning("PDF", "Сначала запустите анализ.")
            return

        out_path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF", "*.pdf")],
            initialfile=f"EEG_Lab5_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        )
        if not out_path:
            return

        self._pdf_busy = True
        self._busy(True, "Экспорт PDF…")

        self._pdf_thread = threading.Thread(target=self._export_pdf_worker, args=(out_path,), daemon=True)
        self._pdf_thread.start()

    def _on_electrodes_changed(self, electrodes):
        if not electrodes:
            self.cbo_channel["values"] = ["auto"]
            self.cbo_channel.set("auto")
            return

        vals = ["auto"] + electrodes
        self.cbo_channel["values"] = vals

        if self.cbo_channel.get() not in vals:
            self.cbo_channel.set(electrodes[0])
    def _export_pdf_worker(self, out_path: str):
        try:
            font_path = font_manager.findfont("DejaVu Sans")
            try:
                pdfmetrics.registerFont(TTFont("DejaVuSans", font_path))
                base_font = "DejaVuSans"
            except Exception:
                base_font = "Helvetica"

            styles = getSampleStyleSheet()
            styles["Normal"].fontName = base_font
            styles["Heading1"].fontName = base_font
            styles["Heading2"].fontName = base_font
            if "H3" not in styles:
                styles.add(ParagraphStyle(name="H3", fontName=base_font, fontSize=12, leading=14, spaceBefore=10, spaceAfter=6))

            doc = SimpleDocTemplate(
                out_path,
                pagesize=A4,
                leftMargin=1.5 * cm,
                rightMargin=1.5 * cm,
                topMargin=1.5 * cm,
                bottomMargin=1.5 * cm,
                title="EEG Lab5 Report",
            )

            story = []
            story.append(Paragraph("Отчёт: Лямбда-ритмы ЭЭГ при разных воздействиях", styles["Heading1"]))
            story.append(Spacer(1, 0.3 * cm))
            story.append(Paragraph(f"Расположение электродов: {self.eeg_montage.get()}", styles["Normal"]))
            story.append(Paragraph(f"Канал (подсказка): {self.eeg_channel_hint.get()}", styles["Normal"]))
            story.append(Paragraph(f"FS (Гц): {self._last_fs_user}", styles["Normal"]))
            story.append(Paragraph(f"λ-диапазон: {LAMBDA_BAND_HZ[0]}–{LAMBDA_BAND_HZ[1]} Гц", styles["Normal"]))
            story.append(Paragraph(f"α-диапазон: {ALPHA_BAND_HZ[0]}–{ALPHA_BAND_HZ[1]} Гц", styles["Normal"]))
            story.append(Spacer(1, 0.5 * cm))

            story.append(Paragraph("Таблица 1: Мощности по диапазонам (Welch PSD)", styles["Heading2"]))
            story.append(_df_to_rl_table(self.band_power_df, base_font))
            story.append(Spacer(1, 0.5 * cm))

            story.append(Paragraph("Таблица 2: Статистики мощности λ(t)", styles["Heading2"]))
            story.append(_df_to_rl_table(self.lambda_time_df, base_font))
            story.append(Spacer(1, 0.5 * cm))

            story.append(Paragraph("Таблица 3: Сводная", styles["Heading2"]))
            story.append(_df_to_rl_table(self.summary_df, base_font))
            story.append(PageBreak())
            # --- НОВОЕ: Анализ и выводы (в PDF) ---
            story.append(Paragraph("Анализ и выводы", styles["Heading2"]))
            text = getattr(self, "conclusions_text", "")
            if not text:
                text = "Выводы не сформированы. Запустите анализ перед экспортом отчёта."
            for line in text.split("\n"):
                line = line.strip()
                if not line:
                    story.append(Spacer(1, 0.15 * cm))
                else:
                    story.append(Paragraph(line, styles["Normal"]))
            story.append(PageBreak())
            # --- /НОВОЕ ---
            with tempfile.TemporaryDirectory() as tmpdir:
                self._ui_queue.put(("status", "PDF: графики…"))

                bars_path = os.path.join(tmpdir, "bars.png")
                fig_b = make_bars_figure(self.summary_df)
                _save_figure_png_threadsafe(fig_b, bars_path, dpi=160)
                try:
                    plt.close(fig_b)
                except Exception:
                    pass

                story.append(Paragraph("Сравнение условий: mean/max/min мощности λ(t)", styles["Heading2"]))
                story.append(_rl_image(bars_path, max_width_cm=17.5))
                story.append(PageBreak())

                for idx, r in enumerate(self._last_records, start=1):
                    self._ui_queue.put(("status", f"PDF: файл {idx}/{len(self._last_records)}…"))

                    name = r["name"]
                    t = r["t"]
                    x = r["x"]
                    fs_hz = r["fs"]

                    story.append(Paragraph(f"Файл: {name}", styles["Heading2"]))
                    story.append(Paragraph(f"Канал: {r['sig_col']} | FS: {fs_hz}", styles["Normal"]))
                    story.append(Spacer(1, 0.25 * cm))

                    raw_path = os.path.join(tmpdir, f"{name}_raw.png")
                    psd_path = os.path.join(tmpdir, f"{name}_psd.png")
                    lam_path = os.path.join(tmpdir, f"{name}_lambda.png")

                    fig1 = make_raw_figure(t, x, fs_hz, name, for_pdf=True)
                    _save_figure_png_threadsafe(fig1, raw_path, dpi=160)
                    try:
                        plt.close(fig1)
                    except Exception:
                        pass

                    fig2 = make_psd_figure(x, fs_hz, name, for_pdf=True)
                    _save_figure_png_threadsafe(fig2, psd_path, dpi=160)
                    try:
                        plt.close(fig2)
                    except Exception:
                        pass

                    fig3 = make_lambda_figure(t, x, fs_hz, name, for_pdf=True)
                    _save_figure_png_threadsafe(fig3, lam_path, dpi=160)
                    try:
                        plt.close(fig3)
                    except Exception:
                        pass

                    story.append(Paragraph("Сигнал (общий вид + зум)", styles["H3"]))
                    story.append(_rl_image(raw_path, max_width_cm=17.5))
                    story.append(Spacer(1, 0.4 * cm))

                    story.append(Paragraph("Спектр (Welch PSD) + выделение диапазонов", styles["H3"]))
                    story.append(_rl_image(psd_path, max_width_cm=17.5))
                    story.append(Spacer(1, 0.4 * cm))

                    story.append(Paragraph("λ-ритм (4–6 Гц) + мощность λ(t)", styles["H3"]))
                    story.append(_rl_image(lam_path, max_width_cm=17.5))
                    story.append(PageBreak())

            self._ui_queue.put(("status", "PDF: сборка…"))
            doc.build(story)

            self._ui_queue.put(("pdf_done", out_path))
        except Exception as e:
            self._ui_queue.put(("pdf_error", str(e)))

class TenTwentySelector(ttk.Frame):
    """
    Кликабельная схема 10–20.
    Выбор одного или нескольких электродов.
    """

    ELECTRODES = {
        "Fp1": (0.35, 0.18), "Fp2": (0.65, 0.18),
        "F7": (0.18, 0.30), "F3": (0.40, 0.30), "Fz": (0.50, 0.28),
        "F4": (0.60, 0.30), "F8": (0.82, 0.30),
        "T3": (0.14, 0.50), "C3": (0.38, 0.50), "Cz": (0.50, 0.50),
        "C4": (0.62, 0.50), "T4": (0.86, 0.50),
        "T5": (0.18, 0.70), "P3": (0.40, 0.70), "Pz": (0.50, 0.72),
        "P4": (0.60, 0.70), "T6": (0.82, 0.70),
        "O1": (0.40, 0.86), "Oz": (0.50, 0.88), "O2": (0.60, 0.86),
    }

    def __init__(self, parent, on_change=None, width=340, height=260):
        super().__init__(parent)
        self.on_change = on_change
        self.selected = set()

        self.canvas = tk.Canvas(
            self, width=width, height=height,
            bg=UI["panel2"], highlightthickness=1,
            highlightbackground=UI["border"]
        )
        self.canvas.pack(fill="both", expand=False)

        self._hit = {}
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<Configure>", lambda e: self._draw())

        self._draw()

    def get_selected(self):
        return sorted(self.selected)

    def clear(self):
        self.selected.clear()
        self._draw()
        if self.on_change:
            self.on_change(self.get_selected())

    def _on_click(self, event):
        item = self.canvas.find_closest(event.x, event.y)
        if not item:
            return
        item = item[0]
        if item not in self._hit:
            return

        name = self._hit[item]
        if name in self.selected:
            self.selected.remove(name)
        else:
            self.selected.add(name)

        self._draw()
        if self.on_change:
            self.on_change(self.get_selected())

    def _draw(self):
        self.canvas.delete("all")
        self._hit.clear()

        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        pad = 12

        # голова
        self.canvas.create_oval(
            pad, pad, w - pad, h - pad,
            outline=UI["border"], width=2, fill=UI["panel"]
        )

        r = 11
        for name, (nx, ny) in self.ELECTRODES.items():
            x = int(nx * (w - 2 * pad) + pad)
            y = int(ny * (h - 2 * pad) + pad)

            sel = name in self.selected
            fill = UI["accent"] if sel else UI["panel2"]
            outline = UI["accent"] if sel else UI["border"]
            text_col = "white" if sel else UI["text"]

            cid = self.canvas.create_oval(
                x - r, y - r, x + r, y + r,
                fill=fill, outline=outline, width=2
            )
            self._hit[cid] = name

            self.canvas.create_text(
                x, y, text=name,
                fill=text_col, font=("SF Pro Text", 10, "bold")
            )
# ---------------------------
# Построение графиков
# ---------------------------
def make_raw_figure(t: np.ndarray, x: np.ndarray, fs_hz: float, title: str, for_pdf: bool = False) -> Figure:
    fig = Figure(figsize=(10, 4), dpi=110 if not for_pdf else 120)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    style_axes(ax1)
    style_axes(ax2)

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
def load_time_and_channel(csv_path: str, channel: str):
    df = _try_read_csv(csv_path)

    def to_num(s):
        if s.dtype == object:
            s = s.astype(str).str.replace(",", ".", regex=False)
        return pd.to_numeric(s, errors="coerce")

    time_col = None
    for c in df.columns:
        if "time" in str(c).lower() or "время" in str(c).lower():
            time_col = c
            break

    if channel not in df.columns:
        raise ValueError(f"Нет канала {channel}")

    x = to_num(df[channel]).to_numpy()

    if time_col:
        t = to_num(df[time_col]).to_numpy()
        mask = np.isfinite(t) & np.isfinite(x)
        t, x = t[mask], x[mask]
        if len(t) > 2 and not np.all(np.diff(t) > 0):
            t = np.arange(len(x)) / FS_HZ_DEFAULT
            time_col = "synthetic_time"
    else:
        t = np.arange(len(x)) / FS_HZ_DEFAULT
        time_col = "synthetic_time"

    n = min(len(t), len(x))
    return t[:n], x[:n], time_col, channel

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
    style_axes(ax1)
    style_axes(ax2)

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
    df = summary_df.copy()
    df = df.sort_values(metric, ascending=True)

    labels = df["Файл"].astype(str).values
    vals = df[metric].values

    # Автовысота
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

    # “мягкая” сетка по x
    ax.grid(True, axis="x", alpha=0.55)
    ax.grid(False, axis="y")

    fig.tight_layout()
    return fig

def build_conclusions(
    summary_df: pd.DataFrame,
    records: list,
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

    # Качество данных
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

    # Ранжирование по доле лямбда-энергии
    lines.append("3. Сравнение условий по выраженности λ-ритма")
    df = summary_df.copy()

    # защита от отсутствующих колонок
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

    # α сравнение
    lines.append("")
    lines.append("5. Сопоставление λ и α диапазонов")
    df_a = df.sort_values("P_α / P_total", ascending=False).reset_index(drop=True)
    lines.append(f"• Наибольшая относительная мощность α: {df_a.iloc[0]['Файл']} (P_α/P_total = {float(df_a.iloc[0]['P_α / P_total']):.6f})")
    lines.append("• Если α-доля растёт, это может отражать более выраженную альфа-активность по сравнению с λ при данном условии.")

    lines.append("")
    lines.append("6. Итоговый вывод")
    lines.append("• В рамках выбранных условий наблюдаются различия в относительной мощности λ-диапазона (4–6 Гц).")
    lines.append("• Полученные различия можно использовать для сравнения режимов (покой/визуальная фиксация/поиск/когнитивная нагрузка),")
    lines.append("  при этом корректность интерпретации зависит от качества записи и согласованности FS/канала.")
    lines.append("• Рекомендуется сохранять одинаковые параметры фильтрации и длину записи для корректного сравнения между условиями.")

    return "\n".join(lines)
# ---------------------------
# ReportLab helpers
# ---------------------------
def _df_to_rl_table(df: Optional[pd.DataFrame], font_name: str):
    styles = getSampleStyleSheet()
    if df is None or df.empty:
        return Paragraph("Нет данных.", styles["Normal"])

    show_df = df.copy()
    for c in show_df.columns:
        if pd.api.types.is_numeric_dtype(show_df[c]):
            show_df[c] = show_df[c].round(6)

    data = [list(show_df.columns)] + show_df.astype(str).values.tolist()

    tbl = Table(data, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), font_name),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#FFEDD5")),  # шапка
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#FED7AA")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#FFF7ED")]),
    ]))
    return tbl


def _rl_image(path: str, max_width_cm: float = 17.0):
    img = RLImage(path)
    max_w = max_width_cm * cm
    if img.drawWidth > max_w:
        scale = max_w / img.drawWidth
        img.drawWidth *= scale
        img.drawHeight *= scale
    return img


def main():
    app = EEGApp()
    app.mainloop()


if __name__ == "__main__":
    main()