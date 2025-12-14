# eeg_lab5_app.py
# GUI app: Live Arduino -> CSV, and Lab5 analysis (lambda rhythm) like in ipynb
# Version: scrollable plots (one plot block per file, stacked vertically) + PDF report export

import os
import time
import csv
import threading
import queue
import tempfile
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Tuple
import matplotlib.pyplot as plt


import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import font_manager

from scipy.signal import welch, butter, filtfilt

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Optional serial
try:
    import serial
    SERIAL_OK = True
except Exception:
    SERIAL_OK = False

# Report (PDF)
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


# ---------------------------
# Lab 5 parameters (as ipynb)
# ---------------------------
FS_HZ_DEFAULT = 250.0
LAMBDA_BAND_HZ = (4.0, 6.0)
ALPHA_BAND_HZ = (7.0, 13.0)


# ---------------------------
# Utilities: robust CSV read
# ---------------------------
def _try_read_csv(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, sep=None, engine="python")
        return df
    except Exception:
        pass

    for sep in [",", ";", "\t"]:
        for dec in [".", ","]:
            try:
                df = pd.read_csv(path, sep=sep, decimal=dec, engine="python")
                return df
            except Exception:
                continue

    return pd.read_csv(path, engine="python")


def load_time_and_signal(path: str) -> Tuple[np.ndarray, np.ndarray, str, str]:
    """
    Loads CSV with two columns (time, A0) OR tries to infer from numeric columns.
    Returns: t_seconds, x_signal, time_col_name, sig_col_name
    """
    df = _try_read_csv(path)
    cols = list(df.columns)

    time_candidates = [c for c in cols if "время" in str(c).lower() or "time" in str(c).lower()]
    sig_candidates = [c for c in cols if "a0" in str(c).lower() or "eeg" in str(c).lower() or "amp" in str(c).lower()]

    def to_numeric_series(s: pd.Series) -> pd.Series:
        if s.dtype == object:
            s = s.astype(str).str.replace(",", ".", regex=False)
        return pd.to_numeric(s, errors="coerce")

    numeric_cols = []
    for c in cols:
        sn = to_numeric_series(df[c])
        if sn.notna().sum() >= max(5, int(0.05 * len(df))):
            numeric_cols.append(c)

    if not numeric_cols:
        raise ValueError(f"{os.path.basename(path)}: нет числовых столбцов")

    time_col = None
    sig_col = None

    for c in time_candidates:
        if c in numeric_cols:
            time_col = c
            break
    for c in sig_candidates:
        if c in numeric_cols:
            sig_col = c
            break

    if time_col is None or sig_col is None:
        time_like = []
        signal_like = []
        for c in numeric_cols:
            s = to_numeric_series(df[c]).dropna()
            if len(s) < 10:
                continue
            is_monotonic = s.is_monotonic_increasing
            unique_ratio = s.nunique() / max(1, len(s))
            if is_monotonic and unique_ratio > 0.9:
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
            x_tmp = to_numeric_series(df[sig_col]).dropna().to_numpy(dtype=float)
            t_tmp = np.arange(len(x_tmp)) / FS_HZ_DEFAULT
            return t_tmp, x_tmp, "synthetic_time", sig_col

    t = to_numeric_series(df[time_col]).to_numpy(dtype=float)
    x = to_numeric_series(df[sig_col]).to_numpy(dtype=float)

    mask = np.isfinite(t) & np.isfinite(x)
    t = t[mask]
    x = x[mask]

    if len(t) >= 3 and not np.all(np.diff(t) > 0):
        x = x.astype(float)
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
# Lab 5 computations
# ---------------------------
def compute_psd(x: np.ndarray, fs_hz: float, nperseg: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    freqs_hz, psd = welch(x, fs=fs_hz, nperseg=min(nperseg, max(64, len(x))))
    return freqs_hz, psd


def integrate_band_power(freqs_hz: np.ndarray, psd: np.ndarray, band_hz: Tuple[float, float]) -> float:
    low, high = band_hz
    mask = (freqs_hz >= low) & (freqs_hz <= high)
    if not np.any(mask):
        return 0.0
    return float(np.trapezoid(psd[mask], freqs_hz[mask]))


def butter_bandpass(data: np.ndarray, low_hz: float, high_hz: float, fs_hz: float, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs_hz
    low = low_hz / nyq
    high = high_hz / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, data)


def extract_lambda_signal(x: np.ndarray, fs_hz: float) -> np.ndarray:
    return butter_bandpass(np.asarray(x, dtype=float), LAMBDA_BAND_HZ[0], LAMBDA_BAND_HZ[1], fs_hz=fs_hz, order=4)


def sliding_window_power(x: np.ndarray, fs_hz: float, window_sec: float = 2.0, overlap: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    win = int(window_sec * fs_hz)
    if win < 1:
        raise ValueError("Слишком маленькое окно")
    step = max(1, int(win * (1.0 - overlap)))

    t_vals, p_vals = [], []
    for start in range(0, len(x) - win, step):
        seg = x[start:start + win]
        p_vals.append(float(np.mean(seg ** 2)))
        t_vals.append(start / fs_hz)
    return np.asarray(t_vals), np.asarray(p_vals)


# ---------------------------
# Plot styling
# ---------------------------
def style_axes(ax):
    ax.set_facecolor("#f7f7f7")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.25, linewidth=0.6)


# ---------------------------
# Scrollable frame for plots
# ---------------------------
class ScrollablePlotArea(ttk.Frame):
    """
    A scrollable container: Canvas + scrollbar + inner frame.
    We place multiple matplotlib canvases inside the inner frame (stacked vertically).
    """
    def __init__(self, parent):
        super().__init__(parent)

        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.vbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vbar.set)

        self.vbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.inner = ttk.Frame(self.canvas)
        self.inner_id = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")

        self.inner.bind("<Configure>", self._on_frame_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        # Mouse wheel support (Win/Mac/Linux)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)       # Windows / Mac
        self.canvas.bind_all("<Button-4>", self._on_mousewheel_linux)   # Linux up
        self.canvas.bind_all("<Button-5>", self._on_mousewheel_linux)   # Linux down

    def _on_frame_configure(self, _event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self.canvas.itemconfigure(self.inner_id, width=event.width)

    def _on_mousewheel(self, event):
        if self.winfo_containing(event.x_root, event.y_root) in (self.canvas, self.inner) or self._is_child_of_inner(event):
            delta = int(-1 * (event.delta / 120))
            self.canvas.yview_scroll(delta, "units")

    def _on_mousewheel_linux(self, event):
        if self.winfo_containing(event.x_root, event.y_root) in (self.canvas, self.inner) or self._is_child_of_inner(event):
            if event.num == 4:
                self.canvas.yview_scroll(-2, "units")
            elif event.num == 5:
                self.canvas.yview_scroll(2, "units")

    def _is_child_of_inner(self, event) -> bool:
        w = self.winfo_containing(event.x_root, event.y_root)
        while w is not None:
            if w == self.inner:
                return True
            w = w.master
        return False

    def clear(self):
        for child in list(self.inner.winfo_children()):
            child.destroy()


# ---------------------------
# Arduino (pySerial) streamer
# ---------------------------
@dataclass
class SerialConfig:
    port: str = "COM3"
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
        self.ser = serial.Serial(self.cfg.port, self.cfg.baudrate, timeout=1)
        time.sleep(2)
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
                if self.cfg.delimiter in line:
                    parts = line.split(self.cfg.delimiter)
                else:
                    parts = [line]
                raw = parts[0].strip().replace(",", ".")
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


# ---------------------------
# GUI App
# ---------------------------
class EEGApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("EEG Lab5 — λ-ритм (4–6 Гц) — GUI")
        self.geometry("1200x800")

        self.serial_queue: "queue.Queue[Tuple[float, float]]" = queue.Queue()
        self.streamer: Optional[ArduinoSerialStreamer] = None

        self.live_t: List[float] = []
        self.live_x: List[float] = []
        self.live_max_sec = 10.0

        self.loaded_files: List[str] = []

        self.band_power_df = None
        self.lambda_time_df = None
        self.summary_df = None

        # keep last analysis records for PDF
        self._last_records = None
        self._last_fs_user = FS_HZ_DEFAULT

        self._build_ui()
        self.after(50, self._poll_serial_queue)

    def _build_ui(self):
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True)

        self.tab_live = ttk.Frame(nb)
        self.tab_files = ttk.Frame(nb)
        self.tab_analysis = ttk.Frame(nb)

        nb.add(self.tab_live, text="Live (Arduino)")
        nb.add(self.tab_files, text="Файлы CSV")
        nb.add(self.tab_analysis, text="Анализ ЛР5")

        self._build_live_tab()
        self._build_files_tab()
        self._build_analysis_tab()

    # -------- LIVE TAB --------
    def _build_live_tab(self):
        frm = ttk.Frame(self.tab_live, padding=10)
        frm.pack(fill="both", expand=True)

        top = ttk.Frame(frm)
        top.pack(fill="x")

        ttk.Label(top, text="COM порт:").pack(side="left")
        self.ent_port = ttk.Entry(top, width=14)
        self.ent_port.insert(0, "COM3" if os.name == "nt" else "/dev/tty.usbmodem*")
        self.ent_port.pack(side="left", padx=6)

        ttk.Label(top, text="Baud:").pack(side="left")
        self.ent_baud = ttk.Entry(top, width=10)
        self.ent_baud.insert(0, "115200")
        self.ent_baud.pack(side="left", padx=6)

        ttk.Label(top, text="Окно (с):").pack(side="left")
        self.ent_win = ttk.Entry(top, width=8)
        self.ent_win.insert(0, "10")
        self.ent_win.pack(side="left", padx=6)

        self.btn_start = ttk.Button(top, text="▶ Старт", command=self.start_stream)
        self.btn_start.pack(side="left", padx=6)

        self.btn_stop = ttk.Button(top, text="■ Стоп", command=self.stop_stream, state="disabled")
        self.btn_stop.pack(side="left", padx=6)

        self.btn_save = ttk.Button(top, text="💾 Сохранить CSV", command=self.save_live_csv, state="disabled")
        self.btn_save.pack(side="left", padx=6)

        ttk.Separator(frm, orient="horizontal").pack(fill="x", pady=10)

        self.fig_live = Figure(figsize=(10, 4), dpi=110)
        self.ax_live = self.fig_live.add_subplot(111)
        style_axes(self.ax_live)
        self.ax_live.set_title("Live сигнал (последние N секунд)")
        self.ax_live.set_xlabel("Время, с")
        self.ax_live.set_ylabel("A0 (В) / у.е.")
        self.line_live, = self.ax_live.plot([], [], linewidth=1.2)

        self.canvas_live = FigureCanvasTkAgg(self.fig_live, master=frm)
        self.canvas_live.get_tk_widget().pack(fill="both", expand=True)

        self.lbl_live_status = ttk.Label(frm, text="Статус: не запущено")
        self.lbl_live_status.pack(anchor="w", pady=6)

    def start_stream(self):
        if not SERIAL_OK:
            messagebox.showerror("Ошибка", "pyserial не установлен. Установи: pip install pyserial")
            return

        port = self.ent_port.get().strip()
        try:
            baud = int(self.ent_baud.get().strip())
        except Exception:
            messagebox.showerror("Ошибка", "Неверный baudrate")
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
        self.lbl_live_status.config(text=f"Статус: подключение/чтение с {port} @ {baud}")

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
                    messagebox.showerror("Ошибка Serial", "Не удалось подключиться к порту")
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

        self.after(50, self._poll_serial_queue)

    def _update_live_plot(self):
        if not self.live_t:
            return
        t = np.asarray(self.live_t)
        x = np.asarray(self.live_x)

        tmax = t[-1]
        mask = t >= max(0.0, tmax - self.live_max_sec)
        t2, x2 = t[mask], x[mask]

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
            w.writerow(["Время (с)", "A0 (В)"])
            for t, x in zip(self.live_t, self.live_x):
                w.writerow([f"{t:.6f}", f"{x:.6f}"])

        messagebox.showinfo("Сохранено", f"CSV сохранён:\n{path}")

    # -------- FILES TAB --------
    def _build_files_tab(self):
        frm = ttk.Frame(self.tab_files, padding=10)
        frm.pack(fill="both", expand=True)

        top = ttk.Frame(frm)
        top.pack(fill="x")

        ttk.Button(top, text="➕ Добавить CSV", command=self.add_csv_files).pack(side="left")
        ttk.Button(top, text="🗑 Удалить выбранный", command=self.remove_selected_file).pack(side="left", padx=6)
        ttk.Button(top, text="Очистить список", command=self.clear_file_list).pack(side="left", padx=6)

        ttk.Separator(frm, orient="horizontal").pack(fill="x", pady=10)

        mid = ttk.Frame(frm)
        mid.pack(fill="both", expand=True)

        self.lst_files = tk.Listbox(mid, height=12)
        self.lst_files.pack(side="left", fill="both", expand=True)

        scr = ttk.Scrollbar(mid, orient="vertical", command=self.lst_files.yview)
        scr.pack(side="left", fill="y")
        self.lst_files.configure(yscrollcommand=scr.set)

        hint = ttk.Label(frm, text="Добавь 1–5 CSV (каждый — отдельное условие). Потом перейди во вкладку “Анализ ЛР5”.")
        hint.pack(anchor="w", pady=8)

    def add_csv_files(self):
        paths = filedialog.askopenfilenames(filetypes=[("CSV", "*.csv"), ("All", "*.*")])
        if not paths:
            return
        for p in paths:
            if p not in self.loaded_files:
                self.loaded_files.append(p)
                self.lst_files.insert("end", p)

    def remove_selected_file(self):
        sel = self.lst_files.curselection()
        if not sel:
            return
        idx = sel[0]
        path = self.lst_files.get(idx)
        self.lst_files.delete(idx)
        self.loaded_files = [p for p in self.loaded_files if p != path]

    def clear_file_list(self):
        self.lst_files.delete(0, "end")
        self.loaded_files.clear()

    # -------- ANALYSIS TAB --------
    def _build_analysis_tab(self):
        frm = ttk.Frame(self.tab_analysis, padding=10)
        frm.pack(fill="both", expand=True)

        controls = ttk.Frame(frm)
        controls.pack(fill="x")

        ttk.Label(controls, text="FS (Гц):").pack(side="left")
        self.ent_fs = ttk.Entry(controls, width=10)
        self.ent_fs.insert(0, "250")
        self.ent_fs.pack(side="left", padx=6)

        ttk.Button(controls, text="▶ Запустить анализ ЛР5", command=self.run_lab5).pack(side="left", padx=6)

        # NEW: export report button
        self.btn_report = ttk.Button(controls, text="📄 Экспорт отчёта PDF", command=self.export_report_pdf)
        self.btn_report.pack(side="left", padx=6)

        self.lbl_an_status = ttk.Label(controls, text="Статус: готово")
        self.lbl_an_status.pack(side="left", padx=12)

        ttk.Separator(frm, orient="horizontal").pack(fill="x", pady=10)

        pan = ttk.PanedWindow(frm, orient="horizontal")
        pan.pack(fill="both", expand=True)

        left = ttk.Frame(pan)
        right = ttk.Frame(pan)
        pan.add(left, weight=1)
        pan.add(right, weight=3)

        ttk.Label(left, text="Таблицы (как в ipynb)").pack(anchor="w")

        self.tbl = ttk.Treeview(left, show="headings")
        self.tbl.pack(fill="both", expand=True, pady=6)

        self.tbl_scr = ttk.Scrollbar(left, orient="vertical", command=self.tbl.yview)
        self.tbl_scr.pack(side="right", fill="y")
        self.tbl.configure(yscrollcommand=self.tbl_scr.set)

        self.cbo_table = ttk.Combobox(left, state="readonly", values=["band_power_df", "lambda_time_df", "summary_df"])
        self.cbo_table.current(2)
        self.cbo_table.pack(fill="x", pady=6)
        self.cbo_table.bind("<<ComboboxSelected>>", lambda e: self._render_current_table())

        self.fig_nb = ttk.Notebook(right)
        self.fig_nb.pack(fill="both", expand=True)

        self.plot_areas = {}

        for name in ["Raw (overview+zoom)", "PSD + bands", "λ filter + λ(t)"]:
            tab = ttk.Frame(self.fig_nb)
            self.fig_nb.add(tab, text=name)
            area = ScrollablePlotArea(tab)
            area.pack(fill="both", expand=True)
            self.plot_areas[name] = area

        tab_bars = ttk.Frame(self.fig_nb)
        self.fig_nb.add(tab_bars, text="Bars (mean/max/min)")
        self.fig_bars = Figure(figsize=(10, 7), dpi=110)
        self.canvas_bars = FigureCanvasTkAgg(self.fig_bars, master=tab_bars)
        self.canvas_bars.get_tk_widget().pack(fill="both", expand=True)

    def _set_status(self, text: str):
        self.lbl_an_status.config(text=f"Статус: {text}")
        self.update_idletasks()

    def run_lab5(self):
        if not self.loaded_files:
            messagebox.showwarning("Нет файлов", "Добавь CSV во вкладке “Файлы CSV”.")
            return

        try:
            fs_user = float(self.ent_fs.get().strip())
        except Exception:
            fs_user = FS_HZ_DEFAULT
        if fs_user <= 0:
            fs_user = FS_HZ_DEFAULT
        self._last_fs_user = fs_user

        for area in self.plot_areas.values():
            area.clear()

        self._set_status("чтение CSV…")

        records = []
        for path in self.loaded_files:
            try:
                t, x, tcol, xcol = load_time_and_signal(path)
                fs_est = estimate_fs_from_time(t, fallback=fs_user)
                fs_hz = fs_user if fs_user > 0 else fs_est
                name = os.path.splitext(os.path.basename(path))[0]
                records.append({
                    "name": name,
                    "path": path,
                    "t": t,
                    "x": x,
                    "fs": fs_hz,
                    "time_col": tcol,
                    "sig_col": xcol,
                })
            except Exception as e:
                messagebox.showerror("Ошибка чтения", f"{os.path.basename(path)}:\n{e}")
                return

        self._last_records = records

        self._set_status("PSD + band powers…")
        band_rows = []
        for r in records:
            freqs, psd = compute_psd(r["x"], fs_hz=r["fs"], nperseg=1024)
            p_total = integrate_band_power(freqs, psd, (0.5, 40.0))
            p_lambda = integrate_band_power(freqs, psd, LAMBDA_BAND_HZ)
            p_alpha = integrate_band_power(freqs, psd, ALPHA_BAND_HZ)
            band_rows.append({
                "Файл": r["name"],
                "Канал": r["sig_col"],
                "FS_HZ": r["fs"],
                "P_total": p_total,
                "P_lambda": p_lambda,
                "P_alpha": p_alpha,
                "P_lambda / P_total": (p_lambda / p_total) if p_total > 0 else np.nan,
                "P_alpha / P_total": (p_alpha / p_total) if p_total > 0 else np.nan,
            })
        self.band_power_df = pd.DataFrame(band_rows)

        self._set_status("λ(t) stats…")
        lambda_rows = []
        for r in records:
            lam = extract_lambda_signal(r["x"], fs_hz=r["fs"])
            t_win, p_win = sliding_window_power(lam, fs_hz=r["fs"], window_sec=2.0, overlap=0.5)
            lambda_rows.append({
                "Файл": r["name"],
                "Средняя мощность λ(t)": float(np.mean(p_win)) if len(p_win) else np.nan,
                "Макс мощность λ(t)": float(np.max(p_win)) if len(p_win) else np.nan,
                "Мин мощность λ(t)": float(np.min(p_win)) if len(p_win) else np.nan,
            })
        self.lambda_time_df = pd.DataFrame(lambda_rows)

        self.summary_df = (
            self.band_power_df
            .merge(self.lambda_time_df, on=["Файл"])
            .sort_values("P_lambda / P_total", ascending=False)
            .reset_index(drop=True)
        )

        self._set_status("рисую Raw…")
        for r in records:
            self._add_raw_block(r)

        self._set_status("рисую PSD…")
        for r in records:
            self._add_psd_block(r)

        self._set_status("рисую λ + λ(t)…")
        for r in records:
            self._add_lambda_block(r)

        self._set_status("рисую bar charts…")
        self._plot_bars(self.summary_df)

        self._render_current_table()
        self._set_status("готово ✅")

    # ---------------------------
    # Plot blocks (GUI)
    # ---------------------------
    def _add_raw_block(self, r):
        area = self.plot_areas["Raw (overview+zoom)"]

        title = r["name"]
        t = r["t"]
        x = r["x"]
        fs_hz = r["fs"]

        wrap = ttk.Frame(area.inner, padding=(6, 8))
        wrap.pack(fill="x", expand=True)

        lab = ttk.Label(wrap, text=f"{title}", font=("Arial", 12, "bold"))
        lab.pack(anchor="w", pady=(0, 6))

        fig = make_raw_figure(t, x, fs_hz, title)
        canv = FigureCanvasTkAgg(fig, master=wrap)
        canv.get_tk_widget().pack(fill="x", expand=True)
        canv.draw()

        ttk.Separator(area.inner, orient="horizontal").pack(fill="x", padx=10, pady=6)

    def _add_psd_block(self, r):
        area = self.plot_areas["PSD + bands"]

        title = r["name"]
        x = r["x"]
        fs_hz = r["fs"]

        wrap = ttk.Frame(area.inner, padding=(6, 8))
        wrap.pack(fill="x", expand=True)

        lab = ttk.Label(wrap, text=f"{title}", font=("Arial", 12, "bold"))
        lab.pack(anchor="w", pady=(0, 6))

        fig = make_psd_figure(x, fs_hz, title)
        canv = FigureCanvasTkAgg(fig, master=wrap)
        canv.get_tk_widget().pack(fill="x", expand=True)
        canv.draw()

        ttk.Separator(area.inner, orient="horizontal").pack(fill="x", padx=10, pady=6)

    def _add_lambda_block(self, r):
        area = self.plot_areas["λ filter + λ(t)"]

        title = r["name"]
        t = r["t"]
        x = r["x"]
        fs_hz = r["fs"]

        wrap = ttk.Frame(area.inner, padding=(6, 8))
        wrap.pack(fill="x", expand=True)

        lab = ttk.Label(wrap, text=f"{title}", font=("Arial", 12, "bold"))
        lab.pack(anchor="w", pady=(0, 6))

        fig = make_lambda_figure(t, x, fs_hz, title)
        canv = FigureCanvasTkAgg(fig, master=wrap)
        canv.get_tk_widget().pack(fill="x", expand=True)
        canv.draw()

        ttk.Separator(area.inner, orient="horizontal").pack(fill="x", padx=10, pady=6)

    def _plot_bars(self, summary_df: pd.DataFrame):
        self.fig_bars.clear()

        ax1 = self.fig_bars.add_subplot(3, 1, 1)
        ax2 = self.fig_bars.add_subplot(3, 1, 2)
        ax3 = self.fig_bars.add_subplot(3, 1, 3)
        for ax in (ax1, ax2, ax3):
            style_axes(ax)

        df = summary_df.sort_values("Средняя мощность λ(t)", ascending=False)
        labels = df["Файл"].astype(str).values
        x = np.arange(len(labels))

        ax1.bar(x, df["Средняя мощность λ(t)"].values)
        ax1.set_title("Средняя мощность λ(t)")
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)

        ax2.bar(x, df["Макс мощность λ(t)"].values)
        ax2.set_title("Макс мощность λ(t)")
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)

        ax3.bar(x, df["Мин мощность λ(t)"].values)
        ax3.set_title("Мин мощность λ(t)")
        ax3.set_xticks(x)
        ax3.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)

        self.fig_bars.tight_layout()
        self.canvas_bars.draw_idle()

    # ---------- Table render ----------
    def _render_current_table(self):
        name = self.cbo_table.get().strip()
        if name == "band_power_df":
            df = self.band_power_df
        elif name == "lambda_time_df":
            df = self.lambda_time_df
        else:
            df = self.summary_df

        if df is None or df.empty:
            return

        self.tbl.delete(*self.tbl.get_children())

        cols = list(df.columns)
        self.tbl["columns"] = cols
        for c in cols:
            self.tbl.heading(c, text=c)
            self.tbl.column(c, width=150, anchor="w")

        show_df = df.copy()
        for c in show_df.columns:
            if pd.api.types.is_numeric_dtype(show_df[c]):
                show_df[c] = show_df[c].round(6)

        for _, row in show_df.iterrows():
            self.tbl.insert("", "end", values=[row[c] for c in cols])

    # ---------------------------
    # PDF report export
    # ---------------------------
    def export_report_pdf(self):
        if not REPORTLAB_OK:
            messagebox.showerror("Нет reportlab", "Установи reportlab:\n\npip install reportlab")
            return

        if self.summary_df is None or self._last_records is None:
            messagebox.showwarning("Нет анализа", "Сначала нажми “Запустить анализ ЛР5”, потом экспортируй отчёт.")
            return

        out_path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF", "*.pdf")],
            initialfile=f"EEG_Lab5_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        )
        if not out_path:
            return

        try:
            self._set_status("экспорт PDF…")

            # Register Cyrillic-capable font (DejaVu Sans from matplotlib)
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
            styles.add(ParagraphStyle(name="H3", fontName=base_font, fontSize=12, leading=14, spaceBefore=10, spaceAfter=6, bold=True))

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

            story.append(Paragraph("Отчёт: ЛР5 — Лямбда-ритм ЭЭГ (4–6 Гц)", styles["Heading1"]))
            story.append(Paragraph(f"Дата: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}", styles["Normal"]))
            story.append(Spacer(1, 0.3 * cm))
            story.append(Paragraph(f"FS (Гц): {self._last_fs_user}", styles["Normal"]))
            story.append(Paragraph(f"λ-диапазон: {LAMBDA_BAND_HZ[0]}–{LAMBDA_BAND_HZ[1]} Гц", styles["Normal"]))
            story.append(Paragraph(f"α-диапазон: {ALPHA_BAND_HZ[0]}–{ALPHA_BAND_HZ[1]} Гц", styles["Normal"]))
            story.append(Spacer(1, 0.5 * cm))

            # Tables
            story.append(Paragraph("Таблица 1: Мощности в диапазонах (Welch PSD)", styles["Heading2"]))
            story.append(_df_to_rl_table(self.band_power_df, base_font))
            story.append(Spacer(1, 0.5 * cm))

            story.append(Paragraph("Таблица 2: Статистики мощности λ(t)", styles["Heading2"]))
            story.append(_df_to_rl_table(self.lambda_time_df, base_font))
            story.append(Spacer(1, 0.5 * cm))

            story.append(Paragraph("Таблица 3: Итоговая сводная", styles["Heading2"]))
            story.append(_df_to_rl_table(self.summary_df, base_font))
            story.append(PageBreak())

            # Figures: per file
            with tempfile.TemporaryDirectory() as tmpdir:
                # Bars
                bars_path = os.path.join(tmpdir, "bars.png")
                fig_b = make_bars_figure(self.summary_df)
                fig_b.savefig(bars_path, dpi=160, bbox_inches="tight")
                plt.close(fig_b)


                story.append(Paragraph("Итоговые графики (mean/max/min мощности λ(t))", styles["Heading2"]))
                story.append(_rl_image(bars_path, max_width_cm=17.5))
                story.append(PageBreak())

                for r in self._last_records:
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
                    fig1.savefig(raw_path, dpi=160, bbox_inches="tight")
                    plt.close(fig1)

                    fig2 = make_psd_figure(x, fs_hz, name, for_pdf=True)
                    fig2.savefig(psd_path, dpi=160, bbox_inches="tight")
                    plt.close(fig2)

                    fig3 = make_lambda_figure(t, x, fs_hz, name, for_pdf=True)
                    fig3.savefig(lam_path, dpi=160, bbox_inches="tight")
                    plt.close(fig3)

                    story.append(Paragraph("Raw (общий вид + зум)", styles["H3"]))
                    story.append(_rl_image(raw_path, max_width_cm=17.5))
                    story.append(Spacer(1, 0.4 * cm))

                    story.append(Paragraph("PSD (Welch) + выделение диапазонов", styles["H3"]))
                    story.append(_rl_image(psd_path, max_width_cm=17.5))
                    story.append(Spacer(1, 0.4 * cm))

                    story.append(Paragraph("λ-фильтр (4–6 Гц) + мощность λ(t)", styles["H3"]))
                    story.append(_rl_image(lam_path, max_width_cm=17.5))
                    story.append(PageBreak())

            doc.build(story)

            self._set_status("готово ✅")
            messagebox.showinfo("Готово", f"PDF отчёт сохранён:\n{out_path}")

        except Exception as e:
            self._set_status("ошибка")
            messagebox.showerror("Ошибка экспорта PDF", str(e))


# ---------------------------
# Figures builders (used both in GUI and PDF)
# ---------------------------
def make_raw_figure(t: np.ndarray, x: np.ndarray, fs_hz: float, title: str, for_pdf: bool = False) -> Figure:
    fig = Figure(figsize=(10, 4), dpi=110 if not for_pdf else 120)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    style_axes(ax1)
    style_axes(ax2)

    max_over = min(len(x), int(10 * fs_hz))
    max_zoom = min(len(x), int(4 * fs_hz))

    ax1.plot(t[:max_over], x[:max_over], linewidth=1.2)
    ax1.axhline(0, linestyle="--", linewidth=0.8, alpha=0.7)
    ax1.set_title(f"{title}\nПервые {max_over/fs_hz:.1f} с", fontsize=10)
    ax1.set_xlabel("Время, с")
    ax1.set_ylabel("Амплитуда")

    ax2.plot(t[:max_zoom], x[:max_zoom], linewidth=1.3)
    ax2.axhline(0, linestyle="--", linewidth=0.8, alpha=0.7)
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
    ax.semilogy(freqs, psd, linewidth=1.2, label="PSD ЭЭГ")
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

    ax1.plot(t[:max_samp], x[:max_samp], linewidth=1.1, label="ЭЭГ")
    ax1.plot(t[:max_samp], lam[:max_samp], linewidth=1.3, alpha=0.9, label="λ (4–6 Гц)")
    ax1.axhline(0, linestyle="--", linewidth=0.8, alpha=0.7)
    ax1.set_title(f"{title}\nЭЭГ и λ-ритм", fontsize=10)
    ax1.set_xlabel("Время, с")
    ax1.set_ylabel("Амплитуда")
    ax1.legend(fontsize=9)

    t_win, p_win = sliding_window_power(lam, fs_hz=fs_hz, window_sec=2.0, overlap=0.5)
    ax2.plot(t_win, p_win, linewidth=1.3)
    ax2.set_title(f"{title}\nМощность λ(t)", fontsize=10)
    ax2.set_xlabel("Время, с")
    ax2.set_ylabel("mean(x²)")

    fig.tight_layout()
    return fig


def make_bars_figure(summary_df: pd.DataFrame) -> Figure:
    fig = Figure(figsize=(10, 7), dpi=120)
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2)
    ax3 = fig.add_subplot(3, 1, 3)
    for ax in (ax1, ax2, ax3):
        style_axes(ax)

    df = summary_df.sort_values("Средняя мощность λ(t)", ascending=False)
    labels = df["Файл"].astype(str).values
    x = np.arange(len(labels))

    ax1.bar(x, df["Средняя мощность λ(t)"].values)
    ax1.set_title("Средняя мощность λ(t)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)

    ax2.bar(x, df["Макс мощность λ(t)"].values)
    ax2.set_title("Макс мощность λ(t)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)

    ax3.bar(x, df["Мин мощность λ(t)"].values)
    ax3.set_title("Мин мощность λ(t)")
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)

    fig.tight_layout()
    return fig


# ---------------------------
# Report helpers (ReportLab)
# ---------------------------
def _df_to_rl_table(df: pd.DataFrame, font_name: str):
    if df is None or df.empty:
        return Paragraph("Нет данных.", getSampleStyleSheet()["Normal"])

    show_df = df.copy()

    # round numeric
    for c in show_df.columns:
        if pd.api.types.is_numeric_dtype(show_df[c]):
            show_df[c] = show_df[c].round(6)

    data = [list(show_df.columns)] + show_df.astype(str).values.tolist()

    tbl = Table(data, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), font_name),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E6E6E6")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#FAFAFA")]),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
    ]))
    return tbl


def _rl_image(path: str, max_width_cm: float = 17.0):
    # Keep aspect ratio, scale by width
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
