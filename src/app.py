from __future__ import annotations

import os
import time
import csv
import threading
import queue
import tempfile
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Tuple, Any, Dict

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import font_manager

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Optional MNE (for 10–20 montage coordinates)
try:
    import mne
    MNE_OK = True
except Exception:
    mne = None  # type: ignore
    MNE_OK = False

# --- optional deps ---
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

from config import (
    UI, FONT_MAIN, FONT_TITLE, FONT_H2, FONT_SMALL,
    FS_HZ_DEFAULT, LAMBDA_BAND_HZ, ALPHA_BAND_HZ,
    apply_mpl_style, style_axes, _blend, save_figure_png_threadsafe
)
from io_csv import load_time_and_signal, estimate_fs_from_time
from signal_analysis import (
    compute_psd, integrate_band_power, extract_lambda_signal, sliding_window_power, build_conclusions
)
from plotting import make_raw_figure, make_psd_figure, make_lambda_figure, make_bars_figure


# ---------------------------
# Scrollable helpers
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
    def __init__(self, cfg: SerialConfig, out_queue: "queue.Queue[tuple[Any, Any]]"):
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

    def close_now(self):
        try:
            if self.ser and getattr(self.ser, "is_open", False):
                self.ser.close()
        except Exception:
            pass

    def stop(self):
        self._stop.set()
        self.close_now()

    def run(self):
        try:
            self.connect()
        except Exception as e:
            self.q.put(("__ERROR__", str(e)))
            return

        try:
            while not self._stop.is_set():
                try:
                    line = self.ser.readline().decode("utf-8", errors="ignore").strip()
                except Exception:
                    break
                if not line:
                    continue

                parts = [p.strip() for p in line.split(self.cfg.delimiter)]
                if len(parts) < self.cfg.channels:
                    continue

                vals: List[float] = []
                ok = True
                for i in range(self.cfg.channels):
                    token = parts[i].replace(",", ".")
                    try:
                        vals.append(float(token))
                    except Exception:
                        ok = False
                        break
                if not ok:
                    continue

                t = time.time() - (self.t0 or time.time())
                self.q.put((t, vals))
        finally:
            self.close_now()


_BaseTk = TkinterDnD.Tk if DND_OK else tk.Tk


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
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#FFEDD5")),
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


# ---------------------------
# App
# ---------------------------
class EEGApp(_BaseTk):
    def __init__(self):
        super().__init__()
        apply_mpl_style()

        # --- LIVE defaults (важно: до _build_ui)
        self.live_channels = 1
        self.live_xs: List[List[float]] = [[]]  # список каналов
        self.live_t: List[float] = []

        self.title("Лямбда-ритмы ЭЭГ при разных воздействиях")
        self.geometry("1240x820")
        self.configure(bg=UI["bg"])

        self.serial_queue: "queue.Queue[tuple[Any, Any]]" = queue.Queue()
        self.streamer: Optional[ArduinoSerialStreamer] = None

        self.loaded_files: List[str] = []
        self._last_records: Optional[List[Dict[str, Any]]] = None
        self._last_fs_user = FS_HZ_DEFAULT

        self.eeg_montage = tk.StringVar(value="(не выбрано)")
        self.eeg_channel_hint = tk.StringVar(value="Авто (по CSV)")

        self.band_power_df: Optional[pd.DataFrame] = None
        self.lambda_time_df: Optional[pd.DataFrame] = None
        self.summary_df: Optional[pd.DataFrame] = None
        self.conclusions_text: str = ""

        self._analysis_thread: Optional[threading.Thread] = None
        self._analysis_busy = False
        self._ui_queue: "queue.Queue[Tuple[str, object]]" = queue.Queue()

        self._pdf_thread: Optional[threading.Thread] = None
        self._pdf_busy = False

        # ---- 10–20 electrode selection (global state) ----
        self.selected_electrodes: List[str] = []
        self.max_selected_electrodes: int = 3
        self.stream_labels: List[str] = []
        self._live_running = False
        self._stream_time_offset = 0.0
        self._last_live_draw_ts = 0.0

        # сравнение (BARS) доступно/нет
        self._compare_allowed: bool = False
        self._compare_reason: str = ""

        self._setup_style()
        self._build_ui()
        self._sync_electrode_selection_ui()

        self.after(60, self._poll_serial_queue)
        self.after(60, self._poll_ui_queue)

    # ----------------------------
    # Compare mode availability
    # ----------------------------
    def _try_read_csv_quick(self, p: str) -> Optional[pd.DataFrame]:
        try:
            return pd.read_csv(p, sep=None, engine="python")
        except Exception:
            pass
        for sep in [",", ";", "\t"]:
            for dec in [".", ","]:
                try:
                    return pd.read_csv(p, sep=sep, decimal=dec, engine="python")
                except Exception:
                    continue
        try:
            return pd.read_csv(p, engine="python")
        except Exception:
            return None

    def _to_num(self, s: pd.Series) -> pd.Series:
        if s.dtype == object:
            s = s.astype(str).str.replace(",", ".", regex=False)
        return pd.to_numeric(s, errors="coerce")

    def _pick_time_col(self, cols: List[str]) -> Optional[str]:
        for c in cols:
            lc = str(c).lower()
            if "time" in lc or "время" in lc:
                return str(c)
        return None

    def _signal_col_count_for_file(self, path: str) -> Optional[int]:
        df = self._try_read_csv_quick(path)
        if df is None or df.empty:
            return None
        cols = [str(c) for c in df.columns]
        time_col = self._pick_time_col(cols)

        numeric_cols: List[str] = []
        for c in cols:
            sn = self._to_num(df[c])
            if sn.notna().sum() >= max(5, int(0.05 * len(df))):
                numeric_cols.append(c)

        if not numeric_cols:
            return None

        signal_cols = [c for c in numeric_cols if (time_col is None or c != time_col)]
        if not signal_cols:
            signal_cols = [numeric_cols[0]]
        return len(signal_cols)

    def _update_compare_mode_availability(self) -> None:
        """
        Правило:
        Сравнение (BARS) показываем/включаем только если:
        - загружено >= 2 файла
        - и количество "сигнальных" колонок (без времени) одинаковое во всех файлах
        """
        if len(self.loaded_files) < 2:
            self._compare_allowed = False
            self._compare_reason = "Сравнение доступно только при 2+ файлах."
        else:
            counts: List[int] = []
            for p in self.loaded_files:
                c = self._signal_col_count_for_file(p)
                if c is None:
                    # если файл плохой — безопасно считаем, что сравнение нельзя
                    self._compare_allowed = False
                    self._compare_reason = "Не удалось прочитать структуру CSV для сравнения."
                    break
                counts.append(int(c))

            if counts:
                if len(set(counts)) != 1:
                    self._compare_allowed = False
                    self._compare_reason = "Сравнение отключено: разное число колонок в файлах."
                else:
                    self._compare_allowed = True
                    self._compare_reason = ""

        # Применяем к UI (если радиокнопка уже создана)
        if hasattr(self, "rb_bars") and self.rb_bars is not None:
            try:
                self.rb_bars.config(state=("normal" if self._compare_allowed else "disabled"))
            except Exception:
                pass

        # Если сейчас выбран режим BARS, а он стал недоступен -> переключаем на RAW
        if hasattr(self, "plot_mode"):
            try:
                if self.plot_mode.get() == "BARS" and not self._compare_allowed:
                    self.plot_mode.set("RAW")
                    self._render_plots()
            except Exception:
                pass

    # ---------------------------
    # shared helpers
    # ---------------------------
    def _reset_live_plot_lines(self, n_channels: int, labels: Optional[List[str]] = None):
        n_channels = max(1, int(n_channels))
        self.live_channels = n_channels

        self.live_t = []
        self.live_xs = [[] for _ in range(n_channels)]

        self.ax_live.cla()
        style_axes(self.ax_live)
        self.ax_live.set_title("Сигнал в реальном времени")
        self.ax_live.set_xlabel("Время, с")
        self.ax_live.set_ylabel("Амплитуда (у.е.)")

        self.lines_live = []
        for i in range(n_channels):
            lab = (labels[i] if labels and i < len(labels) else f"CH{i + 1}")
            (ln,) = self.ax_live.plot([], [], linewidth=1.8, label=lab)
            self.lines_live.append(ln)

        self.ax_live.legend(fontsize=9, loc="upper right")

        if hasattr(self, "canvas_live") and self.canvas_live is not None:
            self.canvas_live.draw_idle()

    def _selected_electrodes_str(self) -> str:
        return ", ".join(self.selected_electrodes) if self.selected_electrodes else "(не выбрано)"

    def _sync_electrode_selection_ui(self) -> None:
        if hasattr(self, "_analysis_elec_var"):
            try:
                self._analysis_elec_var.set(self._selected_electrodes_str())
            except Exception:
                pass
        if hasattr(self, "ax_live"):
            try:
                s = self._selected_electrodes_str()
                self.ax_live.set_ylabel(f"A0 / {s}" if s != "(не выбрано)" else "A0 (у.е.)")
                if hasattr(self, "canvas_live"):
                    self.canvas_live.draw_idle()
            except Exception:
                pass

    def _infer_channel_capacity(self) -> int:
        try:
            if self.streamer is not None and hasattr(self.streamer, "cfg"):
                return max(1, int(getattr(self.streamer.cfg, "channels", 1)))
        except Exception:
            pass

        if getattr(self, "loaded_files", None):
            counts = []
            for p in self.loaded_files:
                c = self._signal_col_count_for_file(p)
                if c is not None:
                    counts.append(int(c))
            if counts:
                return max(1, min(counts))
        return 3

    # ---------------------------
    # style
    # ---------------------------
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
        style.map("Treeview.Heading", background=[("active", UI["hover"])])

        style.configure("Seg.TRadiobutton",
                        background=UI["panel2"],
                        foreground=UI["muted"],
                        padding=(12, 8))
        style.map("Seg.TRadiobutton",
                  background=[("selected", UI["hover"]), ("active", UI["hover"])],
                  foreground=[("selected", UI["text"]), ("active", UI["text"])])

        self.option_add("*TCombobox*Listbox.background", UI["panel2"])
        self.option_add("*TCombobox*Listbox.foreground", UI["text"])
        self.option_add("*TCombobox*Listbox.selectBackground", UI["hover"])
        self.option_add("*TCombobox*Listbox.selectForeground", UI["text"])
        self.option_add("*TCombobox*Listbox.font", FONT_MAIN)
        self.option_add("*TCombobox*Listbox.borderWidth", 0)
        self.option_add("*TCombobox*Listbox.highlightThickness", 1)
        self.option_add("*TCombobox*Listbox.highlightBackground", UI["border"])

    # ---------------------------
    # UI
    # ---------------------------
    def _build_ui(self):
        topbar = ttk.Frame(self)
        topbar.pack(fill="x", padx=12, pady=(12, 0))
        ttk.Button(topbar, text="Инструкция", command=self._show_help_dialog).pack(side="right")

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=12, pady=(8, 12))

        self.tab_live = ttk.Frame(nb)
        self.tab_1020 = ttk.Frame(nb)
        self.tab_files = ttk.Frame(nb)
        self.tab_analysis = ttk.Frame(nb)

        nb.add(self.tab_live, text="Онлайн")
        nb.add(self.tab_1020, text="Система 10–20")
        nb.add(self.tab_files, text="Файлы")
        nb.add(self.tab_analysis, text="Анализ ЛР5")

        self._build_live_tab()
        self._build_1020_tab()
        self._build_files_tab()
        self._build_analysis_tab()

        # важное: после сборки — обновить доступность сравнения
        self._update_compare_mode_availability()

    # ---------------- help ----------------
    def _show_help_dialog(self) -> None:
        win = tk.Toplevel(self)
        win.title("Инструкция")
        win.transient(self)
        win.grab_set()
        win.geometry("720x560")
        win.configure(bg=UI["bg"])

        header = tk.Frame(win, bg=UI["panel"], padx=16, pady=14)
        header.pack(fill="x")
        tk.Label(header, text="Инструкция", bg=UI["panel"], fg=UI["text"], font=FONT_TITLE).pack(anchor="w")
        tk.Label(
            header,
            text="Коротко: подключись → выбери электроды → загрузи файл → запусти анализ → сохрани PDF",
            bg=UI["panel"], fg=UI["muted"], font=FONT_MAIN
        ).pack(anchor="w", pady=(6, 0))

        container = tk.Frame(win, bg=UI["bg"], padx=16, pady=14)
        container.pack(fill="both", expand=True)

        card = tk.Frame(container, bg=UI["panel2"], highlightthickness=1, highlightbackground=UI["border"])
        card.pack(fill="both", expand=True)

        yscroll = ttk.Scrollbar(card, orient="vertical")
        yscroll.pack(side="right", fill="y")

        txt = tk.Text(
            card, wrap="word", yscrollcommand=yscroll.set,
            bg=UI["panel2"], fg=UI["text"], insertbackground=UI["text"],
            borderwidth=0, highlightthickness=0, padx=14, pady=12, font=FONT_MAIN
        )
        txt.pack(side="left", fill="both", expand=True)
        yscroll.config(command=txt.yview)

        txt.tag_configure("h", font=FONT_H2, foreground=UI["text"], spacing3=8)
        txt.tag_configure("sub", font=FONT_MAIN, foreground=UI["muted"], spacing3=6)
        txt.tag_configure("stepn", font=FONT_MAIN, foreground=UI["accent"], spacing1=2)
        txt.tag_configure("step", font=FONT_MAIN, foreground=UI["text"], lmargin1=24, lmargin2=24, spacing1=2)
        txt.tag_configure("bullet", font=FONT_MAIN, foreground=UI["text"], lmargin1=24, lmargin2=24, spacing1=2)
        txt.tag_configure("note", font=FONT_MAIN, foreground=UI["muted"], lmargin1=24, lmargin2=24, spacing1=2)
        txt.tag_configure("pill", font=FONT_SMALL, foreground=UI["muted"],
                          background=_blend(UI["panel2"], UI["hover"], 0.45))

        def add(line: str, tag: str | Tuple[str, ...] | None = None):
            if tag is None:
                txt.insert("end", line)
            else:
                txt.insert("end", line, tag)

        add("Шаги работы\n", "h")
        steps = [
            "Онлайн — подключитесь к Arduino по COM-порту и убедитесь, что сигнал идёт.",
            "Система 10–20 — выберите до 3 электродов (это позиции на голове).",
            "Файлы — загрузите CSV (проводник или drag&drop).",
            "Анализ — запустите расчёты (лямбда-ритм/PSD) и посмотрите результаты.",
            "Экспорт PDF — сохраните отчёт.",
        ]
        for i, s in enumerate(steps, 1):
            add(f"{i}) ", "stepn")
            add(s + "\n", "step")
        add("\n")

        add("Коротко про термины\n", "h")
        add("• ", "bullet"); add("Канал", ("bullet", "pill"))
        add(" — один столбец сигнала в CSV или один поток чисел в онлайн-режиме.\n", "bullet")
        add("• ", "bullet"); add("Электрод", ("bullet", "pill"))
        add(" — точка на голове по системе 10–20 (O1, Oz, O2 и т.д.).\n", "bullet")
        add("\n")

        add("Про режим «Сравнение»\n", "h")
        add("• Доступен только если загружено минимум 2 файла.\n", "note")
        add("• И если во всех файлах одинаковое число сигнальных колонок (без времени).\n", "note")
        add("\n")

        add("Подсказка\n", "h")
        add("Для λ-ритма обычно выбирают затылочные электроды: O1 / Oz / O2.\n", "sub")

        txt.config(state="disabled")

        footer = tk.Frame(container, bg=UI["bg"], pady=10)
        footer.pack(fill="x")
        ttk.Button(footer, text="Закрыть", command=win.destroy).pack(side="right")

    # ---------------- live ----------------
    def _build_live_tab(self):
        root = ttk.Frame(self.tab_live)
        root.pack(fill="both", expand=True)

        header = ttk.Frame(root)
        header.pack(fill="x", pady=(0, 10))
        ttk.Label(header, text="Онлайн запись (Arduino)", style="Title.TLabel").pack(side="left")

        card = ttk.Frame(root, padding=14, style="Card.TFrame")
        card.pack(fill="x")

        ttk.Label(card, text="Порт:", style="Muted.TLabel").pack(side="left", padx=(0, 6))
        self.cbo_port = ttk.Combobox(card, width=28, state="readonly")
        self.cbo_port.pack(side="left", padx=(0, 16))

        ttk.Label(card, text="Скорость (baud):", style="Muted.TLabel").pack(side="left", padx=(0, 6))
        self.ent_baud = ttk.Entry(card, width=10)
        self.ent_baud.insert(0, "115200")
        self.ent_baud.pack(side="left", padx=(0, 16))

        self.btn_start = ttk.Button(card, text="▶ Старт", command=self.start_stream, style="Primary.TButton")
        self.btn_start.pack(side="left", padx=(0, 8))

        self.btn_stop = ttk.Button(card, text="⏸ Стоп", command=self.stop_stream, state="disabled", style="Danger.TButton")
        self.btn_stop.pack(side="left", padx=(0, 8))

        self.btn_reset_live = ttk.Button(card, text="↺ Сброс", command=self.reset_live, style="Ghost.TButton")
        self.btn_reset_live.pack(side="left", padx=(0, 8))

        self.btn_save = ttk.Button(card, text="💾 Сохранить CSV", command=self.save_live_csv,
                                   state="disabled", style="Ghost.TButton")
        self.btn_save.pack(side="left", padx=(0, 8))

        self.btn_add_to_files = ttk.Button(
            card, text="➕ Добавить в анализ", command=self.add_live_to_files,
            state="disabled", style="Ghost.TButton"
        )
        self.btn_add_to_files.pack(side="left")

        plot_card = ttk.Frame(root, padding=14, style="Card.TFrame")
        plot_card.pack(fill="both", expand=True, pady=(12, 0))

        from matplotlib.figure import Figure
        self.fig_live = Figure(figsize=(10, 4), dpi=110)
        self.ax_live = self.fig_live.add_subplot(111)
        style_axes(self.ax_live)

        self.canvas_live = FigureCanvasTkAgg(self.fig_live, master=plot_card)
        self.canvas_live.get_tk_widget().pack(fill="both", expand=True)

        self.live_max_sec = 10.0
        self._reset_live_plot_lines(1, ["A0"])
        self.canvas_live.draw_idle()

        self.lbl_live_status = ttk.Label(root, text="Статус: не запущено", style="Muted.TLabel")
        self.lbl_live_status.pack(anchor="w", pady=(10, 0))

        self.refresh_ports(silent=True)
        self.after(2000, self._ports_autorefresh_tick)

    def _ports_autorefresh_tick(self):
        try:
            if not getattr(self, "_live_running", False):
                self.refresh_ports(silent=True)
        finally:
            self.after(2000, self._ports_autorefresh_tick)

    def refresh_ports(self, silent: bool = False):
        ports = []
        if SERIAL_OK:
            try:
                ports = [p.device for p in serial.tools.list_ports.comports()]
            except Exception:
                ports = []
        try:
            import glob
            extra = glob.glob("/dev/tty.*") + glob.glob("/dev/cu.*") + glob.glob("/dev/ttys*") + glob.glob("/dev/tty.usb*")
            for p in extra:
                if p not in ports:
                    ports.append(p)
        except Exception:
            pass

        ports = sorted(set(ports))
        self.cbo_port["values"] = ports

        if ports and not self.cbo_port.get():
            self.cbo_port.set(ports[0])

        if not silent:
            self.lbl_live_status.config(text="Статус: список портов обновлён")

    def start_stream(self):
        if getattr(self, "_live_running", False):
            return

        if not SERIAL_OK:
            messagebox.showerror("Serial", "pyserial не установлен.\n\npip install pyserial")
            return

        selected = list(getattr(self, "selected_electrodes", []))
        if not selected:
            messagebox.showinfo("Выбор электродов", "Перед стартом выберите электрод(ы) во вкладке «Система 10–20» (минимум 1).")
            return

        n_channels = max(1, min(len(selected), self.max_selected_electrodes))
        self.stream_labels = selected[:n_channels]

        port = (self.cbo_port.get() or "").strip()
        available_ports = list(self.cbo_port.cget('values') or [])
        if available_ports and port and port not in available_ports:
            port = ''
        if not port:
            messagebox.showerror("Ошибка", "Выбери порт.")
            return

        try:
            baud = int(self.ent_baud.get().strip())
        except Exception:
            messagebox.showerror("Ошибка", "Неверная скорость (baud).")
            return

        if not getattr(self, "live_t", None):
            self._reset_live_plot_lines(n_channels, self.stream_labels)
            self._stream_time_offset = 0.0
        else:
            self._stream_time_offset = float(self.live_t[-1]) if self.live_t else 0.0
            if getattr(self, "live_channels", 1) != n_channels:
                self.reset_live()
                self._reset_live_plot_lines(n_channels, self.stream_labels)
                self._stream_time_offset = 0.0

        self._drain_serial_queue()

        cfg = SerialConfig(port=port, baudrate=baud, delimiter=",", channels=n_channels)
        self.streamer = ArduinoSerialStreamer(cfg, self.serial_queue)
        self.streamer.start()

        self._live_running = True
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")

        self.btn_save.config(state=("normal" if self.live_t else "disabled"))
        self.btn_add_to_files.config(state=("normal" if self.live_t else "disabled"))

        self.lbl_live_status.config(text=f"Статус: поток | каналы: {', '.join(self.stream_labels)}")

    def stop_stream(self):
        if self.streamer:
            try:
                self.streamer.stop()
                self.streamer.close_now()
            except Exception:
                pass
            self.streamer = None

        self._live_running = False
        self._drain_serial_queue()

        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")

        has_data = bool(self.live_t)
        self.btn_save.config(state="normal" if has_data else "disabled")
        self.btn_add_to_files.config(state="normal" if has_data else "disabled")
        self.lbl_live_status.config(text="Статус: остановлено (пауза)")

    def reset_live(self):
        if getattr(self, "_live_running", False):
            self.stop_stream()

        self._drain_serial_queue()

        self.live_t = []
        self.live_xs = [[] for _ in range(1)]
        self.stream_labels = []
        self._stream_time_offset = 0.0
        self._last_live_draw_ts = 0.0
        self._reset_live_plot_lines(1, ["A0"])
        self.btn_save.config(state="disabled")
        self.btn_add_to_files.config(state="disabled")
        self.lbl_live_status.config(text="Статус: сброшено ✅")

    def _drain_serial_queue(self) -> None:
        try:
            while True:
                self.serial_queue.get_nowait()
        except queue.Empty:
            pass

    def _poll_serial_queue(self):
        max_items = 300
        got_any = False
        count = 0

        while count < max_items:
            try:
                item = self.serial_queue.get_nowait()
            except queue.Empty:
                break

            count += 1

            if isinstance(item[0], str) and item[0] == "__ERROR__":
                messagebox.showerror("Ошибка Serial", f"Не удалось подключиться:\n{item[1]}")
                self.stop_stream()
                break

            t, vals = item
            if not isinstance(t, (int, float)) or not isinstance(vals, list):
                continue

            t = float(t) + float(getattr(self, "_stream_time_offset", 0.0))
            self.live_t.append(t)

            need = max(1, len(getattr(self, "stream_labels", [])) or 1)
            if len(self.live_xs) != need:
                self.live_xs = [[] for _ in range(need)]

            for i in range(min(need, len(vals))):
                self.live_xs[i].append(float(vals[i]))

            got_any = True

        if got_any:
            now = time.time()
            if (now - getattr(self, "_last_live_draw_ts", 0.0)) >= 0.08:
                self._last_live_draw_ts = now
                self._update_live_plot()

        self.after(60, self._poll_serial_queue)

    def _update_live_plot(self):
        if len(self.live_t) < 2:
            return

        t = np.asarray(self.live_t, dtype=float)

        ymin, ymax = None, None
        for i, ln in enumerate(self.lines_live):
            x = np.asarray(self.live_xs[i], dtype=float)
            ln.set_data(t, x)

            if len(x):
                lo, hi = float(np.min(x)), float(np.max(x))
                ymin = lo if ymin is None else min(ymin, lo)
                ymax = hi if ymax is None else max(ymax, hi)

        self.ax_live.set_xlim(float(t[0]), float(t[-1]))
        if ymin is not None and ymax is not None:
            pad = 0.05 * (ymax - ymin) if ymax > ymin else 0.5
            self.ax_live.set_ylim(ymin - pad, ymax + pad)

        self.canvas_live.draw_idle()

    def add_live_to_files(self):
        return self.add_live_record_to_files()

    def add_live_record_to_files(self):
        if not self.live_t:
            messagebox.showwarning("Нет данных", "Сначала запиши сигнал (Старт → Стоп).")
            return

        live_count = sum(1 for p in self.loaded_files if os.path.basename(p).startswith("eeg_live_"))
        if live_count >= 6:
            messagebox.showinfo("Лимит", "Можно добавить максимум 6 записей из онлайн-режима.")
            return

        tmpdir = os.path.join(tempfile.gettempdir(), "eeg_app_live_records")
        os.makedirs(tmpdir, exist_ok=True)

        fname = f"eeg_live_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        path = os.path.join(tmpdir, fname)

        labels = self.stream_labels if self.stream_labels else [f"CH{i + 1}" for i in range(len(self.live_xs) or 1)]

        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["time_s"] + labels)
            n = len(self.live_t)
            for k in range(n):
                row = [f"{self.live_t[k]:.6f}"]
                for i in range(len(labels)):
                    row.append(f"{self.live_xs[i][k]:.6f}" if k < len(self.live_xs[i]) else "")
                w.writerow(row)

        path = os.path.abspath(path)
        if path not in self.loaded_files:
            self.loaded_files.append(path)
            if hasattr(self, "lst_files"):
                self.lst_files.insert("end", path)
            if hasattr(self, "_refresh_files_count"):
                self._refresh_files_count()

        # ✅ обновить доступность сравнения
        self._update_compare_mode_availability()

        messagebox.showinfo("Добавлено", "Запись добавлена во вкладку «Файлы» и готова к анализу.")

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
            header_label = ", ".join(self.selected_electrodes) if self.selected_electrodes else "A0"
            w.writerow(["время_с", header_label])

            n = len(self.live_t)
            for k in range(n):
                row = [f"{self.live_t[k]:.6f}"]
                for i in range(self.live_channels):
                    row.append(f"{self.live_xs[i][k]:.6f}" if k < len(self.live_xs[i]) else "")
                w.writerow(row)

        messagebox.showinfo("Сохранено", f"CSV сохранён:\n{path}")

    # ---------------- 10–20 ----------------
    def _build_1020_tab(self):
        root = ttk.Frame(self.tab_1020)
        root.pack(fill="both", expand=True)

        head = ttk.Frame(root)
        head.pack(fill="x", padx=12, pady=(12, 6))

        ttk.Label(head, text="Система 10–20: выбор электродов", font=FONT_H2).pack(side="left")

        cap = max(1, min(self.max_selected_electrodes, self._infer_channel_capacity()))
        self._1020_cap = cap
        ttk.Label(head, text=f"(максимум {cap} электрод(а))", foreground=UI["muted"], font=FONT_SMALL).pack(side="left", padx=(10, 0))

        body = ttk.Frame(root)
        body.pack(fill="both", expand=True, padx=12, pady=10)

        left = tk.Frame(body, bg=UI["panel2"], highlightthickness=1, highlightbackground=UI["border"])
        left.pack(side="left", fill="both", expand=True)

        right = ttk.Frame(body)
        right.pack(side="right", fill="y", padx=(12, 0))

        if not MNE_OK:
            ttk.Label(left, text="Библиотека MNE не установлена.\nСхема 10–20 недоступна.", justify="center").pack(expand=True)
            return

        self._1020_canvas = tk.Canvas(left, bg=UI["panel2"], highlightthickness=0, borderwidth=0)
        self._1020_canvas.pack(fill="both", expand=True)

        self._1020_selected_var = tk.StringVar(value="Выбрано: (ничего)")
        ttk.Label(root, textvariable=self._1020_selected_var, foreground=UI["muted"], font=FONT_MAIN).pack(anchor="w", padx=14, pady=(0, 12))

        montage = mne.channels.make_standard_montage("standard_1020")
        ch_pos = montage.get_positions()["ch_pos"]
        keep = {
            "Fp1","Fp2","F7","F3","Fz","F4","F8",
            "T3","C3","Cz","C4","T4",
            "T5","P3","Pz","P4","T6",
            "O1","Oz","O2"
        }
        self._1020_positions = {k: (float(v[0]), float(v[1])) for k, v in ch_pos.items() if k in keep}

        self._1020_radius = 10
        self._1020_hit_radius = 14
        self._1020_items = {}

        self._1020_selected: List[str] = list(self.selected_electrodes)

        def _update_selected_text():
            self._1020_selected_var.set("Выбрано: " + ", ".join(self._1020_selected) if self._1020_selected else "Выбрано: (ничего)")

        def _project_to_canvas(x: float, y: float, w: int, h: int):
            pad = max(30, int(0.08 * min(w, h)))
            size = min(w, h) - 2 * pad
            size = max(size, 10)
            cx, cy = w / 2, h / 2
            xs = [p[0] for p in self._1020_positions.values()]
            ys = [p[1] for p in self._1020_positions.values()]
            rx = max(abs(min(xs)), abs(max(xs)), 1e-6)
            ry = max(abs(min(ys)), abs(max(ys)), 1e-6)
            nx = x / rx
            ny = y / ry
            px = cx + (nx * (size / 2))
            py = cy - (ny * (size / 2))
            return px, py

        def _draw():
            c = self._1020_canvas
            c.delete("all")
            self._1020_items.clear()

            w = c.winfo_width()
            h = c.winfo_height()
            if w < 60 or h < 60:
                return

            pad = max(28, int(0.075 * min(w, h)))
            base = min(w, h)
            self._1020_radius = max(6, int(base * 0.012))
            self._1020_hit_radius = max(self._1020_radius + 4, int(base * 0.018))

            c.create_oval(pad, pad, w - pad, h - pad, outline=UI["border"], width=2)

            cx = w / 2
            top = pad
            c.create_polygon(cx - 12, top + 2, cx + 12, top + 2, cx, top - 14,
                             outline=UI["border"], fill=UI["panel2"], width=2)

            c.create_oval(pad - 10, h / 2 - 28, pad + 18, h / 2 + 28, outline=UI["border"], width=2, fill=UI["panel2"])
            c.create_oval(w - pad - 18, h / 2 - 28, w - pad + 10, h / 2 + 28, outline=UI["border"], width=2, fill=UI["panel2"])

            for name, (x, y) in self._1020_positions.items():
                px, py = _project_to_canvas(x, y, w, h)
                r = max(6, min(14, int(min(w, h) * 0.018)))

                selected = name in self._1020_selected
                fill = UI["accent"] if selected else UI["panel"]
                outline = UI["accent"] if selected else UI["border"]

                oid = c.create_oval(px - r, py - r, px + r, py + r, fill=fill, outline=outline, width=2)
                show_label = selected or (min(w, h) >= 820 and name in ("Fp1","Fp2","Fz","Cz","Pz","Oz","O1","O2"))
                tid = None
                if show_label:
                    fsz = 9 if min(w, h) >= 820 else 8
                    tid = c.create_text(px, py + r + 10, text=name, fill=UI["muted"], font=("SF Pro Text", fsz))

                self._1020_items[name] = (oid, tid)

            _update_selected_text()

        def _nearest_electrode(xc: float, yc: float):
            c = self._1020_canvas
            w = c.winfo_width()
            h = c.winfo_height()
            best = None
            best_d2 = 10 ** 12
            for name, (x, y) in self._1020_positions.items():
                px, py = _project_to_canvas(x, y, w, h)
                d2 = (px - xc) ** 2 + (py - yc) ** 2
                if d2 < best_d2:
                    best_d2 = d2
                    best = name
            if best is None:
                return None
            if best_d2 <= (self._1020_hit_radius ** 2):
                return best
            return None

        def _toggle(name: str):
            cap = int(getattr(self, "_1020_cap", self.max_selected_electrodes))
            if name in self._1020_selected:
                self._1020_selected.remove(name)
            else:
                if cap <= 1:
                    self._1020_selected[:] = [name]
                else:
                    if len(self._1020_selected) >= cap:
                        messagebox.showinfo("Ограничение выбора", f"Можно выбрать не более {cap} электрод(ов) для текущих данных.")
                        return
                    self._1020_selected.append(name)
            _draw()

        def _on_click(evt):
            name = _nearest_electrode(evt.x, evt.y)
            if name:
                _toggle(name)

        self._1020_canvas.bind("<Button-1>", _on_click)
        self._1020_canvas.bind("<Configure>", lambda _e: _draw())

        ttk.Label(right, text="Управление выбором", font=FONT_H2).pack(anchor="w", pady=(0, 8))

        rec_card = ttk.Frame(right, padding=10, style="Card2.TFrame")
        rec_card.pack(fill="x", pady=(0, 12))
        ttk.Label(rec_card, text="Рекомендации", style="H2.TLabel").pack(anchor="w")
        ttk.Label(
            rec_card,
            text=(
                "• Для регистрации λ-ритма обычно используют затылочные отведения: O1, Oz, O2.\n"
                "• При отсутствии чисто затылочных допускаются теменно-затылочные варианты: Pz вместе с Oz/O2.\n"
                "• Для контрольного сравнения можно выбрать лобные отведения (Fp1/Fp2/Fz).\n\n"
                "Рекомендация: фиксируйте выбранные отведения на время одной записи и не меняйте их «на лету»."
            ),
            style="Muted.TLabel",
            justify="left",
        ).pack(anchor="w")

        def _clear():
            self._1020_selected.clear()
            _draw()

        presets: List[Tuple[str, List[str]]] = [
            ("Затылочные (λ): O1, Oz, O2", ["O1", "Oz", "O2"]),
            ("Теменно-затылочные: Pz, Oz, O2", ["Pz", "Oz", "O2"]),
            ("Теменные: P3, Pz, P4", ["P3", "Pz", "P4"]),
            ("Лобные: Fp1, Fp2, Fz", ["Fp1", "Fp2", "Fz"]),
            ("Центральные: C3, Cz, C4", ["C3", "Cz", "C4"]),
        ]

        self._1020_preset_var = tk.StringVar(value=presets[0][0])

        ttk.Label(right, text="Набор электродов:", style="Muted.TLabel").pack(anchor="w", pady=(6, 4))
        cbo = ttk.Combobox(right, state="readonly", textvariable=self._1020_preset_var,
                           values=[p[0] for p in presets], width=28)
        cbo.pack(fill="x", pady=(0, 8))

        def _apply_preset():
            label = self._1020_preset_var.get()
            rec = None
            for nm, arr in presets:
                if nm == label:
                    rec = arr
                    break
            if not rec:
                return
            self._1020_selected.clear()
            for ch in rec:
                cap = int(getattr(self, "_1020_cap", self.max_selected_electrodes))
                if ch in self._1020_positions and len(self._1020_selected) < cap:
                    self._1020_selected.append(ch)
            _draw()

        def _apply():
            was_running = (self.streamer is not None)
            if was_running:
                self.stop_stream()

            self.selected_electrodes = list(self._1020_selected)
            self.eeg_montage.set(self._selected_electrodes_str())
            self._sync_electrode_selection_ui()
            _update_selected_text()

            messagebox.showinfo("Применено", "Выбор электродов сохранён и будет использован в других вкладках.")

            if was_running:
                self.start_stream()

        ttk.Button(right, text="Применить набор", command=_apply_preset).pack(fill="x", pady=(0, 8))
        ttk.Button(right, text="Очистить", command=_clear).pack(fill="x", pady=(0, 8))
        ttk.Separator(right).pack(fill="x", pady=10)
        ttk.Button(right, text="Применить", command=_apply).pack(fill="x")

        self.after(50, _draw)

    # ---------------- files ----------------
    def _build_files_tab(self):
        root = ttk.Frame(self.tab_files)
        root.pack(fill="both", expand=True)

        sf = ScrollableFrame(root, bg=UI["bg"])
        sf.pack(fill="both", expand=True)
        page = sf.inner

        header = ttk.Frame(page)
        header.pack(fill="x", pady=(0, 10), padx=12)
        ttk.Label(header, text="Файлы CSV", style="Title.TLabel").pack(side="left")

        card = ttk.Frame(page, padding=14, style="Card.TFrame")
        card.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        top = ttk.Frame(card, style="Card.TFrame")
        top.pack(fill="x")

        self.btn_add_files = ttk.Button(top, text="➕ Добавить CSV", command=self.add_csv_files, style="Primary.TButton")
        self.btn_add_files.pack(side="left")

        self.btn_remove_file = ttk.Button(top, text="🗑 Удалить", command=self.remove_selected_file, style="Ghost.TButton")
        self.btn_remove_file.pack(side="left", padx=8)

        self.btn_clear_files = ttk.Button(top, text="Очистить", command=self.clear_file_list, style="Danger.TButton")
        self.btn_clear_files.pack(side="left", padx=8)

        self.lbl_files_count = ttk.Label(top, text="Файлов: 0", style="Muted.TLabel")
        self.lbl_files_count.pack(side="right")

        drop = ttk.Frame(card, padding=12, style="Card2.TFrame")
        drop.pack(fill="x", pady=(12, 10))

        drop_text = "Перетащите CSV сюда" if DND_OK else "Добавьте CSV кнопкой выше"
        self.drop_label = ttk.Label(drop, text=drop_text, style="Muted.TLabel", anchor="center", justify="center")
        self.drop_label.pack(fill="x")

        info = ttk.Frame(card, padding=12, style="Card2.TFrame")
        info.pack(fill="x", pady=(0, 10))
        ttk.Label(info, text="Рекомендации по CSV", style="H2.TLabel").pack(anchor="w")
        ttk.Label(
            info,
            text=(
                "• Желательно иметь колонку времени (time_s / время_с) и 1+ колонок сигнала.\n"
                "• Если времени нет — программа построит синтетическое время по FS.\n"
                "• Для сравнения условий используйте одинаковый FS и одинаковый набор отведений."
            ),
            style="Muted.TLabel",
            justify="left",
        ).pack(anchor="w", pady=(6, 0))

        ttk.Label(card, text="Список добавленных файлов", style="H2.TLabel").pack(anchor="w", pady=(6, 6))

        list_frame = ttk.Frame(card, style="Card.TFrame")
        list_frame.pack(fill="x", pady=(0, 12))
        list_frame.configure(height=170)
        list_frame.pack_propagate(False)

        self.lst_files = tk.Listbox(
            list_frame, height=6,
            bg=UI["panel2"], fg=UI["text"],
            selectbackground=UI.get("hover", UI["accent"]),
            selectforeground=UI["text"],
            highlightthickness=0, bd=0,
        )
        self.lst_files.pack(side="left", fill="both", expand=True, padx=(0, 8))

        scr = ttk.Scrollbar(list_frame, orient="vertical", command=self.lst_files.yview)
        scr.pack(side="left", fill="y")
        self.lst_files.configure(yscrollcommand=scr.set)

        if DND_OK:
            for widget in (self.drop_label, self.lst_files):
                widget.drop_target_register(DND_FILES)
                widget.dnd_bind("<<DropEnter>>", self._on_drop_enter)
                widget.dnd_bind("<<DropLeave>>", self._on_drop_leave)
                widget.dnd_bind("<<Drop>>", self._on_drop_files)

    def _on_drop_enter(self, _event=None):
        self.drop_label.config(text="Отпустите файлы, чтобы добавить")
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
        for p in paths:
            p = os.path.abspath(p)
            if os.path.isfile(p) and p.lower().endswith(".csv") and p not in self.loaded_files:
                self.loaded_files.append(p)
                self.lst_files.insert("end", p)
        self._refresh_files_count()

        # ✅ обновить доступность сравнения
        self._update_compare_mode_availability()

    def remove_selected_file(self):
        sel = self.lst_files.curselection()
        if not sel:
            return
        idx = sel[0]
        path = self.lst_files.get(idx)
        self.lst_files.delete(idx)
        self.loaded_files = [p for p in self.loaded_files if p != path]
        self._refresh_files_count()

        # ✅ обновить доступность сравнения
        self._update_compare_mode_availability()

    def clear_file_list(self):
        self.lst_files.delete(0, "end")
        self.loaded_files.clear()
        self._refresh_files_count()

        # ✅ обновить доступность сравнения
        self._update_compare_mode_availability()

    # ---------------- analysis ----------------
    def _build_analysis_tab(self):
        root = ttk.Frame(self.tab_analysis)
        root.pack(fill="both", expand=True)

        header = ttk.Frame(root)
        header.pack(fill="x", pady=(0, 10))
        ttk.Label(header, text="Лямбда-ритм ЭЭГ при разных воздействиях", style="Title.TLabel").pack(side="left")

        controls = ttk.Frame(root, padding=14, style="Card.TFrame")
        controls.pack(fill="x")

        ttk.Label(controls, text="FS (Гц):", style="Muted.TLabel").pack(side="left", padx=(0, 8))
        self.ent_fs = ttk.Entry(controls, width=10)
        self.ent_fs.insert(0, "250")
        self.ent_fs.pack(side="left", padx=(0, 14))

        ttk.Label(controls, text="Выбранные электроды:", style="Muted.TLabel").pack(side="left", padx=(8, 8))
        self._analysis_elec_var = tk.StringVar(value=self._selected_electrodes_str())
        ttk.Label(controls, textvariable=self._analysis_elec_var, style="Muted.TLabel").pack(side="left", padx=(0, 14))

        self.btn_run = ttk.Button(controls, text="▶ Запустить анализ", command=self.run_lab5, style="Primary.TButton")
        self.btn_run.pack(side="left", padx=(0, 8))

        self.btn_reset_analysis = ttk.Button(controls, text="↺ Сброс анализа", command=self.reset_analysis, style="Ghost.TButton")
        self.btn_reset_analysis.pack(side="left", padx=(0, 8))

        self.btn_report = ttk.Button(controls, text="📄 Экспорт PDF", command=self.export_report_pdf, style="Ghost.TButton")
        self.btn_report.pack(side="left", padx=(0, 12))

        self.pb = ttk.Progressbar(controls, mode="indeterminate", length=180)
        self.pb.pack(side="left", padx=(0, 10))
        self.lbl_an_status = ttk.Label(controls, text="Готово", style="Muted.TLabel")
        self.lbl_an_status.pack(side="left")

        vpan = ttk.PanedWindow(root, orient="vertical")
        vpan.pack(fill="both", expand=True, pady=(12, 0))

        body = ttk.PanedWindow(vpan, orient="horizontal")
        left = ttk.Frame(body)
        right = ttk.Frame(body)
        body.add(left, weight=2)
        body.add(right, weight=5)
        vpan.add(body, weight=5)

        left_card = ttk.Frame(left, padding=14, style="Card.TFrame")
        left_card.pack(fill="both", expand=True)
        ttk.Label(left_card, text="Таблица результатов", style="H2.TLabel").pack(anchor="w")

        self.cbo_table = ttk.Combobox(left_card, state="readonly",
                                      values=["Сводная", "Мощности по диапазонам", "Статистики λ(t)"])
        self.cbo_table.current(0)
        self.cbo_table.pack(fill="x", pady=(10, 10))
        self.cbo_table.bind("<<ComboboxSelected>>", lambda e: self._render_current_table())

        self.tbl = ttk.Treeview(left_card, show="headings")
        self.tbl.pack(fill="both", expand=True)

        self.tbl_scr = ttk.Scrollbar(left_card, orient="vertical", command=self.tbl.yview)
        self.tbl_scr.pack(side="right", fill="y")
        self.tbl.configure(yscrollcommand=self.tbl_scr.set)

        right_card = ttk.Frame(right, padding=14, style="Card.TFrame")
        right_card.pack(fill="both", expand=True)

        seg = ttk.Frame(right_card, padding=10, style="Card2.TFrame")
        seg.pack(fill="x", pady=(0, 12))
        ttk.Label(seg, text="Режим отображения:", style="Muted.TLabel").pack(side="left", padx=(0, 10))

        self.plot_mode = tk.StringVar(value="RAW")

        ttk.Radiobutton(seg, text="Сигнал", value="RAW", variable=self.plot_mode,
                        style="Seg.TRadiobutton", command=self._render_plots).pack(side="left", padx=(0, 8))
        ttk.Radiobutton(seg, text="Спектр (PSD)", value="PSD", variable=self.plot_mode,
                        style="Seg.TRadiobutton", command=self._render_plots).pack(side="left", padx=(0, 8))
        ttk.Radiobutton(seg, text="λ-ритм", value="LAMBDA", variable=self.plot_mode,
                        style="Seg.TRadiobutton", command=self._render_plots).pack(side="left", padx=(0, 8))

        # ✅ сравнение: кнопка есть, но будет disabled если нельзя
        self.rb_bars = ttk.Radiobutton(seg, text="Сравнение", value="BARS", variable=self.plot_mode,
                                       style="Seg.TRadiobutton", command=self._render_plots)
        self.rb_bars.pack(side="left", padx=(0, 8))

        self.plot_host = ttk.Frame(right_card, style="Card2.TFrame")
        self.plot_host.pack(fill="both", expand=True)
        self.plot_area = ScrollablePlotArea(self.plot_host)
        self.plot_area.pack(fill="both", expand=True)

        concl_card = ttk.Frame(vpan, padding=14, style="Card.TFrame")
        vpan.add(concl_card, weight=2)
        ttk.Label(concl_card, text="Анализ и выводы", style="H2.TLabel").pack(anchor="w", pady=(0, 8))

        txt_wrap = ttk.Frame(concl_card, style="Card.TFrame")
        txt_wrap.pack(fill="both", expand=True)
        scr_y = ttk.Scrollbar(txt_wrap, orient="vertical")
        scr_y.pack(side="right", fill="y")

        self.txt_conclusions = tk.Text(
            txt_wrap, height=10, bg=UI["panel2"], fg=UI["text"],
            insertbackground=UI["text"], highlightthickness=1,
            highlightbackground=UI["border"], bd=0, wrap="word",
        )
        self.txt_conclusions.pack(side="left", fill="both", expand=True)
        self.txt_conclusions.configure(yscrollcommand=scr_y.set)
        scr_y.configure(command=self.txt_conclusions.yview)
        self.txt_conclusions.insert("1.0", "Запустите анализ, чтобы сформировать выводы.")
        self.txt_conclusions.config(state="disabled")

        # ✅ сразу применим правило сравнения
        self._update_compare_mode_availability()

    def _set_status(self, text: str):
        self.lbl_an_status.config(text=text)
        self.update_idletasks()

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

    def reset_analysis(self):
        if self._analysis_busy or self._pdf_busy:
            return
        self.band_power_df = None
        self.lambda_time_df = None
        self.summary_df = None
        self._last_records = None
        try:
            self.tbl.delete(*self.tbl.get_children())
            self.tbl["columns"] = []
        except Exception:
            pass
        try:
            self.plot_area.clear()
            ttk.Label(self.plot_area.inner, text="Результаты очищены. Запустите анализ заново.", style="Muted.TLabel").pack(
                padx=12, pady=12, anchor="w"
            )
        except Exception:
            pass
        self.conclusions_text = ""
        try:
            self.txt_conclusions.config(state="normal")
            self.txt_conclusions.delete("1.0", "end")
            self.txt_conclusions.insert("1.0", "Результаты очищены. Нажмите «Запустить анализ». ")
            self.txt_conclusions.config(state="disabled")
        except Exception:
            pass
        self._set_status("Сброс выполнен")

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

        # ✅ перед анализом обновим правило сравнения (на случай свежих файлов)
        self._update_compare_mode_availability()

        self._busy(True, "Анализ: запуск…")
        self._analysis_thread = threading.Thread(target=self._run_lab5_worker, args=(fs_user,), daemon=True)
        self._analysis_thread.start()

    def _run_lab5_worker(self, fs_user: float):
        try:
            self._ui_queue.put(("status", "Чтение CSV…"))

            def _try_read_csv_local(p: str) -> pd.DataFrame:
                df = self._try_read_csv_quick(p)
                if df is None:
                    raise ValueError(f"Не удалось прочитать CSV: {os.path.basename(p)}")
                return df

            def _to_num(s: pd.Series) -> pd.Series:
                return self._to_num(s)

            def _pick_time_col(cols: List[str]) -> Optional[str]:
                return self._pick_time_col(cols)

            records: List[Dict[str, Any]] = []
            selected = list(self.selected_electrodes)

            for path in self.loaded_files:
                df = _try_read_csv_local(path)
                cols = [str(c) for c in df.columns]

                time_col = _pick_time_col(cols)

                numeric_cols: List[str] = []
                for c in cols:
                    sn = _to_num(df[c])
                    if sn.notna().sum() >= max(5, int(0.05 * len(df))):
                        numeric_cols.append(c)

                if not numeric_cols:
                    raise ValueError(f"{os.path.basename(path)}: нет числовых столбцов")

                signal_cols = [c for c in numeric_cols if (time_col is None or c != time_col)]
                if not signal_cols:
                    signal_cols = [numeric_cols[0]]

                channels_to_use: List[str] = []
                electrode_label_for: Dict[str, str] = {}

                if selected:
                    existing = [ch for ch in selected if ch in cols]
                    if existing:
                        channels_to_use = existing[: self.max_selected_electrodes]
                        for ch in channels_to_use:
                            electrode_label_for[ch] = ch
                    else:
                        ch = signal_cols[0]
                        channels_to_use = [ch]
                        electrode_label_for[ch] = ", ".join(selected)
                else:
                    channels_to_use = signal_cols[: max(1, min(len(signal_cols), 8))]
                    for ch in channels_to_use:
                        electrode_label_for[ch] = ch

                for ch in channels_to_use:
                    x = _to_num(df[ch]).to_numpy(dtype=float)

                    if time_col is not None and time_col in cols:
                        t = _to_num(df[time_col]).to_numpy(dtype=float)
                        mask = np.isfinite(t) & np.isfinite(x)
                        t, x = t[mask], x[mask]
                        if len(t) > 2 and not np.all(np.diff(t) > 0):
                            t = np.arange(len(x)) / FS_HZ_DEFAULT
                            time_used = "synthetic_time"
                        else:
                            time_used = time_col
                    else:
                        mask = np.isfinite(x)
                        x = x[mask]
                        t = np.arange(len(x)) / FS_HZ_DEFAULT
                        time_used = "synthetic_time"

                    fs_est = estimate_fs_from_time(t, fallback=fs_user)
                    fs_hz = fs_user if fs_user > 0 else fs_est

                    name = os.path.splitext(os.path.basename(path))[0]
                    dur = float(t[-1] - t[0]) if len(t) > 1 else 0.0
                    nan_ratio = float(np.mean(~np.isfinite(x))) if len(x) else 1.0

                    records.append(
                        {
                            "name": name,
                            "path": path,
                            "t": t,
                            "x": x,
                            "fs": fs_hz,
                            "time_col": time_used,
                            "sig_col": ch,
                            "electrode_label": electrode_label_for.get(ch, ch),
                            "duration_s": dur,
                            "nan_ratio": nan_ratio,
                            "montage": self.eeg_montage.get(),
                            "channel_hint": self.eeg_channel_hint.get(),
                        }
                    )

            self._ui_queue.put(("status", "PSD и мощности диапазонов…"))
            band_rows: List[Dict[str, Any]] = []
            for r in records:
                freqs, psd = compute_psd(r["x"], fs_hz=r["fs"], nperseg=1024)
                p_total = integrate_band_power(freqs, psd, (0.5, 40.0))
                p_lambda = integrate_band_power(freqs, psd, LAMBDA_BAND_HZ)
                p_alpha = integrate_band_power(freqs, psd, ALPHA_BAND_HZ)

                band_rows.append(
                    {
                        "Файл": r["name"],
                        "Электрод/канал": r["electrode_label"],
                        "Канал (CSV)": r["sig_col"],
                        "FS (Гц)": r["fs"],
                        "Длительность (с)": r["duration_s"],
                        "Доля NaN": r["nan_ratio"],
                        "P_total": p_total,
                        "P_λ": p_lambda,
                        "P_α": p_alpha,
                        "P_λ / P_total": (p_lambda / p_total) if (p_total and p_total > 0) else np.nan,
                        "P_α / P_total": (p_alpha / p_total) if (p_total and p_total > 0) else np.nan,
                    }
                )
            band_power_df = pd.DataFrame(band_rows)

            self._ui_queue.put(("status", "λ(t) статистики…"))
            lambda_rows: List[Dict[str, Any]] = []
            for r in records:
                lam = extract_lambda_signal(r["x"], fs_hz=r["fs"])
                t_win, p_win = sliding_window_power(lam, fs_hz=r["fs"], window_sec=2.0, overlap=0.5)
                lambda_rows.append(
                    {
                        "Файл": r["name"],
                        "Электрод/канал": r["electrode_label"],
                        "Средняя мощность λ(t)": float(np.mean(p_win)) if len(p_win) else np.nan,
                        "Максимум λ(t)": float(np.max(p_win)) if len(p_win) else np.nan,
                        "Минимум λ(t)": float(np.min(p_win)) if len(p_win) else np.nan,
                    }
                )
            lambda_time_df = pd.DataFrame(lambda_rows)

            summary_df = (
                band_power_df.merge(lambda_time_df, on=["Файл", "Электрод/канал"], how="left")
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

                # ✅ после анализа ещё раз применим правило сравнения (на всякий)
                self._update_compare_mode_availability()

                self._render_current_table()
                self._render_plots()

                self.conclusions_text = build_conclusions(
                    summary_df=self.summary_df,
                    records=self._last_records,
                    fs_user=self._last_fs_user,
                    montage=self.eeg_montage.get(),
                    channel_hint=self.eeg_channel_hint.get(),
                )
                self.txt_conclusions.config(state="normal")
                self.txt_conclusions.delete("1.0", "end")
                self.txt_conclusions.insert("1.0", self.conclusions_text)
                self.txt_conclusions.config(state="disabled")

                self._busy(False, "Готово ✅")

            elif kind == "pdf_done":
                self._pdf_busy = False
                self._busy(False, "Готово ✅")
                messagebox.showinfo("PDF", f"Отчёт сохранён:\n{payload}")

            elif kind == "pdf_error":
                self._pdf_busy = False
                self._busy(False, "Ошибка")
                messagebox.showerror("PDF", str(payload))

        self.after(60, self._poll_ui_queue)

    def _render_current_table(self):
        choice = self.cbo_table.get()
        if choice == "Мощности по диапазонам":
            df = self.band_power_df
        elif choice == "Статистики λ(t)":
            df = self.lambda_time_df
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

        rows = show_df.values.tolist()
        self._table_rows_cache = rows
        self._table_cols_cache = cols
        self._table_insert_pos = 0

        def _step():
            chunk = 150
            end = min(self._table_insert_pos + chunk, len(self._table_rows_cache))
            for i in range(self._table_insert_pos, end):
                self.tbl.insert("", "end", values=self._table_rows_cache[i])
            self._table_insert_pos = end
            if self._table_insert_pos < len(self._table_rows_cache):
                self.after(1, _step)

        _step()

    def _render_plots(self):
        self.plot_area.clear()
        if self._last_records is None:
            ttk.Label(self.plot_area.inner, text="Запустите анализ, чтобы увидеть графики.", style="Muted.TLabel").pack(
                padx=12, pady=12, anchor="w"
            )
            return

        mode = self.plot_mode.get()

        if mode == "BARS":
            # ✅ если режим недоступен — показываем причину и выходим
            if not self._compare_allowed:
                msg = self._compare_reason or "Сравнение недоступно."
                wrap = ttk.Frame(self.plot_area.inner, padding=14, style="Card.TFrame")
                wrap.pack(fill="x", expand=True, padx=12, pady=12)
                ttk.Label(wrap, text=msg, style="Muted.TLabel").pack(anchor="w")
                return

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
            title = f'{r["name"]} — {r.get("electrode_label","")}'.strip(" —")
            ttk.Label(wrap, text=title, style="H2.TLabel").pack(anchor="w", pady=(0, 2))
            ttk.Label(wrap, text=f'Колонка CSV: {r.get("sig_col","")} | FS: {r.get("fs",""):.2f}', style="Muted.TLabel").pack(anchor="w", pady=(0, 10))

            if mode == "RAW":
                fig = make_raw_figure(r["t"], r["x"], r["fs"], title)
            elif mode == "PSD":
                fig = make_psd_figure(r["x"], r["fs"], title)
            else:
                fig = make_lambda_figure(r["t"], r["x"], r["fs"], title)

            canv = FigureCanvasTkAgg(fig, master=wrap)
            canv.get_tk_widget().pack(fill="x", expand=True)
            canv.draw()

    # ---------------- PDF ----------------
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

        # ✅ на момент экспорта тоже обновим правило
        self._update_compare_mode_availability()

        self._pdf_busy = True
        self._busy(True, "Экспорт PDF…")
        self._pdf_thread = threading.Thread(target=self._export_pdf_worker, args=(out_path,), daemon=True)
        self._pdf_thread.start()

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
                out_path, pagesize=A4,
                leftMargin=1.5 * cm, rightMargin=1.5 * cm,
                topMargin=1.5 * cm, bottomMargin=1.5 * cm,
                title="EEG Lab5 Report",
            )

            story = []
            story.append(Paragraph("Отчёт: Лямбда-ритмы ЭЭГ при разных воздействиях", styles["Heading1"]))
            story.append(Spacer(1, 0.3 * cm))
            story.append(Paragraph(f"Расположение электродов: {self.eeg_montage.get()}", styles["Normal"]))
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

            story.append(Paragraph("Анализ и выводы", styles["Heading2"]))
            text = self.conclusions_text or "Выводы не сформированы. Запустите анализ перед экспортом отчёта."
            for line in text.split("\n"):
                line = line.strip()
                if not line:
                    story.append(Spacer(1, 0.15 * cm))
                else:
                    story.append(Paragraph(line, styles["Normal"]))
            story.append(PageBreak())

            with tempfile.TemporaryDirectory() as tmpdir:
                self._ui_queue.put(("status", "PDF: графики…"))

                # ✅ Сравнение в PDF только если разрешено правилом
                if self._compare_allowed:
                    bars_path = os.path.join(tmpdir, "bars.png")
                    fig_b = make_bars_figure(self.summary_df)
                    save_figure_png_threadsafe(fig_b, bars_path, dpi=160)
                    try:
                        plt.close(fig_b)
                    except Exception:
                        pass

                    story.append(Paragraph("Сравнение условий: mean/max/min мощности λ(t)", styles["Heading2"]))
                    story.append(_rl_image(bars_path, max_width_cm=17.5))
                    story.append(PageBreak())
                else:
                    story.append(Paragraph("Сравнение условий", styles["Heading2"]))
                    story.append(Paragraph(self._compare_reason or "Сравнение недоступно для текущего набора файлов.", styles["Normal"]))
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

                    fig1 = make_raw_figure(t, x, fs_hz, f"{name} — {r.get('electrode_label','')}".strip(" —"), for_pdf=True)
                    save_figure_png_threadsafe(fig1, raw_path, dpi=160)
                    try:
                        plt.close(fig1)
                    except Exception:
                        pass

                    fig2 = make_psd_figure(x, fs_hz, f"{name} — {r.get('electrode_label','')}".strip(" —"), for_pdf=True)
                    save_figure_png_threadsafe(fig2, psd_path, dpi=160)
                    try:
                        plt.close(fig2)
                    except Exception:
                        pass

                    fig3 = make_lambda_figure(t, x, fs_hz, f"{name} — {r.get('electrode_label','')}".strip(" —"), for_pdf=True)
                    save_figure_png_threadsafe(fig3, lam_path, dpi=160)
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


def main():
    app = EEGApp()
    app.mainloop()


if __name__ == "__main__":
    main()