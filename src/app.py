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
# 10-20 selector
# ---------------------------
class TenTwentySelector(ttk.Frame):
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

        self.title("# Лямбда-ритмы ЭЭГ при разных воздействиях")
        self.geometry("1240x820")
        self.configure(bg=UI["bg"])

        self.serial_queue: "queue.Queue[tuple[Any, Any]]" = queue.Queue()
        self.streamer: Optional[ArduinoSerialStreamer] = None
        self.live_t: List[float] = []
        self.live_x: List[float] = []
        self.live_max_sec = 10.0

        self.loaded_files: List[str] = []
        self._last_records: Optional[List[Dict[str, Any]]] = None
        self._last_fs_user = FS_HZ_DEFAULT

        self.eeg_montage = tk.StringVar(value="O1–Oz–O2 (затылочная область)")
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

        self._setup_style()
        self._build_ui()

        self.after(60, self._poll_serial_queue)
        self.after(60, self._poll_ui_queue)

    # ------- style -------
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

    # ------- UI -------
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

        self.btn_stop = ttk.Button(card, text="■ Стоп", command=self.stop_stream, state="disabled", style="Danger.TButton")
        self.btn_stop.pack(side="left", padx=(0, 8))

        self.btn_save = ttk.Button(card, text="💾 Сохранить CSV", command=self.save_live_csv, state="disabled", style="Ghost.TButton")
        self.btn_save.pack(side="left")

        plot_card = ttk.Frame(root, padding=14, style="Card.TFrame")
        plot_card.pack(fill="both", expand=True, pady=(12, 0))

        from matplotlib.figure import Figure
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

        # Drag&Drop
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
                in_brace = True; buf = ""
            elif ch == "}":
                in_brace = False; out.append(buf); buf = ""
            elif ch == " " and not in_brace:
                if buf:
                    out.append(buf); buf = ""
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

    def remove_selected_file(self):
        sel = self.lst_files.curselection()
        if not sel:
            return
        idx = sel[0]
        path = self.lst_files.get(idx)
        self.lst_files.delete(idx)
        self.loaded_files = [p for p in self.loaded_files if p != path]
        self._refresh_files_count()

    def clear_file_list(self):
        self.lst_files.delete(0, "end")
        self.loaded_files.clear()
        self._refresh_files_count()

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

        ttk.Label(controls, text="Расположение электродов:", style="Muted.TLabel").pack(side="left", padx=(8, 8))
        self.cbo_montage = ttk.Combobox(
            controls, textvariable=self.eeg_montage, state="readonly", width=30,
            values=[
                "O1–Oz–O2 (затылочная область)",
                "Pz (теменно-затылочная область)",
                "T5/T6 (височно-затылочная область)",
                "Другое (указать в выводах)",
            ],
        )
        self.cbo_montage.pack(side="left", padx=(0, 14))

        self.btn_run = ttk.Button(controls, text="▶ Запустить анализ", command=self.run_lab5, style="Primary.TButton")
        self.btn_run.pack(side="left", padx=(0, 8))

        self.btn_report = ttk.Button(controls, text="📄 Экспорт PDF", command=self.export_report_pdf, style="Ghost.TButton")
        self.btn_report.pack(side="left", padx=(0, 12))

        self.pb = ttk.Progressbar(controls, mode="indeterminate", length=180)
        self.pb.pack(side="left", padx=(0, 10))
        self.lbl_an_status = ttk.Label(controls, text="Готово", style="Muted.TLabel")
        self.lbl_an_status.pack(side="left")

        body = ttk.PanedWindow(root, orient="horizontal")
        body.pack(fill="both", expand=True, pady=(12, 0))

        left = ttk.Frame(body); right = ttk.Frame(body)
        body.add(left, weight=2)
        body.add(right, weight=5)

        # left: table + 10-20
        left_card = ttk.Frame(left, padding=14, style="Card.TFrame")
        left_card.pack(fill="both", expand=True)
        ttk.Label(left_card, text="Таблица результатов", style="H2.TLabel").pack(anchor="w")

        self.cbo_table = ttk.Combobox(
            left_card, state="readonly",
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

        # right: plots
        right_card = ttk.Frame(right, padding=14, style="Card.TFrame")
        right_card.pack(fill="both", expand=True)

        seg = ttk.Frame(right_card, padding=10, style="Card2.TFrame")
        seg.pack(fill="x", pady=(0, 12))
        ttk.Label(seg, text="Режим отображения:", style="Muted.TLabel").pack(side="left", padx=(0, 10))

        self.plot_mode = tk.StringVar(value="RAW")
        for key, label in [("RAW", "Сигнал"), ("PSD", "Спектр (PSD)"), ("LAMBDA", "λ-ритм"), ("BARS", "Сравнение")]:
            ttk.Radiobutton(seg, text=label, value=key, variable=self.plot_mode,
                            style="Seg.TRadiobutton", command=self._render_plots).pack(side="left", padx=(0, 8))

        self.plot_host = ttk.Frame(right_card, style="Card2.TFrame")
        self.plot_host.pack(fill="both", expand=True)
        self.plot_area = ScrollablePlotArea(self.plot_host)
        self.plot_area.pack(fill="both", expand=True)

        # conclusions
        concl_card = ttk.Frame(root, padding=14, style="Card.TFrame")
        concl_card.pack(fill="x", pady=(12, 0))
        ttk.Label(concl_card, text="Анализ и выводы", style="H2.TLabel").pack(anchor="w", pady=(0, 8))

        self.txt_conclusions = tk.Text(
            concl_card, height=10, bg=UI["panel2"], fg=UI["text"],
            insertbackground=UI["text"], highlightthickness=1,
            highlightbackground=UI["border"], bd=0, wrap="word",
        )
        self.txt_conclusions.pack(fill="both", expand=True)
        self.txt_conclusions.insert("1.0", "Запустите анализ, чтобы сформировать выводы.")
        self.txt_conclusions.config(state="disabled")

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
            records: List[Dict[str, Any]] = []
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
                    "montage": self.eeg_montage.get(),
                    "channel_hint": self.eeg_channel_hint.get(),
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
                band_power_df.merge(lambda_time_df, on=["Файл"])
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

        for _, row in show_df.iterrows():
            self.tbl.insert("", "end", values=[row[c] for c in cols])

    def _render_plots(self):
        self.plot_area.clear()
        if self._last_records is None:
            ttk.Label(self.plot_area.inner, text="Запустите анализ, чтобы увидеть графики.", style="Muted.TLabel").pack(padx=12, pady=12, anchor="w")
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

                bars_path = os.path.join(tmpdir, "bars.png")
                fig_b = make_bars_figure(self.summary_df)
                save_figure_png_threadsafe(fig_b, bars_path, dpi=160)
                try: plt.close(fig_b)
                except Exception: pass

                story.append(Paragraph("Сравнение условий: mean/max/min мощности λ(t)", styles["Heading2"]))
                story.append(_rl_image(bars_path, max_width_cm=17.5))
                story.append(PageBreak())

                for idx, r in enumerate(self._last_records, start=1):
                    self._ui_queue.put(("status", f"PDF: файл {idx}/{len(self._last_records)}…"))

                    name = r["name"]
                    t = r["t"]; x = r["x"]; fs_hz = r["fs"]

                    story.append(Paragraph(f"Файл: {name}", styles["Heading2"]))
                    story.append(Paragraph(f"Канал: {r['sig_col']} | FS: {fs_hz}", styles["Normal"]))
                    story.append(Spacer(1, 0.25 * cm))

                    raw_path = os.path.join(tmpdir, f"{name}_raw.png")
                    psd_path = os.path.join(tmpdir, f"{name}_psd.png")
                    lam_path = os.path.join(tmpdir, f"{name}_lambda.png")

                    fig1 = make_raw_figure(t, x, fs_hz, name, for_pdf=True)
                    save_figure_png_threadsafe(fig1, raw_path, dpi=160)
                    try: plt.close(fig1)
                    except Exception: pass

                    fig2 = make_psd_figure(x, fs_hz, name, for_pdf=True)
                    save_figure_png_threadsafe(fig2, psd_path, dpi=160)
                    try: plt.close(fig2)
                    except Exception: pass

                    fig3 = make_lambda_figure(t, x, fs_hz, name, for_pdf=True)
                    save_figure_png_threadsafe(fig3, lam_path, dpi=160)
                    try: plt.close(fig3)
                    except Exception: pass

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
