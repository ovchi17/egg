from __future__ import annotations
import os
from typing import Tuple
import numpy as np
import pandas as pd
from config import FS_HZ_DEFAULT

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
