# -*- coding: utf-8 -*-
"""Loading and profiling CSV / Excel datasets, plus column typing helpers."""
import os

import numpy as np
import pandas as pd

FILE_EXT = {".csv", ".xlsx", ".xls", ".data"}


def load_file(path, sheet=None):
    """Load a CSV / Excel file into a pandas DataFrame (first sheet default)."""
    path = str(path)
    lower = path.lower()
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    if lower.endswith(".csv"):
        df = pd.read_csv(path)
    elif lower.endswith((".xlsx", ".xls")):
        df = pd.read_excel(path, sheet_name=sheet if sheet is not None else 0)
    elif lower.endswith(".data"):
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {path}. Use CSV or Excel.")
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def excel_sheets(path):
    lower = str(path).lower()
    if lower.endswith((".xlsx", ".xls")):
        try:
            xl = pd.ExcelFile(path)
            return list(xl.sheet_names)
        except Exception:
            return []
    return []


def profile_columns(df):
    """Return per-column metadata: dtype class, unique count, missing %, sample."""
    rows = df.shape[0]
    out = {}
    for c in df.columns:
        s = df[c]
        nun = s.nunique(dropna=True)
        missing = int(s.isna().sum())
        name = s.name
        kind = "numeric"
        if pd.api.types.is_numeric_dtype(s):
            kind = "numeric"
        elif isinstance(s.dtype, pd.Series.dtype.__class__) and pd.api.types.is_datetime64_any_dtype(s):
            kind = "datetime"
        elif hasattr(s, "dt"):
            kind = "datetime"
        else:
            if nun <= 30:
                kind = "categorical"
            else:
                kind = "text"
        out[c] = {
            "dtype": str(s.dtype),
            "unique": int(nun),
            "missing": missing,
            "missing_pct": round(100.0 * missing / rows, 2) if rows else 0.0,
            "kind": kind,
            "sample": [str(v) for v in s.dropna().head(3).tolist()],
        }
    return out


def is_problematic_target(df, target):
    """A column can only be a target if it has at least 2 valid values."""
    if target not in df.columns:
        return True
    return df[target].nunique(dropna=True) < 2


def parse_contents(contents, filename, sheet=None):
    """Decode a base64 `dcc.Upload` payload into a DataFrame.

    Returns (df, None) or (None, error_message).
    """
    import base64
    import io

    if contents is None:
        return None, "No file provided"
    try:
        content_type, content_string = contents.split(",", 1)
        decoded = base64.b64decode(content_string)
        lower = (filename or "").lower()
        if lower.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(decoded))
        elif lower.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(decoded), sheet_name=sheet if sheet is not None else 0)
        else:
            return None, "Unsupported file type - use CSV or Excel"
        df.columns = [str(c).strip() for c in df.columns]
        return df, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"