# utils_io.py
from __future__ import annotations
from pathlib import Path
import json
import joblib
import pandas as pd

__all__ = [
    "save_json", "load_json",
    "save_df", "load_df",
    "save_pickle", "load_pickle",
    "log",
]

def log(msg: str) -> None:
    print(f"[geospatial] {msg}")

def save_json(obj, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path: Path | str):
    with open(Path(path), "r", encoding="utf-8") as f:
        return json.load(f)

def save_df(df: pd.DataFrame, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # use pyarrow (fast + robust)
    df.to_parquet(path, index=False, engine="pyarrow")

def load_df(path: Path | str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path, engine="pyarrow")

def save_pickle(obj, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)

def load_pickle(path: Path | str):
    return joblib.load(path)
