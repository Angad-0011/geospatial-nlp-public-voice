"""Utility helpers for reading/writing files."""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict

import pandas as pd


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def log(message: str) -> None:
    print(f"[geospatial-nlp] {message}")


def save_parquet(df: pd.DataFrame, path: Path | str) -> None:
    path = Path(path)
    ensure_parent(path)
    try:
        df.to_parquet(path, index=False)
    except Exception:
        # Fallback to JSON if parquet support is unavailable
        path.write_text(df.to_json(orient="records", lines=True), encoding="utf-8")
    log(f"Saved {len(df):,} rows to {path}")


def load_parquet(path: Path | str) -> pd.DataFrame:
    path = Path(path)
    try:
        df = pd.read_parquet(path)
    except Exception:
        df = pd.read_json(path, orient="records", lines=True)
    log(f"Loaded {len(df):,} rows from {path}")
    return df


def save_json(data: Dict[str, Any], path: Path | str) -> None:
    path = Path(path)
    ensure_parent(path)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    log(f"Wrote JSON to {path}")


def load_json(path: Path | str) -> Dict[str, Any]:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8"))


def save_pickle(obj: Any, path: Path | str) -> None:
    path = Path(path)
    ensure_parent(path)
    with path.open("wb") as f:
        pickle.dump(obj, f)
    log(f"Pickle saved at {path}")


def load_pickle(path: Path | str) -> Any:
    path = Path(path)
    with path.open("rb") as f:
        return pickle.load(f)


__all__ = [
    "ensure_parent",
    "log",
    "save_parquet",
    "load_parquet",
    "save_json",
    "load_json",
    "save_pickle",
    "load_pickle",
]
