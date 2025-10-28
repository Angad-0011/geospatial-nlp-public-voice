from pathlib import Path

import pandas as pd
import pytest

from utils_io import load_parquet
from config import BASE_DIR


@pytest.fixture(scope="session")
def sample_dir() -> Path:
    return BASE_DIR / "data" / "sample"


@pytest.fixture(scope="session")
def reddit_sample(sample_dir) -> pd.DataFrame:
    return load_parquet(sample_dir / "sample_reddit.parquet")


@pytest.fixture(scope="session")
def news_sample(sample_dir) -> pd.DataFrame:
    return load_parquet(sample_dir / "sample_news.parquet")
