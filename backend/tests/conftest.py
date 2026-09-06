import os
import sys
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))                    # riskml
sys.path.insert(0, str(ROOT / "backend"))        # app

os.environ.setdefault("ARTIFACTS_DIR", str(ROOT / "riskml" / "artifacts"))
os.environ.setdefault("DEMO_DATA", str(ROOT / "data" / "student-por.csv"))
os.environ.setdefault("RATE_LIMIT_PER_MINUTE", "10000")

warnings.filterwarnings("ignore")


@pytest.fixture(scope="session")
def root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def client():
    from app.main import app
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def por_csv(root) -> bytes:
    return (root / "data" / "student-por.csv").read_bytes()


@pytest.fixture(scope="session")
def por_df(root):
    import pandas as pd
    return pd.read_csv(root / "data" / "student-por.csv", sep=";")
