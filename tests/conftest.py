from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def ts_ndvi(data_dir):
    yield (-3000, np.loadtxt(data_dir / "ts-ndvi.txt", dtype="int16"))
