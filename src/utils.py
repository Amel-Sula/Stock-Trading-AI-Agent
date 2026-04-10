from __future__ import annotations
from pathlib import Path
import random
import numpy as np

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
