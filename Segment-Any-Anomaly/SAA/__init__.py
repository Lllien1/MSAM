import sys
from pathlib import Path

# Ensure third-party submodules living inside this repo resolve without pip install.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_GROUNDINGDINO_SRC = _PROJECT_ROOT / "GroundingDINO"
if _GROUNDINGDINO_SRC.exists():
    _src = str(_GROUNDINGDINO_SRC)
    if _src not in sys.path:
        sys.path.insert(0, _src)

from .model import Model
from .hybrid_prompts import *
