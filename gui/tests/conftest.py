from __future__ import annotations

import sys
from pathlib import Path


GUI_DIR = Path(__file__).resolve().parents[1]
if str(GUI_DIR) not in sys.path:
    sys.path.insert(0, str(GUI_DIR))
