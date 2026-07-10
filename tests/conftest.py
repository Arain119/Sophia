from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT

if (PACKAGE_ROOT / "ml").is_dir():
    src = str(PACKAGE_ROOT)
    if src not in sys.path:
        sys.path.insert(0, src)
    existing = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = src if not existing else src + os.pathsep + existing
