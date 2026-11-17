"""Lokálny test rozpoznania typu oblečenia pravidlami (bez GPU a bez AI).

Použitie (v koreni projektu):
    pip install opencv-python numpy      # ak ešte nie sú nainštalované
    python scripts/typ_oblecenia.py tricko.jpg nohavice.png saty.jpg …

Pre každú fotku vypíše typ (upper / lower / overall, alebo „neisté“ → backend
by použil CLIP), dôvod a číselné znaky siluety. Rovnakú funkciu volá backend
pri cloth_type="auto".
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402

from src.cloth_type import decide, features  # noqa: E402
from src.masking import silhouette  # noqa: E402


def main(paths: list[str]) -> int:
    if not paths:
        print(__doc__)
        return 1
    for path in paths:
        fg = silhouette(Image.open(path))
        if fg is None:
            print(f"{Path(path).name}: silueta sa nenašla")
            continue
        f = features(fg)
        label, reason = decide(f)
        print(f"{Path(path).name}: {label or 'neisté'} – {reason}")
        print("  " + "  ".join(f"{k} {v:.2f}" if isinstance(v, float) else f"{k} {v}" for k, v in f.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
