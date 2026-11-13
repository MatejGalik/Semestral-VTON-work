"""Lokálny test analýzy strihu nohavíc (bez GPU a bez modelu).

Použitie (v koreni projektu):
    pip install opencv-python numpy      # ak ešte nie sú nainštalované
    python scripts/analyza_strihu.py nohavice1.jpg nohavice2.png …

Pre každú fotku nohavíc vypíše strih (úzke / rovné / široké), pomer šírky
nohavice k šírke bokov pri kolene a pri leme a celý profil šírky
(0 = rozkrok, 1 = lem). Rovnakú funkciu volá backend pri `lower`.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402

from src.masking import analyze_pants, describe_fit  # noqa: E402


def main(paths: list[str]) -> int:
    if not paths:
        print(__doc__)
        return 1
    for path in paths:
        fit = analyze_pants(Image.open(path))
        print(f"{Path(path).name}: {describe_fit(fit) if fit else 'nohavice sa nenašli'}")
        if fit:
            print("  profil:", " ".join(f"{v:.2f}" for v in fit["profile"]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
