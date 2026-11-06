"""Úprava masky z AutoMaskera (CatVTON: DensePose + SCHP) podľa nového oblečenia.

Postup (adjust_mask):
1. predĺženie   – dlhý vrch (tunika, dlhý sveter) → maska siaha nižšie cez boky
2. šírka        – posuvník v UI: + roztiahne masku (viac priestoru na iný strih),
                  − zúži ju (menej sa zmení okolie)
3. ochrana      – vyreže dlane, pri krátkom rukáve / bez rukávov aj holé ruky,
                  pri roztiahnutí aj tvár a vlasy

Modul neimportuje kód CatVTON – mapy tried mu odovzdáva src/try_on.py.
Masky sú PIL "L" 768×1024 (0 / 255), ešte bez rozmazania okraja.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

PROTECT_DILATE = 2                    # roztiahnutie chránenej oblasti rúk (iterácie)
LONG_EXTEND = 0.35                    # dlhý vrch: maska pokračuje o 35 % výšky trupu nižšie
LONG_SIDE_MARGIN = 0.15               # … a do šírky o 15 % šírky trupu na každú stranu

# Indexy tried (ako v CatVTON model/cloth_masker.py)
DENSEPOSE_TORSO = (1, 2)
DENSEPOSE_HANDS = (3, 4)
SCHP_ARMS = (14, 15)                  # Left-arm, Right-arm – v LIP aj ATR
SCHP_LIP_HEAD = (2, 13)               # Hair, Face
SCHP_ATR_HEAD = (2, 11)               # Hair, Face


def _kernel(size: tuple[int, int], divisor: int) -> np.ndarray:
    k = max(size) // divisor
    k = k if k % 2 == 1 else k + 1
    return np.ones((k, k), np.uint8)


def _as_array(img: Image.Image, size: tuple[int, int]) -> np.ndarray:
    """Mapu tried zjednotí na rozmer masky – bez interpolácie tried."""
    if img.size != size:
        img = img.resize(size, Image.NEAREST)
    return np.array(img)


def _extend_long(m: np.ndarray, densepose: np.ndarray) -> bool:
    """Predĺži masku pod trup (obdĺžnik cez boky). Vráti True, ak sa predĺžila."""
    ys, xs = np.nonzero(np.isin(densepose, DENSEPOSE_TORSO))
    if ys.size == 0:
        return False                  # DensePose nenašiel trup
    top, bottom, left, right = ys.min(), ys.max(), xs.min(), xs.max()
    height, width = bottom - top, right - left
    h, w = m.shape
    y0 = top + height // 2            # od polovice trupu (nadviaže na existujúcu masku)
    y1 = min(h, int(bottom + LONG_EXTEND * height))
    x0 = max(0, int(left - LONG_SIDE_MARGIN * width))
    x1 = min(w, int(right + LONG_SIDE_MARGIN * width))
    m[y0:y1, x0:x1] = True
    return True


def _grow(m: np.ndarray, px: int) -> np.ndarray:
    """Roztiahne (px > 0) alebo zúži (px < 0) masku o px pixelov."""
    if px == 0:
        return m
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * abs(px) + 1,) * 2)
    op = cv2.dilate if px > 0 else cv2.erode
    return op(m.astype(np.uint8), k) > 0


def adjust_mask(
    mask: Image.Image,
    parts: dict,
    cloth_type: str,
    protect_arms: bool = False,
    long_garment: bool = False,
    grow: int = 0,
) -> tuple[Image.Image, list[str], list[str]]:
    """Upraví masku AutoMaskera. `parts` = výstup AutoMaskera (densepose, schp_lip, schp_atr).

    Vráti (maska, chránené časti, vykonané úpravy).
    """
    size = mask.size
    m = np.array(mask.convert("L")) > 127
    densepose = _as_array(parts["densepose"], size)
    lip = _as_array(parts["schp_lip"], size)
    atr = _as_array(parts["schp_atr"], size)
    upper = cloth_type in ("upper", "overall")
    changes: list[str] = []

    # Spodok bez zmeny šírky – maska AutoMaskera ostáva presne ako bola
    if not upper and grow == 0:
        return mask, [], changes

    # 1) Predĺženie pre dlhý vrch
    if cloth_type == "upper" and long_garment and _extend_long(m, densepose):
        changes.append("predĺžená")

    # 2) Šírka masky
    if grow:
        m = _grow(m, grow)
        changes.append(f"šírka {grow:+d} px")

    # 3) Ochrana: dlane vždy, holé ruky pri krátkom rukáve (a pri spodku vždy)
    kernel = _kernel(size, 250)       # rovnaká veľkosť ako v AutoMaskeri
    skin = np.isin(lip, SCHP_ARMS) | np.isin(atr, SCHP_ARMS)
    hands = cv2.dilate(np.isin(densepose, DENSEPOSE_HANDS).astype(np.uint8), kernel) > 0
    protect = hands & skin
    protected = ["dlane"]
    if (upper and protect_arms) or cloth_type == "lower":
        protect |= skin
        protected.append("ruky")
    protect = cv2.dilate(protect.astype(np.uint8), kernel, iterations=PROTECT_DILATE) > 0

    # Tvár a vlasy – len pri roztiahnutí (bez dilatácie, aby ostal golier v maske)
    if grow > 0:
        protect |= np.isin(lip, SCHP_LIP_HEAD) | np.isin(atr, SCHP_ATR_HEAD)
        protected.append("tvár")

    m &= ~protect
    return Image.fromarray((m * 255).astype(np.uint8)), protected, changes


def mask_ratio(mask: Image.Image) -> float:
    """Podiel pixelov v maske (0–1)."""
    return round(float((np.array(mask.convert("L")) > 127).mean()), 3)
