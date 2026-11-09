"""Úprava masky z AutoMaskera (CatVTON: DensePose + SCHP) podľa nového oblečenia.

Postup (adjust_mask):
1. predĺženie   – dlhý vrch (tunika, dlhý sveter) → maska siaha nižšie cez boky
1b. šaty        – `overall`: maska skončí pri leme podľa dĺžky šiat a zúži sa
                  na telo (bez „kabátovej“ siluety AutoMaskera)
2. ochrana      – vyreže dlane, pri krátkom rukáve / bez rukávov aj holé ruky

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

# Šaty (overall): lem podľa dĺžky, relatívne k DensePose (bok → koleno → členok)
DRESS_HEM = {                         # (úsek, podiel úseku)
    "mini": ("thigh", 0.6),           # 60 % stehna pod bokom
    "knee": ("shin", 0.1),            # tesne pod kolenom
    "midi": ("shin", 0.5),            # polovica lýtka
    "maxi": None,                     # dĺžka sa nemení
}
DRESS_HEM_MARGIN = 0.05               # rezerva pod lemom (podiel výšky trupu)
DRESS_SKIRT_MARGIN = 0.25             # sukňa: do šírky o 25 % šírky trupu na každú stranu
DRESS_BODY_DIVISOR = 40               # roztiahnutie tela: jadro max(rozmer)/40 ≈ 25 px

# Indexy tried (ako v CatVTON model/cloth_masker.py)
DENSEPOSE_TORSO = (1, 2)
DENSEPOSE_HANDS = (3, 4)
DENSEPOSE_THIGHS = (7, 8, 9, 10)
DENSEPOSE_SHINS = (11, 12, 13, 14)
DENSEPOSE_ARMS = tuple(range(15, 23))   # nadlaktie + predlaktie
SCHP_ARMS = (14, 15)                  # Left-arm, Right-arm – v LIP aj ATR


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


def _dress_shape(m: np.ndarray, densepose: np.ndarray, length: str | None) -> str | None:
    """Šaty: maska po lem + len telo a sukňa. Vráti popis úpravy alebo None."""
    torso = np.isin(densepose, DENSEPOSE_TORSO)
    if not torso.any():
        return None                   # DensePose nenašiel trup
    ys, xs = np.nonzero(torso)
    top, hip, left, right = ys.min(), ys.max(), xs.min(), xs.max()
    torso_h, torso_w = hip - top, right - left
    h, w = m.shape

    thigh_ys = np.nonzero(np.isin(densepose, DENSEPOSE_THIGHS))[0]
    shin_ys = np.nonzero(np.isin(densepose, DENSEPOSE_SHINS))[0]
    knee = shin_ys.min() if shin_ys.size else (thigh_ys.max() if thigh_ys.size else None)
    ankle = shin_ys.max() if shin_ys.size else None

    hem = h                           # bez rezu (maxi, alebo nohy nie sú na fotke)
    rule = DRESS_HEM.get(length or "")
    if rule and knee is not None:
        part, ratio = rule
        if part == "thigh":
            hem = hip + ratio * (knee - hip)
        elif ankle is not None:
            hem = knee + ratio * (ankle - knee)
        hem = min(h, int(hem + DRESS_HEM_MARGIN * torso_h))

    # Povolená oblasť: roztiahnuté telo (trup, ruky, nohy) + sukňa od pása po lem
    body = np.isin(densepose, DENSEPOSE_TORSO + DENSEPOSE_ARMS + DENSEPOSE_THIGHS + DENSEPOSE_SHINS)
    allowed = cv2.dilate(body.astype(np.uint8), _kernel((w, h), DRESS_BODY_DIVISOR)) > 0
    waist = int(hip - 0.25 * torso_h)
    x0 = max(0, int(left - DRESS_SKIRT_MARGIN * torso_w))
    x1 = min(w, int(right + DRESS_SKIRT_MARGIN * torso_w))
    allowed[waist:hem, x0:x1] = True
    allowed[hem:] = False
    m &= allowed
    return f"šaty: {length or '?'}" + ("" if hem < h else " (bez rezu)")


def adjust_mask(
    mask: Image.Image,
    parts: dict,
    cloth_type: str,
    protect_arms: bool = False,
    long_garment: bool = False,
    dress_length: str | None = None,
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

    # Spodok – maska AutoMaskera ostáva presne ako bola
    if not upper:
        return mask, [], changes

    # 1) Predĺženie pre dlhý vrch
    if cloth_type == "upper" and long_garment and _extend_long(m, densepose):
        changes.append("predĺžená")

    # 1b) Šaty: dĺžka po lem a tvar tela namiesto siluety AutoMaskera
    if cloth_type == "overall" and (change := _dress_shape(m, densepose, dress_length)):
        changes.append(change)

    # 2) Ochrana: dlane vždy, holé ruky pri krátkom rukáve / bez rukávov
    kernel = _kernel(size, 250)       # rovnaká veľkosť ako v AutoMaskeri
    skin = np.isin(lip, SCHP_ARMS) | np.isin(atr, SCHP_ARMS)
    hands = cv2.dilate(np.isin(densepose, DENSEPOSE_HANDS).astype(np.uint8), kernel) > 0
    protect = hands & skin
    protected = ["dlane"]
    if protect_arms:
        protect |= skin
        protected.append("ruky")
    # Pri odhalených rukách menšie roztiahnutie – inak ostane pri dlani kúsok
    # pôvodného rukáva (manžeta) a model podľa neho dokreslí dlhý rukáv.
    iterations = 1 if protect_arms else PROTECT_DILATE
    protect = cv2.dilate(protect.astype(np.uint8), kernel, iterations=iterations) > 0
    m &= ~protect
    return Image.fromarray((m * 255).astype(np.uint8)), protected, changes


def mask_ratio(mask: Image.Image) -> float:
    """Podiel pixelov v maske (0–1)."""
    return round(float((np.array(mask.convert("L")) > 127).mean()), 3)
