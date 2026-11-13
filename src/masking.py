"""Úprava masky z AutoMaskera (CatVTON: DensePose + SCHP) podľa nového oblečenia.

Postup (adjust_mask):
1. predĺženie   – dlhý vrch (tunika, dlhý sveter) → maska siaha nižšie cez boky
1b. šaty        – `overall`: maska skončí pri leme podľa dĺžky šiat a zúži sa
                  na telo (bez „kabátovej“ siluety AutoMaskera)
2. ochrana      – vyreže dlane, pri krátkom rukáve / bez rukávov aj holé ruky
3. nohavice     – `lower`: podľa strihu z fotky nohavíc (analyze_pants) sa
                  maska každej nohy rozšíri; maska AutoMaskera sa nikdy nezužuje

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

# Nohavice (lower): strih z fotky nohavíc → rozšírenie masky
DENSEPOSE_FEET = (5, 6)
DENSEPOSE_LEGS = ((7, 9, 11, 13), (8, 10, 12, 14))   # pravá / ľavá noha: stehno + lýtko
SCHP_LIP_KEEP = (2, 3, 5, 7, 13, 14, 15, 18, 19)      # vlasy, rukavice, vrch, kabát, tvár, ruky, topánky
SCHP_ATR_KEEP = (2, 4, 9, 10, 11, 14, 15, 16)         # vlasy, vrch, topánky, tvár, ruky, taška
FIT_BG_TOLERANCE = 30                 # silueta: min. rozdiel farby od pozadia (0–255)
FIT_WAIST_ROWS = 0.06                 # horných 6 % siluety (pás) sa pri šírke bokov preskočí
FIT_DEFAULT_CROTCH = 0.35             # rozkrok, ak sa nohavice na fotke nerozdelia (podiel výšky)
FIT_SAMPLES = 20                      # počet bodov profilu šírky (rozkrok → lem)
FIT_MAX_LEG = 0.8                     # strop: noha max. 80 % šírky bokov
FIT_MAX_PROFILE = 0.9                 # nohavica širšia ako 90 % bokov → nevierohodné
FIT_MAX_JUMP = 0.2                    # skok šírky medzi susednými bodmi profilu → nevierohodné
FIT_MIN_GAIN = 0.03                   # pod +3 % plochy masky sa maska nemení
FIT_LABELS = ((0.36, "úzke"), (0.46, "rovné"), (9.0, "široké"))  # podľa šírky lemu / bokov


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


def _runs(row: np.ndarray) -> list[int]:
    """Dĺžky súvislých úsekov True v riadku (napr. dve nohavice vedľa seba)."""
    d = np.diff(np.concatenate(([0], row.astype(np.int8), [0])))
    return list(np.nonzero(d == -1)[0] - np.nonzero(d == 1)[0])


def analyze_pants(cloth: Image.Image) -> dict | None:
    """Strih nohavíc z produktovej fotky – len spracovanie obrazu, žiadny model.

    1. silueta: pixely, ktoré sa farbou líšia od pozadia (medián okraja fotky),
    2. rozkrok = prvý riadok, kde sa silueta rozdelí na dve nohavice,
    3. šírka bokov = najširší riadok medzi pásom a rozkrokom,
    4. profil: šírka jednej nohavice / šírka bokov v FIT_SAMPLES bodoch od
       rozkroku po lem (0 = rozkrok, 1 = lem).
    Vráti {"profile", "knee", "hem", "label"} alebo None, ak nohavice nenájde.
    """
    img = np.asarray(cloth.convert("RGB")).astype(np.int16)
    h, w, _ = img.shape
    border = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]])
    fg = np.abs(img - np.median(border, axis=0)).max(axis=-1) > FIT_BG_TOLERANCE
    k = _kernel((w, h), 150)
    fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, k)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg)
    if n < 2:
        return None
    fg = labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    ys = np.nonzero(fg.any(axis=1))[0]
    top, bottom = int(ys.min()), int(ys.max())
    height = bottom - top + 1
    if height < 50 or fg.sum() < 0.02 * h * w:
        return None                   # príliš malá silueta

    min_run = 0.05 * float(np.ptp(np.nonzero(fg.any(axis=0))[0]) + 1)   # ignoruj úzke výbežky (šnúrky)
    split = [y for y in range(top + int(0.15 * height), bottom + 1)
             if len([r for r in _runs(fg[y]) if r >= min_run]) >= 2]
    crotch = split[0] if split else top + int(FIT_DEFAULT_CROTCH * height)
    if bottom - crotch < 0.3 * height:
        return None                   # nohavice príliš krátke (napr. sukňa, šortky)
    # Šírka bokov nohavíc = najširší riadok nad rozkrokom (pás býva stiahnutý
    # gumou alebo na fotke zakrivený, preto nie samotný pás)
    hip = float(max(sum(_runs(fg[y])) for y in range(top + int(FIT_WAIST_ROWS * height), crotch)))

    rows = np.arange(crotch, bottom + 1)
    widths = []
    for y in rows:
        r = sorted((x for x in _runs(fg[y]) if x >= min_run), reverse=True)
        widths.append(np.mean(r[:2]) if len(r) >= 2 else (r[0] / 2 if r else np.nan))
    widths = np.array(widths) / hip
    s = (rows - crotch) / max(1, bottom - crotch)
    grid = np.linspace(0, 1, FIT_SAMPLES)
    bins = np.minimum((s * FIT_SAMPLES).astype(int), FIT_SAMPLES - 1)
    profile = np.array([np.nanmedian(widths[bins == i]) if np.any((bins == i) & ~np.isnan(widths))
                        else np.nan for i in range(FIT_SAMPLES)])
    ok = ~np.isnan(profile)
    if ok.sum() < FIT_SAMPLES // 2:
        return None
    profile = np.interp(grid, grid[ok], profile[ok])
    # Kontrola vierohodnosti: nohavica širšia ako boky alebo skoky v šírke →
    # nie je to produktová fotka nohavíc (napr. osoba, zložené nohavice)
    if profile.max() > FIT_MAX_PROFILE or np.abs(np.diff(profile)).max() > FIT_MAX_JUMP:
        return None
    profile = np.convolve(np.pad(profile, 1, mode="edge"), np.ones(3) / 3, mode="valid")
    hem = float(profile[-5:-1].mean())       # bez úplného spodku (zaoblený lem, manžeta)
    label = next(name for limit, name in FIT_LABELS if hem < limit)
    return {"profile": profile, "knee": float(profile[FIT_SAMPLES // 2]), "hem": hem, "label": label}


def describe_fit(fit: dict | None) -> str:
    """Text do `info`, napr. "široké (koleno 0.52, lem 0.61)"."""
    return f'{fit["label"]} (koleno {fit["knee"]:.2f}, lem {fit["hem"]:.2f})' if fit else "-"


def _widen_legs(m: np.ndarray, densepose: np.ndarray, keep: np.ndarray, fit: dict) -> float:
    """Rozšíri masku nohavíc podľa profilu strihu. Vráti prírastok plochy (podiel).

    Pre každú nohu (DensePose stehno + lýtko) a každý riadok od vrchu stehna po
    členok: stred nohy z DensePose, šírka = profil(výška) × šírka bokov osoby.
    Pridá sa len to, čo nie je v `keep` (ruky, vrch, topánky…); nič sa neuberá.
    """
    h, w = m.shape
    thighs = np.isin(densepose, DENSEPOSE_THIGHS)
    ys = np.nonzero(thighs.any(axis=1))[0]
    if ys.size == 0:
        return 0.0                    # DensePose nenašiel nohy
    top_rows = thighs[ys.min(): ys.min() + max(2, int(0.15 * (ys.max() - ys.min())))]
    xs = np.nonzero(top_rows.any(axis=0))[0]
    hip = float(xs.max() - xs.min() + 1)  # šírka bokov = šírka oboch stehien hore

    grid = np.linspace(0, 1, len(fit["profile"]))
    added = np.zeros_like(m)
    for leg_ids in DENSEPOSE_LEGS:
        leg = np.isin(densepose, leg_ids)
        rows = np.nonzero(leg.any(axis=1))[0]
        if rows.size < 10:
            continue                  # noha nie je na fotke
        y0, y1 = int(rows.min()), int(rows.max())
        ys_all = np.arange(y0, y1 + 1)
        centers = np.array([np.nonzero(leg[y])[0].mean() if leg[y].any() else np.nan for y in ys_all])
        ok = ~np.isnan(centers)
        centers = np.interp(ys_all, ys_all[ok], centers[ok])
        centers = np.convolve(np.pad(centers, 7, mode="edge"), np.ones(15) / 15, mode="valid")
        half = np.minimum(np.interp((ys_all - y0) / max(1, y1 - y0), grid, fit["profile"]), FIT_MAX_LEG) * hip / 2
        for y, c, hw in zip(ys_all, centers, half):
            added[y, max(0, int(c - hw)): min(w, int(c + hw) + 1)] = True

    before = m.sum()
    new = m | (added & ~keep)
    gain = float(new.sum() - before) / max(1, before)
    if gain >= FIT_MIN_GAIN:
        m |= new
        return gain
    return 0.0


def adjust_mask(
    mask: Image.Image,
    parts: dict,
    cloth_type: str,
    protect_arms: bool = False,
    long_garment: bool = False,
    dress_length: str | None = None,
    pants_fit: dict | None = None,
) -> tuple[Image.Image, list[str], list[str]]:
    """Upraví masku AutoMaskera. `parts` = výstup AutoMaskera (densepose, schp_lip, schp_atr).

    `pants_fit` = výsledok analyze_pants() pre `lower` (None → maska bez zmeny).
    Vráti (maska, chránené časti, vykonané úpravy).
    """
    size = mask.size
    m = np.array(mask.convert("L")) > 127
    densepose = _as_array(parts["densepose"], size)
    lip = _as_array(parts["schp_lip"], size)
    atr = _as_array(parts["schp_atr"], size)
    upper = cloth_type in ("upper", "overall")
    changes: list[str] = []

    # Spodok – maska AutoMaskera sa len rozšíri podľa strihu (široké nohavice)
    if not upper:
        if pants_fit is None:
            return mask, [], changes
        hands = cv2.dilate(np.isin(densepose, DENSEPOSE_HANDS + DENSEPOSE_FEET).astype(np.uint8),
                           _kernel(size, 250), iterations=PROTECT_DILATE) > 0
        keep = hands | np.isin(lip, SCHP_LIP_KEEP) | np.isin(atr, SCHP_ATR_KEEP)
        gain = _widen_legs(m, densepose, keep, pants_fit)
        if not gain:
            return mask, [], changes
        changes.append(f"nohavice rozšírené (+{gain:.0%})")
        return Image.fromarray((m * 255).astype(np.uint8)), [], changes

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
