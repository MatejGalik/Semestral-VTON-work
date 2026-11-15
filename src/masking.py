"""Úprava masky z AutoMaskera (CatVTON: DensePose + SCHP) podľa nového oblečenia.

Postup (adjust_mask):
1. predĺženie   – dlhý vrch (tunika, dlhý sveter) → maska siaha nižšie cez boky
1b. šaty        – `overall`: maska skončí pri leme podľa dĺžky šiat a zúži sa
                  na telo (bez „kabátovej“ siluety AutoMaskera)
2. ochrana      – vyreže dlane, pri krátkom rukáve / bez rukávov aj holé ruky
3. nohavice     – `lower`: ak sú nové nohavice (analyze_pants) širšie ako
                  pôvodné na osobe, maska sa smerom von rozšíri o rozdiel;
                  maska AutoMaskera sa nikdy nezužuje

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
SCHP_LIP_PANTS = (9, 12)              # pôvodné nohavice / sukňa na osobe
SCHP_ATR_PANTS = (5, 6)
FIT_BG_TOLERANCE = 30                 # silueta: min. rozdiel farby od pozadia (0–255)
FIT_WAIST_ROWS = 0.06                 # horných 6 % siluety (pás) sa pri šírke bokov preskočí
FIT_DEFAULT_CROTCH = 0.35             # rozkrok, ak sa nohavice na fotke nerozdelia (podiel výšky)
FIT_SAMPLES = 20                      # počet bodov profilu šírky (rozkrok → lem)
FIT_MAX_EXTRA = 0.4                   # strop: okraj sa posunie max. o 40 % šírky bokov na stranu
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


def _runs(row: np.ndarray) -> list[tuple[int, int]]:
    """Súvislé úseky True v riadku ako (začiatok, dĺžka) – napr. dve nohavice vedľa seba."""
    d = np.diff(np.concatenate(([0], row.astype(np.int8), [0])))
    starts, ends = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
    return list(zip(starts, ends - starts))


def _leg_profile(fg: np.ndarray) -> dict | None:
    """Profil nohavíc zo siluety (fotka nohavíc aj nohavice na osobe zo SCHP).

    rozkrok = prvý riadok, kde sa silueta rozdelí na dve nohavice,
    boky = najširší riadok medzi pásom a rozkrokom (pás býva stiahnutý gumou),
    pre každý riadok od rozkroku po lem: šírka jednej nohavice / boky
    a vonkajšie okraje siluety (vľavo, vpravo).
    """
    ys = np.nonzero(fg.any(axis=1))[0]
    if ys.size < 50:
        return None
    top, bottom = int(ys.min()), int(ys.max())
    height = bottom - top + 1
    min_run = 0.05 * float(np.ptp(np.nonzero(fg.any(axis=0))[0]) + 1)  # ignoruj úzke výbežky
    split = [y for y in range(top + int(0.15 * height), bottom + 1)
             if len([r for r in _runs(fg[y]) if r[1] >= min_run]) >= 2]
    crotch = split[0] if split else top + int(FIT_DEFAULT_CROTCH * height)
    if bottom - crotch < 0.3 * height:
        return None                   # príliš krátke (sukňa, šortky)
    hip = float(max(sum(r[1] for r in _runs(fg[y])) for y in range(top + int(FIT_WAIST_ROWS * height), crotch)))

    rows = np.arange(crotch, bottom + 1)
    widths, left, right = [], [], []
    for y in rows:
        r = sorted((x for x in _runs(fg[y]) if x[1] >= min_run), key=lambda x: -x[1])
        widths.append(np.mean([x[1] for x in r[:2]]) if len(r) >= 2 else (r[0][1] / 2 if r else np.nan))
        left.append(min(x[0] for x in r) if r else np.nan)
        right.append(max(x[0] + x[1] for x in r) if r else np.nan)
    return {"top": top, "crotch": crotch, "bottom": bottom, "hip": hip, "rows": rows,
            "ratio": np.array(widths) / hip, "left": np.array(left, float), "right": np.array(right, float)}


def _fill_nan(a: np.ndarray) -> np.ndarray:
    ok = ~np.isnan(a)
    return np.interp(np.arange(a.size), np.nonzero(ok)[0], a[ok]) if ok.any() else a


def _smooth(a: np.ndarray, n: int) -> np.ndarray:
    return np.convolve(np.pad(a, n // 2, mode="edge"), np.ones(n) / n, mode="valid")


def analyze_pants(cloth: Image.Image) -> dict | None:
    """Strih nohavíc z produktovej fotky – len spracovanie obrazu, žiadny model.

    1. silueta: pixely, ktoré sa farbou líšia od pozadia (medián okraja fotky),
    2. profil (_leg_profile): šírka jednej nohavice / šírka bokov
       v FIT_SAMPLES bodoch od rozkroku (0) po lem (1).
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
    if fg.sum() < 0.02 * h * w:
        return None                   # príliš malá silueta
    leg = _leg_profile(fg)
    if leg is None:
        return None

    s = (leg["rows"] - leg["crotch"]) / max(1, leg["bottom"] - leg["crotch"])
    grid = np.linspace(0, 1, FIT_SAMPLES)
    bins = np.minimum((s * FIT_SAMPLES).astype(int), FIT_SAMPLES - 1)
    ratio = leg["ratio"]
    profile = np.array([np.nanmedian(ratio[bins == i]) if np.any((bins == i) & ~np.isnan(ratio))
                        else np.nan for i in range(FIT_SAMPLES)])
    ok = ~np.isnan(profile)
    if ok.sum() < FIT_SAMPLES // 2:
        return None
    profile = np.interp(grid, grid[ok], profile[ok])
    # Kontrola vierohodnosti: nohavica širšia ako boky alebo skoky v šírke →
    # nie je to produktová fotka nohavíc (napr. osoba, zložené nohavice)
    if profile.max() > FIT_MAX_PROFILE or np.abs(np.diff(profile)).max() > FIT_MAX_JUMP:
        return None
    profile = _smooth(profile, 3)
    hem = float(profile[-5:-1].mean())       # bez úplného spodku (zaoblený lem, manžeta)
    label = next(name for limit, name in FIT_LABELS if hem < limit)
    return {"profile": profile, "knee": float(profile[FIT_SAMPLES // 2]), "hem": hem, "label": label}


def describe_fit(fit: dict | None) -> str:
    """Text do `info`, napr. "široké (koleno 0.52, lem 0.61)"."""
    return f'{fit["label"]} (koleno {fit["knee"]:.2f}, lem {fit["hem"]:.2f})' if fit else "-"


def _widen_legs(m: np.ndarray, old: np.ndarray, keep: np.ndarray, fit: dict) -> float:
    """Rozšíri masku nohavíc o rozdiel strihu nových a pôvodných nohavíc.

    `old` = pôvodné nohavice na osobe (SCHP), prípadne nohy z DensePose.
    Ten istý profil (_leg_profile) sa zmeria aj na nich; v každom riadku od
    rozkroku po lem sa vonkajší okraj nohavíc posunie von o
    (nový pomer − pôvodný pomer) × šírka bokov. Pridá sa len to, čo nie je
    v `keep` (ruky, vrch, topánky…); nič sa neuberá. Vráti prírastok plochy.
    """
    leg = _leg_profile(old)
    if leg is None:
        return 0.0
    w = m.shape[1]
    rows, hip = leg["rows"], leg["hip"]
    s = (rows - leg["crotch"]) / max(1, leg["bottom"] - leg["crotch"])
    new_ratio = np.interp(s, np.linspace(0, 1, len(fit["profile"])), fit["profile"])
    extra = np.clip(new_ratio - _fill_nan(leg["ratio"]), 0, FIT_MAX_EXTRA) * hip
    k = max(15, int(0.08 * len(rows)) | 1)       # vyhladenie okraja (~8 % dĺžky nohy)
    extra = _smooth(extra, k)
    left, right = _smooth(_fill_nan(leg["left"]), k), _smooth(_fill_nan(leg["right"]), k)

    added = np.zeros_like(m)
    for y, l, r, e in zip(rows, left, right, extra):
        if e >= 1:
            added[y, max(0, int(l - e)): min(w, int(r + e) + 1)] = True

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
        # Pôvodné nohavice na osobe (SCHP); ak ich niet (šaty, šortky…), nohy z DensePose
        old = np.isin(lip, SCHP_LIP_PANTS) | np.isin(atr, SCHP_ATR_PANTS)
        if old.sum() < 0.01 * old.size:
            old = np.isin(densepose, DENSEPOSE_LEGS[0] + DENSEPOSE_LEGS[1])
        gain = _widen_legs(m, old, keep, pants_fit)
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
