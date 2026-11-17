"""Úprava masky z AutoMaskera (CatVTON: DensePose + SCHP) podľa nového oblečenia.

Postup (adjust_mask):
1. šaty         – `overall`: maska v tvare šiat z fotky produktu (analyze_dress:
                  dĺžka a rozšírenie sukne), priložená na trup osoby; bez
                  fotky šiat len telo + sukňa (bez „kabátovej“ siluety AutoMaskera)
2. ochrana      – vyreže dlane
3. nohavice     – `lower`: podľa voľnosti nových nohavíc (analyze_pants)
                  sa určí cieľová šírka na osobe (noha z DensePose → šírka
                  z fotky); o rozdiel voči pôvodným nohaviciam sa maska
                  rozšíri von; maska AutoMaskera sa nikdy nezužuje

Modul neimportuje kód CatVTON – mapy tried mu odovzdáva src/try_on.py.
Masky sú PIL "L" 768×1024 (0 / 255), ešte bez rozmazania okraja.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

PROTECT_DILATE = 2                    # roztiahnutie chránenej oblasti rúk (iterácie)

# Šaty (overall): maska len na telo + sukňa (bez „kabátovej“ siluety AutoMaskera)
DRESS_SKIRT_MARGIN = 0.25             # sukňa: do šírky o 25 % šírky trupu na každú stranu
DRESS_BODY_DIVISOR = 40               # roztiahnutie tela: jadro max(rozmer)/40 ≈ 25 px
# Šaty podľa fotky produktu (analyze_dress → _dress_from_product)
DRESS_SAMPLES = 20                    # body profilu šírky (hruď → lem)
DRESS_FLAT_TO_BODY = 0.75             # šírka hrude osoby spredu ≈ 0,75 × šírka rozložených šiat
DRESS_DRAPE = 0.9                     # sukňa na osobe ≈ 0,9 × šírka rozloženej sukne
DRESS_BUST_Y = 0.2                    # hruď (začiatok profilu) = 20 % výšky trupu pod jeho vrchom
DRESS_MARGIN = 0.04                   # rezerva okolo tvaru šiat (podiel šírky hrude)
DRESS_WAIST_Y = 0.65                  # pás osoby = 65 % výšky trupu (DensePose) pod jeho vrchom
DRESS_WAIST_MAX = 0.95                # pás na fotke = najužšie miesto, ak je < 95 % šírky hrude

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
FIT_MAX_EXTRA = 0.45                  # strop: okraj sa posunie max. o 45 % šírky bokov na stranu
FIT_MAX_PROFILE = 0.9                 # nohavica širšia ako 90 % bokov → nevierohodné
FIT_MAX_JUMP = 0.2                    # skok šírky medzi susednými bodmi profilu → nevierohodné
FIT_MIN_GAIN = 0.01                   # pod +1 % plochy masky sa maska nemení
FIT_MAX_CROTCH = 0.45                 # produkt: rozdelenie nižšie ako 45 % výšky = nohavice sa
                                      # prekrývajú → rozkrok = FIT_DEFAULT_CROTCH
FIT_TIGHT = (0.47, 0.31)              # profil priliehavých nohavíc na fotke (rozkrok, lem)
FIT_LOOSE_RANGE = 0.12                # o toľko širšie ako priliehavé = úplne voľné
FIT_LOOSE_SCALE = 1.25                # voľná nohavica na osobe ≈ 1,25 × pomer z fotky × boky
FIT_OUTER_SHARE = 0.85                # podiel rozšírenia, ktorý ide von (dnu je maska už medzi nohami)
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


def _dress_shape(m: np.ndarray, densepose: np.ndarray) -> str | None:
    """Šaty: maska len na roztiahnuté telo + sukňa od pása nadol. Vráti popis úpravy alebo None."""
    torso = np.isin(densepose, DENSEPOSE_TORSO)
    if not torso.any():
        return None                   # DensePose nenašiel trup
    ys, xs = np.nonzero(torso)
    top, hip, left, right = ys.min(), ys.max(), xs.min(), xs.max()
    torso_h, torso_w = hip - top, right - left
    h, w = m.shape

    # Povolená oblasť: roztiahnuté telo (trup, ruky, nohy) + sukňa od pása po lem
    body = np.isin(densepose, DENSEPOSE_TORSO + DENSEPOSE_ARMS + DENSEPOSE_THIGHS + DENSEPOSE_SHINS)
    allowed = cv2.dilate(body.astype(np.uint8), _kernel((w, h), DRESS_BODY_DIVISOR)) > 0
    waist = int(hip - 0.25 * torso_h)
    x0 = max(0, int(left - DRESS_SKIRT_MARGIN * torso_w))
    x1 = min(w, int(right + DRESS_SKIRT_MARGIN * torso_w))
    allowed[waist:, x0:x1] = True
    m &= allowed
    return "šaty: tvar tela"


def analyze_dress(cloth: Image.Image) -> dict | None:
    """Tvar šiat z produktovej fotky – len spracovanie obrazu, žiadny model.

    Silueta (ako pri nohaviciach) → hruď = prvý riadok, kde je šírka aspoň
    polovica šírky v strede šiat (nad ním sú ramienka / výstrih), šírka hrude
    = medián nasledujúcich 8 % výšky. Profil = šírka / hruď v DRESS_SAMPLES
    bodoch od hrude po lem, dĺžka = (lem − hruď) / hruď.
    Pás = najužšie miesto medzi hruďou a 60 % výšky; ak je zreteľný,
    `waist_len` = (lem − hruď) / (pás − hruď) – dĺžka podľa pásu, nie šírky.
    Vráti {"profile", "length", "waist_len", "hem"} alebo None.
    """
    fg = silhouette(cloth)
    if fg is None:
        return None
    ys, xs = np.nonzero(fg.any(axis=1))[0], np.nonzero(fg.any(axis=0))[0]
    fg = fg[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]
    h = fg.shape[0]
    widths = fg.sum(axis=1).astype(float)
    core = float(np.median(widths[int(0.3 * h): int(0.6 * h)]))
    bust_y = int(np.argmax(widths >= 0.5 * core))
    if bust_y > 0.35 * h:
        return None                   # nevyzerá ako šaty (široké až nízko)
    bust = float(np.median(widths[bust_y: bust_y + max(1, int(0.08 * h))]))
    rows = widths[bust_y:] / max(bust, 1.0)
    grid = np.linspace(0, len(rows) - 1, DRESS_SAMPLES).astype(int)
    profile = _smooth(rows[grid], 3)
    length = (h - bust_y) / max(bust, 1.0)
    if not (1.0 <= length <= 8.0) or profile.max() > 5.0:
        return None
    smooth = np.convolve(widths, np.ones(5) / 5, "same")
    lo, hi = bust_y + int(0.05 * h), int(0.6 * h)
    waist_len = None
    if hi > lo:
        waist_y = lo + int(np.argmin(smooth[lo:hi]))
        if smooth[waist_y] < DRESS_WAIST_MAX * bust and waist_y > bust_y:
            waist_len = (h - bust_y) / (waist_y - bust_y)
    return {"profile": profile, "length": float(length), "waist_len": waist_len,
            "hem": float(profile[-3:].mean())}


def describe_dress(dress: dict | None) -> str:
    """Text do `info`, napr. "šaty (dĺžka 3.4× hruď, lem 2.10× hruď)"."""
    return f'šaty (dĺžka {dress["length"]:.1f}× hruď, lem {dress["hem"]:.2f}× hruď)' if dress else "-"


def _dress_from_product(m: np.ndarray, densepose: np.ndarray, lip: np.ndarray,
                        atr: np.ndarray, dress: dict) -> str | None:
    """Šaty: maska v tvare šiat z fotky produktu, priložená na osobu.

    Hruď osoby = šírka trupu (DensePose) v DRESS_BUST_Y výšky trupu. Rozložené
    šaty sú širšie ako telo spredu (DRESS_FLAT_TO_BODY) → mierka. Od hrude
    nadol sa kreslí tvar šiat (šírka podľa profilu, sukňa × DRESS_DRAPE) až po
    lem: ak majú šaty zreteľný pás, lem = hruď + waist_len × (pás − hruď) osoby,
    inak podľa šírky hrude; najnižšie nad chodidlami. Nad hruďou ostáva maska AutoMaskera, ruky ostávajú tiež
    (rukáv / holá koža), pôvodné nohavice nad lemom sa pridajú celé (aby z nich
    nebolo nič vidieť). Pod lemom maska končí, topánky sa nikdy nemaskujú.
    """
    torso = np.isin(densepose, DENSEPOSE_TORSO)
    if not torso.any():
        return None
    h, w = m.shape
    ys = np.nonzero(torso.any(axis=1))[0]
    top, hip = int(ys.min()), int(ys.max())
    y0 = int(top + DRESS_BUST_Y * (hip - top))
    xs = np.nonzero(torso[y0])[0]
    if xs.size == 0:
        return None
    person_bust = float(xs.max() - xs.min() + 1)
    cx = float(xs.mean())
    flat = person_bust / DRESS_FLAT_TO_BODY       # šírka rozložených šiat v pixeloch osoby
    if dress.get("waist_len"):                    # dĺžka podľa pásu (spoľahlivejšie)
        waist = top + DRESS_WAIST_Y * (hip - top)
        hem = min(h, int(y0 + dress["waist_len"] * (waist - y0)))
    else:                                         # šaty bez pásu – podľa šírky hrude
        hem = min(h, int(y0 + dress["length"] * flat))
    # Lem najnižšie nad chodidlami (maxi šaty nesmú prekryť topánky)
    feet = np.nonzero((np.isin(densepose, DENSEPOSE_FEET) | np.isin(lip, (18, 19))
                       | np.isin(atr, (9, 10))).any(axis=1))[0]
    feet = feet[feet > y0]
    if feet.size:
        hem = min(hem, int(feet.min()))

    shape = np.zeros_like(m)
    rows = np.arange(y0, hem)
    s = (rows - y0) / max(1, hem - y0)
    ratio = np.interp(s, np.linspace(0, 1, len(dress["profile"])), dress["profile"])
    half = np.maximum(ratio * flat * DRESS_DRAPE, person_bust) / 2 + DRESS_MARGIN * person_bust
    for y, hw in zip(rows, half):
        shape[y, max(0, int(cx - hw)): min(w, int(cx + hw) + 1)] = True

    arms = cv2.dilate(np.isin(densepose, DENSEPOSE_ARMS).astype(np.uint8), _kernel((w, h), 60)) > 0
    pants = np.isin(lip, SCHP_LIP_PANTS) | np.isin(atr, SCHP_ATR_PANTS)
    pants = cv2.dilate(pants.astype(np.uint8), _kernel((w, h), 100)) > 0
    shoes = np.isin(lip, (18, 19)) | np.isin(atr, (9, 10))

    above = np.zeros_like(m)
    above[:y0] = True
    new = (m & above) | shape | (m & arms) | pants
    new[hem:] = False
    new &= ~shoes
    m[:] = new
    return f"šaty: tvar z fotky (lem {hem / h:.0%} výšky)"


def _runs(row: np.ndarray) -> list[tuple[int, int]]:
    """Súvislé úseky True v riadku ako (začiatok, dĺžka) – napr. dve nohavice vedľa seba."""
    d = np.diff(np.concatenate(([0], row.astype(np.int8), [0])))
    starts, ends = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
    return list(zip(starts, ends - starts))


def _leg_profile(fg: np.ndarray, max_crotch: float | None = None) -> dict | None:
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
    if max_crotch is not None and crotch - top > max_crotch * height:
        crotch = top + int(FIT_DEFAULT_CROTCH * height)   # nohavice sa prekrývajú (baggy)
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


def _alpha(img: Image.Image) -> np.ndarray | None:
    """Alfa kanál (0–255), ak má fotka priehľadné pozadie (aspoň 5 % pixelov), inak None."""
    if img.mode not in ("RGBA", "LA", "PA") and "transparency" not in img.info:
        return None
    a = np.asarray(img.convert("RGBA"))[..., 3]
    return a if (a < 128).mean() >= 0.05 else None


def flatten(img: Image.Image) -> Image.Image:
    """RGB na bielom pozadí. Priehľadné pozadie (PNG / WebP produktov) by pri
    obyčajnom convert("RGB") zostalo čierne alebo s náhodnými farbami."""
    if _alpha(img) is None:
        return img.convert("RGB")
    rgba = img.convert("RGBA")
    white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    return Image.alpha_composite(white, rgba).convert("RGB")


def silhouette(cloth: Image.Image) -> np.ndarray | None:
    """Silueta oblečenia na produktovej fotke (bool H×W) alebo None.

    Priehľadné pozadie → priamo alfa kanál. Inak pixely, ktoré sa farbou líšia
    od pozadia (medián okraja fotky). Potom otvorenie / zatvorenie (OpenCV)
    a najväčšia súvislá oblasť. Bez priehľadnosti predpokladá jednoliate pozadie.
    """
    alpha = _alpha(cloth)
    if alpha is not None:
        fg = alpha >= 128
        h, w = fg.shape
    else:
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
    return fg


def analyze_pants(cloth: Image.Image) -> dict | None:
    """Strih nohavíc z produktovej fotky – len spracovanie obrazu, žiadny model.

    1. silueta: pixely, ktoré sa farbou líšia od pozadia (medián okraja fotky),
    2. profil (_leg_profile): šírka jednej nohavice / šírka bokov
       v FIT_SAMPLES bodoch od rozkroku (0) po lem (1).
    Vráti {"profile", "knee", "hem", "label"} alebo None, ak nohavice nenájde.
    """
    fg = silhouette(cloth)
    if fg is None:
        return None
    leg = _leg_profile(fg, max_crotch=FIT_MAX_CROTCH)
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
    # Spodných 15 %: zaoblený / pokrčený lem šírku umelo znižuje → nepoklesne pod hodnotu nad ním
    tail = int(0.85 * FIT_SAMPLES)
    profile[tail:] = np.maximum(profile[tail:], profile[tail - 1])
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


def _interp_s(prof: dict, rows: np.ndarray, crotch: int, bottom: int) -> np.ndarray:
    """Pomer šírky z profilu `prof` (_leg_profile) v riadkoch `rows` iného profilu – podľa
    relatívnej polohy medzi rozkrokom (0) a lemom (1)."""
    s_src = (prof["rows"] - prof["crotch"]) / max(1, prof["bottom"] - prof["crotch"])
    s_dst = (rows - crotch) / max(1, bottom - crotch)
    return np.interp(s_dst, s_src, _fill_nan(prof["ratio"]))


def _widen_legs(m: np.ndarray, old: np.ndarray, legs: np.ndarray, keep: np.ndarray, fit: dict) -> float:
    """Rozšíri masku nohavíc podľa strihu nových nohavíc.

    `old` = pôvodné nohavice na osobe (SCHP), `legs` = nohy z DensePose. Pre
    každý riadok od rozkroku po lem (pomery k šírke bokov osoby):
      voľnosť  = (nový pomer z fotky − priliehavé nohavice FIT_TIGHT) / FIT_LOOSE_RANGE, 0…1
      cieľ     = noha + voľnosť × (FIT_LOOSE_SCALE × nový pomer − noha)
      rozdiel  = cieľ − pôvodné nohavice; vonkajší okraj sa posunie o FIT_OUTER_SHARE × rozdiel.
    Priliehavé nohavice → cieľ = noha → nič; baggy → cieľ ~ šírka z fotky.
    Pridá sa len to, čo nie je v `keep`; nič sa neuberá. Vráti prírastok plochy.
    """
    leg = _leg_profile(old)
    if leg is None:
        return 0.0
    w = m.shape[1]
    rows, hip, crotch, bottom = leg["rows"], leg["hip"], leg["crotch"], leg["bottom"]
    s = (rows - crotch) / max(1, bottom - crotch)
    new = np.interp(s, np.linspace(0, 1, len(fit["profile"])), fit["profile"])
    tight = FIT_TIGHT[0] + (FIT_TIGHT[1] - FIT_TIGHT[0]) * s
    loose = np.clip((new - tight) / FIT_LOOSE_RANGE, 0, 1)
    old_ratio = _fill_nan(leg["ratio"])
    body = _leg_profile(legs) if legs.any() else None
    leg_ratio = np.minimum(_interp_s(body, rows, crotch, bottom) * body["hip"] / hip, old_ratio)         if body is not None else old_ratio
    target = leg_ratio + loose * np.maximum(FIT_LOOSE_SCALE * new - leg_ratio, 0)
    extra = np.clip((target - old_ratio) * FIT_OUTER_SHARE, 0, FIT_MAX_EXTRA) * hip
    k = max(15, int(0.08 * len(rows)) | 1)       # vyhladenie okraja (~8 % dĺžky nohy)
    extra = _smooth(extra, k)
    left, right = _smooth(_fill_nan(leg["left"]), k), _smooth(_fill_nan(leg["right"]), k)

    added = np.zeros_like(m)
    for y, l, r, e in zip(rows, left, right, extra):
        if e >= 1:
            added[y, max(0, int(l - e)): min(w, int(r + e) + 1)] = True

    before = m.sum()
    grown = m | (added & ~keep)
    gain = float(grown.sum() - before) / max(1, before)
    if gain >= FIT_MIN_GAIN:
        m |= grown
        return gain
    return 0.0


def adjust_mask(
    mask: Image.Image,
    parts: dict,
    cloth_type: str,
    pants_fit: dict | None = None,
    dress_fit: dict | None = None,
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
        legs = np.isin(densepose, DENSEPOSE_LEGS[0] + DENSEPOSE_LEGS[1])
        old = np.isin(lip, SCHP_LIP_PANTS) | np.isin(atr, SCHP_ATR_PANTS)
        if old.sum() < 0.01 * old.size:
            old = legs
        gain = _widen_legs(m, old, legs, keep, pants_fit)
        if not gain:
            return mask, [], changes
        changes.append(f"nohavice rozšírené (+{gain:.0%})")
        return Image.fromarray((m * 255).astype(np.uint8)), [], changes

    # 1) Šaty: tvar šiat z fotky produktu; ak nie je, tvar tela namiesto siluety AutoMaskera
    if cloth_type == "overall":
        change = _dress_from_product(m, densepose, lip, atr, dress_fit) if dress_fit else None
        change = change or _dress_shape(m, densepose)
        if change:
            changes.append(change)

    # 2) Ochrana dlaní (holé ruky sa nechránia – model môže nakresliť dlhý rukáv)
    kernel = _kernel(size, 250)       # rovnaká veľkosť ako v AutoMaskeri
    skin = np.isin(lip, SCHP_ARMS) | np.isin(atr, SCHP_ARMS)
    hands = cv2.dilate(np.isin(densepose, DENSEPOSE_HANDS).astype(np.uint8), kernel) > 0
    protected = ["dlane"]
    protect = cv2.dilate((hands & skin).astype(np.uint8), kernel, iterations=PROTECT_DILATE) > 0
    m &= ~protect
    return Image.fromarray((m * 255).astype(np.uint8)), protected, changes


def mask_ratio(mask: Image.Image) -> float:
    """Podiel pixelov v maske (0–1)."""
    return round(float((np.array(mask.convert("L")) > 127).mean()), 3)
