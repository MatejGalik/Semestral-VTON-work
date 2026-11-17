"""Typ oblečenia z produktovej fotky pravidlami nad siluetou – bez AI.

classify(cloth) -> (typ, dôvod) alebo (None, dôvod), keď si pravidlá nie sú isté
(vtedy UI vyzve na ručný výber, API použije "upper"). Typ: "upper" | "lower" | "overall".

Postup:
1. silueta (masking.silhouette) – predpokladá jednoliate pozadie,
2. znaky (features): pomer výška / šírka, šírka v hornej časti a pri leme,
   kde je silueta najširšia, či sa spodok delí na dve nohavice,
3. rozhodovací strom (pravidlá nižšie, prahy RULE_*).

Lokálny test: python scripts/typ_oblecenia.py <fotky>
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from .masking import _runs, silhouette

RULE_LEGS_FROM = 0.35        # dve nohavice sa hľadajú od 35 % výšky po lem (baggy sa pri leme dotýkajú)
RULE_LEGS_SHARE = 0.3        # … aspoň v 30 % týchto riadkov „pár nohavíc“ (3 úseky = rukávy pri tele)
RULE_LEGS_SIMILAR = 0.6      # pár: užší úsek ≥ 60 % širšieho
RULE_LEGS_CENTER = 0.15      # … a medzera najviac 15 % šírky od stredu siluety
RULE_MIN_RUN = 0.05          # úsek užší ako 5 % šírky siluety sa ignoruje (strapce, šnúrky)
RULE_LONG = 1.45             # výška / šírka ≥ 1,45 → dlhý kus (šaty, maxi sukňa)
RULE_SLEEVES_ZONE = 0.5      # najširšie miesto v hornej polovici → rukávy / ramená
RULE_SLEEVES_HEM = 0.85      # … a lem užší ako 85 % max. šírky
RULE_WAISTBAND = 0.42        # horný okraj ako súvislý pás ≥ 42 % max. šírky → sukňa / nohavice
RULE_PANTS_TOP = 0.6         # nohavice: hore súvislý pás ≥ 60 % (inak výstrih / ramená → overal)
RULE_CROTCH_PANTS = 0.48     # rozkrok v hornej časti (≤ 48 % výšky) → nohavice, nie overal
RULE_SHORT = 1.4             # dve nohavice a výška / šírka < 1,4 → šortky / široké nohavice
RULE_WIDE_TOP = 0.7          # krátka rovná silueta s pásom ≥ 70 % hore = asi spodok → neisté
RULE_FLARE = 1.25            # lem ≥ 1,25 × najužšie miesto → rozširuje sa nadol


def features(fg: np.ndarray) -> dict:
    """Číselné znaky siluety (všetko relatívne, nezávisí od rozlíšenia)."""
    ys, xs = np.nonzero(fg.any(axis=1))[0], np.nonzero(fg.any(axis=0))[0]
    fg = fg[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]
    h, w = fg.shape
    min_run = RULE_MIN_RUN * w
    widths = fg.sum(axis=1) / w                        # šírka riadku / max. šírka siluety
    runs = [[r for r in _runs(row) if r[1] >= min_run] for row in fg]

    def band(a: float, b: float) -> slice:
        return slice(int(a * h), max(int(a * h) + 1, int(b * h)))

    def leg_pair(r: list[tuple[int, int]]) -> bool:
        if len(r) != 2:
            return False
        (a, la), (b, lb) = sorted(r)
        gap_center = (a + la + b) / 2
        return (min(la, lb) >= RULE_LEGS_SIMILAR * max(la, lb)
                and abs(gap_center - w / 2) <= RULE_LEGS_CENTER * w)

    legs_rows = runs[band(RULE_LEGS_FROM, 1.0)]
    # Rozkrok: prvý riadok (od 15 % výšky), od ktorého ďalších 5 % riadkov je prevažne pár nohavíc
    pairs = np.array([leg_pair(r) for r in runs], float)
    win = max(1, int(0.05 * h))
    crotch = next((y for y in range(int(0.15 * h), h - win) if pairs[y: y + win].mean() >= 0.6), h)
    top = band(0.02, 0.08)                             # tesne pod horným okrajom
    top_runs = [len(r) for r in runs[top]]
    return {
        "aspect": h / w,
        "legs": float(np.mean([leg_pair(r) for r in legs_rows])),
        "crotch_at": crotch / h,
        "top_width": float(np.median(widths[top])),
        "top_single": float(np.mean([n == 1 for n in top_runs])) >= 0.5,
        "hem_width": float(np.median(widths[band(0.9, 0.98)])),
        "widest_at": float(np.argmax(np.convolve(widths, np.ones(5) / 5, "same")) / h),
        "narrowest": float(widths[band(0.05, 0.6)].min()),
    }


def decide(f: dict) -> tuple[str | None, str]:
    """Rozhodovací strom nad znakmi. Vráti (typ alebo None, dôvod)."""
    sleeves = f["widest_at"] < RULE_SLEEVES_ZONE and f["hem_width"] < RULE_SLEEVES_HEM
    if f["legs"] >= RULE_LEGS_SHARE:
        if f["crotch_at"] <= RULE_CROTCH_PANTS:
            return "lower", "dve nohavice, rozkrok v hornej časti (nohavice)"
        if f["aspect"] < RULE_SHORT:
            return "lower", "dve nohavice, nízky pomer strán (šortky / široké nohavice)"
        if f["top_single"] and f["top_width"] >= RULE_PANTS_TOP and f["widest_at"] >= 0.15:
            return "lower", "dve nohavice, hore pás"
        return "overall", "dve nohavice, rozkrok nízko a hore výstrih / ramená (overal)"
    waistband = f["top_single"] and f["top_width"] >= RULE_WAISTBAND
    flare = f["hem_width"] >= RULE_FLARE * f["narrowest"]
    if f["aspect"] >= RULE_LONG:
        if waistband and flare and not sleeves:
            return "lower", "dlhá, hore rovný pás, nadol sa rozširuje (maxi sukňa)"
        return "overall", "dlhá silueta bez nohavíc (šaty)"
    if sleeves:
        return "upper", "najširšia hore (rukávy / ramená), krátka"
    if waistband and flare:
        return "lower", "hore rovný pás, nadol sa rozširuje (sukňa)"
    if not flare and not (f["top_single"] and f["top_width"] >= RULE_WIDE_TOP):
        return "upper", "krátka, rovná silueta, hore bez širokého pásu (vrch)"
    return None, "krátka, bez jasných znakov (široký pás alebo rozšírenie bez pásu) – neisté"


def classify(cloth: Image.Image) -> tuple[str | None, str]:
    fg = silhouette(cloth)
    if fg is None:
        return None, "silueta sa nenašla (pozadie nie je jednoliate?)"
    return decide(features(fg))
