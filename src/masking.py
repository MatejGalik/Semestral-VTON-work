"""Generovanie masky pre CatVTON – dva maskery na výber a porovnanie.

- "schp":    AutoMasker z CatVTON (DensePose + SCHP) + naša ochrana rúk.
- "sapiens": Meta Sapiens-seg (28 tried Goliath) – vlastná logika podľa
             AutoMaskera, presnejšia pri neštandardných pózach.

Modul neimportuje kód CatVTON – modely mu odovzdáva src/try_on.py.
Všetky masky sú PIL "L" 768×1024 (0 / 255), ešte bez rozmazania okraja.
"""

from __future__ import annotations

import logging
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

log = logging.getLogger(__name__)

PROTECT_DILATE = 2                    # roztiahnutie chránenej oblasti rúk (iterácie) – SCHP
SAPIENS_PROTECT_DILATE = 2            # ako SCHP; 1 → maska príliš tesná (konce dlhého rukáva)
SAPIENS_BLUR_THRESHOLD = 25           # ako AutoMasker – široká maska dáva modelu priestor na iný strih
LEGS_MIN_RATIO = 0.005                # nohy na fotke (podiel) → Lower_Clothing sú naozaj nohavice

# --------------------------------------------------------------------------- #
# SCHP / DensePose (indexy ako v CatVTON model/cloth_masker.py)
# --------------------------------------------------------------------------- #
DENSEPOSE_HANDS = (3, 4)
SCHP_ARMS = (14, 15)                  # Left-arm, Right-arm – v LIP aj ATR

# --------------------------------------------------------------------------- #
# Sapiens-seg (facebook/sapiens-seg-*-torchscript), triedy GOLIATH_CLASSES
# --------------------------------------------------------------------------- #
SAPIENS_REPO = "facebook/sapiens-seg-{size}-torchscript"
SAPIENS_MEAN = (123.5, 116.5, 103.5)  # RGB, hodnoty 0–255 (bez /255)
SAPIENS_STD = (58.5, 57.0, 57.5)

S_BACKGROUND = 0
S_APPAREL = 1
S_FACE = (2, 23, 24, 25, 26, 27)      # Face_Neck, pery, zuby, jazyk
S_HAIR = 3
S_FEET = (4, 13)
S_HANDS = (5, 14)
S_LOWER_ARMS = (6, 15)
S_UPPER_ARMS = (10, 19)
S_LOWER_LEGS = (7, 16)
S_UPPER_LEGS = (11, 20)
S_SHOES_SOCKS = (8, 9, 17, 18)
S_LOWER_CLOTHING = 12
S_TORSO = 21
S_UPPER_CLOTHING = 22

S_ARMS = S_LOWER_ARMS + S_UPPER_ARMS
S_LEGS = S_LOWER_LEGS + S_UPPER_LEGS

# Podľa typu oblečenia (obdoba MASK_CLOTH_PARTS / MASK_DENSE_PARTS / PROTECT_* v AutoMaskeri)
S_MASK_CLOTH = {                      # pôvodné oblečenie – maskuje sa vždy
    "upper": (S_UPPER_CLOTHING, S_APPAREL),
    "lower": (S_LOWER_CLOTHING,),
    "overall": (S_UPPER_CLOTHING, S_LOWER_CLOTHING, S_APPAREL),
}
S_MASK_BODY = {                       # telo, kam nové oblečenie môže siahať
    "upper": (S_TORSO,) + S_ARMS,
    "lower": S_LEGS,
    "overall": (S_TORSO,) + S_ARMS + S_LEGS,
}
S_PROTECT_BODY = {                    # slabá ochrana – časti tela mimo oblečenia
    "upper": S_LEGS,
    "lower": S_ARMS,
    "overall": (),
}
S_PROTECT_CLOTH = {                   # slabá ochrana – iné oblečenie
    "upper": (S_LOWER_CLOTHING,),
    "lower": (S_UPPER_CLOTHING, S_APPAREL),
    "overall": (),
}


# --------------------------------------------------------------------------- #
# Spoločné pomocné funkcie
# --------------------------------------------------------------------------- #
def _kernel(size: tuple[int, int], divisor: int) -> np.ndarray:
    k = max(size) // divisor
    k = k if k % 2 == 1 else k + 1
    return np.ones((k, k), np.uint8)


def _as_array(img: Image.Image, size: tuple[int, int]) -> np.ndarray:
    """Mapu tried zjednotí na rozmer masky – bez interpolácie tried."""
    if img.size != size:
        img = img.resize(size, Image.NEAREST)
    return np.array(img)


def _cut(mask: Image.Image, protect: np.ndarray, kernel: np.ndarray,
         iterations: int = PROTECT_DILATE) -> Image.Image:
    """Z masky vyreže chránenú oblasť (mierne roztiahnutú, aby do nej
    nezasahoval rozmazaný okraj masky)."""
    protect = cv2.dilate(protect.astype(np.uint8), kernel, iterations=iterations) > 0
    m = np.array(mask.convert("L")) > 127
    return Image.fromarray(((m & ~protect) * 255).astype(np.uint8))


def _hull(mask: np.ndarray) -> np.ndarray:
    """Konvexný obal každej súvislej oblasti (ako hull_mask v AutoMaskeri)."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros(mask.shape, np.uint8)
    for c in contours:
        cv2.fillPoly(out, [cv2.convexHull(c)], 1)
    return out > 0


# --------------------------------------------------------------------------- #
# Masker "schp" – AutoMasker + ochrana rúk
# --------------------------------------------------------------------------- #
def protect_schp(mask: Image.Image, parts: dict, protect_arms: bool) -> tuple[Image.Image, list[str]]:
    """Z masky AutoMaskera vyreže dlane (vždy) a holú kožu rúk (ak protect_arms)."""
    size = mask.size
    densepose = _as_array(parts["densepose"], size)
    skin = np.isin(_as_array(parts["schp_lip"], size), SCHP_ARMS) | np.isin(
        _as_array(parts["schp_atr"], size), SCHP_ARMS
    )
    kernel = _kernel(size, 250)  # rovnaká veľkosť ako v AutoMaskeri
    hands = cv2.dilate(np.isin(densepose, DENSEPOSE_HANDS).astype(np.uint8), kernel) > 0
    protect = hands & skin
    protected = ["dlane"]
    if protect_arms:
        protect |= skin
        protected.append("ruky")
    return _cut(mask, protect, kernel), protected


# --------------------------------------------------------------------------- #
# Masker "sapiens"
# --------------------------------------------------------------------------- #
def load_sapiens(size: str, device: str):
    """Stiahne a načíta Sapiens-seg torchscript (fp32). Vráti model alebo vyhodí výnimku."""
    from huggingface_hub import HfApi, hf_hub_download

    repo = SAPIENS_REPO.format(size=size)
    files = [f for f in HfApi().list_repo_files(repo) if f.endswith("_torchscript.pt2") and "goliath" in f]
    if not files:
        raise RuntimeError(f"V {repo} sa nenašiel súbor *_goliath_*_torchscript.pt2")
    path = hf_hub_download(repo, files[0])
    model = torch.jit.load(path, map_location=device).eval()
    log.info("Sapiens-seg %s načítaný (%s)", size, files[0])
    return model


@torch.inference_mode()
def sapiens_parse(model, person: Image.Image, device: str) -> np.ndarray:
    """Mapa tried (H×W, 0–27) pre fotku osoby v rozmere fotky."""
    w, h = person.size
    img = np.array(person.convert("RGB").resize((768, 1024), Image.BILINEAR), dtype=np.float32)
    x = torch.from_numpy(img).permute(2, 0, 1)
    x = (x - torch.tensor(SAPIENS_MEAN).view(3, 1, 1)) / torch.tensor(SAPIENS_STD).view(3, 1, 1)
    out = model(x.unsqueeze(0).to(device))
    if isinstance(out, (list, tuple)):
        out = out[0]
    if out.dim() == 3:
        out = out.unsqueeze(0)
    logits = F.interpolate(out.float(), size=(h, w), mode="bilinear", align_corners=False)
    return logits[0].argmax(dim=0).to(torch.uint8).cpu().numpy()


SAPIENS_STATS = {                     # skupiny tried pre diagnostiku v info
    "upper_clothing": (S_UPPER_CLOTHING,),
    "lower_clothing": (S_LOWER_CLOTHING,),
    "apparel": (S_APPAREL,),
    "torso": (S_TORSO,),
    "arms": S_ARMS,
    "hands": S_HANDS,
    "legs": S_LEGS,
    "face": S_FACE,
    "background": (S_BACKGROUND,),
}


def sapiens_stats(parse: np.ndarray) -> dict[str, float]:
    """Podiel hlavných tried Sapiens na fotke (0–1) – diagnostika."""
    return {k: round(float(np.isin(parse, v).mean()), 3) for k, v in SAPIENS_STATS.items()}


def sapiens_mask(parse: np.ndarray, cloth_type: str, protect_arms: bool) -> tuple[Image.Image, list[str]]:
    """Maska z mapy tried Sapiens – postup podľa AutoMasker.cloth_agnostic_mask,
    hneď s ochranou rúk (dlane vždy, holé ruky pri protect_arms)."""
    h, w = parse.shape
    size = (w, h)
    k_small = _kernel(size, 250)
    blur = max(size) // 25 | 1

    def of(classes) -> np.ndarray:
        return np.isin(parse, classes)

    # Silná ochrana: tvár, dlane, chodidlá (+ holé ruky pri krátkom rukáve)
    strong = of(S_FACE + S_FEET)
    protected = ["tvár", "dlane"]
    hands = of(S_HANDS)
    if protect_arms and cloth_type in ("upper", "overall"):
        hands |= of(S_ARMS)
        protected.append("ruky")
    strong |= hands

    # Lower_Clothing pri vrchu: na fotke do pása Sapiens označí spodok trička ako
    # Lower_Clothing. Chrániť ho len, keď sú vidieť nohy (sú to naozaj nohavice).
    protect_cloth = S_PROTECT_CLOTH[cloth_type]
    mask_cloth = S_MASK_CLOTH[cloth_type]
    if cloth_type == "upper" and of(S_LEGS + S_FEET + S_SHOES_SOCKS).mean() < LEGS_MIN_RATIO:
        protect_cloth = ()
        mask_cloth = mask_cloth + (S_LOWER_CLOTHING,)

    # Slabá ochrana: vlasy, topánky, iné oblečenie, časti tela mimo typu
    weak = (
        strong
        | of((S_HAIR,) + S_SHOES_SOCKS)
        | of(S_PROTECT_BODY[cloth_type])
        | of(protect_cloth)
    )

    cloth = of(mask_cloth)
    body = of(S_MASK_BODY[cloth_type]).astype(np.uint8)
    # Telo roztiahnuť v 1/4 rozlíšení (ako AutoMasker) – priestor pre iný strih
    small = cv2.resize(body, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)
    small = cv2.dilate(small, k_small, iterations=2)
    body = cv2.resize(small, size, interpolation=cv2.INTER_NEAREST) > 0

    area = ((parse != S_BACKGROUND) & ~weak) | body
    area = _hull(area) & ~weak
    area = cv2.GaussianBlur(area.astype(np.uint8) * 255, (blur, blur), 0) >= SAPIENS_BLUR_THRESHOLD
    area = (area | cloth) & ~strong
    area = cv2.dilate(area.astype(np.uint8), k_small, iterations=1) > 0

    # Chránené ruky ešte raz vyrezať – záverečné roztiahnutie do nich zasahuje
    mask = _cut(Image.fromarray((area * 255).astype(np.uint8)), hands, k_small, SAPIENS_PROTECT_DILATE)
    if cloth_type == "upper" and not protect_cloth:
        protected.append("spodok maskovaný")
    return mask, protected


# --------------------------------------------------------------------------- #
# Porovnanie masiek
# --------------------------------------------------------------------------- #
def mask_ratio(mask: Image.Image) -> float:
    """Podiel pixelov v maske (0–1)."""
    return round(float((np.array(mask.convert("L")) > 127).mean()), 3)


def save_comparison(person: Image.Image, masks: dict[str, Image.Image], out_dir: str, name: str) -> str:
    """Uloží vedľa seba osobu a masky (sivé = prekreslí sa) ako PNG. Vráti cestu."""
    os.makedirs(out_dir, exist_ok=True)
    base = np.array(person.convert("RGB"))
    tiles = [base]
    for m in masks.values():
        sel = np.array(m.convert("L").resize(person.size, Image.NEAREST)) > 127
        tile = base.copy()
        tile[sel] = (tile[sel] * 0.3 + 128 * 0.7).astype(np.uint8)
        tiles.append(tile)
    path = os.path.join(out_dir, f"{name}_{'_'.join(masks)}.png")
    Image.fromarray(np.concatenate(tiles, axis=1)).save(path)
    return path
