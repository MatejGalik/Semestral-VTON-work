"""Backend virtuálneho skúšania – CatVTON so vstavaným maskovaním (AutoMasker).

Kontrakt (používa ho app.py a API /try_on):
    try_on(person_img, cloth_img, cloth_type, **options) -> {"result", "mask", "info"}

    person_img, cloth_img : PIL.Image
    cloth_type            : "upper" | "lower" | "overall"
    options               : steps (int), seed (int), paste_face (bool),
                            guidance_scale (float, voliteľné),
                            sleeves ("auto" | "short" | "long" | "sleeveless",
                            voliteľné – ručné určenie rukávov namiesto CLIP)

Ochrana rúk: dlane sa z masky vyrežú vždy, holá koža rúk vtedy, keď nové
oblečenie má krátky rukáv alebo je bez rukávov (určí CLIP z fotky oblečenia).

Model sa načíta raz pri importe. Bez CUDA import zlyhá a app.py sa prepne
do DEMO režimu. Kód CatVTON nie je v repe – očakáva sa naklonovaný v adresári
z premennej CATVTON_DIR (predvolene ./CatVTON vedľa repa alebo v ňom).
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Nastavenia
# --------------------------------------------------------------------------- #
WIDTH, HEIGHT = 768, 1024                     # rozlíšenie, na ktorom bol model trénovaný
BASE_MODEL = "booksforcharlie/stable-diffusion-inpainting"  # kópia SD 1.5 inpainting (runwayml repo bolo zmazané)
CATVTON_REPO = "zhengchong/CatVTON"
ATTN_VERSION = "mix"                          # váhy mix-48k-1024
GUIDANCE_SCALE = 2.5                          # predvolená hodnota z CatVTON
MASK_BLUR = 9                                 # rozmazanie okraja masky (ako v CatVTON demo)

# Ochrana rúk
CLIP_MODEL = "openai/clip-vit-base-patch32"
SLEEVE_PROMPTS = {                            # trieda → popis pre CLIP (zero-shot)
    "short": "a product photo of a short-sleeved top, t-shirt or dress",
    "long": "a product photo of a long-sleeved top, shirt, sweater, jacket or dress",
    "sleeveless": "a product photo of a sleeveless top, tank top or strapless dress",
}
SLEEVE_MIN_CONFIDENCE = 0.6                   # pod touto istotou sa ruky nechránia
PROTECT_DILATE = 2                            # roztiahnutie chránenej oblasti (iterácie)
# Indexy tried (rovnaké ako v CatVTON model/cloth_masker.py)
DENSEPOSE_HANDS = (3, 4)
SCHP_ARMS = (14, 15)                          # Left-arm, Right-arm – v LIP aj ATR


def _find_catvton_dir() -> Path:
    candidates = [os.environ.get("CATVTON_DIR")] if os.environ.get("CATVTON_DIR") else []
    repo_root = Path(__file__).resolve().parent.parent
    candidates += [repo_root / "CatVTON", repo_root.parent / "CatVTON"]
    for c in candidates:
        if c and (Path(c) / "model" / "pipeline.py").is_file():
            return Path(c).resolve()
    raise RuntimeError(
        "Kód CatVTON sa nenašiel. Naklonuj https://github.com/Zheng-Chong/CatVTON "
        "a nastav CATVTON_DIR (skúšané: %s)." % ", ".join(map(str, candidates))
    )


# --------------------------------------------------------------------------- #
# Načítanie modelu (raz pri importe)
# --------------------------------------------------------------------------- #
if not torch.cuda.is_available():
    raise RuntimeError("CUDA nie je k dispozícii – CatVTON potrebuje GPU.")

_CATVTON_DIR = _find_catvton_dir()
# Na začiatok sys.path: CatVTON má top-level moduly `model` a `utils`, ktoré by
# inak mohli prekryť rovnomenné balíky v site-packages.
sys.path.insert(0, str(_CATVTON_DIR))

from diffusers.image_processor import VaeImageProcessor  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402
from model.cloth_masker import AutoMasker  # noqa: E402  (z CatVTON)
from model.pipeline import CatVTONPipeline  # noqa: E402  (z CatVTON)
from utils import repaint_result, resize_and_crop, resize_and_padding  # noqa: E402  (z CatVTON)

_DEVICE = "cuda"
# bf16 len na Ampere+ (compute capability >= 8). Pozor: torch.cuda.is_bf16_supported()
# vracia True aj na T4 (emulácia), no vtedy SDPA nemá úsporný kernel a attention
# spadne na "math" variant, ktorý chce ~36 GB VRAM. T4 / P100 na Kaggle → fp16.
_DTYPE = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
_GPU_NAME = torch.cuda.get_device_name(0)

_t0 = time.perf_counter()
log.info("CatVTON: sťahujem/načítavám váhy…")
_weights_dir = snapshot_download(
    CATVTON_REPO,
    allow_patterns=["mix-48k-1024/*", "SCHP/*", "DensePose/*"],
)

_pipeline = CatVTONPipeline(
    base_ckpt=BASE_MODEL,
    attn_ckpt=_weights_dir,
    attn_ckpt_version=ATTN_VERSION,
    weight_dtype=_DTYPE,
    use_tf32=True,
    device=_DEVICE,
)
_automasker = AutoMasker(
    densepose_ckpt=os.path.join(_weights_dir, "DensePose"),
    schp_ckpt=os.path.join(_weights_dir, "SCHP"),
    device=_DEVICE,
)
_mask_processor = VaeImageProcessor(
    vae_scale_factor=8, do_normalize=False, do_binarize=True, do_convert_grayscale=True
)

# CLIP na určenie dĺžky rukáva. Ak sa nenačíta, backend beží ďalej
# a chráni len dlane (ako bez tejto funkcie).
try:
    from transformers import CLIPModel, CLIPProcessor

    _clip_model = CLIPModel.from_pretrained(CLIP_MODEL, torch_dtype=torch.float16).to(_DEVICE).eval()
    _clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
except Exception as exc:  # noqa: BLE001
    log.warning("CLIP sa nepodarilo načítať, ruky sa nebudú chrániť podľa rukávov: %s", exc)
    _clip_model = _clip_processor = None

log.info("CatVTON pripravený za %.1f s (%s, %s)", time.perf_counter() - _t0, _GPU_NAME, _DTYPE)


# --------------------------------------------------------------------------- #
# Pomocné funkcie
# --------------------------------------------------------------------------- #
def _crop_box(size: tuple[int, int]) -> tuple[int, int, int, int]:
    """Rovnaký výrez ako resize_and_crop z CatVTON (stred, pomer 3:4)."""
    w, h = size
    if w / h < WIDTH / HEIGHT:
        new_w, new_h = w, w * HEIGHT // WIDTH
    else:
        new_w, new_h = h * WIDTH // HEIGHT, h
    return (w - new_w) // 2, (h - new_h) // 2, (w + new_w) // 2, (h + new_h) // 2


def _paste_back(original: Image.Image, generated: Image.Image, box) -> Image.Image:
    """Vráti výsledok v pôvodnom rozmere fotky – vygenerovaný výrez vloží späť."""
    canvas = original.copy()
    canvas.paste(generated.resize((box[2] - box[0], box[3] - box[1]), Image.LANCZOS), box[:2])
    return canvas


def _detect_sleeves(cloth: Image.Image) -> tuple[str | None, float]:
    """CLIP zero-shot: 'short' | 'long' | 'sleeveless' a istota (0–1)."""
    if _clip_model is None:
        return None, 0.0
    labels = list(SLEEVE_PROMPTS)
    inputs = _clip_processor(
        text=[SLEEVE_PROMPTS[k] for k in labels], images=cloth, return_tensors="pt", padding=True
    ).to(_DEVICE)
    inputs["pixel_values"] = inputs["pixel_values"].half()
    probs = _clip_model(**inputs).logits_per_image.softmax(dim=-1)[0].float().cpu()
    best = int(probs.argmax())
    return labels[best], float(probs[best])


def _as_array(img: Image.Image, size: tuple[int, int]) -> np.ndarray:
    """Mapu tried (DensePose/SCHP) zjednotí na rozmer masky – bez interpolácie tried."""
    if img.size != size:
        img = img.resize(size, Image.NEAREST)
    return np.array(img)


def _protect_mask(mask: Image.Image, parts: dict, protect_arms: bool) -> tuple[Image.Image, list[str]]:
    """Z masky vyreže dlane (vždy) a holú kožu rúk (ak protect_arms)."""
    size = mask.size
    densepose = _as_array(parts["densepose"], size)
    skin = np.isin(_as_array(parts["schp_lip"], size), SCHP_ARMS) | np.isin(
        _as_array(parts["schp_atr"], size), SCHP_ARMS
    )

    kernel = np.ones((max(size) // 250 | 1,) * 2, np.uint8)  # rovnaká veľkosť ako v AutoMaskeri
    hands = cv2.dilate(np.isin(densepose, DENSEPOSE_HANDS).astype(np.uint8), kernel) > 0
    protect = hands & skin
    protected = ["dlane"]
    if protect_arms:
        protect |= skin
        protected.append("ruky")

    # Mierne roztiahnuť, aby rozmazaný okraj masky nezasahoval do chránenej oblasti.
    protect = cv2.dilate(protect.astype(np.uint8), kernel, iterations=PROTECT_DILATE) > 0
    m = np.array(mask.convert("L")) > 127
    return Image.fromarray(((m & ~protect) * 255).astype(np.uint8)), protected


# --------------------------------------------------------------------------- #
# Verejné API
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def try_on(person_img: Image.Image, cloth_img: Image.Image, cloth_type: str, **options) -> dict:
    steps = int(options.get("steps", 30))
    seed = int(options.get("seed", 42))
    paste_face = bool(options.get("paste_face", True))
    guidance = float(options.get("guidance_scale", GUIDANCE_SCALE))

    if cloth_type not in ("upper", "lower", "overall"):
        raise ValueError(f"Neznámy typ oblečenia: {cloth_type!r}")

    started = time.perf_counter()
    person = person_img.convert("RGB")
    cloth = cloth_img.convert("RGB")
    box = _crop_box(person.size)

    person_r = resize_and_crop(person, (WIDTH, HEIGHT))
    cloth_r = resize_and_padding(cloth, (WIDTH, HEIGHT))

    # 1) Dĺžka rukáva nového oblečenia (CLIP) – len pre vrch a celé oblečenie
    sleeves_opt = options.get("sleeves", "auto")
    sleeves, confidence = None, 0.0
    if cloth_type in ("upper", "overall"):
        if sleeves_opt in SLEEVE_PROMPTS:
            sleeves, confidence = sleeves_opt, 1.0          # ručne zadané
        else:
            sleeves, confidence = _detect_sleeves(cloth_r)

    # 2) Maska – vstavaný AutoMasker (DensePose + SCHP)
    masker_out = _automasker(person_r, cloth_type)
    mask = masker_out["mask"]

    # 3) Ochrana rúk: dlane vždy, holá koža pri krátkom rukáve / bez rukávov
    protected = []
    if cloth_type in ("upper", "overall"):
        protect_arms = sleeves in ("short", "sleeveless") and confidence >= SLEEVE_MIN_CONFIDENCE
        mask, protected = _protect_mask(mask, masker_out, protect_arms)
    mask_soft = _mask_processor.blur(mask, blur_factor=MASK_BLUR)

    # 4) Generovanie
    generator = torch.Generator(device=_DEVICE).manual_seed(seed)
    result = _pipeline(
        image=person_r,
        condition_image=cloth_r,
        mask=mask_soft,
        num_inference_steps=steps,
        guidance_scale=guidance,
        height=HEIGHT,
        width=WIDTH,
        generator=generator,
    )[0]

    # 5) Mimo masky vrátiť pôvodné pixely (tvár, ruky, pozadie zostanú ostré)
    if paste_face:
        result = repaint_result(result, person_r, mask_soft)

    # 6) Späť do pôvodného rozmeru fotky
    result_full = _paste_back(person, result, box)
    mask_full = _paste_back(Image.new("L", person.size, 0), mask.convert("L"), box)

    info = {
        "model": "CatVTON mix-48k-1024",
        "steps": steps,
        "seed": seed,
        "guidance_scale": guidance,
        "paste_face": paste_face,
        "sleeves": f"{sleeves} ({confidence:.2f})" if sleeves else "-",
        "sleeves_source": "manual" if sleeves_opt in SLEEVE_PROMPTS else "clip",
        "protected": ", ".join(protected) or "-",
        "resolution": f"{WIDTH}x{HEIGHT}",
        "device": f"{_GPU_NAME} ({str(_DTYPE).replace('torch.', '')})",
        "seconds": round(time.perf_counter() - started, 1),
    }
    # UI info nezobrazuje – výpis do konzoly (na Kaggle vo výstupe poslednej bunky).
    log.info("try_on: %s", info)
    return {"result": result_full, "mask": mask_full, "info": info}
