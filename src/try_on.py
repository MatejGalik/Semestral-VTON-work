"""Backend virtuálneho skúšania – CatVTON so vstavaným maskovaním (AutoMasker).

Kontrakt (používa ho app.py a API /try_on):
    try_on(person_img, cloth_img, cloth_type, **options) -> {"result", "mask", "info"}

    person_img, cloth_img : PIL.Image
    cloth_type            : "upper" | "lower" | "overall"
    options               : steps (int), seed (int), paste_face (bool),
                            guidance_scale (float, voliteľné),
                            sleeves ("auto" | "short" | "long" | "sleeveless",
                            voliteľné – ručné určenie rukávov namiesto CLIP),
                            length ("auto" | "cropped" | "regular" | "long",
                            voliteľné – ručné určenie dĺžky vrchu namiesto CLIP)

Maska: AutoMasker z CatVTON (DensePose + SCHP), upravená v src/masking.py podľa
nového oblečenia (CLIP): dlhý vrch → maska siaha nižšie; krátky rukáv / bez
rukávov → holé ruky sa chránia; dlane sa chránia vždy.

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

# Úprava masky podľa nového oblečenia (CLIP zero-shot)
CLIP_MODEL = "openai/clip-vit-base-patch32"
SLEEVE_PROMPTS = {                            # trieda → popis pre CLIP (zero-shot)
    "short": "a product photo of a short-sleeved top, t-shirt or dress",
    "long": "a product photo of a long-sleeved top, shirt, sweater, jacket or dress",
    "sleeveless": "a product photo of a sleeveless top, tank top or strapless dress",
}
SLEEVE_MIN_CONFIDENCE = 0.6                   # pod touto istotou sa ruky nechránia
LENGTH_PROMPTS = {                            # dĺžka vrchu
    "cropped": "a product photo of a cropped top that ends above the waist",
    "regular": "a product photo of a regular-length top or t-shirt that ends at the waist",
    "long": "a product photo of a long top, tunic, long sweater or coat that covers the hips",
}
LENGTH_MIN_CONFIDENCE = 0.6                   # pod touto istotou sa maska nepredĺži


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

from . import masking  # noqa: E402

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
    skip_safety_check=True,                   # NSFW filter SD vypnutý – falošné poplachy (pozri docs/backend.md)
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

# CLIP na určenie rukáva a dĺžky. Ak sa nenačíta, backend beží ďalej
# s maskou AutoMaskera (chráni len dlane, nepredlžuje).
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


def _clip_classify(cloth: Image.Image, prompts: dict[str, str]) -> tuple[str | None, float]:
    """CLIP zero-shot: najpodobnejší kľúč z `prompts` a istota (0–1)."""
    if _clip_model is None:
        return None, 0.0
    labels = list(prompts)
    inputs = _clip_processor(
        text=[prompts[k] for k in labels], images=cloth, return_tensors="pt", padding=True
    ).to(_DEVICE)
    inputs["pixel_values"] = inputs["pixel_values"].half()
    probs = _clip_model(**inputs).logits_per_image.softmax(dim=-1)[0].float().cpu()
    best = int(probs.argmax())
    return labels[best], float(probs[best])


def _classify(cloth: Image.Image, prompts: dict[str, str], manual) -> tuple[str | None, float, str]:
    """Ručná hodnota má prednosť pred CLIP. Vráti (trieda, istota, zdroj)."""
    if manual in prompts:
        return manual, 1.0, "manual"
    label, conf = _clip_classify(cloth, prompts)
    return label, conf, "clip"


def _fmt(label: str | None, conf: float, source: str) -> str:
    return f"{label} ({conf:.2f}, {source})" if label else "-"


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

    # 1) Nové oblečenie (CLIP): rukáv pre vrch aj celé, dĺžka len pre vrch
    sleeves, s_conf, s_src = (None, 0.0, "-")
    length, l_conf, l_src = (None, 0.0, "-")
    if cloth_type in ("upper", "overall"):
        sleeves, s_conf, s_src = _classify(cloth_r, SLEEVE_PROMPTS, options.get("sleeves", "auto"))
    if cloth_type == "upper":
        length, l_conf, l_src = _classify(cloth_r, LENGTH_PROMPTS, options.get("length", "auto"))

    # 2) Maska AutoMaskera + úpravy (predĺženie, ochrana rúk)
    masker_out = _automasker(person_r, cloth_type)
    mask, protected, changes = masking.adjust_mask(
        masker_out["mask"],
        masker_out,
        cloth_type,
        protect_arms=sleeves in ("short", "sleeveless") and s_conf >= SLEEVE_MIN_CONFIDENCE,
        long_garment=length == "long" and l_conf >= LENGTH_MIN_CONFIDENCE,
    )
    mask_soft = _mask_processor.blur(mask, blur_factor=MASK_BLUR)

    # 3) Generovanie
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

    # 4) Mimo masky vrátiť pôvodné pixely (tvár, ruky, pozadie zostanú ostré)
    if paste_face:
        result = repaint_result(result, person_r, mask_soft)

    # 5) Späť do pôvodného rozmeru fotky
    result_full = _paste_back(person, result, box)
    mask_full = _paste_back(Image.new("L", person.size, 0), mask.convert("L"), box)

    info = {
        "model": "CatVTON mix-48k-1024",
        "steps": steps,
        "seed": seed,
        "guidance_scale": guidance,
        "paste_face": paste_face,
        "sleeves": _fmt(sleeves, s_conf, s_src),
        "length": _fmt(length, l_conf, l_src),
        "mask_changes": ", ".join(changes) or "-",
        "mask_ratio": masking.mask_ratio(mask),
        "protected": ", ".join(protected) or "-",
        "resolution": f"{WIDTH}x{HEIGHT}",
        "device": f"{_GPU_NAME} ({str(_DTYPE).replace('torch.', '')})",
        "seconds": round(time.perf_counter() - started, 1),
    }
    # UI info nezobrazuje – výpis do konzoly (na Kaggle vo výstupe poslednej bunky).
    log.info("try_on: %s", info)
    return {"result": result_full, "mask": mask_full, "info": info}
