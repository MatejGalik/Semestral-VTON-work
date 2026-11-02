"""Backend virtuálneho skúšania – CatVTON so vstavaným maskovaním (AutoMasker).

Kontrakt (používa ho app.py a API /try_on):
    try_on(person_img, cloth_img, cloth_type, **options) -> {"result", "mask", "info"}

    person_img, cloth_img : PIL.Image
    cloth_type            : "upper" | "lower" | "overall"
    options               : steps (int), seed (int), paste_face (bool),
                            guidance_scale (float, voliteľné)

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

    # 1) Maska – vstavaný AutoMasker (DensePose + SCHP)
    mask = _automasker(person_r, cloth_type)["mask"]
    mask_soft = _mask_processor.blur(mask, blur_factor=MASK_BLUR)

    # 2) Generovanie
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

    # 3) Mimo masky vrátiť pôvodné pixely (tvár, ruky, pozadie zostanú ostré)
    if paste_face:
        result = repaint_result(result, person_r, mask_soft)

    # 4) Späť do pôvodného rozmeru fotky
    result_full = _paste_back(person, result, box)
    mask_full = _paste_back(Image.new("L", person.size, 0), mask.convert("L"), box)

    return {
        "result": result_full,
        "mask": mask_full,
        "info": {
            "model": "CatVTON mix-48k-1024",
            "steps": steps,
            "seed": seed,
            "guidance_scale": guidance,
            "paste_face": paste_face,
            "resolution": f"{WIDTH}x{HEIGHT}",
            "device": f"{_GPU_NAME} ({str(_DTYPE).replace('torch.', '')})",
            "seconds": round(time.perf_counter() - started, 1),
        },
    }
