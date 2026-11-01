"""Virtuálne skúšanie oblečenia – Gradio UI.

Spustenie:
    python app.py            # lokálne (bez GPU beží v DEMO režime)
    python app.py --share    # Kaggle / Colab – vráti verejný link
    python app.py --demo     # vynúti DEMO režim aj keď backend existuje

API endpoint `/try_on` (pre rozšírenie do prehliadača) – poradie vstupov je pevné:
    person_img, cloth_img, cloth_type ("upper"|"lower"|"overall"),
    steps (int), seed (int, -1 = náhodný), paste_face (bool), show_mask (bool)
výstupy: result, mask, download (súbor PNG), status (text)
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import tempfile
import time
import traceback
from datetime import datetime
from pathlib import Path

import gradio as gr
from PIL import Image, ImageDraw, ImageFont

from config import CLOTH_TYPES, TEXTS as T
from ui.tabs import build_about_tab, build_future_tabs, build_header, build_tryon_tab
from ui.theme import FORCE_LIGHT_JS, build_theme, page_head

log = logging.getLogger("vton")
CSS_PATH = Path(__file__).resolve().parent / "ui" / "styles.css"
RESULTS_DIR = Path(tempfile.gettempdir()) / "vton_results"
VALID_TYPES = {value for _, value in CLOTH_TYPES}


# --------------------------------------------------------------------------- #
# Stub backend (DEMO režim)
# --------------------------------------------------------------------------- #
def _font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans-Bold.ttf", "arialbd.ttf", "Arial Bold.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def _stub_try_on(person_img, cloth_img, cloth_type, **options) -> dict:
    """Vráti fotku osoby s nápisom „DEMO režim“ a hrubú masku podľa typu."""
    time.sleep(1.5)  # nech je vidieť stav „Generujem…“
    img = person_img.convert("RGB").copy()
    w, h = img.size

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    band_h = max(40, h // 10)
    top = h - band_h - h // 20
    draw.rectangle([0, top, w, top + band_h], fill=(28, 28, 28, 170))
    font = _font(int(band_h * 0.5))
    text = T["demo_watermark"]
    tw, th = draw.textbbox((0, 0), text, font=font)[2:]
    draw.text(((w - tw) / 2, top + (band_h - th) / 2 - th * 0.1), text,
              font=font, fill=(255, 255, 255, 255))
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")

    region = {"upper": (0.18, 0.55), "lower": (0.45, 0.95), "overall": (0.18, 0.95)}
    y0, y1 = region.get(cloth_type, (0.18, 0.95))
    mask = Image.new("L", img.size, 0)
    ImageDraw.Draw(mask).rectangle([w * 0.2, h * y0, w * 0.8, h * y1], fill=255)

    return {"result": img, "mask": mask, "info": {"demo": True, **options}}


def _load_backend():
    try:
        from src.try_on import try_on  # noqa: WPS433 – zámerne lenivý import

        return try_on, False
    except Exception as exc:  # ImportError, chýbajúce CUDA, váhy…
        log.warning("Backend sa nepodarilo načítať, beží DEMO režim: %s", exc)
        return _stub_try_on, True


BACKEND, DEMO_MODE = _stub_try_on, True  # nastaví sa v main()


# --------------------------------------------------------------------------- #
# Adaptér – jediné miesto, kde UI volá backend
# --------------------------------------------------------------------------- #
def _warn(message: str) -> None:
    gr.Warning(message, title=T["warning_title"])


def _no_change(status: str = ""):
    return gr.skip(), gr.skip(), gr.skip(), status


def run_tryon(person_img, cloth_img, cloth_type, steps, seed, paste_face, show_mask,
              progress=gr.Progress()):
    if person_img is None:
        _warn(T["err_no_person"])
        return _no_change()
    if cloth_img is None:
        _warn(T["err_no_cloth"])
        return _no_change()
    if cloth_type not in VALID_TYPES:
        _warn(T["err_bad_type"])
        return _no_change()

    seed = int(seed) if seed is not None else -1
    if seed < 0:
        seed = random.randint(0, 2**31 - 1)

    progress(0, desc=T["generating"])
    started = time.perf_counter()
    try:
        out = BACKEND(
            person_img, cloth_img, cloth_type,
            steps=int(steps), seed=seed, paste_face=bool(paste_face),
        )
        result = out["result"]
    except Exception as exc:
        traceback.print_exc()
        if "OutOfMemory" in type(exc).__name__ or "out of memory" in str(exc).lower():
            _warn(T["err_oom"])
        else:
            _warn(T["err_backend"].format(error=exc))
        return _no_change()
    elapsed = time.perf_counter() - started

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"vyskusanie_{datetime.now():%Y%m%d_%H%M%S}_{seed}.png"
    result.save(path)

    mask = out.get("mask")
    status = f'{T["done_in"].format(seconds=elapsed)} · seed {seed}'
    return (
        result,
        gr.update(value=mask, visible=bool(show_mask and mask is not None)),
        gr.update(value=str(path), visible=True),
        status,
    )


# --------------------------------------------------------------------------- #
# Skladanie aplikácie
# --------------------------------------------------------------------------- #
def build_app(demo_mode: bool) -> gr.Blocks:
    with gr.Blocks(title=T["page_title"], fill_width=False) as app:
        with gr.Column(elem_classes="vton-page"):
            build_header(demo_mode)
            with gr.Tabs(elem_classes="vton-tabs"):
                build_tryon_tab(run_tryon)
                build_future_tabs()
                build_about_tab()
    return app


def main(argv=None) -> None:
    global BACKEND, DEMO_MODE

    parser = argparse.ArgumentParser(description="Virtuálne skúšanie oblečenia – UI")
    parser.add_argument("--share", action="store_true", help="verejný Gradio link")
    parser.add_argument("--demo", action="store_true", help="vynútiť DEMO režim")
    parser.add_argument("--host", default=None, help="napr. 0.0.0.0")
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    BACKEND, DEMO_MODE = (_stub_try_on, True) if args.demo else _load_backend()

    app = build_app(DEMO_MODE)
    app.queue(default_concurrency_limit=1)
    app.launch(
        share=args.share,
        server_name=args.host,
        server_port=args.port,
        theme=build_theme(),
        head=page_head(),
        css_paths=[CSS_PATH],
        js=FORCE_LIGHT_JS,
        allowed_paths=[str(RESULTS_DIR)],
        footer_links=["api", "gradio"],
        show_error=True,
    )


if __name__ == "__main__":
    sys.exit(main())
