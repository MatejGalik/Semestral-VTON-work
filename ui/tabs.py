"""Stavba UI – jedna stránka skúšania (bez tabov).

Stránka volá backend len cez funkciu `run_fn` (adaptér `run_tryon` z app.py).
"""

from __future__ import annotations

import html
from typing import Callable

import gradio as gr

from config import (
    CLOTH_TYPES,
    DEFAULT_CLOTH_TYPE,
    PASTE_FACE_DEFAULT,
    RESPECT_FIT_DEFAULT,
    USE_CLIP_DEFAULT,
    SEED_DEFAULT,
    SHOW_MASK_DEFAULT,
    STEPS_DEFAULT,
    TEXTS as T,
)


# --------------------------------------------------------------------------- #
# Hlavička
# --------------------------------------------------------------------------- #
def build_header(demo: bool) -> None:
    badge = (
        f'<span class="vton-badge" title="{html.escape(T["demo_badge_tip"])}">'
        f'{html.escape(T["demo_badge"])}</span>'
        if demo
        else ""
    )
    gr.HTML(
        f"""
        <header class="vton-header">
          <div class="vton-header-row">
            <h1 class="vton-title">{html.escape(T["title"])}</h1>
            {badge}
          </div>
          <p class="vton-subtitle">{html.escape(T["subtitle"])}</p>
        </header>
        """,
        elem_classes="vton-header-wrap",
        padding=False,
    )


def _card_title(text: str) -> None:
    gr.HTML(f'<div class="vton-card-title">{html.escape(text)}</div>', padding=False)


def _separator(symbol: str) -> None:
    with gr.Column(scale=0, min_width=40, elem_classes="vton-sep-col"):
        gr.HTML(f'<div class="vton-sep" aria-hidden="true">{symbol}</div>', padding=False)


# --------------------------------------------------------------------------- #
# Stránka skúšania
# --------------------------------------------------------------------------- #
def build_tryon(run_fn: Callable) -> None:
    # --- Typ oblečenia (nad kartami) ---------------------------------- #
    with gr.Column(elem_classes="vton-controls vton-controls-top"):
        cloth_type = gr.Radio(
            choices=CLOTH_TYPES,
            value=DEFAULT_CLOTH_TYPE,
            label=T["cloth_type"],
            show_label=True,
            elem_classes="vton-segmented",
        )

    # --- Tri karty: OSOBA + PRODUKT = VÝSLEDOK ------------------------- #
    with gr.Row(elem_classes="vton-cards", equal_height=False):
        with gr.Column(scale=1, min_width=260, elem_classes="vton-card"):
            _card_title(T["card_person"])
            person = gr.Image(
                type="pil",
                image_mode="RGB",
                sources=["upload", "clipboard"],
                show_label=False,
                container=False,
                placeholder=T["person_placeholder"],
                height=440,
                elem_classes="vton-image",
            )

        _separator("+")

        with gr.Column(scale=1, min_width=260, elem_classes="vton-card"):
            _card_title(T["card_product"])
            cloth = gr.Image(
                type="pil",
                image_mode="RGBA",  # priehľadné pozadie produktu → silueta z alfa kanála
                sources=["upload", "clipboard"],
                show_label=False,
                container=False,
                placeholder=T["product_placeholder"],
                height=440,
                elem_classes="vton-image",
            )

        _separator("=")

        with gr.Column(scale=1, min_width=260, elem_classes="vton-card vton-card-result"):
            _card_title(T["card_result"])
            result = gr.Image(
                type="pil",
                format="png",
                show_label=False,
                container=False,
                interactive=False,
                buttons=["fullscreen"],
                height=440,
                elem_classes="vton-image",
            )
            mask = gr.Image(
                type="pil",
                format="png",
                label=T["mask_label"],
                interactive=False,
                buttons=[],
                height=200,
                visible=False,
                elem_classes="vton-mask",
            )
            with gr.Row(elem_classes="vton-result-actions"):
                download = gr.DownloadButton(
                    T["download"],
                    variant="secondary",
                    size="md",
                    visible=False,
                    elem_classes="vton-download",
                )
                status = gr.Markdown("", elem_classes="vton-meta")

    # --- Hlavné tlačidlo --------------------------------------------- #
    with gr.Column(elem_classes="vton-controls"):
        submit = gr.Button(
            T["submit"], variant="primary", size="lg", elem_classes="vton-submit"
        )

    # --- Prepínače (pod tlačidlom) ------------------------------------- #
    with gr.Row(elem_classes="vton-controls vton-switches"):
        show_mask = gr.Checkbox(value=SHOW_MASK_DEFAULT, label=T["show_mask"])
        respect_fit = gr.Checkbox(value=RESPECT_FIT_DEFAULT, label=T["respect_fit"],
                                  info=T["respect_fit_info"])
        use_clip = gr.Checkbox(value=USE_CLIP_DEFAULT, label=T["use_clip"],
                               info=T["use_clip_info"])

    # Skryté vstupy – v UI sa nenastavujú, ale sú súčasť API /try_on (poradie vstupov)
    steps = gr.Number(value=STEPS_DEFAULT, precision=0, visible=False)
    seed = gr.Number(value=SEED_DEFAULT, precision=0, visible=False)
    paste_face = gr.Checkbox(value=PASTE_FACE_DEFAULT, visible=False)

    # --- Udalosti ----------------------------------------------------- #
    def _start():
        return gr.update(interactive=False), T["generating"]

    def _finish():
        return gr.update(interactive=True)

    # Poradie vstupov endpointu /try_on je súčasť API – NEMENIŤ.
    api_inputs = [person, cloth, cloth_type, steps, seed, paste_face, show_mask]
    api_outputs = [result, mask, download, status]

    # Tlačidlo v UI posiela navyše prepínače `respect_fit`, `use_clip` (súkromná udalosť).
    (
        submit.click(_start, None, [submit, status], queue=False,
                     api_visibility="private")
        .then(run_fn, api_inputs + [respect_fit, use_clip], api_outputs,
              api_visibility="private", concurrency_limit=1, concurrency_id="try_on",
              show_progress="full", show_progress_on=[result])
        .then(_finish, None, submit, queue=False, api_visibility="private")
    )
    # Verejný endpoint /try_on – skryté tlačidlo, vstupy bez zmeny (strih vždy zapnutý).
    api_trigger = gr.Button(visible=False)
    api_trigger.click(run_fn, api_inputs, api_outputs, api_name="try_on",
                      api_visibility="public", concurrency_limit=1,
                      concurrency_id="try_on")

    show_mask.change(
        lambda show, m: gr.update(visible=bool(show and m is not None)),
        [show_mask, mask], mask, queue=False, api_visibility="private",
    )
