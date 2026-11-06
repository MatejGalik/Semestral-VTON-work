"""Stavba jednotlivých tabov UI.

Tab Skúšanie volá backend len cez funkciu `run_fn` (adaptér `run_tryon` z app.py).
"""

from __future__ import annotations

import html
from typing import Callable

import gradio as gr

from config import (
    ABOUT,
    CLOTH_TYPES,
    DEFAULT_CLOTH_TYPE,
    EXAMPLES_DIR,
    FEATURES,
    PASTE_FACE_DEFAULT,
    SEED_DEFAULT,
    SHOW_MASK_DEFAULT,
    STEPS_DEFAULT,
    STEPS_MAX,
    STEPS_MIN,
    TEXTS as T,
)

# Príklady: (osoba, produkt, typ). Chýbajúce súbory sa preskočia.
EXAMPLES = [
    ("person_1.png", "cloth_upper_1.png", "upper"),
    ("person_2.png", "cloth_lower_1.png", "lower"),
    ("person_3.png", "cloth_overall_1.png", "overall"),
]


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
# Tab: Skúšanie
# --------------------------------------------------------------------------- #
def build_tryon_tab(run_fn: Callable) -> None:
    with gr.Tab(T["tab_tryon"], id="tryon"):
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
                    image_mode="RGB",
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

        # --- Pokročilé nastavenia ----------------------------------------- #
        with gr.Accordion(T["advanced"], open=False, elem_classes="vton-accordion"):
            with gr.Row():
                steps = gr.Slider(
                    STEPS_MIN, STEPS_MAX, value=STEPS_DEFAULT, step=1,
                    label=T["steps"], info=T["steps_info"], min_width=220,
                )
                seed = gr.Number(
                    value=SEED_DEFAULT, precision=0, minimum=-1,
                    label=T["seed"], info=T["seed_info"], min_width=160,
                )
            with gr.Row():
                paste_face = gr.Checkbox(value=PASTE_FACE_DEFAULT, label=T["paste_face"])
                show_mask = gr.Checkbox(value=SHOW_MASK_DEFAULT, label=T["show_mask"])

        # --- Príklady ----------------------------------------------------- #
        examples = [
            [str(EXAMPLES_DIR / p), str(EXAMPLES_DIR / c), t]
            for p, c, t in EXAMPLES
            if (EXAMPLES_DIR / p).exists() and (EXAMPLES_DIR / c).exists()
        ]
        if examples:
            with gr.Accordion(T["examples"], open=False, elem_classes="vton-accordion"):
                gr.Markdown(T["examples_hint"], elem_classes="vton-hint")
                gr.Examples(
                    examples=examples,
                    inputs=[person, cloth, cloth_type],
                    label=None,
                    cache_examples=False,
                    examples_per_page=6,
                )

        # --- Udalosti ----------------------------------------------------- #
        def _start():
            return gr.update(interactive=False), T["generating"]

        def _finish():
            return gr.update(interactive=True)

        # Poradie vstupov endpointu /try_on je súčasť API – NEMENIŤ.
        api_inputs = [person, cloth, cloth_type, steps, seed, paste_face, show_mask]
        api_outputs = [result, mask, download, status]

        (
            submit.click(_start, None, [submit, status], queue=False,
                         api_visibility="private")
            .then(run_fn, api_inputs, api_outputs, api_name="try_on",
                  api_visibility="public", concurrency_limit=1,
                  show_progress="full", show_progress_on=[result])
            .then(_finish, None, submit, queue=False, api_visibility="private")
        )

        show_mask.change(
            lambda show, m: gr.update(visible=bool(show and m is not None)),
            [show_mask, mask], mask, queue=False, api_visibility="private",
        )


# --------------------------------------------------------------------------- #
# Tab: O projekte (vždy viditeľný)
# --------------------------------------------------------------------------- #
def build_about_tab() -> None:
    with gr.Tab(T["tab_about"], id="about"):
        steps = "".join(f"<li>{html.escape(s)}</li>" for s in ABOUT["pipeline"])
        authors = ", ".join(html.escape(a) for a in ABOUT["authors"])
        gr.HTML(
            f"""
            <article class="vton-about">
              <h2>{html.escape(T["tab_about"])}</h2>
              <p>{html.escape(ABOUT["description"])}</p>
              <h3>Ako to funguje</h3>
              <ol>{steps}</ol>
              <h3>Technológie</h3>
              <p>{html.escape(ABOUT["tech"])}</p>
              <h3>Autori</h3>
              <p>{authors}</p>
            </article>
            """,
            padding=False,
        )


# --------------------------------------------------------------------------- #
# Budúce taby – zobrazia sa len pri FEATURES[<názov>] = True
# --------------------------------------------------------------------------- #
def _todo(text: str) -> None:
    gr.Markdown(f"_{text}_", elem_classes="vton-hint")


def build_compare_masks_tab() -> None:
    # TODO: vedľa seba AutoMasker (CatVTON) vs. naša maska pre rovnakú osobu,
    #       prípadne prekrytie masky cez fotku a IoU medzi maskami.
    with gr.Tab(T["tab_compare_masks"], id="compare_masks"):
        _todo("Porovnanie masiek pripravujeme.")


def build_compare_models_tab() -> None:
    # TODO: výber modelu (CatVTON / IDM-VTON) a výsledky oboch vedľa seba.
    #       Backend bude potrebovať voliteľný parameter `model` v try_on().
    with gr.Tab(T["tab_compare_models"], id="compare_models"):
        _todo("Porovnanie modelov pripravujeme.")


def build_history_tab() -> None:
    # TODO: galéria posledných výsledkov v session (gr.State + gr.Gallery),
    #       napojiť na výstup endpointu try_on cez .then().
    with gr.Tab(T["tab_history"], id="history"):
        _todo("Históriu výsledkov pripravujeme.")


def build_metrics_tab() -> None:
    # TODO: ArcFace podobnosť tváre pôvodnej fotky a výsledku
    #       (backend môže vracať hodnotu v result["info"]).
    with gr.Tab(T["tab_metrics"], id="metrics"):
        _todo("Metriky pripravujeme.")


FUTURE_TABS: dict[str, Callable[[], None]] = {
    "compare_masks": build_compare_masks_tab,
    "compare_models": build_compare_models_tab,
    "history": build_history_tab,
    "metrics": build_metrics_tab,
}


def build_future_tabs() -> None:
    for name, builder in FUTURE_TABS.items():
        if FEATURES.get(name):
            builder()
