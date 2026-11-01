"""Vlastná Gradio téma – svetlý, pokojný štýl módneho e-shopu."""

from pathlib import Path

import gradio as gr

from config import COLORS, FONT_BODY, FONT_HEADING, TEXTS

C = COLORS

# Zaoblenie: 12 px pre karty aj tlačidlá
RADIUS = gr.themes.Size(
    xxs="4px", xs="6px", sm="8px", md="12px", lg="12px", xl="12px", xxl="16px"
)

CARD_SHADOW = "0 1px 3px rgba(0,0,0,.06)"


def build_theme() -> gr.themes.Base:
    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.stone,
        secondary_hue=gr.themes.colors.stone,
        neutral_hue=gr.themes.colors.stone,
        radius_size=RADIUS,
        spacing_size=gr.themes.sizes.spacing_lg,
        text_size=gr.themes.sizes.text_md,
        font=[gr.themes.GoogleFont(FONT_BODY, weights=(400, 500, 600)),
              "system-ui", "sans-serif"],
        font_mono=["ui-monospace", "Consolas", "monospace"],
    )

    # Pätkové písmo na nadpisy – načítané tiež cez GoogleFont, CSS ho použije
    # v ui/styles.css (--font-heading).
    heading = gr.themes.GoogleFont(FONT_HEADING, weights=(400, 500, 600))
    theme._stylesheets.append(heading.stylesheet()["url"])

    theme.set(
        # Pozadie a text
        body_background_fill=C["bg"],
        body_text_color=C["text"],
        body_text_color_subdued=C["text_muted"],
        background_fill_primary=C["card"],
        background_fill_secondary=C["bg"],
        border_color_primary=C["border"],
        border_color_accent=C["accent"],
        color_accent=C["accent"],
        color_accent_soft=C["accent_soft"],
        link_text_color=C["accent"],
        link_text_color_hover=C["text"],
        # Karty / bloky
        block_background_fill=C["card"],
        block_border_color=C["border"],
        block_border_width="1px",
        block_shadow=CARD_SHADOW,
        block_radius="12px",
        block_label_background_fill=C["card"],
        block_label_text_color=C["text_muted"],
        block_label_border_color=C["border"],
        block_title_text_color=C["text_muted"],
        block_info_text_color=C["text_muted"],
        panel_background_fill=C["card"],
        panel_border_color=C["border"],
        # Vstupy
        input_background_fill=C["card"],
        input_border_color=C["border"],
        input_border_width="1px",
        input_border_color_focus=C["accent"],
        input_shadow="none",
        input_shadow_focus=f"0 0 0 2px {C['accent_soft']}",
        checkbox_background_color=C["card"],
        checkbox_border_color=C["border"],
        checkbox_background_color_selected=C["accent"],
        checkbox_border_color_selected=C["accent"],
        checkbox_border_color_focus=C["accent"],
        checkbox_label_background_fill=C["card"],
        checkbox_label_background_fill_hover=C["bg"],
        checkbox_label_background_fill_selected=C["accent_soft"],
        checkbox_label_border_color=C["border"],
        checkbox_label_border_color_selected=C["accent"],
        checkbox_label_text_color_selected=C["text"],
        slider_color=C["accent"],
        # Hlavné tlačidlo: čierne, pri hoveri akcent
        button_primary_background_fill=C["text"],
        button_primary_background_fill_hover=C["accent"],
        button_primary_text_color="#FFFFFF",
        button_primary_text_color_hover="#FFFFFF",
        button_primary_border_color=C["text"],
        button_primary_border_color_hover=C["accent"],
        button_primary_shadow="none",
        button_primary_shadow_hover="none",
        # Vedľajšie tlačidlá (Stiahnuť): biele s tenkým okrajom
        button_secondary_background_fill=C["card"],
        button_secondary_background_fill_hover=C["bg"],
        button_secondary_text_color=C["text"],
        button_secondary_border_color=C["border"],
        button_secondary_border_color_hover=C["accent"],
        button_border_width="1px",
        button_large_radius="12px",
        button_medium_radius="12px",
        button_small_radius="8px",
        # Žiadne gradienty v loaderi
        loader_color=C["accent"],
    )
    return theme


def color_vars_css() -> str:
    """CSS premenné --vton-* z config.COLORS (používa ui/styles.css)."""
    names = {
        "bg": "bg", "card": "card", "border": "border", "text": "text",
        "text_muted": "muted", "accent": "accent", "accent_soft": "accent-soft",
    }
    body = "".join(f"--vton-{css}: {C[key]};" for key, css in names.items())
    return f":root {{{body}}}"


PAGE_CSS = Path(__file__).resolve().parent / "page.css"


def page_head() -> str:
    """<style> do <head>: farebné premenné + globálne pravidlá z ui/page.css.

    Ide mimo `css_paths`, pretože Gradio tamojšie pravidlá scopuje pod
    .gradio-container a pravidlá pre body / samotný kontajner by nezabrali.
    """
    return (
        f"<style>{color_vars_css()}\n{toast_titles_css()}\n"
        f"{PAGE_CSS.read_text(encoding='utf-8')}</style>"
    )


def toast_titles_css() -> str:
    """Slovenské nadpisy toastov.

    Gradio 6.x v toaste zobrazuje typ správy („Warning“) a parameter `title`
    ignoruje, preto text nahrádzame cez CSS.
    """
    titles = {"warning": TEXTS["warning_title"], "error": TEXTS["error_title"],
              "info": TEXTS["info_title"]}
    rules = [".toast-title{font-size:0 !important}"]
    rules += [
        f'.toast-title.{kind}::before{{content:"{text}";font-size:var(--text-md,14px)}}'
        for kind, text in titles.items()
    ]
    return "\n".join(rules)


# Gradio pri tmavej systémovej téme pridá triedu .dark – UI je navrhnuté len
# svetlé, preto ju odstraňujeme.
FORCE_LIGHT_JS = """
() => {
  const strip = () => {
    document.documentElement.classList.remove('dark');
    document.body.classList.remove('dark');
  };
  strip();
  new MutationObserver(strip).observe(document.body, {attributes: true, attributeFilter: ['class']});
  new MutationObserver(strip).observe(document.documentElement, {attributes: true, attributeFilter: ['class']});
}
"""
