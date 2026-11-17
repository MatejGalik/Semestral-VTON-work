"""Konfigurácia UI: typy oblečenia, predvolené hodnoty, texty a farby."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent

# --------------------------------------------------------------------------- #
# Typ oblečenia: (popis v UI, hodnota pre backend). Hodnoty sa nesmú meniť –
# používa ich aj API endpoint /try_on. "auto" = typ určia pravidlá (src/cloth_type.py).
# --------------------------------------------------------------------------- #
CLOTH_TYPES = [
    ("Automaticky", "auto"),
    ("Vrchný diel", "upper"),
    ("Spodný diel", "lower"),
    ("Celé / šaty", "overall"),
]
DEFAULT_CLOTH_TYPE = "auto"

# Generovanie – v UI sa nenastavuje (skryté vstupy API /try_on)
STEPS_DEFAULT = 30
SEED_DEFAULT = 42  # -1 = náhodný seed
PASTE_FACE_DEFAULT = True
SHOW_MASK_DEFAULT = False
RESPECT_FIT_DEFAULT = True  # rozšírenie masky podľa strihu (len UI, API vždy zapnuté)

EXPECTED_SECONDS = 15  # orientačný čas generovania na GPU

# --------------------------------------------------------------------------- #
# Farby (zdieľané medzi ui/theme.py a ui/styles.css – CSS ich dostáva ako
# premenné, takže stačí meniť tu).
# --------------------------------------------------------------------------- #
COLORS = {
    "bg": "#FAF8F5",          # slonovinové pozadie
    "card": "#FFFFFF",        # karty
    "border": "#E7E2DA",      # jemné okraje
    "text": "#1C1C1C",        # takmer čierna
    "text_muted": "#7A746C",  # teplá sivá
    "accent": "#B08D57",      # karamelová / zlatá
    "accent_soft": "#F3ECE1", # veľmi jemný tón akcentu (pozadie vybraného)
}

FONT_HEADING = "Cormorant Garamond"
FONT_BODY = "Inter"

# --------------------------------------------------------------------------- #
# Texty UI (všetko po slovensky)
# --------------------------------------------------------------------------- #
TEXTS = {
    "page_title": "Virtuálne skúšanie oblečenia",
    "title": "Virtuálne skúšanie oblečenia",
    "subtitle": "Vyber fotku seba a kus oblečenia.",
    "demo_badge": "Demo",
    "demo_badge_tip": "Model nie je načítaný – výsledky sú len ukážkové.",
    "demo_watermark": "DEMO režim",
    # Taby
    # Karty
    "card_person": "Osoba",
    "card_product": "Produkt",
    "card_result": "Výsledok",
    "person_placeholder": "Nahraj celopostavovú fotku",
    "product_placeholder": "Nahraj fotku oblečenia",
    "result_placeholder": "Tu sa zobrazí výsledok",
    "mask_label": "Maska",
    # Ovládanie
    "cloth_type": "Typ oblečenia",
    "submit": "Vyskúšať",
    "download": "Stiahnuť",
    "generating": f"Generujem… (≈{EXPECTED_SECONDS} s)",
    "done_in": "Vygenerované za {seconds:.1f} s",
    "detected_type": "rozpoznané: {label}",
    "show_mask": "Zobraziť masku",
    "respect_fit": "Respektovať strih",
    "respect_fit_info": "Široké nohavice rozšíria masku. Vypni na porovnanie.",
    # Chyby (gr.Warning)
    "warning_title": "Upozornenie",
    "error_title": "Chyba",
    "info_title": "Info",
    "err_no_person": "Chýba fotka osoby.",
    "err_no_cloth": "Chýba fotka oblečenia.",
    "err_bad_type": "Neznámy typ oblečenia.",
    "err_type_unknown": "Typ oblečenia sa nepodarilo rozpoznať ({reason}). Vyber ho ručne.",
    "err_backend": "Generovanie zlyhalo: {error}",
    "err_oom": "Na GPU došla pamäť. Skús menej krokov alebo menšiu fotku.",
}

