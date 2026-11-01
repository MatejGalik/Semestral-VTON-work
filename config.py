"""Konfigurácia UI: prepínače funkcií, texty a farby.

Zmena flagu vo FEATURES zobrazí príslušný tab bez ďalších úprav kódu.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
ASSETS_DIR = ROOT / "assets"
EXAMPLES_DIR = ASSETS_DIR / "examples"

# --------------------------------------------------------------------------- #
# Budúce funkcie – kým je flag False, tab sa vôbec nevytvorí.
# --------------------------------------------------------------------------- #
FEATURES = {
    "compare_masks": False,   # tab Porovnanie: AutoMasker vs. naša maska
    "compare_models": False,  # výber modelu (CatVTON / IDM-VTON)
    "history": False,         # história posledných výsledkov v session
    "metrics": False,         # zobrazenie ArcFace podobnosti tváre
}

# --------------------------------------------------------------------------- #
# Typ oblečenia: (popis v UI, hodnota pre backend). Hodnoty sa nesmú meniť –
# používa ich aj API endpoint /try_on.
# --------------------------------------------------------------------------- #
CLOTH_TYPES = [
    ("Vrchný diel", "upper"),
    ("Spodný diel", "lower"),
    ("Celé / šaty", "overall"),
]
DEFAULT_CLOTH_TYPE = "upper"

# Pokročilé nastavenia
STEPS_MIN, STEPS_MAX, STEPS_DEFAULT = 20, 50, 30
SEED_DEFAULT = 42  # -1 = náhodný seed
PASTE_FACE_DEFAULT = True
SHOW_MASK_DEFAULT = False

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
    "tab_tryon": "Skúšanie",
    "tab_about": "O projekte",
    "tab_compare_masks": "Porovnanie masiek",
    "tab_compare_models": "Porovnanie modelov",
    "tab_history": "História",
    "tab_metrics": "Metriky",
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
    "advanced": "Pokročilé nastavenia",
    "steps": "Počet krokov",
    "steps_info": "Viac krokov = vyššia kvalita, dlhšie čakanie.",
    "seed": "Seed",
    "seed_info": "Rovnaký seed = rovnaký výsledok. −1 = náhodný.",
    "paste_face": "Vložiť pôvodnú tvár",
    "show_mask": "Zobraziť masku",
    "examples": "Príklady",
    "examples_hint": "Klikni na príklad – predvyplní osobu aj produkt.",
    # Chyby (gr.Warning)
    "warning_title": "Upozornenie",
    "error_title": "Chyba",
    "info_title": "Info",
    "err_no_person": "Chýba fotka osoby.",
    "err_no_cloth": "Chýba fotka oblečenia.",
    "err_bad_type": "Neznámy typ oblečenia.",
    "err_backend": "Generovanie zlyhalo: {error}",
    "err_oom": "Na GPU došla pamäť. Skús menej krokov alebo menšiu fotku.",
}

# Tab „O projekte“
ABOUT = {
    "description": (
        "Semestrálny projekt, ktorý ukazuje virtuálne skúšanie oblečenia. "
        "Používateľ nahrá svoju fotku a fotku produktu a model vygeneruje "
        "realistický obrázok osoby v novom oblečení – bez fyzického skúšania."
    ),
    "pipeline": [
        "Nahratie fotky osoby a fotky oblečenia.",
        "Predspracovanie: zmena veľkosti a orezanie na pomer 3 : 4.",
        "Maska oblasti, ktorá sa má prekresliť (podľa typu oblečenia).",
        "Generovanie difúznym modelom CatVTON na GPU.",
        "Voliteľné vloženie pôvodnej tváre pre zachovanie identity.",
        "Výsledok na stiahnutie.",
    ],
    "authors": ["Matej Gálik"],
    "tech": "CatVTON · PyTorch · Gradio · Kaggle GPU",
}
