# Virtuálne skúšanie oblečenia (VTON)

Semestrálny projekt: používateľ nahrá fotku osoby a fotku oblečenia a model
**CatVTON** vygeneruje fotku osoby v novom oblečení. UI je postavené na
**Gradio 6** (`gr.Blocks`).

## Požiadavky

- Python **3.10+**
- Pre skutočné generovanie GPU s CUDA (napr. Kaggle). Bez GPU beží UI v DEMO režime.

```bash
pip install -r requirements.txt
```

## Spustenie lokálne

```bash
python app.py
```

Otvor <http://127.0.0.1:7860>.

Ak `src/try_on.py` neexistuje alebo jeho import zlyhá (napr. chýba GPU alebo
váhy modelu), aplikácia sa aj tak spustí v **DEMO režime**. V hlavičke je štítok
„Demo“ a výsledkom je fotka osoby s nápisom „DEMO režim“. Dôvod zlyhania importu
sa vypíše do konzoly.

## Spustenie na Windows (Git Bash)

Git Bash je bash na Windows a používa bežný Windows Python. Postup:

```bash
# 1. prejdi do priečinka projektu (disk C: je v Git Bash /c/)
cd /c/REPOS/Semestral-VTON-work

# 2. vytvor a aktivuj virtuálne prostredie (stačí raz)
python -m venv .venv               # na Windows "python", nie "python3"
source .venv/Scripts/activate      # na Windows "Scripts", nie "bin"

# 3. nainštaluj závislosti (stačí raz)
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. spusti
python app.py
```

Potom otvor <http://127.0.0.1:7860>. Aplikáciu zastavíš klávesmi **Ctrl+C**.

Pri ďalšom spustení stačí:

```bash
cd /c/REPOS/Semestral-VTON-work
source .venv/Scripts/activate
python app.py
```

Na začiatku riadku sa po aktivácii objaví `(.venv)`. Prostredie opustíš príkazom
`deactivate`.

**Je virtuálne prostredie nutné?** Na Windows nie, `pip install -r requirements.txt`
funguje aj priamo. Prostredie len oddelí verzie knižníc (Gradio, neskôr PyTorch)
od ostatných projektov a celé sa dá zmazať cez `rm -rf .venv`.

**Časté problémy**

- `python: command not found` alebo sa otvorí Microsoft Store: Python nie je
  v `PATH`. Preinštaluj ho z <https://www.python.org> so zaškrtnutým
  **Add python.exe to PATH**, alebo skús príkaz `py` namiesto `python`.
- Port 7860 je obsadený: `python app.py --port 7861`.
- Chceš link pre niekoho iného: `python app.py --share`.

> **Máš Git Bash alebo WSL?** Spusti `uname -a`. Ak výstup obsahuje `MINGW64`,
> je to Git Bash a platí postup vyššie. Ak obsahuje `Linux … microsoft`, je to
> WSL (skutočný Linux): použi `python3`, `source .venv/bin/activate`
> a virtuálne prostredie je tam povinné.

### Prepínače

| Prepínač        | Význam                                                  |
|-----------------|---------------------------------------------------------|
| `--share`       | vytvorí verejný link `https://….gradio.live` (na 1 týždeň) |
| `--demo`        | vynúti DEMO režim, aj keď backend existuje              |
| `--host 0.0.0.0`| počúva na všetkých rozhraniach (napr. v Dockeri)        |
| `--port 7861`   | iný port, ak je 7860 obsadený                           |

## Spustenie na Kaggle

1. V notebooku zapni **Settings → Accelerator → GPU** a **Internet → On**
   (bez internetu nevznikne verejný link).
2. V bunke:

   ```bash
   !git clone <URL-repozitára> vton
   %cd vton
   !pip install -q -r requirements.txt
   !python app.py --share
   ```

3. Vo výstupe bunky sa objaví `Running on public URL: https://….gradio.live`.
   Tento link otvor v prehliadači.

Bunka beží, kým je aplikácia spustená. Zastavíš ju tlačidlom **Stop** (■).

## Použitie

1. Nahraj fotku osoby (karta **Osoba**) a fotku oblečenia (karta **Produkt**),
   alebo klikni na príklad v sekcii **Príklady**.
2. Vyber typ oblečenia: Vrchný diel / Spodný diel / Celé / šaty.
3. Klikni **Vyskúšať**. Výsledok si stiahneš tlačidlom **Stiahnuť**.

V sekcii **Pokročilé nastavenia** sú počet krokov (20–50), seed (−1 = náhodný),
vloženie pôvodnej tváre a zobrazenie masky.

## API

Aplikácia má jeden verejný endpoint `/try_on` (viditeľný cez „Use via API“
v pätičke). Názov ani poradie vstupov sa nemenia, používa ho rozšírenie do
prehliadača.

```python
from gradio_client import Client, handle_file

client = Client("https://….gradio.live/")
result, mask, download, status = client.predict(
    handle_file("osoba.jpg"),     # person_img
    handle_file("tricko.jpg"),    # cloth_img
    "upper",                      # cloth_type: "upper" | "lower" | "overall"
    30,                           # steps (20–50)
    42,                           # seed (-1 = náhodný)
    True,                         # paste_face
    False,                        # show_mask
    api_name="/try_on",
)
```

## Napojenie modelu

UI volá backend len cez adaptér `run_tryon()` v `app.py`, ktorý importuje:

```python
# src/try_on.py
def try_on(person_img: PIL.Image, cloth_img: PIL.Image, cloth_type: str,
           **options) -> dict:
    # options: steps: int, seed: int, paste_face: bool
    # návrat: {"result": PIL.Image, "mask": PIL.Image | None, "info": dict}
```

Keď tento súbor existuje a dá sa importovať, DEMO režim sa automaticky vypne.

## Štruktúra

```
app.py            # spustenie, adaptér run_tryon(), skladanie tabov
config.py         # FEATURES, texty UI, farby, predvolené hodnoty
ui/theme.py       # Gradio téma (farby, písma)
ui/styles.css     # štýly komponentov (karty, „+“ a „=“, segmentové tlačidlá)
ui/page.css       # globálne štýly stránky (vkladané do <head>)
ui/tabs.py        # build_tryon_tab(), build_about_tab(), budúce taby
assets/examples/  # ukážkové osoby a produkty
src/              # backend (model) – mimo UI
```

### Budúce funkcie

V `config.py` stačí prepnúť flag na `True` a zobrazí sa nový tab:

```python
FEATURES = {
    "compare_masks": False,   # Porovnanie: AutoMasker vs. naša maska
    "compare_models": False,  # CatVTON / IDM-VTON
    "history": False,         # história výsledkov v session
    "metrics": False,         # ArcFace podobnosť tváre
}
```

Obsah tabov sa dopĺňa vo funkciách `build_<nazov>_tab()` v `ui/tabs.py`.
