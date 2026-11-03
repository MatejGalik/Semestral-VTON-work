# CLAUDE.md

Semestrálny projekt **Virtuálne skúšanie oblečenia (VTON)**: Gradio UI + CatVTON
backend, generovanie beží na Kaggle GPU. Prehľad: `docs/architektura.md`,
backend: `docs/backend.md`.

## Komunikácia

- Píš **po slovensky**, stručne a heslovito. Používateľ nechce čítať dlhé texty.
- Analýzy úloh rob subagentom `analytik` (krátke, max. ~1 strana).

## Kontrakty – nemeniť bez výslovného súhlasu

- `try_on(person_img, cloth_img, cloth_type, **options) -> {"result", "mask", "info"}` v `src/try_on.py`.
- API endpoint `/try_on`: názov a poradie vstupov (`person, cloth, cloth_type, steps, seed, paste_face, show_mask`) – používa ho rozšírenie do prehliadača.
- Hodnoty `cloth_type`: `upper` / `lower` / `overall`.
- Lokálne bez GPU musí aplikácia bežať v DEMO režime (import backendu smie zlyhať).

## Prostredie

- Lokálne (Windows) nie je GPU ani torch – backend sa tu nedá spustiť, len skontrolovať (`python -m py_compile`) a overiť DEMO režim.
- Skutočný test: push na GitHub → `kaggle/run.ipynb` → Run All. Kód CatVTON sa klonuje za behu (`CATVTON_DIR`), nie je v repe.
- Kaggle GPU: T4 (16 GB, bez bf16) / P100 – fp16. Torch v `requirements-gpu.txt` nepripínať.

## Dokumentácia sa aktualizuje spolu s kódom

Pri **každej zmene** kódu, závislostí alebo postupu v tom istom kroku uprav aj
dokumentáciu (a commitni ju spolu so zmenou):

| Zmena | Uprav |
|---|---|
| `src/` (backend, maska, model, parametre) | `docs/backend.md` – prehľad, kontrakt, rozhodnutia |
| UI, `config.py`, `app.py`, API, nové súbory/technológie | `docs/architektura.md` |
| spúšťanie, inštalácia, Kaggle notebook, prepínače | `README.md` (+ `docs/architektura.md` ak treba) |
| nové zistenie z testovania (kvalita, chyba, výkon) | `docs/backend.md` – „Zistenia z testovania“ |
| splnený / nový plán | `docs/backend.md` – „Plánované vylepšenia“ |
| návrh dohodnutý v konverzácii (ešte neimplementovaný) | zapísať ako sekciu „Návrh: …“ so stavom *návrh*; pri implementácii priebežne upravovať podľa skutočnosti, po dokončení presunúť do opisu a zmazať stav *návrh* |

Aktualizuj aj dátum „Stav k …“ v upravenom dokumente. Ak zmena dokumentáciu
neovplyvňuje, výslovne to povedz.

## Git

- Commit a push len na požiadanie. Správy commitov po slovensky.
