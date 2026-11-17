# Architektúra a technológie

Stav k 2026-11-17. Celkový pohľad na projekt: z čoho sa skladá, aké technológie
používa, ako sa spúšťa a ako sa na ňom pracuje. Podrobnosti o modeli sú
v [backend.md](backend.md).

## Čo projekt robí

Používateľ nahrá fotku osoby a fotku oblečenia, vyberie typ (vrch / spodok /
celé) a aplikácia vygeneruje fotku osoby v novom oblečení. Ovláda sa cez webové
UI alebo cez API (pre rozšírenie do prehliadača).

## Použité technológie

| Oblasť | Technológia | Na čo |
|---|---|---|
| Jazyk | Python 3.10+ | celý projekt |
| UI | **Gradio 6** (`gr.Blocks`) | webové rozhranie, API endpoint, verejný link (`--share`) |
| Obrázky | Pillow | načítanie, orezanie, maska v DEMO režime |
| Model | **CatVTON** (`mix-48k-1024`) | virtuálne skúšanie – difúzny model na báze Stable Diffusion 1.5 inpainting |
| Maskovanie | AutoMasker z CatVTON: **DensePose** (detectron2) + **SCHP**, upravený v `src/masking.py` | automatická maska oblasti, ktorá sa prekreslí |
| Typ oblečenia (auto) | vlastné pravidlá nad siluetou (OpenCV, numpy), **bez modelu**; pri neistote ručný výber | z fotky oblečenia určí `upper` / `lower` / `overall` |
| Strih nohavíc | vlastný algoritmus (OpenCV, numpy), **bez modelu** | z fotky nohavíc zmeria profil šírky → rozšírenie masky pre široké nohavice |
| ML knižnice | PyTorch, diffusers, transformers, accelerate | beh modelu na GPU (fp16) |
| Váhy | Hugging Face Hub (`huggingface_hub`) | sťahovanie váh CatVTON a SD inpainting |
| GPU | **Kaggle Notebooks** (T4 16 GB / P100) | skutočné generovanie; lokálne PC nemá GPU |
| Verzie | Git + GitHub (`MatejGalik/Semestral-VTON-work`) | kód; Kaggle si ho vždy stiahne |
| Vývoj | VS Code, Claude Code (subagent `analytik`) | písanie kódu, analýzy úloh |
| Písma | Google Fonts: Cormorant Garamond, Inter | vzhľad UI |

## Architektúra

```
┌─────────────────────────────┐      ┌──────────────────────────┐
│ Prehliadač                   │      │ Rozšírenie do prehliadača│
│ (Gradio UI)                  │      │ (gradio_client / HTTP)   │
└──────────────┬──────────────┘      └────────────┬─────────────┘
               │ klik „Vyskúšať“                    │ /try_on
               ▼                                    ▼
┌─────────────────────────────────────────────────────────────┐
│ app.py                                                       │
│  build_app() – skladá UI z ui/                               │
│  run_tryon() – adaptér: validácia, seed, chyby, uloženie PNG │
│  _load_backend() – src.try_on, inak DEMO stub                │
└──────────────┬──────────────────────────────────────────────┘
               │ try_on(person, cloth, cloth_type, **options)
               ▼
┌─────────────────────────────┐   ┌─────────────────────────────┐
│ src/try_on.py (GPU)          │   │ _stub_try_on (DEMO, bez GPU) │
│ CatVTON + maska              │   │ fotka s nápisom + hrubá maska│
└─────────────────────────────┘   └─────────────────────────────┘
```

Princíp: **UI nevie nič o modeli.** Komunikuje len cez funkciu `try_on()`
s pevným kontraktom. Backend sa dá vymeniť (iný model, iná maska) bez zmeny UI.

## Súbory

| Súbor | Obsah |
|---|---|
| `app.py` | spustenie (argumenty `--share`, `--demo`, `--host`, `--port`), DEMO stub, adaptér `run_tryon()`, `build_app()` |
| `config.py` | typy oblečenia, predvolené hodnoty (kroky 50, seed 42, …), farby, písma, všetky texty UI (slovensky) |
| `ui/tabs.py` | hlavička a stránka skúšania (karty Osoba + Produkt = Výsledok, prepínače, skryté vstupy API) – bez tabov |
| `ui/theme.py` | Gradio téma, CSS premenné z `config.COLORS`, `<head>` s písmami, vynútenie svetlého režimu |
| `ui/styles.css`, `ui/page.css` | vzhľad komponentov a stránky |
| `src/try_on.py` | backend s CatVTON ([backend.md](backend.md)) |
| `src/masking.py` | úprava masky AutoMaskera: predĺženie pre dlhý vrch, tvar šiat, ochrana rúk, analýza strihu nohavíc a rozšírenie masky pre široké nohavice |
| `src/cloth_type.py` | typ oblečenia pravidlami nad siluetou (bez AI) pre `cloth_type="auto"` |
| `scripts/analyza_strihu.py` | lokálny test analýzy strihu nohavíc (bez GPU) |
| `scripts/typ_oblecenia.py` | lokálny test rozpoznania typu oblečenia (bez GPU) |
| `assets/examples/` | ukážkové osoby (`person_*.png`) a oblečenie (`cloth_*.png`) na lokálne testy (UI ich už neponúka) |
| `kaggle/run.ipynb` | spúšťač na Kaggle |
| `requirements.txt` / `requirements-gpu.txt` | závislosti UI / backendu |
| `docs/` | dokumentácia, `docs/analyzy/` analýzy úloh |
| `.claude/agents/analytik.md` | subagent Claude Code na stručné analýzy |

## UI

- **Jedna stránka bez tabov** (taby Skúšanie / O projekte, Pokročilé nastavenia
  a Príklady odstránené 2026-11-17).
- Karta Osoba + karta Produkt (RGBA – priehľadné pozadie
  produktu sa zachová pre siluetu, CatVTON dostane fotku na bielom) = karta Výsledok, typ
  oblečenia (Automaticky / Vrchný diel / Spodný diel / Celé / šaty;
  predvolene Automaticky – typ určia pravidlá a stavový riadok ho vypíše),
  tlačidlo Vyskúšať, stiahnutie výsledku (PNG).
- **Prepínače pod tlačidlom:** zobraziť masku, respektovať strih (predvolene
  zapnuté; vypnuté → `fit="off"`, maska bez rozšírenia podľa strihu). Posiela
  ho len UI, API `/try_on` ho nemá.
- Kroky (50), seed (42) a vloženie tváre (áno) sú **skryté vstupy** – v UI sa
  nenastavujú, ale ostávajú vo vstupoch API `/try_on` (kontrakt sa nemení).
- Generovanie beží po jednom (`concurrency_limit=1`), chyby a nedostatok
  pamäte GPU sa zobrazia ako upozornenie.
- Priebeh: `gr.Progress(track_tqdm=True)` v `run_tryon` zachytí `tqdm` krokov
  difúzie v pipeline CatVTON (bez úpravy jeho kódu) → percentá a krok X/50.
  Odhad času v texte „Generujem…“ = `config.EXPECTED_SECONDS` (100 s na T4, 50 krokov).
- **DEMO režim:** keď sa backend nedá načítať, v hlavičke je štítok „Demo“
  a výsledok je fotka osoby s nápisom „DEMO režim“.

## API

Jeden verejný endpoint **`/try_on`**. Názov a poradie vstupov sa **nesmú
meniť** – používa ho rozšírenie do prehliadača.

- Vstupy: `person_img`, `cloth_img`, `cloth_type` (`upper`/`lower`/`overall`/`auto`),
  `steps`, `seed`, `paste_face`, `show_mask`
- Výstupy: `result`, `mask`, `download` (PNG), `status` (text)
- Endpoint visí na skrytom tlačidle (`ui/tabs.py`), tlačidlo Vyskúšať má
  vlastnú súkromnú udalosť s prepínačom „Respektovať strih“ navyše. API preto
  prepínač nemá – strih je pri volaní cez API vždy zapnutý. Obe udalosti
  zdieľajú frontu (`concurrency_id="try_on"`, po jednom).

Príklad volania cez `gradio_client` je v [README](../README.md#api).

## Režimy spustenia

| Kde | Príkaz | Výsledok |
|---|---|---|
| Lokálne (bez GPU) | `python app.py` | UI na `http://127.0.0.1:7860`, DEMO režim |
| Kaggle (GPU) | `kaggle/run.ipynb` → Run All | skutočné generovanie, verejný link `gradio.live` |
| Vynútené demo | `python app.py --demo` | DEMO aj keď backend existuje |

## Beh na Kaggle (workflow)

**Notebook `kaggle/run.ipynb` – Run All:**

```
1. Nastavenia      repo, vetva, PREFETCH_WEIGHTS
2. GPU             nvidia-smi
3. kód projektu    git clone, pri opakovanom behu len aktualizácia na origin/<vetva>
4. CatVTON         kód z HF Space zhengchong/CatVTON (pripnutý commit) + CATVTON_DIR
5. závislosti      pip install requirements.txt + requirements-gpu.txt
6. váhy (HF)       CatVTON (mix-48k-1024, SCHP, DensePose)
7. spustenie       python app.py --share → odkaz gradio.live
```

**Štart aplikácie** – raz sa načíta do GPU: SD 1.5 inpainting + CatVTON (fp16,
bez NSFW filtra), AutoMasker (DensePose + SCHP). CLIP sa od 2026-11-17
nepoužíva. Zlyhá backend → DEMO
režim.

**Každé „Vyskúšať“ (`try_on()`):**

```
1. predspracovanie   osoba → výrez 3:4, 768×1024; oblečenie → biely okraj
   typ oblečenia     len pri auto: pravidlá nad siluetou (bez AI), neisté → ručne
2. strih nohavíc     profil šírky z fotky nohavíc (lower, vlastný algoritmus)
3. maska             AutoMasker (DensePose + SCHP) podľa typu
4. úprava masky      tvar šiat, + rozšírenie (široké nohavice), − dlane
5. blur okraja masky
6. CatVTON           50 krokov difúzie – najdlhší krok
7. paste_face        mimo masky pôvodné pixely
8. výstup            späť do pôvodnej veľkosti → UI; do konzoly „try_on: {...}“
```

Podrobnosti: [backend.md](backend.md).

## Pracovný postup tímu

1. Úloha → (voliteľne) analýza subagentom `analytik` → `docs/analyzy/`.
2. Kód sa píše lokálne, UI sa dá skúšať v DEMO režime.
3. `git push` → na Kaggle *Run All* → test na GPU cez `gradio.live` link.
4. Zistenia a rozhodnutia sa zapisujú do `docs/`.

Každý člen tímu má vlastný Kaggle účet a vlastnú GPU kvótu; kód je len
v GitHub repe, na Kaggle sa nič needituje.
