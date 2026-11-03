# Architektúra a technológie

Stav k 2026-11-03. Celkový pohľad na projekt: z čoho sa skladá, aké technológie
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
| Maskovanie | AutoMasker z CatVTON: **DensePose** (detectron2) + **SCHP** | automatická maska oblasti, ktorá sa prekreslí |
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
│ CatVTON + AutoMasker         │   │ fotka s nápisom + hrubá maska│
└─────────────────────────────┘   └─────────────────────────────┘
```

Princíp: **UI nevie nič o modeli.** Komunikuje len cez funkciu `try_on()`
s pevným kontraktom. Backend sa dá vymeniť (iný model, iná maska) bez zmeny UI.

## Súbory

| Súbor | Obsah |
|---|---|
| `app.py` | spustenie (argumenty `--share`, `--demo`, `--host`, `--port`), DEMO stub, adaptér `run_tryon()`, `build_app()` |
| `config.py` | prepínače `FEATURES`, typy oblečenia, predvolené hodnoty (kroky 20–50, seed 42, …), farby, písma, všetky texty UI (slovensky), obsah tabu „O projekte“ |
| `ui/tabs.py` | hlavička, tab Skúšanie (karty Osoba + Produkt = Výsledok, pokročilé nastavenia, príklady), tab O projekte, budúce taby |
| `ui/theme.py` | Gradio téma, CSS premenné z `config.COLORS`, `<head>` s písmami, vynútenie svetlého režimu |
| `ui/styles.css`, `ui/page.css` | vzhľad komponentov a stránky |
| `src/try_on.py` | backend s CatVTON ([backend.md](backend.md)) |
| `assets/examples/` | ukážkové osoby (`person_*.png`) a oblečenie (`cloth_*.png`) |
| `kaggle/run.ipynb` | spúšťač na Kaggle |
| `requirements.txt` / `requirements-gpu.txt` | závislosti UI / backendu |
| `docs/` | dokumentácia, `docs/analyzy/` analýzy úloh |
| `.claude/agents/analytik.md` | subagent Claude Code na stručné analýzy |

## UI

- **Tab Skúšanie:** karta Osoba + karta Produkt = karta Výsledok, typ
  oblečenia (Vrchný diel / Spodný diel / Celé / šaty), tlačidlo Vyskúšať,
  stiahnutie výsledku (PNG).
- **Pokročilé nastavenia:** počet krokov (20–50, predvolene 30), seed
  (−1 = náhodný), vložiť pôvodnú tvár, zobraziť masku.
- **Príklady:** obrázky z `assets/examples/`.
- **Tab O projekte:** popis pipeline, autori, technológie (texty v `config.ABOUT`).
- **Budúce taby:** zapínajú sa flagom v `config.FEATURES` (`compare_masks`,
  `compare_models`, `history`, `metrics`); obsah v `build_<nazov>_tab()`.
- Generovanie beží po jednom (`concurrency_limit=1`), chyby a nedostatok
  pamäte GPU sa zobrazia ako upozornenie.
- **DEMO režim:** keď sa backend nedá načítať, v hlavičke je štítok „Demo“
  a výsledok je fotka osoby s nápisom „DEMO režim“.

## API

Jeden verejný endpoint **`/try_on`**. Názov a poradie vstupov sa **nesmú
meniť** – používa ho rozšírenie do prehliadača.

- Vstupy: `person_img`, `cloth_img`, `cloth_type` (`upper`/`lower`/`overall`),
  `steps`, `seed`, `paste_face`, `show_mask`
- Výstupy: `result`, `mask`, `download` (PNG), `status` (text)

Príklad volania cez `gradio_client` je v [README](../README.md#api).

## Režimy spustenia

| Kde | Príkaz | Výsledok |
|---|---|---|
| Lokálne (bez GPU) | `python app.py` | UI na `http://127.0.0.1:7860`, DEMO režim |
| Kaggle (GPU) | `kaggle/run.ipynb` → Run All | skutočné generovanie, verejný link `gradio.live` |
| Vynútené demo | `python app.py --demo` | DEMO aj keď backend existuje |

## Pracovný postup tímu

1. Úloha → (voliteľne) analýza subagentom `analytik` → `docs/analyzy/`.
2. Kód sa píše lokálne, UI sa dá skúšať v DEMO režime.
3. `git push` → na Kaggle *Run All* → test na GPU cez `gradio.live` link.
4. Zistenia a rozhodnutia sa zapisujú do `docs/`.

Každý člen tímu má vlastný Kaggle účet a vlastnú GPU kvótu; kód je len
v GitHub repe, na Kaggle sa nič needituje.
