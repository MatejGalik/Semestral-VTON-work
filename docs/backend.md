# Backend – CatVTON (`src/try_on.py`)

Stav k 2026-11-03. Popisuje, ako funguje generovanie, prečo je spravené tak,
ako je, a čo sme zistili pri testovaní.

## Prehľad

```
UI (app.py, run_tryon)
   │  person_img, cloth_img, cloth_type, steps, seed, paste_face
   ▼
src/try_on.py: try_on()
   1. predspracovanie   – osoba: výrez 3:4 + zmena na 768×1024
                          oblečenie: doplnenie bielym okrajom na 768×1024
   2. maska             – AutoMasker (DensePose + SCHP) podľa cloth_type
                          + rozmazanie okraja (blur 9)
   3. generovanie       – CatVTONPipeline (SD 1.5 inpainting + attention váhy)
   4. paste_face        – mimo masky vráti pôvodné pixely (tvár, ruky, pozadie)
   5. späť do pôvodného rozmeru fotky
   ▼
{"result": PIL.Image, "mask": PIL.Image, "info": dict}
```

## Kontrakt

```python
try_on(person_img, cloth_img, cloth_type, **options) -> dict
```

| Parameter | Typ | Význam |
|---|---|---|
| `person_img`, `cloth_img` | `PIL.Image` | fotka osoby a oblečenia |
| `cloth_type` | `"upper"` \| `"lower"` \| `"overall"` | ktorú časť tela maskovať |
| `steps` | `int` (predvolene 30) | počet krokov difúzie; CatVTON odporúča 50 |
| `seed` | `int` (predvolene 42) | rovnaký seed = rovnaký výsledok |
| `paste_face` | `bool` (predvolene `True`) | vrátiť pôvodné pixely mimo masky |
| `guidance_scale` | `float` (predvolene 2.5) | voliteľné, UI ho zatiaľ neposiela |

Návrat: `result` (výsledok v pôvodnom rozmere fotky), `mask` (maska v pôvodnom
rozmere), `info` (model, kroky, seed, rozlíšenie, GPU, čas v sekundách).

API endpoint `/try_on` v `app.py` sa týmto nemení – rozšírenie do prehliadača
funguje ďalej.

## Načítanie modelu

Model sa načíta **raz pri importe** modulu, nie pri každom volaní:

1. Bez CUDA vyhodí `RuntimeError` → `app._load_backend()` prepne aplikáciu do
   **DEMO režimu** (lokálne na PC bez GPU je to očakávané správanie).
2. Nájde kód CatVTON: premenná `CATVTON_DIR`, inak `./CatVTON` v repe alebo
   vedľa neho. Pridá ho na začiatok `sys.path` (CatVTON má top-level moduly
   `model` a `utils`).
3. `snapshot_download("zhengchong/CatVTON")` – len priečinky `mix-48k-1024`,
   `SCHP`, `DensePose`.
4. `CatVTONPipeline` so základným modelom
   `booksforcharlie/stable-diffusion-inpainting` (pôvodný `runwayml` repozitár
   bol zmazaný; rovnakú kópiu používa oficiálne demo CatVTON).
5. `AutoMasker` s váhami DensePose a SCHP.

Prvé spustenie sťahuje niekoľko GB váh a dlho nič nevypisuje – nie je
zaseknuté. Po načítaní sa vypíše `CatVTON pripravený za … s (GPU, dtype)`.

## Dôležité rozhodnutia

| Rozhodnutie | Prečo |
|---|---|
| Kód CatVTON sa neprepisuje ani nekopíruje do repa, klonuje sa za behu | menej kódu na údržbu, jednoduchá aktualizácia |
| Váhy `mix-48k-1024` | univerzálne (vrch, spodok, šaty), rozlíšenie 768×1024 |
| Pevné rozlíšenie 768×1024 | na tom bol model trénovaný |
| **fp16 na T4/P100**, bf16 len pri compute capability ≥ 8 | pozri „OOM na T4“ nižšie |
| `paste_face` = `repaint_result` z CatVTON | jednoduché a funkčné: tvár a pozadie zostanú ostré a pôvodné |
| Výsledok sa vracia v pôvodnom rozmere fotky | sedí s UI; vygenerovaný výrez sa vloží späť do originálu |
| Torch sa v `requirements-gpu.txt` nepripína | Kaggle má torch s CUDA predinštalovaný, preinštalovanie by ho rozbilo |

## Závislosti

- `requirements.txt` – UI (Gradio, Pillow), stačí lokálne.
- `requirements-gpu.txt` – navyše pre CatVTON: `diffusers`, `transformers`,
  `accelerate`, `huggingface_hub` a závislosti detectron2/DensePose (`fvcore`,
  `omegaconf`, `cloudpickle`, `pycocotools`, `av`).
- detectron2 a DensePose sú **priamo v repe CatVTON**, netreba ich kompilovať.
  Kompilované rozšírenie `detectron2._C` nie je potrebné (DensePose R50-FPN ho
  nepoužíva a import je ošetrený).

## Zistenia z testovania

**OOM na T4 (opravené).** Prvá verzia chcela alokovať 36 GB VRAM.
`torch.cuda.is_bf16_supported()` vracia `True` aj na T4 (bf16 len emuluje),
takže sa zvolil bf16. Pre bf16 na T4 nemá PyTorch úsporný kernel attention
a spadne na „math“ variant, ktorý drží celú attention maticu
(CatVTON spája osobu a oblečenie → 24 576 tokenov). Oprava: rozhodovať podľa
`torch.cuda.get_device_capability()`. Veľkosť nahranej fotky s tým nesúvisí –
backend ju vždy zmenší na 768×1024.

**Kvalita závisí najmä od vstupu.**

- Osoba odfotená **spredu, ruky pri tele** → veľmi dobrý výsledok.
- **Ruky za chrbtom / prekrížené** → rozmazané rukávy a ruky. Model bol
  trénovaný na VITON-HD a DressCode, kde sú takmer všetky fotky v štandardnej
  póze; DensePose pri neštandardnej póze zle určí ruky a maska okolo ramien
  nedáva zmysel.
- Oblečenie najlepšie ako produktová fotka na bielom pozadí.
- 50 krokov je o niečo lepšie ako 30.

**Maska je zámerne hrubá.** AutoMasker pre `upper` zakrýva trup aj celé ruky,
aby sa dali meniť aj rukávy (napr. tričko → dlhý rukáv). Chyba by bola, keby
zasahovala do tváre/pozadia, nepokrývala pôvodné oblečenie alebo bola prázdna
(DensePose nenašiel osobu).

**Možné rozmazanie pri veľkých fotkách.** Výsledok vzniká v 768×1024 a pri
vložení späť do veľkej fotky (napr. 3000×4000) sa zväčšuje. Pri zapnutom
`paste_face` je okolie ostrý originál, takže rozdiel je viditeľný. Zatiaľ
neriešené.

## Plánované vylepšenia

1. **Ochrana rúk v maske podľa dĺžky rukáva (ďalší krok).** CLIP (zero-shot)
   z fotky oblečenia určí krátky / dlhý rukáv / bez rukávov. Dlane (DensePose)
   sa chránia vždy; holé ruky (SCHP) sa chránia, keď nové oblečenie má krátky
   rukáv alebo je bez rukávov a osoba má holé ruky. Rozhodnutie a istota sa
   zapíšu do `info`. UI ani API sa nemení. Cieľ: lepšie výsledky pri
   prekrížených rukách.
2. Ručná úprava masky v UI (štetec) – CatVTON vlastnú masku podporuje.
3. Riešenie rozmazania pri veľkých fotkách (nezväčšovať výsledok / voľba
   „vrátiť v rozlíšení modelu“).
4. Porovnanie s iným modelom (IDM-VTON) – flag `compare_models` v `config.py`.
