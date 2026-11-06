# Backend – CatVTON (`src/try_on.py`)

Stav k 2026-11-06. Popisuje, ako funguje generovanie, prečo je spravené tak,
ako je, a čo sme zistili pri testovaní.

## Prehľad

```
UI (app.py, run_tryon)
   │  person_img, cloth_img, cloth_type, steps, seed, paste_face
   ▼
src/try_on.py: try_on()
   1. predspracovanie   – osoba: výrez 3:4 + zmena na 768×1024
                          oblečenie: doplnenie bielym okrajom na 768×1024
   2. dĺžka rukáva      – CLIP z fotky oblečenia (len upper / overall)
   3. maska             – src/masking.py: Sapiens-seg (predvolene) alebo
                          AutoMasker (DensePose + SCHP) podľa cloth_type
   4. ochrana rúk       – z masky vyreže dlane, pri krátkom rukáve aj holé ruky
                          (len upper / overall), potom rozmazanie okraja (blur 9)
   5. generovanie       – CatVTONPipeline (SD 1.5 inpainting + attention váhy)
   6. paste_face        – mimo masky vráti pôvodné pixely (tvár, ruky, pozadie)
   7. späť do pôvodného rozmeru fotky
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
| `sleeves` | `"auto"` \| `"short"` \| `"long"` \| `"sleeveless"` | voliteľné, ručné určenie rukávov namiesto CLIP (na testovanie); UI ho neposiela |
| `masker` | `"sapiens"` \| `"schp"` | voliteľné, ktorý masker použiť (predvolene `VTON_MASKER`, inak `sapiens`); UI ho neposiela |
| `compare_masks` | `bool` | voliteľné, vytvorí obe masky a uloží porovnanie (predvolene `VTON_COMPARE_MASKS`) |

Návrat: `result` (výsledok v pôvodnom rozmere fotky), `mask` (upravená maska
v pôvodnom rozmere), `info` (model, kroky, seed, rozlíšenie, GPU, čas, rukávy,
chránené časti, masker, čas masky, pri porovnaní podiel pixelov oboch masiek
a cesta k PNG). `info` UI nezobrazuje – vypisuje sa do konzoly ako
`try_on: {...}` (na Kaggle vo výstupe poslednej bunky).

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
6. CLIP (`openai/clip-vit-base-patch32`, fp16) na určenie dĺžky rukáva. Ak sa
   nenačíta, backend beží ďalej a chráni len dlane (varovanie v konzole).
7. Sapiens-seg (`facebook/sapiens-seg-{SAPIENS_SIZE}-torchscript`, fp32) – len
   ak je masker `sapiens` alebo je zapnuté porovnanie. Ak sa nenačíta, maska
   bude z AutoMaskera (varovanie v konzole).

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
| Ochrana rúk podľa rukáva nového oblečenia (CLIP), nie podľa fotky osoby | chráni sa len holá koža podľa SCHP, takže stačí poznať nové oblečenie; pozri „Ochrana rúk“ |
| Maska predvolene zo **Sapiens-seg**, AutoMasker ostáva ako záloha a na porovnanie | SCHP (2019) zle určí ruky pri neštandardných pózach; pozri „Maska zo Sapiens“ |
| NSFW filter SD vypnutý (`skip_safety_check=True`) | falošné poplachy pri koži / rukách na hrudi; ušetrí aj VRAM. Pozor: odkaz `--share` je verejný |
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

**NSFW filter – falošné poplachy (vypnuté 2026-11-06).** CatVTON má
predvolene zapnutý safety checker zo Stable Diffusion; keď ho výsledok
„zaujme“, nahradí ho obrázkom `resource/img/NSFW.jpg` s varovaním. Stalo sa to
pri prekrížených rukách s krátkym rukávom (veľa kože na hrudi), hoci maska bola
správna. Riešenie: `skip_safety_check=True` v `CatVTONPipeline`.

**Prázdna maska zo Sapiens (2026-11-06, príčina neznáma).** Pri prvom behu
s prekríženými rukami bola maska celá čierna (prázdna) a výsledok bol pôvodná
fotka. Ďalšie behy (ruky pri tele, neskôr aj prekrížené) fungovali. Možné
príčiny: zlé triedy zo Sapiens, prvé volanie torchscriptu. Navrhnutá poistka:
pri maske < 2 % prepnúť na AutoMasker + diagnostika tried v `info`.

**Možné rozmazanie pri veľkých fotkách.** Výsledok vzniká v 768×1024 a pri
vložení späť do veľkej fotky (napr. 3000×4000) sa zväčšuje. Pri zapnutom
`paste_face` je okolie ostrý originál, takže rozdiel je viditeľný. Zatiaľ
neriešené.

## Ochrana rúk podľa dĺžky rukáva

_Stav: **implementované 2026-11-03, čaká na test na Kaggle.**_

**Problém.** Maska pre `upper` zakrýva trup aj celé ruky. Pri prekrížených
rukách (ruky ležia na hrudi) ich model musí nakresliť nanovo a vzniká rozmazaný
chaos.

**Riešenie.** Z masky AutoMaskera sa vyrežú časti, ktoré sa meniť nemajú,
a `paste_face` ich vráti z pôvodnej fotky:

- **dlane – vždy**: DensePose (triedy 3, 4) ∩ koža rúk podľa SCHP. AutoMasker
  dlane chráni aj sám, no jeho záverečné roztiahnutie masky do nich zasahuje –
  tu sa vyrežú až po ňom.
- **holá koža rúk – len ak nové oblečenie má krátky rukáv alebo je bez rukávov**:
  SCHP LIP aj ATR (triedy 14, 15 = Left-arm, Right-arm).

Netreba zisťovať, aký rukáv má osoba na fotke: chráni sa len holá koža. Ak má
osoba dlhý rukáv, holej kože je málo a rukáv sa prekreslí ako doteraz.
Chránená oblasť sa roztiahne o 2 iterácie (jadro 5×5), aby do nej nezasahoval
rozmazaný okraj masky.

**Dĺžka rukáva – CLIP (zero-shot).** `openai/clip-vit-base-patch32` (~600 MB)
porovná fotku oblečenia s textami pre `short` / `long` / `sleeveless`
(`SLEEVE_PROMPTS` v `src/try_on.py`) a vráti najpodobnejší s istotou. Pri
istote < 0,6 (`SLEEVE_MIN_CONFIDENCE`) sa holé ruky nechránia, dlane áno.
Voľba `sleeves="short"` atď. obíde CLIP (na testovanie).

**Kedy čo beží** (`try_on()`):

```
pri štarte:   CatVTON + AutoMasker + CLIP sa načítajú raz do GPU
pri volaní:
1. predspracovanie       osoba a oblečenie → 768×1024
2. CLIP                  fotka oblečenia → krátky / dlhý / bez      ~0,05 s
3. AutoMasker            fotka osoby → maska + DensePose + SCHP
4. ochrana rúk           maska − dlane − (holé ruky, ak krátky rukáv), potom blur
5. CatVTON               osoba + oblečenie + upravená maska         najdlhší krok
6. paste_face            mimo masky vráti originál → ruky pôvodné
7. späť do pôvodnej veľkosti
```

- Kroky 2 a 3 sú nezávislé, spája ich až krok 4.
- Pri `lower` sa kroky 2 a 4 preskočia – maska pre spodok ruky nezahŕňa.
- Masky DensePose a SCHP vracia AutoMasker (`densepose`, `schp_lip`,
  `schp_atr`), nič sa nepočíta navyše okrem CLIP.

**Výstup.** V `info` (konzola): `"sleeves": "short (0.87)"`,
`"sleeves_source": "clip"`, `"protected": "dlane, ruky"`. UI zobrazí upravenú
masku (prepínač „zobraziť masku“). UI ani API `/try_on` sa nemení.

**Overenie (treba spraviť).** ~10 fotiek (spredu, prekrížené ruky, ruky za
chrbtom) × krátky a dlhý rukáv, porovnanie pred / po s rovnakým seedom.
Vhodné na obhajobu a pre tab `compare_masks` (flag v `config.FEATURES`).

**Riziká.**

- CLIP sa pomýli pri zložených / pokrčených kúskoch → istota v `info`, neskôr
  ručný prepínač v UI.
- SCHP zle označí kožu (napr. tričko v telovej farbe).
- Dlhý rukáv + prekrížené ruky: pomôže sa len dlaniam, rukávy model kreslí sám.
- Náklady: ~600 MB VRAM navyše, zlomok sekundy na volanie.

## Maska zo Sapiens

_Stav: **implementované 2026-11-06, čaká na test na Kaggle.**_ Analýza:
[analyzy/2026-11-06-maska-parser.md](analyzy/2026-11-06-maska-parser.md).

**Prečo.** SCHP (2019) pri prekrížených rukách či rukách za chrbtom označí
ruku ako oblečenie a maska nedáva zmysel. Meta **Sapiens-seg** (2024) je
trénovaný na oveľa rozmanitejších pózach a má 28 tried Goliath – zvlášť
vrch/spodok oblečenia, trup, rameno, predlaktie a dlaň pre každú ruku.

**Ako** (`src/masking.py`, `sapiens_mask`): rovnaký postup ako
`AutoMasker.cloth_agnostic_mask`, len s triedami Sapiens namiesto SCHP
a DensePose:

| Časť | upper | lower | overall |
|---|---|---|---|
| maskuje sa vždy (pôvodné oblečenie) | Upper_Clothing, Apparel | Lower_Clothing | obe + Apparel |
| telo (roztiahnuté v ¼ rozlíšení) | trup, ruky | nohy | všetko |
| slabá ochrana | nohy, spodok | ruky, vrch, Apparel | – |
| silná ochrana (vždy) | tvár + ústa, dlane, chodidlá | ← | ← |
| holé ruky (CLIP: krátky / bez rukávov) | áno | – | áno |

Vlasy, topánky a ponožky sú vždy chránené. Potom konvexný obal, Gaussovo
rozmazanie a prah, pridanie pôvodného oblečenia, záverečné roztiahnutie
a znova vyrezanie chránených rúk.

**Vstup modelu:** RGB 1024×768, bez /255, normalizácia mean
`(123.5, 116.5, 103.5)`, std `(58.5, 57.0, 57.5)`; výstup logity → bilineárne
na rozmer fotky → argmax.

**Prepínače** (premenné prostredia, nastavuje ich `kaggle/run.ipynb`):

| Premenná | Hodnoty | Význam |
|---|---|---|
| `VTON_MASKER` | `sapiens` (predvolene) / `schp` | ktorý masker |
| `SAPIENS_SIZE` | `0.3b` / `0.6b` (predvolene) / `1b` | veľkosť modelu (fp32: ~1,2 / 2,4 / 4,7 GB) |
| `VTON_COMPARE_MASKS` | `1` | pri každom volaní vytvorí obe masky, uloží PNG (osoba, maska, maska) |
| `VTON_MASK_DIR` | cesta | kam ukladať porovnania (predvolene `/kaggle/working/masks`) |

**Záloha.** Ak sa Sapiens nenačíta (sťahovanie, VRAM), backend použije
AutoMasker a v `info` je `"masker": "schp"`.

**Riziká / overiť na Kaggle.**

- VRAM na T4: CatVTON + CLIP + Sapiens 0.6B (fp32) – sledovať `nvidia-smi`.
- Formát výstupu torchscriptu (tensor / tuple, rozlíšenie) – kód zvláda oba.
- Trieda `Apparel` – predpoklad, že ide o vrchné oblečenie (bunda, kabát).
- HF pri `.pt2` súbore ukazuje štítok „Suspicious“ (pickle v torchscripte);
  ide o oficiálny repozitár `facebook/`.
- Licencia CC-BY-NC 4.0 – pre školský projekt v poriadku.

## Plánované vylepšenia

1. Otestovať ochranu rúk a masku zo Sapiens na Kaggle (`COMPARE_MASKS = True`,
   ~10 fotiek: spredu, prekrížené ruky, ruky za chrbtom, z boku) a výsledky
   zapísať do Zistení.
2. Ručný prepínač rukávov (auto / krátke / dlhé) v UI, ak sa CLIP často mýli.
3. Ručná úprava masky v UI (štetec) – CatVTON vlastnú masku podporuje.
4. Riešenie rozmazania pri veľkých fotkách (nezväčšovať výsledok / voľba
   „vrátiť v rozlíšení modelu“).
5. Porovnanie s iným modelom (IDM-VTON) – flag `compare_models` v `config.py`.
