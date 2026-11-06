# Backend – CatVTON (`src/try_on.py`)

Stav k 2026-11-06. Popisuje, ako funguje generovanie, prečo je spravené tak,
ako je, a čo sme zistili pri testovaní.

## Prehľad

```
UI (app.py, run_tryon)
   │  person_img, cloth_img, cloth_type, steps, seed, paste_face, mask_grow
   ▼
src/try_on.py: try_on()
   1. predspracovanie   – osoba: výrez 3:4 + zmena na 768×1024
                          oblečenie: doplnenie bielym okrajom na 768×1024
   2. nové oblečenie    – CLIP z fotky oblečenia: rukáv (upper / overall),
                          dĺžka vrchu (upper)
   3. maska             – AutoMasker (DensePose + SCHP) podľa cloth_type
   4. úprava masky      – src/masking.py: predĺženie (dlhý vrch), šírka
                          (mask_grow), ochrana dlaní a pri krátkom rukáve aj
                          holých rúk, potom rozmazanie okraja (blur 9)
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
| `length` | `"auto"` \| `"cropped"` \| `"regular"` \| `"long"` | voliteľné, ručné určenie dĺžky vrchu namiesto CLIP (na testovanie); UI ho neposiela |
| `mask_grow` | `int` px (predvolene 0, orezané na ±40) | šírka masky: + roztiahne, − zúži; UI posuvník „Šírka masky“ |

Návrat: `result` (výsledok v pôvodnom rozmere fotky), `mask` (upravená maska
v pôvodnom rozmere), `info` (model, kroky, seed, rozlíšenie, GPU, čas, rukávy, dĺžka,
šírka masky, vykonané úpravy masky, podiel masky, chránené časti). `info` UI nezobrazuje – vypisuje sa do konzoly ako
`try_on: {...}` (na Kaggle vo výstupe poslednej bunky).

API endpoint `/try_on` v `app.py` sa týmto nemení – rozšírenie do prehliadača
funguje ďalej (`mask_grow` je vtedy 0). Posuvník šírky posiela len UI cez
súkromný endpoint `try_on_ui` (pozri `docs/architektura.md`, API).

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
6. CLIP (`openai/clip-vit-base-patch32`, fp16) na určenie rukáva a dĺžky
   nového oblečenia. Ak sa nenačíta, backend beží ďalej s maskou AutoMaskera
   (chráni len dlane, nepredlžuje; varovanie v konzole).

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
| Maska z **AutoMaskera** (SCHP + DensePose), Sapiens-seg vyskúšaný a odstránený | CatVTON bol trénovaný s maskami AutoMaskera; pri rukách pri tele je SCHP spoľahlivý; pozri „Sapiens-seg – vyskúšaný a odstránený“ |
| Maska sa upravuje podľa **nového** oblečenia (rukáv, dĺžka – CLIP) a posuvníkom šírky | rieši zmenu strihu (krátky ↔ dlhý rukáv, tunika) bez ďalšieho modelu; pozri „Úprava masky podľa nového oblečenia“ |
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

**Prekrížené ruky nefungujú ani po oprave masky (2026-11-06) – vypustené
z plánu.** Ani Sapiens s opravou spodku a lemu nedal použiteľný výsledok.
Projekt počíta s fotkou osoby s rukami pri tele (podmienka na vstup).

**Sapiens-seg – vyskúšaný a odstránený (2026-11-06).** Cieľ: presnejšia
maska pri neštandardných pózach (analýza
[analyzy/2026-11-06-maska-parser.md](analyzy/2026-11-06-maska-parser.md)).
Meta Sapiens-seg 0.6B (28 tried) nahradil SCHP + DensePose, logika masky bola
prepísaná podľa AutoMaskera. Výsledky na Kaggle:

- prvý beh s prekríženými rukami: **prázdna maska** → nezmenená fotka
  (príčina neistá – triedy alebo prvé volanie torchscriptu),
- fotka do pása: **spodok trička ostal** (Sapiens ho označil ako
  `Lower_Clothing`, ktoré sa pri vrchu chráni) a **svetlý lem** okolo rúk,
- po sprísnení masky (prah 127, menšie roztiahnutie): tričko a ruky
  **stiahnuté k stredu**, pri zápästiach **ostali konce dlhého rukáva** –
  tesná maska nedáva modelu priestor na iný strih,
- prekrížené ruky nepomohol vyriešiť ani Sapiens.

Záver: bez prekrížených rúk Sapiens nemá výhodu, stojí ~2,4 GB VRAM a čas pri
štarte a jeho maska sa líši od masiek, s ktorými bol CatVTON trénovaný. Kód aj
prepínače (`VTON_MASKER`, `SAPIENS_SIZE`, `COMPARE_MASKS`) boli odstránené;
história je v gite (commity `a561a6b`–`34ee6f7`). Poučenie: **maska má byť
široká** (ako AutoMasker), presnejší parser kvalitu nezlepší.

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
2. CLIP                  rukáv (krátky / dlhý / bez) + dĺžka vrchu   ~0,1 s
3. AutoMasker            fotka osoby → maska + DensePose + SCHP
4. úprava masky          + predĺženie, ± šírka, − dlane − (holé ruky), potom blur
5. CatVTON               osoba + oblečenie + upravená maska         najdlhší krok
6. paste_face            mimo masky vráti originál → ruky pôvodné
7. späť do pôvodnej veľkosti
```

- Kroky 2 a 3 sú nezávislé, spája ich až krok 4.
- Pri `lower` sa krok 2 preskočí a krok 4 beží len pri `mask_grow ≠ 0`
  (vtedy sa chránia dlane aj ruky, ako v AutoMaskeri).
- Masky DensePose a SCHP vracia AutoMasker (`densepose`, `schp_lip`,
  `schp_atr`), nič sa nepočíta navyše okrem CLIP.

**Výstup.** V `info` (konzola): `"sleeves": "short (0.87)"`,
`"sleeves_source": "clip"`, `"protected": "dlane, ruky"`. UI zobrazí upravenú
masku (prepínač „zobraziť masku“). UI ani API `/try_on` sa nemení.

**Overenie (treba spraviť).** ~10 fotiek s rukami pri tele × krátky a dlhý
rukáv, porovnanie pred / po s rovnakým seedom.
Vhodné na obhajobu a pre tab `compare_masks` (flag v `config.FEATURES`).

**Riziká.**

- CLIP sa pomýli pri zložených / pokrčených kúskoch → istota v `info`, neskôr
  ručný prepínač v UI.
- SCHP zle označí kožu (napr. tričko v telovej farbe).
- Dlhý rukáv + prekrížené ruky: pomôže sa len dlaniam, rukávy model kreslí sám.
- Náklady: ~600 MB VRAM navyše, zlomok sekundy na volanie.

## Úprava masky podľa nového oblečenia (dĺžka, šírka)

_Stav: **implementované 2026-11-06, čaká na test na Kaggle.**_

**Problém.** Maska AutoMaskera vychádza z oblečenia, ktoré má osoba na fotke.
Keď je nové oblečenie iného strihu, model nemá kam kresliť: dlhá tunika cez
krátke tričko skončí v páse, pri širšom kúsku ostanú zvyšky starého oblečenia
pri okraji masky.

**1. Dĺžka vrchu – CLIP (len `upper`).** `LENGTH_PROMPTS` v `src/try_on.py`:
`cropped` / `regular` / `long`. Pri `long` s istotou ≥ 0,6
(`LENGTH_MIN_CONFIDENCE`) `masking.adjust_mask` pridá do masky obdĺžnik pod
trupom podľa DensePose: od polovice trupu o 35 % jeho výšky nižšie
(`LONG_EXTEND`) a o 15 % šírky do strán (`LONG_SIDE_MARGIN`). Dlane, pri
krátkom rukáve aj holé ruky, sa vyrežú až potom, takže ruky pri bokoch ostanú.
`cropped` masku **neskracuje** – staré oblečenie musí byť v maske celé, inak by
ho `paste_face` vrátil. Voľba `length="long"` obíde CLIP.

**2. Šírka masky – posuvník v UI** („Pokročilé nastavenia → Šírka masky“,
−20 až +40 px, krok 5; backend orezáva na ±40, `MASK_GROW_LIMIT`). Kladná
hodnota roztiahne masku eliptickým jadrom (viac priestoru na iný strih,
zmiznú zvyšky starého oblečenia), záporná ju zúži (menej sa zmení okolie).
Pri roztiahnutí sa z masky vyreže aj tvár a vlasy (SCHP, bez dilatácie, aby
golier ostal v maske). Verejné API `/try_on` posiela vždy 0.

**Poradie v `adjust_mask`:** predĺženie → šírka → ochrana rúk (dlane vždy,
holé ruky pri krátkom rukáve / bez rukávov, pri `lower` vždy) → tvár (len pri
roztiahnutí).

**Výstup v `info`:** `"length": "long (0.74, clip)"`, `"mask_grow": 10`,
`"mask_changes": "predĺžená, šírka +10 px"`, `"mask_ratio": 0.31`.

**Riziká / overiť na Kaggle.**

- CLIP pri dĺžke na produktovej fotke (bez postavy) – istota v `info`; ak sa
  často mýli, ručný prepínač v UI.
- Predĺženie je obdĺžnik – pri tunike cez nohavice sa prekreslí aj časť
  nohavíc a pozadia (zámerne, model tam nakreslí tuniku alebo pozadie).
- Veľké roztiahnutie (+40) → model mení aj pozadie okolo postavy.

## Plánované vylepšenia

1. Otestovať na Kaggle ochranu rúk, predĺženie pre dlhý vrch a posuvník šírky
   masky (~10 fotiek s rukami pri tele; krátky ↔ dlhý rukáv, tričko → tunika)
   a výsledky zapísať do Zistení.
2. Ručný prepínač rukávov a dĺžky (auto / …) v UI, ak sa CLIP často mýli.
3. Ručná úprava masky v UI (štetec) – CatVTON vlastnú masku podporuje.
4. Riešenie rozmazania pri veľkých fotkách (nezväčšovať výsledok / voľba
   „vrátiť v rozlíšení modelu“).
5. Porovnanie s iným modelom (IDM-VTON) – flag `compare_models` v `config.py`.
