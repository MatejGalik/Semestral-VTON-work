# Backend – CatVTON (`src/try_on.py`)

Stav k 2026-11-12. Popisuje, ako funguje generovanie, prečo je spravené tak,
ako je, a čo sme zistili pri testovaní.

## Prehľad

```
UI (app.py, run_tryon)
   │  person_img, cloth_img, cloth_type, steps, seed, paste_face
   ▼
src/try_on.py: try_on()
   1. predspracovanie   – osoba: výrez 3:4 + zmena na 768×1024
                          oblečenie: doplnenie bielym okrajom na 768×1024
   2. nové oblečenie    – CLIP z fotky oblečenia: rukáv (upper / overall),
                          dĺžka vrchu (upper), dĺžka šiat (overall)
   3. maska             – AutoMasker (DensePose + SCHP) podľa cloth_type
   4. úprava masky      – src/masking.py: predĺženie (dlhý vrch), tvar
                          a dĺžka šiat (overall), ochrana
                          dlaní a pri krátkom rukáve aj holých rúk, potom
                          rozmazanie okraja (blur 9)
   5. generovanie       – CatVTONPipeline (SD 1.5 inpainting + attention váhy)
   6. paste_face        – mimo masky vráti pôvodné pixely (tvár, ruky, pozadie)
   7. späť do pôvodného rozmeru fotky
   ▼
{"result": PIL.Image, "mask": PIL.Image, "info": dict}
```

## Kroky: algoritmus vs. AI model

V projekte **nie je žiadne LLM**. Používame modely na obraz (CLIP, DensePose,
SCHP, difúzny model). 🟢 = klasický algoritmus, 🔵 = natrénovaný AI model.

| # | Krok | Typ | Dá sa bez AI? |
|---|---|---|---|
| 1 | Predspracovanie | 🟢 | už je |
| 2 | Rukáv a dĺžka nového oblečenia (CLIP) | 🔵 | áno, silueta oblečenia (len na čistom pozadí) |
| 3a | Póza / časti tela (DensePose) | 🔵 | nie |
| 3b | Segmentácia (SCHP) | 🔵 | prakticky nie |
| 3c | Zloženie masky (AutoMasker) | 🟢 | už je |
| 4 | Úprava masky (`src/masking.py`) | 🟢 | už je |
| 5 | Generovanie (CatVTON) | 🔵 | nie, bez AI len zdeformovaná „nálepka“ |
| 6 | `paste_face` | 🟢 | už je |
| 7 | Späť do pôvodného rozmeru | 🟢 | už je |

**1. Predspracovanie 🟢** (`resize_and_crop`, `resize_and_padding` z CatVTON)
- Osoba: zo stredu fotky sa vyreže časť s pomerom 3:4 (`_crop_box`) a zmenší
  sa na 768×1024, na takom rozlíšení bol model trénovaný.
- Oblečenie: zmenší sa tak, aby sa celé zmestilo do 768×1024, a zvyšok sa
  doplní bielym okrajom, takže sa z oblečenia nič neodreže.

**2. Rukáv a dĺžka nového oblečenia 🔵** (`_clip_classify`)
- CLIP (`clip-vit-base-patch32`) prevedie obrázok aj textové popisy na vektory
  (embeddingy) a porovná ich podobnosť. Ide o zero-shot klasifikáciu: model sa
  na našich dátach netrénoval, triedy sú zadané len textom.
- Popisy tried: rukáv `short` / `long` / `sleeveless`, dĺžka vrchu `cropped` /
  `regular` / `long`, dĺžka šiat `mini` / `knee` / `midi` / `maxi`.
- Softmax nad podobnosťami určí víťaznú triedu a jej istotu. Pod prahom (0,6;
  pri šatách 0,4) sa úprava masky nepoužije, šaty dostanú `knee`.
- *Bez AI:* oddeliť oblečenie od bieleho pozadia prahom, z masky siluety
  spočítať šírku po riadkoch (rukávy = široké „ramená“) a pomer výška/šírka
  (dĺžka). Na fotke s rušným pozadím alebo oblečenia na človeku zlyhá.

**3a. Póza / časti tela 🔵** (DensePose, v AutoMaskeri)
- Neurónová sieť priradí každému pixelu osoby časť tela (trup, nadlaktie,
  predlaktie, dlaň, stehno, lýtko…), spolu 24 častí.
- Z tejto mapy sa určí, kde je telo, aj keď ho zakrýva oblečenie.

**3b. Segmentácia 🔵** (SCHP, modely LIP a ATR, v AutoMaskeri)
- Neurónová sieť označí každý pixel triedou: vrch, nohavice, šaty, koža rúk,
  tvár, vlasy, topánky…
- Vie, čo je teraz *oblečené*, čo je holá koža a čo sa nesmie meniť (tvár,
  vlasy).

**3c. Zloženie masky 🟢** (AutoMasker, `cloth_masker.py` v CatVTON)
- Pravidlá nad mapami 3a a 3b: podľa `cloth_type` sa vyberú oblasti na
  prekreslenie (`upper`: trup a ruky; `lower`: nohy; `overall`: oboje).
- Chránené oblasti (tvár, vlasy, a pri `upper` aj nohy) sa z masky vyrežú.
- Maska sa roztiahne (dilatácia) a vyplnia sa v nej diery. Je zámerne široká,
  aby model mohol nakresliť iný strih.

**4. Úprava masky 🟢** (`src/masking.py: adjust_mask`)
- Dlhý vrch: pod trupom (ohraničujúci obdĺžnik trupu z DensePose) sa pridá
  obdĺžnik, ktorý siaha o 35 % výšky trupu nižšie a o 15 % širšie.
- Šaty: lem sa vypočíta z bodov DensePose (bok → koleno → členok) podľa dĺžky
  šiat. Maska sa oreže na roztiahnuté telo plus obdĺžnik sukne od pása po lem.
- Ochrana rúk: dlane (DensePose ∩ koža zo SCHP) sa vyrežú vždy. Pri krátkom
  rukáve alebo bez rukávov sa vyreže aj všetka holá koža rúk.
- Nakoniec rozmazanie okraja (blur 9), aby nebol vidieť ostrý prechod.
- Použité operácie: prieniky a zjednotenia binárnych máp (numpy), dilatácia
  (OpenCV), ohraničujúce obdĺžniky.

**5. Generovanie 🔵** (`CatVTONPipeline`)
- Difúzny model: SD 1.5 inpainting UNet s dotrénovanými attention váhami
  CatVTON. Textový vstup (prompt) sa nepoužíva.
- Fotka osoby so zakrytou maskou a fotka oblečenia sa spoja **vedľa seba**
  (concat) a VAE ich zakóduje do latentného priestoru.
- Model postupne (`steps` krokov) odšumuje oblasť masky. Cez self-attention
  „kuká“ na oblečenie vedľa a prenáša jeho vzor, farbu a strih na telo
  (záhyby, tiene, póza).
- `guidance_scale` 2,5 určuje, ako silno sa výsledok drží oblečenia.
  `seed` zaručí opakovateľnosť.
- *Bez AI:* TPS deformácia (warp) fotky oblečenia podľa bodov tela a vloženie
  na osobu, ako pri starších metódach. Výsledok nemá záhyby, tiene ani
  prekrytie rukami.

**6. `paste_face` 🟢** (`repaint_result`)
- Alfa zmes: `výsledok × maska + originál × (1 − maska)` s rozmazanou maskou.
- Mimo masky ostanú pôvodné pixely (tvár, pozadie, chránené ruky), takže sa
  nestratí ostrosť a nezmenia sa detaily.

**7. Späť do pôvodného rozmeru 🟢** (`_paste_back`)
- Výsledok 768×1024 sa zväčší na veľkosť výrezu z kroku 1 (LANCZOS) a vloží
  sa späť do pôvodnej fotky. Rovnako sa spracuje maska pre UI.

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

Návrat: `result` (výsledok v pôvodnom rozmere fotky), `mask` (upravená maska
v pôvodnom rozmere), `info` (model, kroky, seed, rozlíšenie, GPU, čas, rukávy, dĺžka,
vykonané úpravy masky, podiel masky, chránené časti). `info` UI nezobrazuje – vypisuje sa do konzoly ako
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
6. CLIP (`openai/clip-vit-base-patch32`, fp16) na určenie rukáva a dĺžky
   nového oblečenia. Ak sa nenačíta, backend beží ďalej s maskou AutoMaskera
   (chráni len dlane, nepredlžuje; varovanie v konzole).

Prvé spustenie sťahuje niekoľko GB váh a dlho nič nevypisuje – nie je
zaseknuté. Po načítaní sa vypíše `CatVTON pripravený za … s (GPU, dtype)`.

## Dôležité rozhodnutia

| Rozhodnutie | Prečo |
|---|---|
| Kód CatVTON sa neprepisuje ani nekopíruje do repa, klonuje sa za behu z **HF Space `zhengchong/CatVTON`** | menej kódu na údržbu; GitHub repo `Zheng-Chong/CatVTON` bolo zmazané (pozri „Zistenia z testovania“) |
| Váhy `mix-48k-1024` | univerzálne (vrch, spodok, šaty), rozlíšenie 768×1024 |
| Pevné rozlíšenie 768×1024 | na tom bol model trénovaný |
| **fp16 na T4/P100**, bf16 len pri compute capability ≥ 8 | pozri „OOM na T4“ nižšie |
| `paste_face` = `repaint_result` z CatVTON | jednoduché a funkčné: tvár a pozadie zostanú ostré a pôvodné |
| Výsledok sa vracia v pôvodnom rozmere fotky | sedí s UI; vygenerovaný výrez sa vloží späť do originálu |
| Ochrana rúk podľa rukáva nového oblečenia (CLIP), nie podľa fotky osoby | chráni sa len holá koža podľa SCHP, takže stačí poznať nové oblečenie; pozri „Ochrana rúk“ |
| Maska z **AutoMaskera** (SCHP + DensePose), Sapiens-seg vyskúšaný a odstránený | CatVTON bol trénovaný s maskami AutoMaskera; pri rukách pri tele je SCHP spoľahlivý; pozri „Sapiens-seg – vyskúšaný a odstránený“ |
| Maska sa upravuje podľa **nového** oblečenia (rukáv, dĺžka – CLIP) | rieši zmenu strihu (krátky ↔ dlhý rukáv, tunika) bez ďalšieho modelu; pozri „Úprava masky podľa nového oblečenia“ |
| NSFW filter SD vypnutý (`skip_safety_check=True`) | falošné poplachy pri koži / rukách na hrudi; ušetrí aj VRAM. Pozor: odkaz `--share` je verejný |
| Torch sa v `requirements-gpu.txt` nepripína | Kaggle má torch s CUDA predinštalovaný, preinštalovanie by ho rozbilo |

## Závislosti

- `requirements.txt` – UI (Gradio, Pillow), stačí lokálne.
- `requirements-gpu.txt` – navyše pre CatVTON: `diffusers`, `transformers`,
  `accelerate`, `huggingface_hub` a závislosti detectron2/DensePose (`fvcore`,
  `omegaconf`, `cloudpickle`, `pycocotools`, `av`).
- detectron2 a DensePose sú **priamo v kóde CatVTON** (HF Space, pripnutý commit `c35ec9e`), netreba ich kompilovať.
  Kompilované rozšírenie `detectron2._C` nie je potrebné (DensePose R50-FPN ho
  nepoužíva a import je ošetrený).

## Zistenia z testovania

**GitHub repo CatVTON zmazané (opravené 2026-11-12).** Bunka 3 notebooku
padla na `git clone` s hláškou `could not read Username for 'https://github.com'`.
To GitHub vypíše pri neexistujúcom repe: `Zheng-Chong/CatVTON` už nie je
dostupné. Rovnaký kód (`model/pipeline.py`, `model/cloth_masker.py`,
`utils.py`, rovnaké API) je v HF Space `zhengchong/CatVTON`. Notebook klonuje
odtiaľ, s `GIT_LFS_SKIP_SMUDGE=1` (bez ukážkových obrázkov) a
`GIT_TERMINAL_PROMPT=0` (chyba namiesto čakania na meno). Váhy
(HF model `zhengchong/CatVTON`) to neovplyvnilo.

**Space bez detectron2 → DEMO režim (opravené 2026-11-12).** Po prechode na
Space backend padol na `No module named 'detectron2'`. Najnovší commit Space
(`24a6952`, „Admin maintenance“) zmazal pribalený `detectron2/` a inštaluje ho
kompiláciou z GitHubu. Notebook preto sťahuje **pripnutý commit `c35ec9e`**
(`CATVTON_COMMIT`), ktorý `detectron2/` ešte má; kód `model/` a `utils.py` má
rovnaké API. Ak by commit zmizol, náhrada je
`pip install --no-build-isolation --no-deps git+https://github.com/facebookresearch/detectron2.git`
(kompilácia ~5–10 min).

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

**Posuvník „Šírka masky“ nefungoval – odstránený (2026-11-06).** Roztiahnutie
/ zúženie masky (−20 až +40 px) v UI na Kaggle neprinieslo viditeľné
zlepšenie, preto bol odstránený z UI aj backendu (commit `37331f7` ho
pridal). Dĺžka vrchu (CLIP) ostáva.

**Šaty na mužovi → nohavice a čierny kardigán (2026-11-09).** Muž vo svetri
s dlhým rukávom a džínsoch + krátke červené šaty na ramienka (`overall`).
Výsledok: z šiat vznikol **overal s nohavicami** a navrch **tmavý kardigán
s dlhým rukávom**. Príčina je maska, nie šaty:

- AutoMasker pre `overall` zakryl jednu veľkú plochu od ramien **po členky**
  vrátane priestoru medzi rukami a telom (silueta dlhého kabáta),
- CatVTON vyplní celú masku oblečením a z okraja masky číta strih → krátke
  šaty natiahol na nohavice, „kabátovú“ plochu a ruky vyplnil kardigánom,
- pri dlaniach ostali manžety svetra (roztiahnutá ochrana dlaní) → nápoveda
  „tu pokračuje rukáv“; holá koža rúk nebola, takže `protect_arms` nepomohol.

Oprava: maska šiat podľa dĺžky a tvaru tela (sekcia nižšie). Obmedzenie
modelu ostáva: holé ruky namiesto dlhého rukáva (a nohy namiesto nohavíc)
model nevytvorí spoľahlivo – na šaty je vhodná fotka v tričku / tielku.

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
4. úprava masky          + predĺženie, − dlane − (holé ruky), potom blur
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

**Overenie (treba spraviť).** ~10 fotiek s rukami pri tele × krátky a dlhý
rukáv, porovnanie pred / po s rovnakým seedom.
Vhodné na obhajobu a pre tab `compare_masks` (flag v `config.FEATURES`).

**Riziká.**

- CLIP sa pomýli pri zložených / pokrčených kúskoch → istota v `info`, neskôr
  ručný prepínač v UI.
- SCHP zle označí kožu (napr. tričko v telovej farbe).
- Dlhý rukáv + prekrížené ruky: pomôže sa len dlaniam, rukávy model kreslí sám.
- Náklady: ~600 MB VRAM navyše, zlomok sekundy na volanie.

## Úprava masky podľa nového oblečenia (dĺžka)

_Stav: **implementované 2026-11-06, čaká na test na Kaggle.**_

**Problém.** Maska AutoMaskera vychádza z oblečenia, ktoré má osoba na fotke.
Keď je nové oblečenie iného strihu, model nemá kam kresliť: dlhá tunika cez
krátke tričko skončí v páse.

**1. Dĺžka vrchu – CLIP (len `upper`).** `LENGTH_PROMPTS` v `src/try_on.py`:
`cropped` / `regular` / `long`. Pri `long` s istotou ≥ 0,6
(`LENGTH_MIN_CONFIDENCE`) `masking.adjust_mask` pridá do masky obdĺžnik pod
trupom podľa DensePose: od polovice trupu o 35 % jeho výšky nižšie
(`LONG_EXTEND`) a o 15 % šírky do strán (`LONG_SIDE_MARGIN`). Dlane, pri
krátkom rukáve aj holé ruky, sa vyrežú až potom, takže ruky pri bokoch ostanú.
`cropped` masku **neskracuje** – staré oblečenie musí byť v maske celé, inak by
ho `paste_face` vrátil. Voľba `length="long"` obíde CLIP.

**Poradie v `adjust_mask`:** predĺženie → ochrana rúk (dlane vždy, holé ruky
pri krátkom rukáve / bez rukávov). Pri `lower` sa maska nemení.

Posuvník šírky masky bol skúšaný a odstránený – pozri Zistenia.

**Výstup v `info`:** `"length": "long (0.74, clip)"`,
`"mask_changes": "predĺžená"`, `"mask_ratio": 0.31`.

**Riziká / overiť na Kaggle.**

- CLIP pri dĺžke na produktovej fotke (bez postavy) – istota v `info`; ak sa
  často mýli, ručný prepínač v UI.
- Predĺženie je obdĺžnik – pri tunike cez nohavice sa prekreslí aj časť
  nohavíc a pozadia (zámerne, model tam nakreslí tuniku alebo pozadie).

## Maska pre šaty (`overall`)

_Stav: **implementované 2026-11-09, čaká na test na Kaggle.**_

**Problém.** Pozri Zistenia „Šaty na mužovi“ – maska AutoMaskera po členky
a so siluetou kabáta → nohavice a kardigán namiesto krátkych šiat.

**1. Dĺžka šiat – CLIP.** `DRESS_LENGTH_PROMPTS` v `src/try_on.py`:
`mini` / `knee` / `midi` / `maxi`. Pri istote < 0,4
(`DRESS_LENGTH_MIN_CONFIDENCE`, 4 triedy) sa použije `knee`
(`DRESS_LENGTH_FALLBACK`). Voľba `length="mini"` atď. obíde CLIP.

**2. Lem podľa DensePose** (`masking._dress_shape`, `DRESS_HEM`): bok = spodok
trupu, koleno = vrch lýtok, členok = spodok lýtok.

| dĺžka | lem |
|---|---|
| mini | bok + 60 % stehna |
| knee | koleno + 10 % lýtka |
| midi | koleno + 50 % lýtka |
| maxi | bez rezu |

\+ rezerva 5 % výšky trupu. Pod lemom maska končí → nohy (aj nohavice)
ostanú z pôvodnej fotky. Ak DensePose nenájde nohy, dĺžka sa nemení.

**3. Tvar tela.** Maska AutoMaskera sa oreže na povolenú oblasť:
DensePose trup + ruky + nohy roztiahnuté o ~25 px (`DRESS_BODY_DIVISOR`)
\+ obdĺžnik sukne od ¾ trupu po lem, širší o 25 % šírky trupu na každú stranu
(`DRESS_SKIRT_MARGIN`, priestor pre áčkový strih). Masku len zmenšuje, nikdy
nepridáva.

**4. Ochrana dlaní pri odhalených rukách.** Ak má nové oblečenie krátky rukáv /
je bez rukávov, chránená oblasť sa roztiahne len o 1 iteráciu (inak 2) – aby
pri dlani neostala manžeta pôvodného rukáva.

**Výstup v `info`:** `"length": "mini (0.62, clip)"`,
`"mask_changes": "šaty: mini"`.

**Riziká / overiť na Kaggle.**

- Tesnejšia maska môže „stiahnuť“ šaty k telu (poučenie zo Sapiens) – ak áno,
  zväčšiť `DRESS_BODY_DIVISOR` / `DRESS_SKIRT_MARGIN`.
- CLIP pri dĺžke šiat – istota v `info`.
- Pôvodné voľné oblečenie mimo roztiahnutého tela (široký kabát) ostane.

## Plánované vylepšenia

1. Otestovať na Kaggle ochranu rúk a predĺženie pre dlhý vrch (~10 fotiek
   s rukami pri tele; krátky ↔ dlhý rukáv, tričko → tunika) a masku pre šaty
   (mini / po kolená / maxi, muž aj žena) a výsledky zapísať do Zistení.
2. Ručný prepínač rukávov a dĺžky (auto / …) v UI, ak sa CLIP často mýli.
3. Ručná úprava masky v UI (štetec) – CatVTON vlastnú masku podporuje.
4. Riešenie rozmazania pri veľkých fotkách (nezväčšovať výsledok / voľba
   „vrátiť v rozlíšení modelu“).
5. Porovnanie s iným modelom (IDM-VTON) – flag `compare_models` v `config.py`.
