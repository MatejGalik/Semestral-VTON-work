# Backend – CatVTON (`src/try_on.py`)

Stav k 2026-11-17. Popisuje, ako funguje generovanie, prečo je spravené tak,
ako je, a čo sme zistili pri testovaní.

## Prehľad

```
UI (app.py, run_tryon)
   │  person_img, cloth_img, cloth_type, steps, seed, paste_face
   ▼
src/try_on.py: try_on()
   1. predspracovanie   – osoba: výrez 3:4 + zmena na 768×1024
                          oblečenie: doplnenie bielym okrajom na 768×1024
   1b. typ oblečenia    – len pri cloth_type="auto": pravidlá nad siluetou
                          (src/cloth_type.py, bez AI); neisté → UI vyzve na
                          ručný výber (API → upper)
   2. strih nohavíc     – lower: profil šírky z fotky nohavíc (vlastný
                          algoritmus, bez modelu)
   3. maska             – AutoMasker (DensePose + SCHP) podľa cloth_type
   4. úprava masky      – src/masking.py: tvar šiat (overall), rozšírenie
                          pre široké nohavice (lower), ochrana dlaní, potom
                          rozmazanie okraja (blur 9)
   5. generovanie       – CatVTONPipeline (SD 1.5 inpainting + attention váhy)
   6. paste_face        – mimo masky vráti pôvodné pixely (tvár, ruky, pozadie)
   7. späť do pôvodného rozmeru fotky
   ▼
{"result": PIL.Image, "mask": PIL.Image, "info": dict}
```

## Kroky: algoritmus vs. AI model

V projekte **nie je žiadne LLM**. Používame len modely na obraz z CatVTON
(DensePose, SCHP, difúzny model); CLIP bol odstránený 2026-11-17. 🟢 = klasický algoritmus, 🔵 = natrénovaný AI model.

| # | Krok | Typ | Dá sa bez AI? |
|---|---|---|---|
| 1 | Predspracovanie | 🟢 | už je |
| 1b | Typ oblečenia pri `auto` – pravidlá nad siluetou | 🟢 | už je |
| 2 | Strih nohavíc (`analyze_pants`) | 🟢 | už je |
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
- Šaty: maska sa oreže na roztiahnuté telo (DensePose) plus obdĺžnik sukne
  od pása nadol.
- Ochrana dlaní: dlane (DensePose ∩ koža zo SCHP) sa vyrežú vždy. Holé ruky
  sa nechránia, model môže nakresliť dlhý rukáv.
- Nohavice: podľa voľnosti strihu z fotky (krok 2) sa určí cieľová šírka
  a vonkajšie okraje masky sa posunú von. Maska sa len rozširuje. Pozri „Rozšírenie masky podľa strihu nohavíc“.
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
- `guidance_scale` 2,5 (predvolené z CatVTON) určuje, ako silno sa výsledok
  drží oblečenia. Vyššie hodnoty (2,6–3,5) skúšané 2026-11-17 – horšie.
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
| `cloth_type` | `"upper"` \| `"lower"` \| `"overall"` \| `"auto"` | ktorú časť tela maskovať; `"auto"` = typ určia pravidlá nad siluetou fotky oblečenia, bez AI (predvolené v UI) |
| `steps` | `int` (predvolene 30) | počet krokov difúzie; CatVTON odporúča 50 |
| `seed` | `int` (predvolene 42) | rovnaký seed = rovnaký výsledok |
| `paste_face` | `bool` (predvolene `True`) | vrátiť pôvodné pixely mimo masky |
| `guidance_scale` | `float` (predvolene 2.5) | voliteľné, UI ho zatiaľ neposiela |
| `fit` | `"auto"` \| `"off"` | voliteľné, `"off"` vypne rozšírenie masky podľa strihu nohavíc (na porovnanie pred / po); UI ho posiela podľa prepínača „Respektovať strih“, API `/try_on` ho nemá (vždy `auto`) |

Návrat: `result` (výsledok v pôvodnom rozmere fotky), `mask` (upravená maska
v pôvodnom rozmere), `info` (model, typ oblečenia – napr. `"lower (rules)"`, kroky, seed, rozlíšenie, GPU, čas,
vykonané úpravy masky, podiel masky, chránené časti, strih nohavíc). `info` UI nezobrazuje – vypisuje sa do konzoly ako
`try_on: {...}` (na Kaggle vo výstupe poslednej bunky).

API endpoint `/try_on` v `app.py` sa týmto nemení – rozšírenie do prehliadača
funguje ďalej (hodnota `"auto"` len pribudla, pôvodné tri ostávajú).

## Automatické rozpoznanie typu oblečenia (`cloth_type="auto"`)

_Stav: **implementované 2026-11-17, overené lokálne v DEMO režime, čaká na test na Kaggle.**_

Len **vlastné pravidlá, bez AI** (CLIP sa na typ nepoužíva). Neisté → UI
vyzve na ručný výber a negeneruje; cez API → `CLOTH_TYPE_FALLBACK` = `upper`.

**1. Pravidlá nad siluetou** (`src/cloth_type.py`, ~10 ms, len OpenCV / numpy)

- Silueta ako pri strihu nohavíc (`masking.silhouette`): fotka s priehľadným
  pozadím (PNG / WebP) → **alfa kanál**; inak farba ≠ pozadie (medián okraja).
  Najväčšia oblasť, orezaná na ohraničujúci obdĺžnik.
- Znaky (relatívne, bez mierky): `aspect` výška / šírka, `legs` podiel riadkov
  od 35 % výšky po lem, kde je „pár nohavíc“ – presne dva úseky, podobne široké
  (≥ 60 %), medzera pri strede siluety (baggy sa pri leme dotýkajú; tri úseky
  alebo nesymetrický pár = rukávy),
  `top_width` / `top_single` šírka a súvislosť horného okraja (pás vs.
  výstrih / ramienka), `hem_width` šírka pri leme, `widest_at` kde je silueta
  najširšia, `narrowest` najužšie miesto v hornej časti.
- Rozhodovací strom (prahy `RULE_*`):
  1. dve nohavice: rozkrok ≤ 48 % výšky (`crotch_at`) alebo výška / šírka < 1,4
     alebo hore súvislý pás → `lower` (nohavice, šortky); inak `overall` (overal),
  2. dlhá (výška / šírka ≥ 1,45): rovný pás + rozširuje sa nadol → `lower`
     (maxi sukňa), inak `overall` (šaty),
  3. krátka, najširšia hore a lem užší → `upper` (rukávy / ramená),
  4. krátka, rovný pás + rozširuje sa → `lower` (sukňa),
  5. krátka, rovná a hore **bez** širokého pásu (< 70 %) → `upper`; inak
     **neisté**.
- Bez siluety (rušné pozadie) → neisté.
- UI (`app.run_tryon`) spustí pravidlá pred generovaním; pri neistote ukáže
  upozornenie s dôvodom a negeneruje. DEMO režim používa tie isté pravidlá.

**Výstup:** `info["cloth_type"]` = typ a zdroj (`rules` / `manual` /
`fallback`), `info["cloth_type_reason"]` = dôvod z pravidiel; UI
ukáže typ v stavovom riadku („rozpoznané: Spodný diel“). Ručný výber ostáva.

**Lokálne overenie (2026-11-17).** `python scripts/typ_oblecenia.py <fotky>`
vypíše typ, dôvod a znaky. Ukážky z `assets/examples/` (3/3) a 12 umelých
siluet (tričko, mikina, dlhý rukáv rozpažený aj pri tele, šortky, skinny,
mini a maxi sukňa, šaty áčkové / puzdrové / tričkové, overal s ramienkami aj
bez): správne 14 z 15. Chyba: dlhé úzke tielko → šaty.

**Oprava po teste na Kaggle (2026-11-17).** Baggy nohavice →
`upper (1.00, rules)`, dôvod „krátka, rovná silueta (vrch)“. Široké nohavice
majú nízky pomer strán a pri leme sa dotýkajú, takže test „dve nohavice“
v spodných 25 % zlyhal a padli do pravidla 5. Oprava: pár nohavíc sa hľadá od
35 % výšky, pravidlo 5 vyžaduje hore úzky okraj (široký pás → neisté → CLIP).
Umelé baggy siluety (dotýkajúce sa aj rozkročené) → `lower`; ostatné bez
zmeny, sveter s rukávmi pri tele → neisté.

**2. test na Kaggle (2026-11-17).** Tie isté baggy nohavice → `overall`
(„dve nohavice a hore výstrih / ramená“). Pár nohavíc sa už našiel, ale horný
okraj nevyzeral ako rovný pás (guma, šnúrka, zakrivenie). Oprava: nový znak
`crotch_at` – poloha rozkroku (nohavice ~0,3–0,42 výšky, overal ~0,5–0,6);
tvar horného okraja rozhoduje až pri nízkom rozkroku. Umelá silueta baggy
s gumou a šnúrkou → `lower`, overaly ostávajú `overall`. Riziko: overal
s dlhým trupom vs. nohavice s vysokým pásom pri hranici 0,48.

**3. test – skutočné fotky (2026-11-17).** Priečinok s 10 produktovými
fotkami (skinny, rovné, 4× baggy, 2 tričká, šaty) + fotky ľudí. Príčina
chýb na Kaggle: 3 fotky (vrátane baggy z testu) majú **priehľadné pozadie**
(RGBA WebP). UI (`image_mode="RGB"`) aj `convert("RGB")` alfa zahodili →
pozadie čierne / s náhodnými farbami → rozbitá silueta. Rovnaké skreslené
pozadie dostával aj CatVTON ako fotku oblečenia. Oprava: UI posiela RGBA,
silueta z alfa kanála, CatVTON dostane fotku na bielom (`masking.flatten`);
CLIP z určenia typu odstránený (neisté → ručný výber). Výsledok: **10/10
produktových fotiek správne**. Fotky ľudí (nie produkt) pravidlá
klasifikujú nezmyselne (osoba v rifliach → `lower`) – na fotku osoby
v karte Produkt pravidlá nie sú určené.

**Riziká / overiť na skutočných fotkách.** Prahy sú odladené na umelých
siluetách. Tielko vs. šaty (bez mierky rozhoduje len pomer strán), dlhý kabát
alebo tunika → šaty, fotka na modelke / figuríne alebo rušné pozadie (silueta
nesedí → CLIP).

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
| Kód CatVTON sa neprepisuje ani nekopíruje do repa, klonuje sa za behu z **HF Space `zhengchong/CatVTON`** | menej kódu na údržbu; GitHub repo `Zheng-Chong/CatVTON` bolo zmazané (pozri „Zistenia z testovania“) |
| Váhy `mix-48k-1024` | univerzálne (vrch, spodok, šaty), rozlíšenie 768×1024 |
| Pevné rozlíšenie 768×1024 | na tom bol model trénovaný |
| **fp16 na T4/P100**, bf16 len pri compute capability ≥ 8 | pozri „OOM na T4“ nižšie |
| `paste_face` = `repaint_result` z CatVTON | jednoduché a funkčné: tvár a pozadie zostanú ostré a pôvodné |
| Výsledok sa vracia v pôvodnom rozmere fotky | sedí s UI; vygenerovaný výrez sa vloží späť do originálu |
| Maska z **AutoMaskera** (SCHP + DensePose), Sapiens-seg vyskúšaný a odstránený | CatVTON bol trénovaný s maskami AutoMaskera; pri rukách pri tele je SCHP spoľahlivý; pozri „Sapiens-seg – vyskúšaný a odstránený“ |
| **Bez CLIP** (odstránený 2026-11-17) | ochrana rúk pri krátkom rukáve a predĺženie / rez podľa dĺžky viac škodili, ako pomáhali (tričká aj mikiny vyzerali bez CLIP lepšie); jeden model menej na GPU; pozri „Odstránené: CLIP“ |
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

**Široké (baggy) nohavice vyšli úzke (2026-11-13).** Muž v úzkych džínsoch
\+ veľmi široké tepláky (`lower`). Výsledok mal správnu farbu a materiál, ale
nohavice boli úzke ako pôvodné džínsy. Príčina: maska AutoMaskera pre `lower`
kopíruje pôvodné nohavice (konvexný obal + roztiahnutie) a `src/masking.py`
ju pri `lower` nemenil. CatVTON kreslí len vnútri masky, mimo nej `paste_face`
vráti pôvodné pozadie, takže širšie nohavice nemal kam nakresliť. Oprava:
sekcia „Rozšírenie masky podľa strihu nohavíc“.

**Rozšírenie nohavíc – 1. test na Kaggle (2026-11-13).** Tie isté baggy
tepláky, seed 42: `fit` = široké (0.55 / 0.55), ale `mask_changes` = `-`.
Maska AutoMaskera pre `lower` je rovný stĺpec od bokov po členky (konvexný
obal pôvodných nohavíc, `mask_ratio` 0.144); model ho vyplnil celý → rovné
nohavice. Cieľová šírka prvej verzie (od tela z DensePose) sa celá zmestila
do stĺpca → prírastok < 3 % → bez zmeny. Oprava: rozdiel voči pôvodným
nohaviciam (sekcia nižšie). Pozitívne zistenie: model tvar masky poslúcha.

**Mikina s „vyhrnutými“ rukávmi (2026-11-17).** Mikina (`upper`) na osobe
v tričku s krátkym rukávom vyšla, akoby mala vyhrnuté rukávy. Podozrenie:
CLIP určil krátky rukáv → holé ruky sa chránili (vyrezali z masky) → model
nemal kam nakresliť dlhý rukáv. Pridaný prepínač `clip="off"` – aj bez CLIP
rukáv len po lakeť, hoci maska pokrývala celé ruky → **obmedzenie modelu**
(holé predlaktia pri tele si model „nechá“). Tričká však bez CLIP vyzerali
lepšie → CLIP odstránený (sekcia „Odstránené: CLIP“).

**Možné rozmazanie pri veľkých fotkách.** Výsledok vzniká v 768×1024 a pri
vložení späť do veľkej fotky (napr. 3000×4000) sa zväčšuje. Pri zapnutom
`paste_face` je okolie ostrý originál, takže rozdiel je viditeľný. Zatiaľ
neriešené.

## Odstránené: CLIP (rukávy, dĺžka) – 2026-11-17

Od 2026-11-06 do 2026-11-17 backend používal **CLIP**
(`openai/clip-vit-base-patch32`, zero-shot) na tri úpravy masky podľa fotky
nového oblečenia:

- **rukáv** krátky / dlhý / bez → pri krátkom a bez rukávov sa z masky vyrezala
  holá koža rúk (SCHP), aby model nedokreslil pôvodný dlhý rukáv,
- **dĺžka vrchu** → pri dlhom (tunika) sa maska predĺžila o obdĺžnik cez boky,
- **dĺžka šiat** mini / po kolená / midi / maxi → maska šiat sa orezala pri leme
  (body DensePose bok → koleno → členok).

**Prečo odstránený.** Na Kaggle vyzerali tričká aj mikiny bez CLIP lepšie
(test s prepínačom `clip="off"`). Chybné určenie rukáva vyrezalo ruky z masky,
dĺžky boli na produktových fotkách nespoľahlivé. Navyše jeden model menej
na GPU a rýchlejší štart. Typ oblečenia aj strih nohavíc sa už určujú vlastnými
pravidlami bez AI.

**Čo ostalo:** ochrana dlaní (vždy), tvar šiat (bez rezu podľa dĺžky).

## Maska pre šaty (`overall`)

_Stav: **implementované 2026-11-09; rez podľa dĺžky odstránený 2026-11-17 (bez CLIP).**_

**Problém.** Pozri Zistenia „Šaty na mužovi“ – maska AutoMaskera po členky
a so siluetou kabáta → nohavice a kardigán namiesto krátkych šiat.

**Tvar tela.** Maska AutoMaskera sa oreže na povolenú oblasť:
DensePose trup + ruky + nohy roztiahnuté o ~25 px (`DRESS_BODY_DIVISOR`)
\+ obdĺžnik sukne od ¾ trupu nadol, širší o 25 % šírky trupu na každú stranu
(`DRESS_SKIRT_MARGIN`, priestor pre áčkový strih). Masku len zmenšuje, nikdy
nepridáva.

Rez pri leme podľa dĺžky šiat (CLIP) bol odstránený 2026-11-17 – maska
siaha po celej dĺžke tela, dĺžku šiat určí model z fotky.

**Výstup v `info`:** `"mask_changes": "šaty: tvar tela"`.

**Riziká / overiť na Kaggle.**

- Tesnejšia maska môže „stiahnuť“ šaty k telu (poučenie zo Sapiens) – ak áno,
  zväčšiť `DRESS_BODY_DIVISOR` / `DRESS_SKIRT_MARGIN`.
- Pôvodné voľné oblečenie mimo roztiahnutého tela (široký kabát) ostane.

## Rozšírenie masky podľa strihu nohavíc (`lower`)

_Stav: **implementované 2026-11-13, 3. verzia (cieľ podľa voľnosti strihu, 2026-11-17) otestovaná lokálne, čaká na test na Kaggle.**_

**Problém.** Pozri Zistenia „Široké (baggy) nohavice vyšli úzke“ – maska
AutoMaskera kopíruje pôvodné nohavice, model nemá priestor na širší strih.

**Riešenie – vlastný algoritmus, žiadny ďalší model.** AutoMasker beží ako
doteraz; jeho maska sa potom len **rozšíri**, nikdy nezúži (staré nohavice
musia ostať v maske celé, inak by ich `paste_face` vrátil).

**1. Analýza strihu z fotky nohavíc** (`masking.analyze_pants`, ~10 ms)

```
fotka nohavíc
 → silueta: pixely, ktoré sa farbou líšia od pozadia (medián okraja fotky)
            + otvorenie / zatvorenie (OpenCV), najväčšia súvislá oblasť
 → rozkrok: prvý riadok, kde sa silueta rozdelí na dve nohavice
            (ak sa nerozdelí alebo až pod 45 % výšky – nohavice sa
            prekrývajú, baggy – 35 % výšky, FIT_DEFAULT_CROTCH)
 → šírka bokov: najširší riadok medzi pásom a rozkrokom
            (pás sa vynechá – býva stiahnutý gumou)
 → profil: šírka jednej nohavice / šírka bokov v 20 bodoch od rozkroku po lem
```

- Strih (len do `info`): podľa šírky pri leme – `úzke` < 0,36 ≤ `rovné` <
  0,46 ≤ `široké` (`FIT_LABELS`).
- **Kontrola vierohodnosti:** nohavica širšia ako 90 % bokov
  (`FIT_MAX_PROFILE`) alebo skok v profile > 0,2 (`FIT_MAX_JUMP`) → nie je to
  produktová fotka nohavíc (napr. osoba, zložené nohavice) → bez zmeny masky.
  Rovnako, ak sa silueta nenájde alebo sú nohavice príliš krátke (šortky).

- Spodných 15 % profilu nepoklesne pod hodnotu nad nimi (zaoblený /
  pokrčený lem by šírku umelo znižoval); kontrola skokov až potom.

**2. Rozšírenie masky** (`masking._widen_legs`) – **cieľová šírka podľa
voľnosti strihu**

- Na osobe: pôvodné nohavice = SCHP (LIP / ATR nohavice, sukňa; ak ich niet,
  nohy z DensePose), nohy = DensePose. Na oboch profil `_leg_profile`.
- V každom riadku od rozkroku po lem (pomery k šírke bokov osoby):
  - **voľnosť** = (nový pomer − priliehavé nohavice) / 0,12, orezané 0…1;
    priliehavé = profil skinny na fotke, lineárne 0,47 → 0,31 (`FIT_TIGHT`),
  - **cieľ** = noha + voľnosť × (1,25 × nový pomer − noha)
    (`FIT_LOOSE_SCALE`: voľná látka visí, na osobe je ~ako na fotke),
  - **rozdiel** = cieľ − pôvodné nohavice; vonkajší okraj sa posunie von
    o 85 % rozdielu (`FIT_OUTER_SHARE`; dnu je maska už medzi nohami).
- Skinny → cieľ = noha → nič. Baggy na osobe v úzkych riflách → široko.
  Baggy na osobe v baggy → pôvodné sú už široké → málo / nič.
- Strop 45 % bokov na stranu (`FIT_MAX_EXTRA`), okraj sa vyhladí (~8 % dĺžky nohy).
- Pridá sa len to, čo nie je chránené: dlane a chodidlá (DensePose,
  roztiahnuté), vlasy, vrch, kabát, tvár, ruky, topánky (SCHP LIP aj ATR).
- Ak prírastok plochy < 1 % (`FIT_MIN_GAIN`), maska sa nemení.

_2. verzia (2026-11-13) posúvala okraj o (nový pomer − pôvodný pomer) × boky.
Na Kaggle (2026-11-17) zelené menčestrové baggy na chalanovi v skinny riflách
vyšli ako skinny: rozkrok na fotke sa našiel až pri 68 % výšky (nohavice sa
prekrývajú), pokrčený lem znížil profil (0,28 pri leme) → „rovné“, rozdiel
voči riflám pod 3 % → maska bez zmeny. Navyše pomer z fotky (rozložené)
a z osoby (spredu) nie sú priamo porovnateľné – priliehavé sú na osobe užšie,
voľné širšie. Preto 3. verzia počíta cieľ podľa voľnosti (vyššie)._

_Prvá verzia (2026-11-13) počítala cieľovú šírku absolútne – od stredu nohy
z DensePose × šírka stehien (tela). Na Kaggle sa celá zmestila do masky
AutoMaskera (pozri Zistenia „Rozšírenie nohavíc – 1. test na Kaggle“), preto
sa prešlo na rozdiel voči pôvodným nohaviciam._

**Výstup v `info`:** `"fit": "široké (koleno 0.55, lem 0.55)"`,
`"mask_changes": "nohavice rozšírené (+22%)"`. Voľba `fit="off"` rozšírenie
vypne (porovnanie pred / po s rovnakým seedom) – v UI prepínač **Respektovať
strih** v Pokročilých nastaveniach.

**Lokálne overenie (2026-11-13, bez GPU).**

- `python scripts/analyza_strihu.py <fotky nohavíc>` vypíše strih a profil.
- Baggy tepláky z testu vyššie: `široké (koleno 0.55, lem 0.55)`; umelé
  siluety: úzke 0,26, rovné 0,41, široké 0,57 pri leme; fotka osoby a fotka
  šiat → „nohavice sa nenašli“ (maska bez zmeny).
- Na fotke muža v džínsoch (DensePose / SCHP nahradené ručne vyrobenými mapami,
  maska ako rovný stĺpec ako na Kaggle) pridali baggy tepláky +22 % plochy
  masky (rozšírenie rastie smerom k lemu), rovné +7 %, úzke nič.
- Maska pre `upper` a `overall` je bit po bite rovnaká ako pred zmenou.

**Lokálne overenie 3. verzie (2026-11-17, bez GPU).** Fotky nohavíc:
4× baggy → `široké` (lem 0,47–0,83; predtým zelené menčestráky `rovné`),
rovné 0,38–0,40, skinny 0,32. Simulácia na fotke chalana v skinny riflách
(SCHP / DensePose nahradené segmentáciou riflí podľa farby, maska = konvexný
obal riflí): baggy +18 až +40 % plochy masky (smerom k lemu viac), rovné
+1 %, skinny 0 %.

**Kaggle (2026-11-17): lepšie, ale stále úzke.** Zelené baggy na chalanovi
v skinny riflách vyšli širšie ako predtým, ale stále užšie, ako by mali.
Úprava: `FIT_LOOSE_SCALE` 1,15 → 1,35, `FIT_OUTER_SHARE` 0,75 → 1,0,
`FIT_MAX_EXTRA` 0,4 → 0,5. Simulácia: prírastok masky +18 % → +40 %
(skúšané aj +56 %, vyzeralo ako sukňa). Skinny / rovné bez zmeny.

**Širšia maska, nohavice stále úzke (2026-11-17).** Maska je širšia, ale
CatVTON v nej nohavice nakreslí úzko a zvyšok vyplní pozadím – vo vnútri masky
nič nevidí a tvar odhaduje z okolia (úzke boky, topánky pri sebe). Pokus:
`guidance_scale` 2,5 → 3,5 (silnejšie držanie sa fotky oblečenia). Overiť na
Kaggle: širšie nohavice? prepálené farby / artefakty pri vrchu a šatách?
Ďalšia možnosť, ak nepomôže: nápoveda tvaru vo vnútri masky (úprava pipeline).

**Výsledok na Kaggle (2026-11-17):** 3,5 baggy nohavice rozšírilo, ale
efekt bol už príliš silný → 3,1 → 2,8 → 2,6 – všetky hodnoty nad 2,5 dávali podobný, divný výsledok
→ **vrátené na 2,5**. Guidance scale tvar nohavíc nerieši; ďalší krok:
nápoveda tvaru vo vnútri masky.

**Príčina „divného“ výsledku (2026-11-17).** Aj s guidance 2,5 rovnaký
výsledok ako s 3,5 → nespôsobovala ho guidance, ale súčasne zmenená širšia
maska (+40 %, commit `5175e71`). Maska nastavená na stred: `FIT_LOOSE_SCALE`
1,25, `FIT_OUTER_SHARE` 0,85, `FIT_MAX_EXTRA` 0,45 → zelené baggy +27 %
(predtým +18 % „lepšie“, +40 % „príliš“), rovné +2 %, skinny 0 %.

**Riziká / overiť na Kaggle.**

- Model nemusí širokú masku vyplniť celú (užšie nohavice + pozadie okolo).
- Pôvodné nohavice sú zo SCHP – ak ich SCHP zle oddelí (ruky pred nohami,
  dlhý vrch cez boky), profil pôvodných nohavíc môže byť nepresný.
- Na fotke nohavíc je pomer šírok z rozloženého kusu, na osobe z pohľadu
  spredu – nie je to to isté, rozdiel je len odhad.
- Silueta predpokladá jednoliate pozadie produktovej fotky; pri rušnom
  pozadí analýza vráti „nenašli“ alebo nepresný profil.
- Dĺžka nohavíc sa nemení (lem = členok podľa DensePose, topánky chránené).

## Plánované vylepšenia

1. Otestovať na Kaggle masku pre šaty (mini / po kolená / maxi, muž aj žena,
   bez rezu podľa dĺžky), rozšírenie masky pre široké
   nohavice (`fit="auto"` vs. `"off"`, rovnaký seed) a automatické
   rozpoznanie typu (`cloth_type="auto"`, ~15 fotiek vrch / spodok / šaty)
   a výsledky zapísať do Zistení.
2. Nápoveda tvaru vo vnútri masky (hrubá silueta vo farbe produktu) – baggy
   nohavice a dlhé rukávy, ktoré model kreslí užšie / kratšie, než dovoľuje maska.
3. Ručná úprava masky v UI (štetec) – CatVTON vlastnú masku podporuje.
4. Riešenie rozmazania pri veľkých fotkách (nezväčšovať výsledok / voľba
   „vrátiť v rozlíšení modelu“).
5. Porovnanie s iným modelom (IDM-VTON).
