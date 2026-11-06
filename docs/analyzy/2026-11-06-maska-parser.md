# Analýza: Variant C – presnejší parser namiesto SCHP pri maske
_2026-11-06_

## Zhrnutie
Nahradiť SCHP (LIP/ATR) + DensePose v našej maske parserom **Meta Sapiens-seg** (28 tried Goliath: vrch/spodok oblečenia, predlaktie/rameno/dlaň zvlášť L/P), ktorý je výrazne robustnejší pri neštandardných pózach. Odporúčanie: `src/masking.py` s voľbou `masker="schp"|"sapiens"`, Sapiens **0.6B/1B torchscript (fp32)**, AutoMasker ponechať ako fallback a pre porovnanie. Najväčšie riziko: VRAM na T4 (CatVTON + CLIP + Sapiens fp32) a prepísanie heuristík AutoMaskera tak, aby maska nebola horšia ako dnes.

Kandidáti (stručne):
- **Sapiens-seg 0.3B/0.6B/1B** – trénovaný na 300M ľudských fotkách, najlepší pri zlých pózach; triedy Upper_Clothing, Lower_Clothing, Apparel, Torso, Left/Right_Hand, _Lower_Arm, _Upper_Arm, Face_Neck, Hair… [presný zoznam/indexy predpoklad – overiť v repe]; vstup 1024×768 (sedí na náš 768×1024); váhy na HF (`facebook/sapiens-seg-1b-torchscript`, menšie [predpoklad]); licencia **CC-BY-NC 4.0** (školský projekt OK); torchscript = len torch, fp32 na každej GPU, bf16 verzia len A100 → na T4 nepoužiteľná.
- **SegFormer b2 clothes** (`mattmdjaga/segformer_b2_clothes`) – 27M param., cez `transformers` (už máme), rýchly, ~1 GB VRAM; ale trénovaný na ATR (rovnaký zdroj ako SCHP) → pri zlých pózach podobné chyby; nemá dlane zvlášť (len Left/Right-arm); mIoU 0.69; licencia NVlabs (nekomerčná). Vhodný len ako lacná alternatíva.
- **DensePose** – po prechode na Sapiens netreba (dlane/predlaktia má Sapiens); ponechať len v režime `schp`.

## Postup
1. Na Kaggle prečítať `model/cloth_masker.py` v klone CatVTON a spísať, čo AutoMasker robí (ochrana tváre/vlasov/dlaní, roztiahnutie oblasti oblečenia, zlúčenie, vyplnenie dier) – prebrať len tieto kroky.
2. Vytvoriť `src/masking.py`: `build_mask(person_r, cloth_type, sleeves, masker) -> (mask, parts, protected)`; vetva `schp` = dnešný AutoMasker + `_protect_mask` presunuté z `try_on.py`.
3. Vetva `sapiens`: načítať torchscript cez `hf_hub_download` + `torch.jit.load`, inferencia 1024×768 fp32, argmax → mapa tried.
4. Mapovanie: maska = oblečenie podľa `cloth_type` (upper: Upper_Clothing+Apparel+Torso+ramená; lower: Lower_Clothing+nohy; overall: obe) → dilatácia → mínus chránené (Face_Neck, Hair, dlane vždy; predlaktia/ramená pri CLIP short/sleeveless) → `_mask_processor.blur` ako dnes.
5. `try_on()` číta `options.get("masker", "sapiens")` a `compare_masks` (vráti v `info` čas a podiel pixelov oboch masiek; masky voliteľne uloží do `/kaggle/working`) – podpis a návrat bez zmeny.
6. Ak Sapiens zlyhá pri načítaní (váhy, OOM), log + fallback na `schp` (ako CLIP).
7. Test v `kaggle/run.ipynb` na sade 5–10 fotiek (prekrížené ruky, ruky za chrbtom, bok) – porovnať masky `schp` vs `sapiens`; zapísať do `docs/backend.md`.

## Riziká a na čo si dávať pozor
- VRAM: 1B fp32 ≈ 4 GB váh + aktivácie [predpoklad] → začať 0.6B, Sapiens držať na GPU, `torch.cuda.empty_cache()` po inferencii; pri OOM 0.3B.
- Formát váh `.pt2`/torchscript a verzia torchu na Kaggle → overiť načítanie ako prvý krok (torch nepripínať).
- Indexy tried Sapiens sa ľahko pomýlia → mapovanie vziať z `sapiens/lite` (`GOLIATH_CLASSES`), nie z pamäti.
- Sapiens čisto podľa parsingu nevie „kam“ nové oblečenie siaha (dlhšie/voľnejšie ako pôvodné) → zachovať roztiahnutie oblasti ako AutoMasker (bbox/konvexný obal torza).
- Čas sťahovania 1B (~4–5 GB) na dočasný disk Kaggle → cache v `/kaggle/working` alebo menší model.
- Licencia CC-BY-NC → spomenúť v README, žiadne komerčné použitie.

## Dopady
- Mení sa: `src/try_on.py` (maskovanie vyčlenené), nový `src/masking.py`, `kaggle/run.ipynb` (test porovnania), `docs/backend.md`, `README.md` (licencia, prepínač).
- Závislosti: žiadne nové (torch + `huggingface_hub`); pri SegFormer len `transformers` (už je).
- VRAM +2–6 GB, čas +~0.5–2 s na obrázok [predpoklad]; DEMO režim nedotknutý (import v backende).
- Kontrakt `try_on()` a `/try_on` sa nemení – `masker`/`compare_masks` sú len `**options`, UI/API ich neposiela (predvolené).

## Otvorené otázky
1. Veľkosť Sapiens? – predvolene **0.6B**, 1B len ak VRAM na T4 vydrží.
2. Predvolený masker? – **sapiens** s fallbackom na schp.
3. Pridať voľbu maskera do UI? – **nie**, len cez `options` / notebook (API kontrakt nemeniť).
4. Ponechať DensePose/AutoMasker natrvalo? – **áno**, kým porovnanie nepotvrdí Sapiens; potom zvážiť odstránenie (úspora VRAM).
5. Ukladať porovnanie masiek? – **áno**, PNG do `/kaggle/working/masks/` pri `compare_masks=True`.

Odhad náročnosti: ~1–2 dni (0,5 d načítanie + mapovanie, 0,5 d heuristiky AutoMaskera, 0,5–1 d testovanie na Kaggle).

## Zdroje
- https://github.com/facebookresearch/sapiens (lite/README – torchscript fp32, bf16 len A100)
- https://huggingface.co/facebook/sapiens-seg-1b-torchscript (CC-BY-NC 4.0, 28 tried, 1024×768)
- https://huggingface.co/mattmdjaga/segformer_b2_clothes (18 tried, ATR, mIoU 0.69)
