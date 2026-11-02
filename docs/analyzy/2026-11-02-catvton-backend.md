# Analýza: Bod 3 – backend `src/try_on.py` s CatVTON
_2026-11-02_

## Zhrnutie
Vytvoriť `src/try_on.py`, ktorý pri importe načíta CatVTON (SD 1.5 inpainting + attention váhy `mix-48k-1024`) a AutoMasker (DensePose + SCHP) a cez `try_on()` vráti `{"result","mask","info"}`. Odporúčanie: kód CatVTON neprepisovať, na Kaggle ho naklonovať a pridať do `sys.path`, váhy stiahnuť cez `snapshot_download` len s potrebnými priečinkami. Najväčšie riziko: inštalácia závislostí (detectron2/DensePose, verzie torch/diffusers) na Kaggle.
Pozn.: „bod 3“ v repe nie je nikde explicitne očíslovaný (v `ABOUT["pipeline"]` je bod 3 maska), analýza vychádza zo zadania.

## Postup
1. Pridať `src/__init__.py` a `src/try_on.py`, na začiatku `if not torch.cuda.is_available(): raise RuntimeError(...)`, aby `app._load_backend()` lokálne prešiel do DEMO režimu.
2. Na Kaggle v notebooku: `git clone https://github.com/Zheng-Chong/CatVTON` do `/kaggle/working/CatVTON`, cestu odovzdať cez env premennú `CATVTON_DIR` (predvolená `./CatVTON`) a v `try_on.py` ju pridať do `sys.path`.
3. Váhy: `snapshot_download("zhengchong/CatVTON", allow_patterns=["mix-48k-1024/*","SCHP/*","DensePose/*"])` + base `runwayml/stable-diffusion-inpainting` (príp. mirror, ak runwayml repo chýba) [predpoklad].
4. Pri importe raz vytvoriť `CatVTONPipeline` (fp16 – Kaggle T4/P100 nemá bf16) a `AutoMasker`, podľa vzoru z `app.py` v repe CatVTON [predpoklad: názvy tried a parametrov].
5. `try_on()`: resize/crop osoby a oblečenia na 768×1024 (š×v), maska `automasker(person, cloth_type)` (hodnoty `upper|lower|overall` sedia s CatVTON [predpoklad]), inferencia s `num_inference_steps=steps`, `generator=torch.Generator("cuda").manual_seed(seed)`, `guidance_scale≈2.5` [predpoklad].
6. `paste_face`: v1 jednoducho vložiť pôvodnú oblasť tváre z SCHP parsingu (label face/hair) cez masku; ak to nestihneme, ignorovať a uviesť v `info`.
7. Výsledok vrátiť v pôvodnom rozmere osoby; `info` = `{"model":"CatVTON mix-48k-1024","steps","seed","resolution","device"}`.
8. Zapísať `requirements-gpu.txt` (torch už je na Kaggle, doplniť diffusers/transformers/accelerate/huggingface_hub/detectron2 podľa repa) a doplniť Kaggle bunky do README.

## Riziká a na čo si dávať pozor
- detectron2 nemá wheel pre Kaggle torch/Python → skúsiť `pip install 'git+https://github.com/facebookresearch/detectron2.git'` (kompilácia ~5–10 min) alebo DensePose z vendorovanej kópie v repe CatVTON [predpoklad].
- Pinnuté staré verzie v `requirements.txt` CatVTON (Python 3.9) môžu rozbiť Kaggle torch → neinštalovať ich naslepo, len chýbajúce balíky.
- `requirements.txt` projektu nesmie dostať torch/diffusers → lokálna inštalácia by bola ťažká a DEMO režim má fungovať bez nich.
- Pomalý import (sťahovanie ~5 GB váh) blokuje štart UI → stiahnuť váhy v samostatnej bunke pred `python app.py --share`, cache na `/kaggle/working` alebo HF cache.
- OOM pri 50 krokoch / dvoch GPU procesoch → `torch.no_grad()`, fp16, `app.queue(concurrency=1)` už je; OOM chybu nechať prebublať (adaptér ju rozpozná).
- Licencia CC BY-NC-SA 4.0 → uviesť v README/„O projekte“, len nekomerčné použitie.

## Dopady
- Nové: `src/__init__.py`, `src/try_on.py`, `requirements-gpu.txt`; úprava README (Kaggle postup). `app.py`/`ui/` bez zmien.
- VRAM: <8 GB pri 1024×768 podľa README (+ DensePose/SCHP ~1–2 GB [predpoklad]); T4 16 GB stačí. Čas ~10–20 s / 30 krokov na T4 [predpoklad], `EXPECTED_SECONDS=15` overiť.
- Disk: ~5–6 GB váh na dočasnom disku Kaggle, po reštarte sa sťahujú znova.
- Kontrakt `try_on(...) -> dict` a endpoint `/try_on` sa nemenia; `show_mask` rieši UI, backend vracia masku vždy.

## Otvorené otázky
1. Ktoré váhy? – predvolene `mix-48k-1024` (univerzálne, aj šaty).
2. CatVTON naklonovať za behu, alebo vendorovať potrebné súbory do `src/catvton/`? – predvolene klon za behu (menej kódu v repe).
3. `paste_face` v prvej verzii? – predvolene áno, jednoduché vloženie tváre podľa SCHP masky; inak no-op s poznámkou v `info`.
4. Rozlíšenie – pevne 768×1024, alebo voliteľne 384×512 pre rýchlosť? – predvolene pevne 768×1024.
5. Model načítať pri importe, alebo lenivo pri prvom volaní? – predvolene pri importe (chyba → DEMO režim hneď pri štarte).

## Zdroje
- https://github.com/Zheng-Chong/CatVTON
- https://huggingface.co/zhengchong/CatVTON (súbory: `mix-48k-1024/`, `vitonhd-16k-512/`, `dresscode-16k-512/`, `SCHP/`, `DensePose/`, `flux-lora/`)
- https://huggingface.co/runwayml/stable-diffusion-inpainting
