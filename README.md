# 🧠 Mesterséges Intelligencia Képfeldolgozó Pipeline

Ez a projekt egy háromlépcsős képfeldolgozó pipeline-t valósít meg PyTorch és Hugging Face Transformers segítségével. A rendszer képes:

1. 🔍 Objektumokat detektálni (DETR)
2. 🖼️ Képosztályozást végezni (ViT)
3. 📝 Képleírást generálni (BLIP)

---

## 📁 Projekt felépítése

- `main.py` – A fő futtatható Python szkript
- `images/5.picture.jpg` – A bemeneti kép (tetszőlegesen cserélhető)
- `requirements.txt` – A szükséges Python csomagok listája
- `.gitignore` – A verziókövetésből kizárt fájlok

---

## ⚙️ Telepítés

1. **Környezeti feltételek:**
   - Python 3.10 vagy újabb
   

2. **Virtuális környezet létrehozása (ajánlott):**

   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
3. **Függőségek telepítése**
   
   ```bash
   pip install -r requirements.txt
   
## Használat
1. **Helyezz el egy képet az images/ mappába, és nevezd el 5.picture.jpg-nek (vagy módosítsd az IMAGE_PATH változót a kódban).**

2. **Futtasd a programot:**
   
   ```bash
   python main.py

4. **A konzolban megjelenik:**

A detektált objektumok listája

A legvalószínűbb képosztályozási címkék

A generált képleírás

## Használt modellek
DETR (facebook/detr-resnet-50) – Objektumdetektálás

ViT (google/vit-base-patch16-224) – Képosztályozás

BLIP (Salesforce/blip-image-captioning-base) – Képleírás generálás

## Beállítások

**A következő változók módosíthatók a main.py fájlban:**
IMAGE_PATH = "images/5.picture.jpg"  # Bemeneti kép elérési útja
DET_SCORE_THR = 0.90                 # DETR küszöbérték
TOPK = 3                             # Top-k osztályozási eredmények
USE_FP16 = True                      # Használjon-e félprecíziós számítást (csak CUDA-n)

## 📜 Licenc

Ez a projekt az MIT licenc alatt érhető el. Szabadon felhasználható, módosítható és terjeszthető, amennyiben a szerzőt megjelölik.

A használt modellek (DETR, ViT, BLIP) saját licencük alá tartoznak – kérlek, ellenőrizd a Hugging Face modellek oldalait.


