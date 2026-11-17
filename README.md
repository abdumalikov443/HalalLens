# HalalLens 🥢

Koreyadagi musulmonlar uchun menyu va taom skaneri. Koreyscha menyuni yoki
taomni suratga olasiz — HalalLens nomini tarjima qiladi, odatiy tarkibini
aytadi va belgilaydi: 🟢 ehtimol halol · 🟡 so'rang · 🔴 harom.

> ⚠️ Bu halollik sertifikati emas. Shubha bo'lsa, restorandan so'rang.

## Qanday ishlaydi

```
Menyu rasmi ──► EasyOCR (ko+en) ──► qatorlarga ajratish ──► taom nomi + narx
                                                              │
                         ┌────────────────────────────────────┴───────────┐
                         ▼                                                ▼
              Qoidalar (kalit so'zlar)                        Claude (tarjima, tarkib,
              돼지, 삼겹살, 소주, 미림 …                        verdikt, sabab) — JSON
                         └──────────────► birlashtirish ◄─────────────────┘
                                 (AI qoida natijasini faqat qattiqlashtira oladi)

Taom rasmi ──► MobileNetV2 (12 ta koreys taomi) ──► dishes.json ──► verdikt
```

**Muhim dizayn qarori.** Kalit so'z qoidalari har doim birinchi ishlaydi. AI
`haram` natijasini hech qachon `likely_ok`ga tushira olmaydi. Model xato
qilsa ham, foydalanuvchi aniq cho'chqa go'shti yoki spirtni o'tkazib yubormaydi.

## Loyiha tuzilmasi

```
halallens/
├── app/
│   ├── main.py          FastAPI: /api/scan-menu, /api/scan-dish
│   ├── pipeline.py      OCR → qoidalar → Claude → yakuniy natija
│   ├── halal/rules.py   koreyscha/inglizcha kalit so'zlar
│   ├── menu_parser.py   OCR qutilarini menyu qatorlariga aylantirish
│   ├── ocr.py           EasyOCR
│   ├── llm.py           Claude (structured JSON output)
│   ├── classifier.py    MobileNetV2 bilan bashorat
│   ├── dishes.json      12 taom haqida ma'lumot
│   └── static/          veb interfeys (telefonga moslashgan)
├── training/            scrape → split → train → evaluate
└── tests/               38 ta test (OCR va AI'siz ishlaydi)
```

---

## Qadam-baqadam

### 1-qadam. O'rnatish
**Python 3.12 tavsiya qilinadi** (3.10–3.12). Python 3.13/3.14 da `torch`,
`easyocr` va `scikit-image` uchun tayyor paketlar bo'lmasligi mumkin.
Versiyani tekshiring: `python --version`.

Windows:
```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt        # yengil: API, qoidalar, Claude, testlar
pip install -r requirements-cv.txt     # og'ir: OCR, PyTorch, o'qitish
```
macOS / Linux: `python3.12 -m venv .venv && source .venv/bin/activate`, keyin
yuqoridagi ikki `pip install`.

`requirements.txt` o'zi bilan testlar va server ishlaydi. Menyu rasmini
o'qish (OCR) va taom modeli uchun `requirements-cv.txt` kerak.

### 2-qadam. Claude API kalitini sozlash
[console.anthropic.com](https://console.anthropic.com) dan kalit oling.
```bash
cp .env.example .env               # .env ichiga kalitni yozing
export ANTHROPIC_API_KEY=sk-ant-...   # Windows: set ANTHROPIC_API_KEY=...
```
Kalit bo'lmasa ham ilova ishlaydi, faqat kalit so'z qoidalari bilan
(tarjima va tarkib chiqmaydi).

### 3-qadam. Testlarni ishga tushirish
```bash
pytest -q
```
38 ta test o'tishi kerak. Ular OCR modeli va API kalitisiz ishlaydi.

### 4-qadam. Menyu skanerini ishga tushirish
```bash
uvicorn app.main:app --reload
```
Brauzerda http://localhost:8000 ni oching, "Menyu" bo'limida koreyscha menyu
rasmini yuklang. Birinchi so'rovda EasyOCR modellari yuklab olinadi (~100 MB,
bir marta).

### 5-qadam. Taom rasmlarini yig'ish
```bash
python training/scrape.py --per-class 150
```
**Eng muhim qism:** `data/raw/<taom>/` papkalarini oching va noto'g'ri
rasmlarni qo'lda o'chiring (logotiplar, menyu rasmlari, boshqa taomlar).
Modelning aniqligi shu tozalashga bog'liq.

### 6-qadam. Datasetni bo'lish
```bash
python training/split_dataset.py
```
Buzilgan va takroriy rasmlar olib tashlanadi, qolgani 70/15/15 nisbatda
train/val/test'ga bo'linadi.

### 7-qadam. Modelni o'qitish
```bash
python training/train.py --epochs 15
```
- 1–5 epoch: faqat yangi klassifikator qatlami o'qitiladi.
- 6–15 epoch: MobileNetV2'ning oxirgi 4 bloki kichik learning rate bilan
  fine-tune qilinadi.

Eng yaxshi model `models/dish_classifier.pt` ga saqlanadi. GPU bo'lmasa,
Google Colab'da ishlating.

### 8-qadam. Natijani o'lchash
```bash
python training/evaluate.py
```
Test to'plamidagi aniqlik va `models/confusion_matrix.png` chiqadi.
**Portfolioga shu raqamlarni yozing.**

### 9-qadam. Taom skanerini sinash
Serverni qayta ishga tushiring va "Taom" bo'limida taom rasmini yuklang.
Model ishonchi 55% dan past bo'lsa, ilova "taniy olmadim" deydi. Bu chegara
`HALALLENS_MIN_CONF` bilan sozlanadi.

---

## Portfolio uchun o'lchanadigan narsalar
- Taom klassifikatori: test aniqligi va har bir klass bo'yicha aniqlik.
- Menyu: 20–30 ta haqiqiy menyu rasmida taom nomlarini to'g'ri ajratish foizi.
- Verdikt: 50 ta taomni qo'lda belgilab, tizim natijasi bilan solishtirish.
- Tezlik: bitta menyu rasmini qayta ishlash vaqti.

## Keyingi qadamlar (ixtiyoriy)
- Telegram bot interfeysi.
- Ko'proq taom klassi (30+).
- Halol sertifikatli restoranlar xaritasi.
