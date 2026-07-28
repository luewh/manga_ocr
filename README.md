# Manga OCR + Translation Workflow

Detects speech/thought/shout/narration bubbles on a manga or webtoon page,
OCRs the text inside each one, translates it, and outputs an annotated page
with a color-coded legend plus a JSON file with all the extracted/translated
text. SFX and captions are excluded structurally (see "How it works" below),
not by trying to guess which text looks like SFX.

## Setup

```bash
pip install -r requirements.txt
```

- `transformers`/`torch` are needed for the default bubble-location model
  (`--detector comic`) -- it downloads weights from Hugging Face on first
  run (~170MB, needs internet access once, then cached locally).
- If you use `--backend manga-ocr` for Japanese pages, also run
  `pip install manga-ocr` (separate one-time model download).

## Usage

```bash
# single page, English text (default)
python pipeline.py --input page001.png --output out/

# whole folder, French
python pipeline.py --input pages/ --output out/ --langs fr --target en

# Korean
python pipeline.py --input pages/ --output out/ --langs ko --target en

# Japanese -- use the specialized higher-accuracy OCR backend
python pipeline.py --input pages/ --output out/ --backend manga-ocr --target en

# no internet / don't want the extra ~170MB model download: fall back to
# pure-OpenCV detection (works best on clean black-and-white line art;
# see the warning about full-color/webtoon art below)
python pipeline.py --input pages/ --output out/ --detector classical

# test detection + legend only, no OCR/translation dependencies needed
python pipeline.py --input pages/ --output out/ --no-ocr --no-translate
```

For each page you get:
- `<page>_annotated.png` — original page with bubble outlines color-coded
  by shape, an `#id type` label per bubble, and a legend strip at the bottom
- `<page>.json` — structured data: per-bubble id, type, confidence, bbox,
  original text, and translated text

## How it works

Detection is a **two-stage** process:

1. **`comic_detector.py`** (default, `--detector comic`) — wraps
  [`ogkalu/comic-text-and-bubble-detector`](https://huggingface.co/ogkalu/comic-text-and-bubble-detector),
  an RT-DETR-v2 model fine-tuned on ~11k manga/webtoon/manhua/western-comic
  pages specifically for this task. It returns three classes per page:
  `bubble`, `text_bubble` (text inside a bubble), and `text_free` (text
  *outside* any bubble — SFX, captions, signage). Only `bubble` locations
  feed into the rest of the pipeline; `text_free` hits are recorded in the
  JSON's `excluded_non_bubble_text_count` but never OCR'd or shown in the
  legend. **This is how SFX exclusion works now** — it's a distinct class
  the model was trained to recognize, not a heuristic guess. Tall webtoon
  pages are sliced into overlapping chunks before detection (matching how
  the model itself was trained) and merged back with NMS-based dedup.

2. **`bubble_detector.find_local_bubble_contour`** — takes each `bubble`
  box from step 1 and, in a small local crop around just that box, finds
  the precise outline contour and classifies its shape:
  - `speech` — smooth round/oval outline
  - `thought` — cloud/scalloped outline (shallow, repeated bumps)
  - `shout` — jagged/spiky outline (deep, sharp bumps) — yelling/impact
  - `narration` — rectangular box

  This local-crop step matters: classifying shape *directly on a full,
  busy, full-color page* is unreliable (textured art, gradients, and
  screentones create spurious contours everywhere — an early version of
  this tool tried exactly that and it didn't hold up on real pages). Doing
  it on a small, already-located crop is a completely different, much
  easier problem, and is well-tested here (`test_synthetic.py`, plus manual
  verification against a real uploaded page).

3. **`ocr.py`** — masks out everything outside the found bubble contour,
  then runs one of two OCR backends:
  - `EasyOCRBackend` (default) — general-purpose, ~80 languages. Pass the
    language(s) you expect via `--langs` (EasyOCR codes, e.g. `en`, `fr`,
    `de`, `ko`, `ch_sim`, `ch_tra`, `ar`, `ru`...). EasyOCR groups languages
    into compatible sets — most Latin-script languages combine freely, but
    Japanese/Korean/Chinese each only combine with English, not with each
    other. Full list: https://www.jaided.ai/easyocr/
  - `MangaOCRBackend` (`--backend manga-ocr`) — Japanese only, a
    transformer trained specifically on manga typography (vertical text,
    furigana, stylized fonts). Use this for Japanese pages specifically.

4. **`translator.py`** — translates each extracted string. Source language
   defaults to `auto` (handy for mixed-language material), target defaults
   to `en` via `--target`. Default backend is free (Google via
   `deep-translator`); a DeepL backend is included too (usually higher
   quality, needs an API key) — swap it in `pipeline.py`.

5. **`pipeline.py`** — orchestrates detection → OCR → translation →
   annotated output + JSON.

## `--detector classical` (fallback, no model download)

`bubble_detector.detect_bubbles` also still works standalone: pure OpenCV
(Otsu threshold + contour shape analysis), no downloads, runs offline. It's
well-tested on clean black-and-white line-art pages (see
`test_synthetic.py`). **On real full-color manga/webtoon art it's
unreliable** — textured backgrounds, gradients, and screentones cause
whole panels or art details (foliage, cloak folds) to be misdetected as
bubbles, and it has no way to distinguish SFX from real bubble text other
than shape guessing. Use `--detector comic` (the default) for anything
that isn't clean line art.

## A note on testing

The `comic_detector.py` model call (`ogkalu/comic-text-and-bubble-detector`)
could not be run end-to-end in the environment this was built in (no
Hugging Face access there). What *is* tested against a real uploaded page:
the chunk-slicing/coordinate-merging math, the NMS dedup logic, and —
critically — the local shape-contour classification step, which was
verified to correctly find and classify all three real dialogue bubbles on
a real full-color webtoon page once given their approximate locations. The
one untested link is the model's own bbox accuracy on your specific art
style; if bubble locations come back off, that's the piece to look at
first (try lowering the score threshold in `ComicDetector(score_threshold=...)`,
or check the model's Hugging Face page for known limitations).

## Tuning / known limitations

- All classical-CV shape thresholds live in `DetectorConfig` (top of
  `bubble_detector.py`).
- Reading order is a simple top-row/right-to-left heuristic
  (`bubbles.sort(...)`), matching typical right-to-left manga layout —
  adjust the sort key if your source reads differently.
- `find_local_bubble_contour`'s padding (default 15px) assumes the comic
  detector's box is reasonably tight around the bubble; if bubbles are
  borderless or the model's boxes run small/large for your art style, that
  padding may need adjusting.

## Files

| File | Purpose |
|---|---|
| `comic_detector.py` | Manga/comic-specific bubble+text location model (RT-DETR-v2) |
| `bubble_detector.py` | Local shape classification (speech/thought/shout/narration) + classical-CV fallback detector |
| `ocr.py` | OCR backends (EasyOCR default, manga-ocr optional) with masked bubble-interior cropping |
| `translator.py` | Translation backends (Google default, DeepL optional) |
| `pipeline.py` | CLI entry point tying it all together |
| `test_synthetic.py` | Sanity-check script proving shape classification + SFX exclusion on synthetic shapes |
