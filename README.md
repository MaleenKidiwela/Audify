# Audify

A fully local, offline Speechify-style reader for scientific papers on Apple Silicon.
Kokoro-82M (via Apple MLX) speaks; the current word highlights as it's read; PDFs come
in with correct two-column reading order, margin line numbers stripped, and scientific
notation spoken properly. No cloud, no fees.

## Status

All four risky pieces are proven working end to end (2026-07-02):

| Piece | Where | Result |
|---|---|---|
| Local TTS + per-word "speech marks" | `spike/tts_word_timing.py` | Kokoro pred_dur → word timestamps natively; RTF ~0.14–0.21 on M2 |
| Streaming playback + live highlight | `spike/streaming_server.py` + `spike/player.html` | first audio ~300 ms, pause/resume, click-word-to-seek |
| Scientific PDF extraction | `spike/pdf_extract.py` | two-column reading order, margin line-number stripping, dehyphenation |
| Notation normalization | `spike/normalize.py` | `1.5e-9`, `10^8`, `<`, `m/s`, `µm`, Greek, °C — with offset maps back to source text |
| Desktop shell | `audify-app/` | Tauri v2 app spawning the Python engine as a sidecar |

## Run it

```bash
# engine + browser player
.venv/bin/python spike/streaming_server.py
# then open http://127.0.0.1:8765

# desktop app (spawns the engine itself)
cd audify-app && npm run tauri dev

# extract a paper to clean text
.venv/bin/python spike/pdf_extract.py spike/papers/bert.pdf
```

## Setup notes (Python 3.13, macOS arm64)

The venv was built with these hard-won details (full pins in `spike/requirements.lock.txt`):

- `pip install mlx-audio soundfile fastapi uvicorn pymupdf misaki num2words spacy phonemizer-fork espeakng-loader`
- **Do not** `pip install misaki[en]` — its `spacy-curated-transformers` extra needs a
  thinc/blis source build that fails on Python 3.13. The extra is only for `trf=True` G2P,
  which we don't use. Install `misaki` bare plus deps individually.
- **Use `phonemizer-fork`, not `phonemizer`** — misaki needs `EspeakWrapper.set_data_path`,
  which only the fork has. Paired with `espeakng-loader` (pip-bundled espeak-ng, no
  homebrew), OOD words fall back to espeak instead of crashing G2P.
- `python -m spacy download en_core_web_sm` happens automatically on first misaki use.
- `spike/mlx_audio_patches.py` must be imported before synthesis: mlx-audio 0.4.4's
  SineGen crashes on certain input lengths (FP-poisoned `ceil` in `interpolate`;
  "Hello world." reproduces it). Patch trims the sine track to f0 length.
  Worth filing upstream at Blaizzy/mlx-audio.

## Architecture

```
Tauri app (audify-app/)  ── spawns ──►  Python engine (spike/streaming_server.py)
  webview UI: player + live highlight     1. normalize.py   spoken text + offset maps
  Web Audio: schedules PCM chunks         2. KokoroPipeline  MLX synthesis per sentence
  fetch: NDJSON stream from engine        3. speech marks    pred_dur → word times,
                                                             remapped to original text
PDF path: pdf_extract.py → clean prose → the same /tts endpoint
```

Speech-mark format matches Speechify's API: `{value, startTime, endTime, start, end}`
(audio ms + char offsets into the submitted text). Words produced by a normalization
expansion (e.g. all of "one point five times ten to the negative ninth") are merged
into a single mark spanning the original `1.5e-9`, so the on-screen token stays
highlighted for the whole utterance.

## Standalone build

The distributable .app freezes the whole Python engine (PyInstaller) into the bundle:

```bash
.venv/bin/pyinstaller --noconfirm --name audify-engine \
  --distpath build/dist --workpath build/work --specpath build \
  --paths spike/vendor --hidden-import multi_column \
  --add-data "../spike/player.html:." \
  --collect-all mlx --collect-all mlx_audio --collect-all misaki \
  --collect-all espeakng_loader --collect-all en_core_web_sm \
  --collect-all spacy --collect-all thinc --collect-all phonemizer \
  --collect-all language_data --collect-all language_tags \
  --hidden-import num2words --hidden-import fitz \
  spike/streaming_server.py
rm -rf audify-app/src-tauri/engine && cp -R build/dist/audify-engine audify-app/src-tauri/engine
cd audify-app && npm run tauri build && ./fix-bundle.sh
```

`fix-bundle.sh` is required: Tauri's bundler dereferences the symlinks PyInstaller
relies on (`_internal/libmlx.dylib -> mlx/lib/libmlx.dylib`), which breaks MLX's
metallib lookup ("Failed to load the default metallib"). The script restores the
symlinks, re-signs ad hoc, and produces the DMG.

Sharing notes: Apple Silicon only; first launch downloads the Kokoro model
(~330 MB, one-time, to `~/.cache/huggingface`) then runs fully offline; the app
is unsigned so recipients must right-click → Open the first time.

## Next
- Nougat/Marker for equation-heavy papers + SRE LaTeX→speech
- Dictionary-checked dehyphenation ("fine-\ntuned" currently rejoins as "finetuned")
- Voice picker (54 Kokoro presets), speed control UI
