"""Spike 2: streaming TTS server with per-word speech marks.

Proves the Speechify-style streaming architecture: the client starts
playback as soon as the first sentence is synthesized, while later
sentences are still being generated.

POST /tts  {"text": "..."}  ->  NDJSON stream, one line per sentence chunk:
  {"pcm": <base64 int16 mono>, "sampleRate": 24000,
   "marks": [{"value","startTime","endTime","start","end"}, ...]}

startTime/endTime are absolute ms from the start of the whole utterance;
start/end are character offsets into the submitted text.

Run:  .venv/bin/python spike/streaming_server.py   then open http://127.0.0.1:8765
"""

import base64
import json
import re
import subprocess
import tempfile
import threading
import time
import wave
from pathlib import Path

import numpy as np

# Bundled espeak-ng for the G2P fallback (must run before misaki loads).
import espeakng_loader
from phonemizer.backend.espeak.wrapper import EspeakWrapper

EspeakWrapper.set_library(espeakng_loader.get_library_path())
EspeakWrapper.set_data_path(espeakng_loader.get_data_path())

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from pdf_extract import extract_blocks

from mlx_audio.tts.models.kokoro import KokoroPipeline
from mlx_audio.tts.utils import load_model

import mlx_audio_patches  # noqa: F401  (fixes SineGen off-by-one-frame crash)
from normalize import normalize

REPO_ID = "prince-canuma/Kokoro-82M"
VOICE = "af_heart"
SAMPLE_RATE = 24000
# Split into sentence-sized chunks so the first audio arrives fast.
SPLIT_PATTERN = r"(?<=[.!?;:])\s+|\n+"

# curated Kokoro presets; new voices download once (~500 KB) on first use
VOICES = {
    "af_heart": "Heart · US female",
    "af_bella": "Bella · US female",
    "af_nicole": "Nicole · US female (soft)",
    "af_sky": "Sky · US female",
    "am_adam": "Adam · US male",
    "am_michael": "Michael · US male",
    "am_puck": "Puck · US male",
    "bf_emma": "Emma · UK female",
    "bf_isabella": "Isabella · UK female",
    "bm_george": "George · UK male",
    "bm_fable": "Fable · UK male",
    "bm_lewis": "Lewis · UK male",
}

print(f"Loading {REPO_ID} ...")
t0 = time.perf_counter()
model = load_model(REPO_ID)
# one pipeline per accent: 'a' (US) and 'b' (UK) differ in G2P
_pipelines: dict[str, KokoroPipeline] = {}


def get_pipeline(voice: str) -> KokoroPipeline:
    lang = voice[0] if voice[:1] in ("a", "b") else "a"
    if lang not in _pipelines:
        _pipelines[lang] = KokoroPipeline(lang_code=lang, model=model, repo_id=REPO_ID)
    return _pipelines[lang]


pipeline = get_pipeline(VOICE)
print(f"Model ready in {time.perf_counter() - t0:.1f}s")

app = FastAPI()
# the Tauri webview calls from tauri://localhost; localhost-only server, so
# a permissive CORS policy is fine
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TTSRequest(BaseModel):
    text: str
    speed: float = 1.0
    start: int = 0  # char offset to begin reading from (click-to-seek)
    voice: str = VOICE


@app.get("/voices")
def voices():
    return {"voices": [{"id": k, "label": v} for k, v in VOICES.items()],
            "default": VOICE}


@app.get("/")
def index():
    # no-store: the Tauri webview otherwise serves a stale cached UI across
    # engine upgrades
    return FileResponse(
        Path(__file__).parent / "player.html",
        headers={"Cache-Control": "no-store"},
    )


# ---- audio export (save .m4a to ~/Downloads for AirDrop etc.) ----
EXPORT = {"status": "idle", "progress": 0.0, "path": None, "error": None}
_export_lock = threading.Lock()


class ExportRequest(BaseModel):
    text: str
    speed: float = 1.0
    filename: str = "audify-audio"
    voice: str = VOICE


def _run_export(req: ExportRequest):
    try:
        voice = req.voice if req.voice in VOICES else VOICE
        pipe = get_pipeline(voice)
        spoken, _, _ = normalize(req.text)
        total = max(len(spoken), 1)
        done = 0
        chunks = []
        for result in pipe(
            spoken, voice=voice, speed=req.speed, split_pattern=SPLIT_PATTERN
        ):
            chunks.append(np.asarray(result.audio).squeeze())
            done += len(result.graphemes)
            EXPORT["progress"] = min(done / total, 0.99)
        audio = np.concatenate(chunks)
        pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)

        safe = re.sub(r"[^\w\s-]", "", req.filename).strip()[:60] or "audify-audio"
        out = Path.home() / "Downloads" / f"{safe}.m4a"
        i = 1
        while out.exists():
            out = Path.home() / "Downloads" / f"{safe} {i}.m4a"
            i += 1

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            with wave.open(tmp.name, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(SAMPLE_RATE)
                w.writeframes(pcm16.tobytes())
            subprocess.run(
                ["afconvert", "-f", "m4af", "-d", "aac", "-b", "64000",
                 tmp.name, str(out)],
                check=True, capture_output=True,
            )
            Path(tmp.name).unlink(missing_ok=True)

        EXPORT.update(status="done", progress=1.0, path=str(out))
        subprocess.run(["open", "-R", str(out)])  # reveal in Finder for AirDrop
    except Exception as e:  # surface any failure to the UI
        EXPORT.update(status="error", error=str(e))


@app.post("/export")
def export(req: ExportRequest):
    with _export_lock:
        if EXPORT["status"] == "running":
            return {"status": "running"}
        EXPORT.update(status="running", progress=0.0, path=None, error=None)
    threading.Thread(target=_run_export, args=(req,), daemon=True).start()
    return {"status": "started"}


@app.get("/export/status")
def export_status():
    return EXPORT


@app.post("/extract")
async def extract_pdf(request: Request):
    """Raw PDF bytes in -> structured read-mode blocks out."""
    data = await request.body()
    blocks = await run_in_threadpool(extract_blocks, bytes(data))
    return {
        "blocks": blocks,
        "text": "\n\n".join(b["text"] for b in blocks),
    }


@app.post("/tts")
def tts(req: TTSRequest):
    voice = req.voice if req.voice in VOICES else VOICE
    pipe = get_pipeline(voice)
    base = max(0, min(req.start, len(req.text)))
    text = req.text[base:]

    # normalize scientific notation etc.; synthesis runs on `spoken`,
    # while mark offsets map back into the user's original text
    spoken, to_start, to_end = normalize(text)

    def generate():
        char_cursor = 0
        elapsed = 0.0  # seconds of audio emitted so far
        t_start = time.perf_counter()
        for result in pipe(
            spoken, voice=voice, speed=req.speed, split_pattern=SPLIT_PATTERN
        ):
            audio = np.asarray(result.audio).squeeze()
            marks = []
            for token in result.tokens or []:
                if token.start_ts is None or token.end_ts is None:
                    continue
                s = spoken.find(token.text, char_cursor)
                if s == -1:
                    s = char_cursor
                e = s + len(token.text)
                char_cursor = e
                orig_s = to_start[min(s, len(to_start) - 1)] + base
                orig_e = to_end[min(e - 1, len(to_end) - 1)] + base
                mark = {
                    "value": token.text,
                    "startTime": round((elapsed + token.start_ts) * 1000),
                    "endTime": round((elapsed + token.end_ts) * 1000),
                    "start": orig_s,
                    "end": orig_e,
                }
                prev = marks[-1] if marks else None
                if prev and prev["start"] == orig_s and prev["end"] == orig_e:
                    # words of one normalized expression: merge into a single
                    # mark spanning the whole spoken expansion
                    prev["endTime"] = mark["endTime"]
                    prev["value"] = req.text[orig_s:orig_e]
                else:
                    marks.append(mark)
            pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
            elapsed += len(audio) / SAMPLE_RATE
            yield json.dumps(
                {
                    "pcm": base64.b64encode(pcm16.tobytes()).decode(),
                    "sampleRate": SAMPLE_RATE,
                    "marks": marks,
                    "synthElapsedMs": round((time.perf_counter() - t_start) * 1000),
                    "audioElapsedMs": round(elapsed * 1000),
                }
            ) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="warning")
