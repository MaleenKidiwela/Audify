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
import time
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

print(f"Loading {REPO_ID} ...")
t0 = time.perf_counter()
model = load_model(REPO_ID)
pipeline = KokoroPipeline(lang_code="a", model=model, repo_id=REPO_ID)
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


@app.get("/")
def index():
    # no-store: the Tauri webview otherwise serves a stale cached UI across
    # engine upgrades
    return FileResponse(
        Path(__file__).parent / "player.html",
        headers={"Cache-Control": "no-store"},
    )


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
    base = max(0, min(req.start, len(req.text)))
    text = req.text[base:]

    # normalize scientific notation etc.; synthesis runs on `spoken`,
    # while mark offsets map back into the user's original text
    spoken, to_start, to_end = normalize(text)

    def generate():
        char_cursor = 0
        elapsed = 0.0  # seconds of audio emitted so far
        t_start = time.perf_counter()
        for result in pipeline(
            spoken, voice=VOICE, speed=req.speed, split_pattern=SPLIT_PATTERN
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
