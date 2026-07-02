"""Spike: Kokoro-82M on MLX with per-word speech marks.

Proves the two risky pieces of Audify in one script:
1. Local TTS synthesis via mlx-audio (Kokoro-82M).
2. Speechify-style word-level timing ("speech marks") derived from the
   model's own phoneme duration predictions -- no forced alignment needed.

Outputs:
  spike/out/spike.wav          -- synthesized audio
  spike/out/speech_marks.json  -- [{value, startTime, endTime, start, end}, ...]
"""

import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf

# Point phonemizer at the pip-bundled espeak-ng before misaki loads, so the
# G2P fallback works without a homebrew install (and can ship in the app).
import espeakng_loader
from phonemizer.backend.espeak.wrapper import EspeakWrapper

EspeakWrapper.set_library(espeakng_loader.get_library_path())
EspeakWrapper.set_data_path(espeakng_loader.get_data_path())

from mlx_audio.tts.models.kokoro import KokoroPipeline
from mlx_audio.tts.utils import load_model

REPO_ID = "prince-canuma/Kokoro-82M"
VOICE = "af_heart"
SAMPLE_RATE = 24000

TEXT = (
    "Audify reads scientific papers aloud, entirely on your own machine. "
    "The gradient of the loss decreases by roughly one point five percent "
    "per iteration, which is a surprisingly stable result."
)

OUT_DIR = Path(__file__).parent / "out"


def main():
    OUT_DIR.mkdir(exist_ok=True)

    print(f"Loading {REPO_ID} ...")
    t0 = time.perf_counter()
    model = load_model(REPO_ID)
    pipeline = KokoroPipeline(lang_code="a", model=model, repo_id=REPO_ID)
    print(f"Model loaded in {time.perf_counter() - t0:.1f}s")

    t0 = time.perf_counter()
    audio_chunks = []
    speech_marks = []
    char_cursor = 0  # running char offset into TEXT across segments
    elapsed_offset = 0.0  # running audio offset across segments

    for result in pipeline(TEXT, voice=VOICE):
        audio = np.asarray(result.audio).squeeze()
        audio_chunks.append(audio)
        segment_duration = len(audio) / SAMPLE_RATE

        for token in result.tokens or []:
            if token.start_ts is None or token.end_ts is None:
                continue
            # char offsets: find token text in TEXT from the cursor
            start = TEXT.find(token.text, char_cursor)
            if start == -1:
                start = char_cursor
            end = start + len(token.text)
            char_cursor = end
            speech_marks.append(
                {
                    "value": token.text,
                    "startTime": round((elapsed_offset + token.start_ts) * 1000),
                    "endTime": round((elapsed_offset + token.end_ts) * 1000),
                    "start": start,
                    "end": end,
                }
            )
        elapsed_offset += segment_duration

    synth_time = time.perf_counter() - t0
    full_audio = np.concatenate(audio_chunks)
    audio_seconds = len(full_audio) / SAMPLE_RATE

    sf.write(OUT_DIR / "spike.wav", full_audio, SAMPLE_RATE)
    (OUT_DIR / "speech_marks.json").write_text(json.dumps(speech_marks, indent=2))

    print(f"\nSynthesized {audio_seconds:.1f}s of audio in {synth_time:.1f}s "
          f"(RTF {synth_time / audio_seconds:.3f})")
    print(f"Words with timestamps: {len(speech_marks)}")
    print("\nFirst 12 speech marks:")
    for m in speech_marks[:12]:
        print(f"  {m['startTime']:>6}ms - {m['endTime']:>6}ms  "
              f"[{m['start']:>3}:{m['end']:>3}]  {m['value']}")

    # sanity checks
    assert speech_marks, "no word timestamps produced"
    monotonic = all(
        a["startTime"] <= b["startTime"]
        for a, b in zip(speech_marks, speech_marks[1:])
    )
    last_end = speech_marks[-1]["endTime"] / 1000
    print(f"\nMonotonic timestamps: {monotonic}")
    print(f"Last word ends at {last_end:.2f}s vs audio length {audio_seconds:.2f}s")


if __name__ == "__main__":
    main()
