"""Generate built-in German demo audio samples (native + deliberate learner mistakes) and reference audio."""

from __future__ import annotations

import asyncio
import subprocess
import tempfile
from pathlib import Path

import edge_tts

from german_aussprachen.presets import PRACTICE_PRESETS, get_samples_dir

SAMPLE_SPECS: list[tuple[str, str, str]] = [
    # (filename, spoken_text, edge_tts_voice)
    ("huette_native.wav", "Ich möchte eine Hütte mieten.", "de-DE-KatjaNeural"),
    ("huette_isch_hut.wav", "Isch möschte eine Hutte mieten.", "de-DE-KatjaNeural"),
    ("huette_english_accent.wav", "Ich möchte eine Hütte mieten.", "en-US-GuyNeural"),
    ("huette_dropped_word.wav", "Ich möchte mieten.", "de-DE-KatjaNeural"),
    ("staat_native.wav", "Der Staat baut eine neue Stadt.", "de-DE-ConradNeural"),
    ("staat_swapped.wav", "Der Stadt baut eine neue Staat.", "de-DE-ConradNeural"),
    ("bach_native.wav", "Zwölf große Pferde springen über den Bach.", "de-DE-ConradNeural"),
    ("bach_back_error.wav", "Zwölf große Pferde springen über den Back.", "de-DE-ConradNeural"),
]


async def synth_to_wav(text: str, voice: str, out_wav: Path) -> None:
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as tmp:
        await edge_tts.Communicate(text, voice).save(tmp.name)
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", tmp.name, "-ar", "16000", "-ac", "1", str(out_wav)],
            check=True,
        )


async def generate_all() -> None:
    samples_dir = get_samples_dir()
    samples_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for filename, spoken_text, voice in SAMPLE_SPECS:
        out_path = samples_dir / filename
        tasks.append((filename, synth_to_wav(spoken_text, voice, out_path)))

    # Also generate native reference audio for every practice preset
    voices = ["de-DE-KatjaNeural", "de-DE-ConradNeural"]
    for i, preset in enumerate(PRACTICE_PRESETS):
        ref_name = f"ref_{preset.id}.wav"
        out_path = samples_dir / ref_name
        tasks.append((ref_name, synth_to_wav(preset.text, voices[i % 2], out_path)))

    for name, coro in tasks:
        print(f"Generating {name}...")
        await coro
    print(f"Done! Generated {len(tasks)} audio files in {samples_dir}")


if __name__ == "__main__":
    asyncio.run(generate_all())
