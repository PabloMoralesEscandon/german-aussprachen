"""Audio loading, decoding (WAV/FLAC/MP3/WebM/OGG via ffmpeg), and resampling to 16kHz mono."""

from __future__ import annotations

import io
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

TARGET_SAMPLE_RATE = 16000


def _to_mono_float32(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        raise ValueError("Audio clip is empty.")
    # Remove DC offset
    audio = audio - float(np.mean(audio))
    return audio


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    tensor = torch.from_numpy(audio).unsqueeze(0)
    resampled = torchaudio.functional.resample(tensor, orig_sr, target_sr)
    return resampled.squeeze(0).numpy().astype(np.float32)


def _decode_with_ffmpeg(data: bytes, filename: str = "input.webm") -> tuple[np.ndarray, int]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            "ffmpeg is required to decode non-WAV audio formats (webm/mp3/ogg/m4a)."
        )
    suffix = Path(filename).suffix or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(data)
        tmp.flush()
        proc = subprocess.run(
            [
                ffmpeg,
                "-v",
                "error",
                "-i",
                tmp.name,
                "-ac",
                "1",
                "-ar",
                str(TARGET_SAMPLE_RATE),
                "-f",
                "f32le",
                "pipe:1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    if proc.returncode != 0 or not proc.stdout:
        err = proc.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"Failed to decode audio with ffmpeg: {err or 'empty output'}")
    audio = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    return _to_mono_float32(audio), TARGET_SAMPLE_RATE


def load_audio_bytes(data: bytes, filename: str = "audio.wav") -> tuple[np.ndarray, int]:
    """Load audio bytes of any common format into 16kHz mono float32 waveform."""
    if not data:
        raise ValueError("Received empty audio payload.")
    try:
        audio, sr = sf.read(io.BytesIO(data), dtype="float32")
        audio = _to_mono_float32(audio)
        audio = _resample(audio, sr, TARGET_SAMPLE_RATE)
        return audio, TARGET_SAMPLE_RATE
    except Exception:
        return _decode_with_ffmpeg(data, filename=filename)


def load_audio_file(path: str | Path) -> tuple[np.ndarray, int]:
    """Load an audio file from disk into 16kHz mono float32 waveform."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Audio file not found: {p}")
    return load_audio_bytes(p.read_bytes(), filename=p.name)


def audio_rms_db(audio: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(audio)) + 1e-12))
    return float(20.0 * np.log10(rms + 1e-12))
