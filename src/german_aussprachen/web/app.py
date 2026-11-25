"""FastAPI web server for the German Pronunciation Assessment Demo."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from german_aussprachen.presets import (
    DEMO_SAMPLES,
    get_samples_dir,
    list_available_samples,
    list_presets_dict,
)
from german_aussprachen.scorer import get_default_scorer

STATIC_DIR = Path(__file__).resolve().parent / "static"
_MODEL_READY = False
_MODEL_ERROR: str | None = None


async def _preload_model() -> None:
    global _MODEL_READY, _MODEL_ERROR
    try:
        scorer = get_default_scorer()
        await asyncio.to_thread(scorer.ensure_loaded)
        _MODEL_READY = True
    except Exception as exc:  # pragma: no cover
        _MODEL_ERROR = str(exc)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    task = asyncio.create_task(_preload_model())
    try:
        yield
    finally:
        if not task.done():
            task.cancel()


def create_app() -> FastAPI:
    app = FastAPI(
        title="German Aussprache Lab",
        description="Open-source German pronunciation assessment using wav2vec2 phoneme recognition, CTC forced alignment, and GOP scoring.",
        version="0.1.0",
        lifespan=lifespan,
    )

    samples_dir = get_samples_dir()
    if samples_dir.exists():
        app.mount("/samples", StaticFiles(directory=str(samples_dir)), name="samples")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/health")
    async def health() -> dict:
        scorer = get_default_scorer()
        ready = _MODEL_READY or scorer._model is not None
        return {
            "status": "error" if _MODEL_ERROR else ("ready" if ready else "loading"),
            "model_id": scorer.model_id,
            "device": scorer.device,
            "error": _MODEL_ERROR,
        }

    @app.get("/api/presets")
    async def presets() -> dict:
        samples_dir_local = get_samples_dir()
        items = list_presets_dict()
        for item in items:
            ref_file = samples_dir_local / f"ref_{item['id']}.wav"
            item["reference_audio_url"] = f"/samples/ref_{item['id']}.wav" if ref_file.exists() else None
        return {"presets": items}

    @app.get("/api/samples")
    async def samples() -> dict:
        return {"samples": list_available_samples()}

    @app.post("/api/analyze")
    async def analyze(
        reference_text: str = Form(...),
        audio: UploadFile = File(...),
    ) -> JSONResponse:
        text = reference_text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="reference_text cannot be empty.")

        raw_bytes = await audio.read()
        if not raw_bytes:
            raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")

        scorer = get_default_scorer()
        try:
            result = await asyncio.to_thread(
                scorer.score_bytes,
                raw_bytes,
                text,
                audio.filename or "recording.webm",
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc

        return JSONResponse(content=result.to_dict())

    @app.post("/api/analyze-sample/{sample_id}")
    async def analyze_sample(
        sample_id: str,
        reference_text: str | None = Form(default=None),
    ) -> JSONResponse:
        sample = next((s for s in DEMO_SAMPLES if s.id == sample_id), None)
        if sample is None:
            raise HTTPException(status_code=404, detail=f"Unknown sample_id: {sample_id}")

        sample_path = get_samples_dir() / sample.filename
        if not sample_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Sample audio file not found: {sample.filename}. Run scripts/generate_samples.py.",
            )

        text = (reference_text or sample.reference_text).strip()
        scorer = get_default_scorer()
        try:
            result = await asyncio.to_thread(scorer.score_file, sample_path, text)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc

        payload = result.to_dict()
        payload["sample"] = {
            "id": sample.id,
            "label": sample.label,
            "kind": sample.kind,
            "description": sample.description,
            "audio_url": f"/samples/{sample.filename}",
        }
        return JSONResponse(content=payload)

    return app


app = create_app()
