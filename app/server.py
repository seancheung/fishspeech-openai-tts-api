from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response

from .audio import CONTENT_TYPES, encode
from .config import get_settings
from .engine import TTSEngine
from .schemas import (
    DirectRequest,
    HealthResponse,
    SpeechRequest,
    VoiceInfo,
    VoiceList,
)
from .voices import VoiceCatalog

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper())
    app.state.settings = settings
    app.state.catalog = VoiceCatalog(settings.voices_path)
    app.state.engine = None
    try:
        app.state.engine = TTSEngine(settings)
    except Exception:
        log.exception("failed to load Fish-Speech model")
        raise
    yield


app = FastAPI(
    title="Fish-Speech OpenAI-TTS API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/healthz", response_model=HealthResponse)
async def healthz(request: Request) -> HealthResponse:
    settings = request.app.state.settings
    engine: TTSEngine | None = request.app.state.engine
    if engine is None:
        return HealthResponse(status="loading", model=settings.fishspeech_model)
    return HealthResponse(
        status="ok",
        model=settings.fishspeech_model,
        device=engine.device,
        sample_rate=engine.sample_rate,
        quantization=engine.quantization,
    )


@app.get("/v1/audio/voices", response_model=VoiceList)
async def list_voices(request: Request) -> VoiceList:
    catalog: VoiceCatalog = request.app.state.catalog
    base = str(request.base_url).rstrip("/")
    voices = catalog.scan()
    data = [
        VoiceInfo(
            id=v.id,
            preview_url=f"{base}/v1/audio/voices/preview?id={quote(v.id, safe='')}",
            prompt_text=v.prompt_text,
        )
        for v in voices.values()
    ]
    return VoiceList(data=data)


@app.get("/v1/audio/voices/preview")
async def preview_voice(id: str, request: Request):
    catalog: VoiceCatalog = request.app.state.catalog
    voice = catalog.get(id)
    if voice is None:
        raise HTTPException(status_code=404, detail=f"voice '{id}' not found")
    return FileResponse(
        path=str(voice.wav_path), media_type="audio/wav", filename=f"{id}.wav"
    )


def _validate_text(raw: str, max_chars: int) -> str:
    text = (raw or "").strip()
    if not text:
        raise HTTPException(status_code=422, detail="input is empty")
    if len(text) > max_chars:
        raise HTTPException(status_code=413, detail=f"input exceeds {max_chars} chars")
    return text


def _validate_format(fmt: str) -> None:
    if fmt not in CONTENT_TYPES:
        raise HTTPException(status_code=422, detail=f"unsupported response_format: {fmt}")


def _require_voice(catalog: VoiceCatalog, voice_id: str):
    voice = catalog.get(voice_id)
    if voice is None:
        raise HTTPException(status_code=404, detail=f"voice '{voice_id}' not found")
    return voice


def _encode_response(samples, sample_rate: int, fmt: str) -> Response:
    try:
        audio_bytes, content_type = encode(samples, sample_rate, fmt)
    except Exception as e:
        log.exception("encoding failed")
        raise HTTPException(status_code=500, detail=f"encoding failed: {e}") from e
    return Response(content=audio_bytes, media_type=content_type)


@app.post("/v1/audio/speech")
async def create_speech(body: SpeechRequest, request: Request):
    settings = request.app.state.settings
    engine: TTSEngine = request.app.state.engine
    catalog: VoiceCatalog = request.app.state.catalog

    text = _validate_text(body.input, settings.max_input_chars)
    _validate_format(body.response_format)
    voice = _require_voice(catalog, body.voice)

    try:
        sr, samples = await engine.synthesize_clone(
            text,
            prompt_wav=str(voice.wav_path),
            prompt_text=voice.prompt_text,
            temperature=body.temperature,
            top_p=body.top_p,
            repetition_penalty=body.repetition_penalty,
            max_new_tokens=body.max_new_tokens,
            chunk_length=body.chunk_length,
            normalize=body.normalize,
            seed=body.seed,
        )
    except HTTPException:
        raise
    except Exception as e:
        log.exception("inference failed")
        raise HTTPException(status_code=500, detail=f"inference failed: {e}") from e

    return _encode_response(samples, sr, body.response_format)


@app.post("/v1/audio/direct")
async def create_direct(body: DirectRequest, request: Request):
    settings = request.app.state.settings
    engine: TTSEngine = request.app.state.engine

    text = _validate_text(body.input, settings.max_input_chars)
    _validate_format(body.response_format)

    try:
        sr, samples = await engine.synthesize_direct(
            text,
            temperature=body.temperature,
            top_p=body.top_p,
            repetition_penalty=body.repetition_penalty,
            max_new_tokens=body.max_new_tokens,
            chunk_length=body.chunk_length,
            normalize=body.normalize,
            seed=body.seed,
        )
    except HTTPException:
        raise
    except Exception as e:
        log.exception("inference failed")
        raise HTTPException(status_code=500, detail=f"inference failed: {e}") from e

    return _encode_response(samples, sr, body.response_format)
