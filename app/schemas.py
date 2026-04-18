from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


ResponseFormat = Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]


class _SamplerFields(BaseModel):
    temperature: Optional[float] = Field(default=None, ge=0.1, le=1.0)
    top_p: Optional[float] = Field(default=None, ge=0.1, le=1.0)
    repetition_penalty: Optional[float] = Field(default=None, ge=0.9, le=2.0)
    max_new_tokens: Optional[int] = Field(default=None, ge=32, le=8192)
    chunk_length: Optional[int] = Field(default=None, ge=100, le=1000)
    normalize: Optional[bool] = Field(default=None)
    seed: Optional[int] = Field(default=None)


class SpeechRequest(_SamplerFields):
    """OpenAI-compatible `/v1/audio/speech` — voice cloning via reference audio + transcript."""

    model: Optional[str] = Field(default=None, description="Accepted for OpenAI compatibility; ignored.")
    input: str = Field(..., description="Text to synthesize.")
    voice: str = Field(..., description="Voice id matching a file pair in the voices directory.")
    response_format: ResponseFormat = Field(default="mp3")
    speed: float = Field(
        default=1.0,
        ge=0.25,
        le=4.0,
        description="Accepted for OpenAI compatibility; ignored (Fish-Speech has no speed control).",
    )


class DirectRequest(_SamplerFields):
    """`/v1/audio/direct` — unconditional TTS without voice reference."""

    input: str = Field(..., description="Text to synthesize.")
    response_format: ResponseFormat = Field(default="mp3")


class VoiceInfo(BaseModel):
    id: str
    preview_url: str
    prompt_text: str


class VoiceList(BaseModel):
    object: Literal["list"] = "list"
    data: list[VoiceInfo]


class HealthResponse(BaseModel):
    status: Literal["ok", "loading", "error"]
    model: str
    device: Optional[str] = None
    sample_rate: Optional[int] = None
    quantization: Optional[Literal["none", "int8", "int4"]] = None
