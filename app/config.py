from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


Quantization = Literal["none", "int8", "int4"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", case_sensitive=False, extra="ignore")

    fishspeech_model: str = Field(default="fishaudio/s2-pro")
    fishspeech_checkpoints_dir: str = Field(default="/checkpoints")
    fishspeech_decoder_config_name: str = Field(default="modded_dac_vq")
    fishspeech_device: Literal["auto", "cuda", "mps", "cpu"] = Field(default="auto")
    fishspeech_cuda_index: int = Field(default=0)
    fishspeech_half: bool = Field(default=False)
    fishspeech_compile: bool = Field(default=False)
    fishspeech_quantization: Quantization = Field(
        default="none",
        description=(
            "Apply Fish-Speech's weight-only quantization to the LLaMA backbone. "
            "`int8` uses symmetric per-channel quantization (CPU- or GPU-friendly); "
            "`int4` uses groupwise affine quantization and requires CUDA at "
            "quantization time. Quantization runs once on first startup and the "
            "resulting checkpoint is cached alongside the original in "
            "`FISHSPEECH_CHECKPOINTS_DIR`."
        ),
    )
    fishspeech_int4_groupsize: int = Field(
        default=128,
        description="Group size used when `fishspeech_quantization=int4`. One of 32, 64, 128, 256.",
    )
    fishspeech_cache_dir: Optional[str] = Field(default=None)

    fishspeech_voices_dir: str = Field(default="/voices")

    fishspeech_temperature: float = Field(default=0.8, ge=0.1, le=1.0)
    fishspeech_top_p: float = Field(default=0.8, ge=0.1, le=1.0)
    fishspeech_repetition_penalty: float = Field(default=1.1, ge=0.9, le=2.0)
    fishspeech_max_new_tokens: int = Field(default=1024, ge=32, le=8192)
    fishspeech_chunk_length: int = Field(default=200, ge=100, le=1000)
    fishspeech_normalize: bool = Field(default=True)
    fishspeech_use_memory_cache: bool = Field(default=True)
    fishspeech_max_seq_len: int = Field(
        default=4096,
        ge=512,
        le=32768,
        description=(
            "Upper bound for the LLaMA KV cache / causal mask. Fish-Speech's "
            "config.json ships with 32768, which pre-allocates ~3-4 GB of KV "
            "cache + a 1 GB causal mask before inference even starts. On "
            "12 GB cards that overflows into shared/system memory, driving "
            "generation below 0.1 tok/s. TTS rarely needs >2048 tokens in a "
            "single chunk (chunk_length default is 200), so clamping to 4096 "
            "is safe. Raise only if you see sequence-length assertion errors."
        ),
    )
    fishspeech_warmup_tokens: int = Field(
        default=64,
        ge=0,
        le=2048,
        description=(
            "Tokens generated during the startup warm-up. Upstream hard-codes 1024, "
            "which on larger models (e.g. s2-pro) can stall container readiness for "
            "a minute or more. Set to 0 to skip warm-up entirely (first request is "
            "then slower, especially with torch.compile)."
        ),
    )

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    log_level: str = Field(default="info")
    max_input_chars: int = Field(default=8000)
    default_response_format: Literal[
        "mp3", "opus", "aac", "flac", "wav", "pcm"
    ] = Field(default="mp3")

    @property
    def voices_path(self) -> Path:
        return Path(self.fishspeech_voices_dir)

    @property
    def checkpoints_path(self) -> Path:
        return Path(self.fishspeech_checkpoints_dir)

    @property
    def model_basename(self) -> str:
        m = self.fishspeech_model
        if "/" in m:
            return m.split("/", 1)[1]
        return m

    @property
    def local_model_dir(self) -> Path:
        return self.checkpoints_path / self.model_basename

    @property
    def resolved_device(self) -> str:
        import torch

        if self.fishspeech_device == "auto":
            if torch.cuda.is_available():
                return f"cuda:{self.fishspeech_cuda_index}"
            mps = getattr(torch.backends, "mps", None)
            if mps is not None and mps.is_available():
                return "mps"
            return "cpu"
        if self.fishspeech_device == "cuda":
            return f"cuda:{self.fishspeech_cuda_index}"
        return self.fishspeech_device


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
