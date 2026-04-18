from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .config import Settings

log = logging.getLogger(__name__)


def _ensure_fish_speech_on_path() -> None:
    """Make the `fish_speech` / `tools` packages importable.

    In the Docker image `/opt/api/fish-speech` is on `PYTHONPATH`; outside the
    container (e.g. running `uvicorn app.server:app` from the repo root) we
    fall back to the submodule path next to `app/`.
    """
    candidates = [
        Path("/opt/api/fish-speech"),
        Path(__file__).resolve().parent.parent / "fish-speech",
    ]
    for c in candidates:
        if c.is_dir() and str(c) not in sys.path:
            sys.path.insert(0, str(c))


_ensure_fish_speech_on_path()

# Must run after fish_speech is importable and before any model load.
from ._fish_tokenizer_patch import apply_patch as _apply_tokenizer_patch  # noqa: E402
from ._warmup_patch import apply_patch as _apply_warmup_patch  # noqa: E402

_apply_tokenizer_patch()


def _download_if_missing(model_id: str, target_dir: Path) -> None:
    codec_path = target_dir / "codec.pth"
    if codec_path.is_file():
        log.info("model already present at %s, skip download", target_dir)
        return

    log.info("downloading model %s → %s", model_id, target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=model_id,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
    )


class TTSEngine:
    def __init__(self, settings: Settings):
        self.settings = settings

        if settings.fishspeech_cache_dir:
            os.environ.setdefault("HF_HOME", settings.fishspeech_cache_dir)

        local_dir = settings.local_model_dir
        _download_if_missing(settings.fishspeech_model, local_dir)

        device = settings.resolved_device

        effective_quant = settings.fishspeech_quantization
        if effective_quant != "none" and not device.startswith("cuda"):
            log.warning(
                "quantization=%s requires CUDA (device=%s); falling back to none",
                effective_quant,
                device,
            )
            effective_quant = "none"
        self.quantization = effective_quant

        llama_dir = self._ensure_quantized_checkpoint(
            local_dir, effective_quant, settings.fishspeech_int4_groupsize
        )

        log.info(
            "loading Fish-Speech model=%s device=%s half=%s compile=%s quant=%s warmup_tokens=%d",
            settings.fishspeech_model,
            device,
            settings.fishspeech_half,
            settings.fishspeech_compile,
            self.quantization,
            settings.fishspeech_warmup_tokens,
        )

        _apply_warmup_patch(settings.fishspeech_warmup_tokens)

        from tools.server.model_manager import ModelManager

        decoder_checkpoint = local_dir / "codec.pth"
        if not decoder_checkpoint.is_file():
            raise FileNotFoundError(
                f"decoder checkpoint not found at {decoder_checkpoint}. "
                "Make sure the HuggingFace repo contains `codec.pth` or mount a "
                "pre-downloaded checkpoint directory at "
                f"{settings.fishspeech_checkpoints_dir}."
            )

        self.manager = ModelManager(
            mode="tts",
            device=device,
            half=settings.fishspeech_half,
            compile=settings.fishspeech_compile,
            llama_checkpoint_path=str(llama_dir),
            decoder_checkpoint_path=str(decoder_checkpoint),
            decoder_config_name=settings.fishspeech_decoder_config_name,
        )
        self.engine = self.manager.tts_inference_engine
        self.device = self.manager.device

        decoder_model = self.engine.decoder_model
        if hasattr(decoder_model, "spec_transform") and hasattr(
            decoder_model.spec_transform, "sample_rate"
        ):
            self.sample_rate = int(decoder_model.spec_transform.sample_rate)
        else:
            self.sample_rate = int(getattr(decoder_model, "sample_rate", 44100))

        self._lock = asyncio.Lock()

    @staticmethod
    def _quantized_dir_name(src_name: str, mode: str, groupsize: int) -> str:
        # Fish-Speech's llama loader picks the quant path by checking whether
        # the checkpoint directory name contains "int8" or "int4". For int4 it
        # additionally parses `g<size>` from the second-to-last `-`-separated
        # path segment, so the trailing `-q` below is a required placeholder.
        if mode == "int8":
            return f"{src_name}-int8"
        if mode == "int4":
            return f"{src_name}-int4-g{groupsize}-q"
        raise ValueError(f"invalid quantization mode: {mode}")

    def _ensure_quantized_checkpoint(
        self, src_dir: Path, mode: str, groupsize: int
    ) -> Path:
        if mode == "none":
            return src_dir

        dst_dir = src_dir.parent / self._quantized_dir_name(src_dir.name, mode, groupsize)
        if (dst_dir / "model.pth").is_file():
            log.info("quantized checkpoint already present at %s", dst_dir)
            return dst_dir

        self._quantize_checkpoint(src_dir, dst_dir, mode, groupsize)
        return dst_dir

    def _quantize_checkpoint(
        self, src_dir: Path, dst_dir: Path, mode: str, groupsize: int
    ) -> None:
        import gc
        import shutil

        import torch
        from fish_speech.models.text2semantic.inference import init_model

        # int4's packing routine hard-codes `.to("cuda")`; int8 works on CPU.
        load_device = "cuda" if mode == "int4" else "cpu"
        log.info(
            "quantizing %s → %s (mode=%s, load_device=%s)",
            src_dir,
            dst_dir,
            mode,
            load_device,
        )

        model, _ = init_model(
            checkpoint_path=str(src_dir),
            device=load_device,
            precision=torch.bfloat16,
            compile=False,
        )

        if mode == "int8":
            from tools.llama.quantize import WeightOnlyInt8QuantHandler

            handler = WeightOnlyInt8QuantHandler(model)
        elif mode == "int4":
            from tools.llama.quantize import WeightOnlyInt4QuantHandler

            handler = WeightOnlyInt4QuantHandler(model, groupsize)
        else:
            raise ValueError(f"invalid quantization mode: {mode}")

        quantized_state_dict = handler.create_quantized_state_dict()

        if dst_dir.exists():
            shutil.rmtree(dst_dir)
        shutil.copytree(str(src_dir), str(dst_dir))

        # Drop any pre-existing weight files so fish-speech loads `model.pth`.
        for stale in ("model.pth", "model.safetensors", "model.safetensors.index.json"):
            p = dst_dir / stale
            if p.is_file():
                p.unlink()
        for shard in dst_dir.glob("model-*.safetensors"):
            shard.unlink()
        # codec.pth is always loaded from the un-quantized directory.
        codec = dst_dir / "codec.pth"
        if codec.is_file():
            codec.unlink()

        target = dst_dir / "model.pth"
        log.info("saving quantized weights to %s", target)
        torch.save(quantized_state_dict, str(target))

        del model, handler, quantized_state_dict
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _build_request(
        self,
        *,
        text: str,
        references: list,
        temperature: Optional[float],
        top_p: Optional[float],
        repetition_penalty: Optional[float],
        max_new_tokens: Optional[int],
        chunk_length: Optional[int],
        normalize: Optional[bool],
        seed: Optional[int],
    ):
        from fish_speech.utils.schema import ServeTTSRequest

        s = self.settings
        return ServeTTSRequest(
            text=text,
            chunk_length=chunk_length if chunk_length is not None else s.fishspeech_chunk_length,
            format="wav",
            references=references,
            reference_id=None,
            seed=seed,
            use_memory_cache="on" if s.fishspeech_use_memory_cache else "off",
            normalize=normalize if normalize is not None else s.fishspeech_normalize,
            streaming=False,
            max_new_tokens=(
                max_new_tokens if max_new_tokens is not None else s.fishspeech_max_new_tokens
            ),
            top_p=top_p if top_p is not None else s.fishspeech_top_p,
            repetition_penalty=(
                repetition_penalty
                if repetition_penalty is not None
                else s.fishspeech_repetition_penalty
            ),
            temperature=temperature if temperature is not None else s.fishspeech_temperature,
        )

    def _run_sync(self, req) -> Tuple[int, np.ndarray]:
        sample_rate = self.sample_rate
        segments: list[np.ndarray] = []
        final_audio: Optional[np.ndarray] = None

        for result in self.engine.inference(req):
            if result.code == "error":
                raise RuntimeError(str(result.error) if result.error else "inference error")
            if result.code == "header":
                continue
            if result.code == "segment":
                if isinstance(result.audio, tuple):
                    sample_rate, audio = result.audio
                    segments.append(np.asarray(audio))
                continue
            if result.code == "final":
                if isinstance(result.audio, tuple):
                    sample_rate, final_audio = result.audio
                break

        if final_audio is not None:
            return int(sample_rate), np.asarray(final_audio)
        if segments:
            return int(sample_rate), np.concatenate(segments, axis=0)
        raise RuntimeError("no audio generated")

    async def synthesize_clone(
        self,
        text: str,
        *,
        prompt_wav: str,
        prompt_text: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        chunk_length: Optional[int] = None,
        normalize: Optional[bool] = None,
        seed: Optional[int] = None,
    ) -> Tuple[int, np.ndarray]:
        from fish_speech.utils.schema import ServeReferenceAudio

        wav_bytes = Path(prompt_wav).read_bytes()
        references = [ServeReferenceAudio(audio=wav_bytes, text=prompt_text)]
        req = self._build_request(
            text=text,
            references=references,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
            chunk_length=chunk_length,
            normalize=normalize,
            seed=seed,
        )
        async with self._lock:
            return await asyncio.to_thread(self._run_sync, req)

    async def synthesize_direct(
        self,
        text: str,
        *,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        chunk_length: Optional[int] = None,
        normalize: Optional[bool] = None,
        seed: Optional[int] = None,
    ) -> Tuple[int, np.ndarray]:
        req = self._build_request(
            text=text,
            references=[],
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
            chunk_length=chunk_length,
            normalize=normalize,
            seed=seed,
        )
        async with self._lock:
            return await asyncio.to_thread(self._run_sync, req)
