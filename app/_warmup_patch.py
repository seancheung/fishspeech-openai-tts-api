"""Shorten or skip fish-speech's hard-coded warm-up run.

Upstream `ModelManager.warm_up` generates 1024 tokens at startup, which for
the 2.5B s2-pro int8 model takes on the order of a minute and blocks
container readiness. This patch lets us set a shorter length (default 64),
or skip warm-up entirely when `FISHSPEECH_WARMUP_TOKENS=0`.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

_PATCHED = False


def apply_patch(warmup_tokens: int) -> None:
    global _PATCHED
    if _PATCHED:
        return

    from fish_speech.utils.schema import ServeTTSRequest
    from tools.server import model_manager as mm
    from tools.server.inference import inference_wrapper

    def patched_warmup(self, tts_inference_engine) -> None:
        if warmup_tokens <= 0:
            log.info("warm-up skipped (FISHSPEECH_WARMUP_TOKENS=0)")
            return
        log.info("warm-up running (max_new_tokens=%d) ...", warmup_tokens)
        request = ServeTTSRequest(
            text="Hello.",
            references=[],
            reference_id=None,
            max_new_tokens=warmup_tokens,
            chunk_length=200,
            top_p=0.7,
            repetition_penalty=1.2,
            temperature=0.7,
            format="wav",
        )
        try:
            list(inference_wrapper(request, tts_inference_engine))
            log.info("warm-up done")
        except Exception:
            log.exception("warm-up raised (continuing without warm-up)")

    mm.ModelManager.warm_up = patched_warmup
    _PATCHED = True
