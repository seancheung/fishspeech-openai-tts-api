"""Force `DualARTransformer.from_pretrained` to honor our max_seq_len cap.

The upstream `fish_speech.models.text2semantic.inference.init_model` calls
`DualARTransformer.from_pretrained(path, load_weights=True)` without passing
`max_length=...`. That leaves `config.max_seq_len` at whatever
`config.json` ships with — 32768 for s2-pro — which forces the model's
`__init__` to allocate a 32768×32768 causal mask (~1 GB) and later pins the
KV cache to 32K. On a 12 GB card that overflows into shared memory.

We can't modify `init_model` directly (submodule), but the upstream
`from_pretrained` already supports a `max_length` override. This patch
intercepts the call and, when the caller didn't specify one, fills in the
user's FISHSPEECH_MAX_SEQ_LEN.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

_PATCHED = False


def apply_patch(max_seq_len: int) -> None:
    global _PATCHED
    if _PATCHED:
        return
    if max_seq_len <= 0:
        return

    from fish_speech.models.text2semantic import llama as fs_llama

    original = fs_llama.DualARTransformer.from_pretrained

    def patched_from_pretrained(path, load_weights=False, max_length=None, **kwargs):
        if max_length is None:
            max_length = max_seq_len
            log.info(
                "from_pretrained: injecting max_length=%d (FISHSPEECH_MAX_SEQ_LEN)",
                max_length,
            )
        return original(
            path, load_weights=load_weights, max_length=max_length, **kwargs
        )

    # `from_pretrained` is a @staticmethod on the class; reassign directly.
    fs_llama.DualARTransformer.from_pretrained = staticmethod(patched_from_pretrained)
    _PATCHED = True
