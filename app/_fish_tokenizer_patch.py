"""Teach fish-speech's FishTokenizer to load raw tiktoken checkpoints.

Fish-Audio's `fishaudio/s1-mini` repo ships only `tokenizer.tiktoken` +
`special_tokens.json` (no HuggingFace standard `tokenizer_config.json` /
`tokenizer.json`). Upstream's `FishTokenizer.__init__` calls
`AutoTokenizer.from_pretrained(dir)` which fails on that layout; the failure
is silently swallowed by `llama.py:DualARTransformer.from_pretrained`,
leaving `tokenizer=None` and blowing up later inside the warm-up path when
`tokenizer.encode(...)` runs.

This patch detects the tiktoken layout and builds an equivalent tokenizer
directly from `tokenizer.tiktoken` + `special_tokens.json`. Models like
`fishaudio/s2-pro` that ship standard HF tokenizer files fall through to
the original AutoTokenizer path unchanged.
"""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import List, Union

log = logging.getLogger(__name__)

# Qwen2-family pre-tokenization regex. `config.json` reports
# vocab_size=155776 == 151643 (Qwen2 base BPE) + 12 control specials + 4096
# semantic tokens + padding, so the pre-tokenizer regex is Qwen's.
_QWEN_PAT_STR = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"|[^\r\n\p{L}\p{N}]?\p{L}+"
    r"|\p{N}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)


def _load_bpe_file(path: Path) -> dict:
    """Read a tiktoken BPE file without pulling in `blobfile`.

    Format: each non-empty line is `<base64 token> <rank>`. `tiktoken.load.
    load_tiktoken_bpe` does the same but hard-requires `blobfile` even for
    local paths, which we don't want as a runtime dep.
    """
    ranks: dict = {}
    with open(path, "rb") as f:
        for line in f.read().splitlines():
            if not line:
                continue
            token_b64, rank = line.split()
            ranks[base64.b64decode(token_b64)] = int(rank)
    return ranks


class _TikTokenBackend:
    """Minimal subset of `PreTrainedTokenizerBase` needed by FishTokenizer."""

    def __init__(self, tiktoken_file: Path, special_tokens_file: Path):
        import tiktoken

        mergeable_ranks = _load_bpe_file(tiktoken_file)
        with open(special_tokens_file, "r", encoding="utf-8") as f:
            special_tokens: dict = json.load(f)
        if not isinstance(special_tokens, dict):
            raise ValueError(
                f"expected {{'<|token|>': id, ...}} mapping in {special_tokens_file}, "
                f"got {type(special_tokens).__name__}"
            )
        special_tokens = {str(k): int(v) for k, v in special_tokens.items()}

        self._encoding = tiktoken.Encoding(
            name="fish_speech",
            pat_str=_QWEN_PAT_STR,
            mergeable_ranks=mergeable_ranks,
            special_tokens=special_tokens,
        )

        # FishTokenizer only reads get_vocab() to locate semantic tokens —
        # those are all special tokens, so exposing just the special map is
        # enough (BPE pieces aren't sensibly representable as str keys here).
        self._vocab = dict(special_tokens)
        self._pad_token_id = special_tokens.get("<|pad|>")
        self._eos_token_id = special_tokens.get("<|endoftext|>") or special_tokens.get(
            "<|end_of_text|>"
        )

    # --- PreTrainedTokenizerBase-like surface ---

    def get_vocab(self) -> dict:
        return dict(self._vocab)

    @property
    def vocab_size(self) -> int:
        return self._encoding.n_vocab

    @property
    def pad_token_id(self):
        return self._pad_token_id

    @property
    def eos_token_id(self):
        return self._eos_token_id

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
        allowed_special: Union[str, set] = "all",
        **_,
    ) -> List[int]:
        if allowed_special == "all" or add_special_tokens:
            return self._encoding.encode(
                text, allowed_special=self._encoding.special_tokens_set
            )
        if isinstance(allowed_special, (set, frozenset)):
            return self._encoding.encode(text, allowed_special=allowed_special)
        return self._encoding.encode(text, disallowed_special=())

    def decode(self, tokens: Union[List[int], int], **_) -> str:
        if isinstance(tokens, int):
            tokens = [tokens]
        return self._encoding.decode(list(tokens))

    def convert_tokens_to_ids(self, token: str):
        if token in self._vocab:
            return self._vocab[token]
        ids = self._encoding.encode(
            token, allowed_special=self._encoding.special_tokens_set
        )
        return ids[0] if len(ids) == 1 else ids


_PATCHED = False


def apply_patch() -> None:
    """Idempotently patch FishTokenizer.__init__."""
    global _PATCHED
    if _PATCHED:
        return

    from fish_speech import tokenizer as fs_tok

    original_init = fs_tok.FishTokenizer.__init__

    def patched_init(self, model_path):
        p = Path(str(model_path))

        tiktoken_file = None
        special_file = None

        if p.is_dir():
            tiktoken_file = p / "tokenizer.tiktoken"
            special_file = p / "special_tokens.json"
        elif p.is_file() and p.name == "tokenizer.tiktoken":
            tiktoken_file = p
            special_file = p.parent / "special_tokens.json"

        if (
            tiktoken_file is not None
            and tiktoken_file.is_file()
            and special_file is not None
            and special_file.is_file()
        ):
            log.info(
                "FishTokenizer: loading via tiktoken backend from %s", tiktoken_file
            )
            import torch

            self._tokenizer = _TikTokenBackend(tiktoken_file, special_file)
            self.semantic_id_to_token_id = {}

            vocab = self._tokenizer.get_vocab()
            valid_ids: list[int] = []
            for code_idx in range(4096):
                token = fs_tok.SEMANTIC_TOKEN_TEMPLATE.format(i=code_idx)
                if token in vocab:
                    token_id = vocab[token]
                    self.semantic_id_to_token_id[code_idx] = token_id
                    valid_ids.append(token_id)

            if not valid_ids:
                log.error(
                    "FishTokenizer (tiktoken): no semantic tokens found in vocab"
                )
                self.semantic_begin_id = 0
                self.semantic_end_id = 0
                self.semantic_map_tensor = torch.zeros(4096, dtype=torch.long)
            else:
                self.semantic_begin_id = min(valid_ids)
                self.semantic_end_id = max(valid_ids)
                self.semantic_map_tensor = torch.zeros(4096, dtype=torch.long)
                for k, v in self.semantic_id_to_token_id.items():
                    self.semantic_map_tensor[k] = v
            log.info(
                "FishTokenizer (tiktoken): semantic range %d..%d",
                self.semantic_begin_id,
                self.semantic_end_id,
            )
            return

        original_init(self, str(model_path))

    fs_tok.FishTokenizer.__init__ = patched_init
    _PATCHED = True
