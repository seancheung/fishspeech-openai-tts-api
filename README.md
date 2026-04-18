# Fish-Speech OpenAI-TTS API

**English** · [中文](./README.zh.md)

An [OpenAI TTS](https://platform.openai.com/docs/api-reference/audio/createSpeech)-compatible HTTP service wrapping [Fish-Speech](https://github.com/fishaudio/fish-speech) — Fish Audio's open-source TTS system featuring the OpenAudio S1 lineage, with zero-shot voice cloning driven by files dropped into a mounted directory.

## Features

- **OpenAI TTS compatible** — `POST /v1/audio/speech` with the same request shape as the OpenAI SDK
- **Voice cloning** — each voice is a `xxx.wav` + `xxx.txt` pair in a mounted directory; Fish-Speech continues the reference speaker to reproduce timbre and prosody
- **Direct TTS** — extra `POST /v1/audio/direct` endpoint for reference-free synthesis (default timbre)
- **2 images** — `cuda` and `cpu`
- **Model weights downloaded at runtime** — nothing heavy baked into the image; checkpoints and the HuggingFace cache are mounted for reuse
- **Multiple output formats** — `mp3`, `opus`, `aac`, `flac`, `wav`, `pcm`

## Available images

| Image | Device |
|---|---|
| `ghcr.io/seancheung/fishspeech-openai-tts-api:cuda-latest` | CUDA 12.8 |
| `ghcr.io/seancheung/fishspeech-openai-tts-api:latest`      | CPU |

Images are built for `linux/amd64`.

## Quick start

### 1. Prepare the voices directory

```
voices/
├── alice.wav     # reference audio, mono, 16kHz+, ~3-20s recommended
├── alice.txt     # UTF-8 text: the exact transcript of alice.wav
├── bob.wav
└── bob.txt
```

**Rules**: a voice is valid only when both files with the same stem exist; the stem is the voice id; unpaired or extra files are ignored. Voices are consumed by `/v1/audio/speech`; the `/v1/audio/direct` endpoint does not need the `voices/` directory.

### 2. Run the container

GPU (recommended):

```bash
docker run --rm -p 8000:8000 --gpus all \
  -v $PWD/voices:/voices:ro \
  -v $PWD/checkpoints:/checkpoints \
  -v $PWD/cache:/root/.cache \
  ghcr.io/seancheung/fishspeech-openai-tts-api:cuda-latest
```

CPU:

```bash
docker run --rm -p 8000:8000 \
  -v $PWD/voices:/voices:ro \
  -v $PWD/checkpoints:/checkpoints \
  -v $PWD/cache:/root/.cache \
  ghcr.io/seancheung/fishspeech-openai-tts-api:latest
```

Model weights are pulled from HuggingFace on first start into `/checkpoints/<model-name>/`. Mounting `/checkpoints` persists them across container restarts.

> **Gated model**: `fishaudio/s2-pro` requires accepting the license on HuggingFace. After that, pass a token to the container so `huggingface_hub` can download:
>
> ```bash
> -e HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
> ```
>
> Alternatively pre-download to the host and mount `-v ./checkpoints:/checkpoints` — the service skips the download if `codec.pth` is already present.

> **GPU prerequisites**: NVIDIA driver + [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) on Linux. On Windows use Docker Desktop + WSL2 + NVIDIA Windows driver; no host CUDA toolkit required. `s2-pro` (~9 GB safetensors, 4B params) wants ≥16 GB VRAM in bf16, or ≥10 GB with `FISHSPEECH_QUANTIZATION=int8`. **12 GB cards (RTX 4070 Ti etc.) must use int8.**

### 3. docker-compose

See [`docker/docker-compose.example.yml`](./docker/docker-compose.example.yml).

## API usage

The service listens on port `8000` by default.

### GET `/v1/audio/voices`

List all usable voices.

```bash
curl -s http://localhost:8000/v1/audio/voices | jq
```

Response:

```json
{
  "object": "list",
  "data": [
    {
      "id": "alice",
      "preview_url": "http://localhost:8000/v1/audio/voices/preview?id=alice",
      "prompt_text": "Hello, this is a reference audio sample."
    }
  ]
}
```

### GET `/v1/audio/voices/preview?id={id}`

Returns the raw reference wav (`audio/wav`), suitable for a browser `<audio>` element.

### POST `/v1/audio/speech`

OpenAI TTS-compatible endpoint — voice cloning mode. Both the voice's `wav` and `txt` are passed to Fish-Speech, which continues the reference speaker to reproduce timbre, rhythm, and emotional nuance.

```bash
curl -s http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "fish-speech",
    "input": "Hello world, this is a test.",
    "voice": "alice",
    "response_format": "mp3"
  }' \
  -o out.mp3
```

Request fields:

| Field | Type | Description |
|---|---|---|
| `model` | string | Accepted but ignored (for OpenAI SDK compatibility) |
| `input` | string | Text to synthesize, up to 8000 characters |
| `voice` | string | Voice id — must match an entry from `/v1/audio/voices` |
| `response_format` | string | `mp3` (default) / `opus` / `aac` / `flac` / `wav` / `pcm` |
| `speed` | float | Accepted for OpenAI SDK compatibility but **ignored** — Fish-Speech has no speed control |
| `temperature` | float | Optional sampling temperature (`0.1 - 1.0`, default `0.8`) |
| `top_p` | float | Optional nucleus sampling (`0.1 - 1.0`, default `0.8`) |
| `repetition_penalty` | float | Optional repetition penalty (`0.9 - 2.0`, default `1.1`) |
| `max_new_tokens` | int | Optional LLaMA generation cap (default `1024`) |
| `chunk_length` | int | Optional long-text chunk size (`100 - 1000`, default `200`) |
| `normalize` | bool | Optional en/zh text normalization (default `true`) |
| `seed` | int | Optional RNG seed for reproducible output |

Output audio is mono; sample rate is decided by the model's decoder (typically 44.1 kHz for `s2-pro`). `pcm` is raw s16le.

### POST `/v1/audio/direct`

Non-standard endpoint for reference-free synthesis. Fish-Speech picks its default timbre; useful when you don't need voice cloning.

```bash
curl -s http://localhost:8000/v1/audio/direct \
  -H 'Content-Type: application/json' \
  -d '{
    "input": "Hello, this is the default Fish-Speech voice.",
    "response_format": "mp3"
  }' \
  -o out_direct.mp3
```

Request fields: same as `/v1/audio/speech` minus `model`, `voice`, and `speed`.

### Using the OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="sk-noop")

with client.audio.speech.with_streaming_response.create(
    model="fish-speech",
    voice="alice",
    input="Hello world",
    response_format="mp3",
) as resp:
    resp.stream_to_file("out.mp3")
```

Extensions (`temperature`, `top_p`, `repetition_penalty`, `chunk_length`, …) can be passed through `extra_body={...}`.

### GET `/healthz`

Returns model name, device, sample rate and status for health checks.

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `FISHSPEECH_MODEL` | `fishaudio/s2-pro` | HuggingFace repo id or a local directory name under `/checkpoints/` |
| `FISHSPEECH_CHECKPOINTS_DIR` | `/checkpoints` | Directory where checkpoints live (and where they're downloaded to on first start) |
| `FISHSPEECH_DECODER_CONFIG_NAME` | `modded_dac_vq` | Decoder config, matches the `codec.pth` shipped with the model |
| `FISHSPEECH_DEVICE` | `auto` | `auto` → CUDA > MPS > CPU. Or `cuda` / `mps` / `cpu` |
| `FISHSPEECH_CUDA_INDEX` | `0` | Selects `cuda:N` when device is `cuda` or `auto` |
| `FISHSPEECH_HALF` | `false` | `true` → `torch.half` (fp16); `false` → `torch.bfloat16` |
| `FISHSPEECH_COMPILE` | `false` | Enable `torch.compile` — first request is slow, later ones are faster |
| `FISHSPEECH_QUANTIZATION` | `none` | `none` / `int8` / `int4`. Weight-only quantization for the LLaMA backbone. Quantization runs once on first startup, the result is cached as `<model>-int8` / `<model>-int4-g<size>-q/` next to the original checkpoint. `int4` requires CUDA at quantize time; non-CUDA devices fall back to `none`. **⚠️ `int4` is broken on torch ≥ 2.4** because fish-speech upstream hasn't updated to the new `_convert_weight_to_int4pack` API — use `int8` instead. |
| `FISHSPEECH_INT4_GROUPSIZE` | `128` | Group size for `int4` (one of `32` / `64` / `128` / `256`). |
| `FISHSPEECH_CACHE_DIR` | — | Sets `HF_HOME` before model load |
| `FISHSPEECH_VOICES_DIR` | `/voices` | Voices directory |
| `FISHSPEECH_TEMPERATURE` | `0.8` | Default sampling temperature (`0.1 - 1.0`) |
| `FISHSPEECH_TOP_P` | `0.8` | Default `top_p` (`0.1 - 1.0`) |
| `FISHSPEECH_REPETITION_PENALTY` | `1.1` | Default `repetition_penalty` (`0.9 - 2.0`) |
| `FISHSPEECH_MAX_NEW_TOKENS` | `1024` | Default generation cap |
| `FISHSPEECH_CHUNK_LENGTH` | `200` | Default long-text chunk size (`100 - 1000`) |
| `FISHSPEECH_NORMALIZE` | `true` | Default value of the `normalize` request field |
| `FISHSPEECH_USE_MEMORY_CACHE` | `true` | Cache reference-audio VQ codes between requests (per-instance) |
| `FISHSPEECH_WARMUP_TOKENS` | `64` | Tokens to generate during startup warm-up. Upstream hard-codes 1024 which blocks container readiness for 30-90s on large models (s2-pro). Set to `0` to skip entirely — first request will then be slower, especially with `FISHSPEECH_COMPILE=true`. |
| `FISHSPEECH_MAX_SEQ_LEN` | `4096` | Override `DualARTransformer`'s `max_seq_len` (ships at 32768) to limit KV cache + causal-mask pre-allocation. The shipped 32768 burns ~3-4 GB VRAM before inference, which on 12 GB cards spills into shared memory and kills throughput. **Minimum 2560** — upstream hard-codes a 2048-token reserve for generation, so allowed prompt length is `max_seq_len - 2048`; setting this ≤ 2048 raises `Prompt is too long` immediately. |
| `MAX_INPUT_CHARS` | `8000` | Upper bound for the `input` field |
| `DEFAULT_RESPONSE_FORMAT` | `mp3` | |
| `HOST` | `0.0.0.0` | |
| `PORT` | `8000` | |
| `LOG_LEVEL` | `info` | |

## Building images locally

Initialize the submodule first (the workflow does this automatically).

```bash
git submodule update --init --recursive

# CUDA image
docker buildx build -f docker/Dockerfile.cuda \
  -t fishspeech-openai-tts-api:cuda .

# CPU image
docker buildx build -f docker/Dockerfile.cpu \
  -t fishspeech-openai-tts-api:cpu .
```

## Performance tuning

### Reference configuration — RTX 4070 Ti (12 GB VRAM)

Tested with `s2-pro` + int8 + `torch.compile`:

```bash
-e FISHSPEECH_MODEL=fishaudio/s2-pro
-e FISHSPEECH_QUANTIZATION=int8
-e FISHSPEECH_MAX_SEQ_LEN=2560
-e FISHSPEECH_COMPILE=true
-e FISHSPEECH_WARMUP_TOKENS=128
```

### Observed performance

| Metric | Value |
|---|---|
| Steady-state generation | **~34 tok/s** |
| Real-time factor | **~1.6×** (generates 1 s audio in ~0.64 s wall-clock) |
| GPU memory | ~11.8 GB (tight on 12 GB; no shared-memory spill) |
| Cold start (load → warm-up → compile) | ~3 min |
| Warm start (container restart, caches hit) | ~60 s |
| Any request (after warm-up) | ~5-6 s for ~8 s audio |

Note: fish-speech logs "Compilation time: X seconds" on every request. That's misleading — it measures the first sample's total generate time, not a recompile. `torch.compile` only wraps `decode_one_token`, whose input shape is fixed `[1, codebook_dim, 1]` regardless of prompt length, so **varying prompt / reference length does NOT trigger recompile**.

### Without compile (faster startup, slower inference)

| Metric | `compile=false` | `compile=true` |
|---|---|---|
| Tok/s | ~6.9 | **~34** (5× faster) |
| Real-time factor | 0.31× | **1.6×** |
| GPU memory | ~10 GB | ~11.8 GB |
| Cold start | ~40 s | ~3 min |

Disable compile only if your deployment restarts the container frequently (K8s HPA, CI, edge short-lived instances). For long-lived services the compile tax pays itself off after the first few requests.

### Scaling to other GPUs

| VRAM | Recommended config | Notes |
|---|---|---|
| **< 10 GB** | Not supported | s2-pro int8 alone is ~8 GB + overhead |
| **10-12 GB** | Above reference config | Tight but works |
| **16-20 GB** | `MAX_SEQ_LEN=4096`, keep int8 + compile | More prompt headroom, less chance of recompile |
| **24 GB+** | Drop `QUANTIZATION=none` for bf16 | Full-precision quality; compile still recommended |

### Trade-offs

- **Compile memory budget**: `torch.compile` adds ~1-2 GB for the Triton kernel cache. On 12 GB cards this is why `MAX_SEQ_LEN=2560` (not 4096) is recommended
- **Reference audio length matters**: a 10 s reference ≈ 230 VQ tokens of prompt. Keeping references ≤ 5-7 s speeds up each generation step by 10-20 %

## Caveats

- **`speed` is a no-op.** Fish-Speech has no native speed control, but the field is kept in the schema so that OpenAI's Python SDK default request body (which always sends `speed=1.0`) does not 422. If you need tempo control, post-process the returned audio.
- **No built-in OpenAI voice names** (`alloy`, `echo`, `fable`, …). Fish-Speech is zero-shot; to get a stable voice under those names, drop `alloy.wav` + `alloy.txt` into `voices/`.
- **Concurrency**: a single Fish-Speech instance is not thread-safe; the service serializes inference with an asyncio Lock. Scale out by running more containers behind a load balancer.
- **Long text**: requests whose `input` exceeds `MAX_INPUT_CHARS` (default 8000) return 413. Fish-Speech itself handles long text by chunking on `chunk_length`.
- **Streaming is not supported** on the HTTP layer — the endpoint returns the complete audio when generation finishes. (Fish-Speech does support streaming internally; exposing it here is future work.)
- **`FISHSPEECH_COMPILE=true`** speeds up subsequent requests via `torch.compile` but makes the first inference considerably slower (and costs extra GPU memory). Leave it off for low-latency first-request use cases.
- **Quantization is one-shot & cached.** The first startup with `FISHSPEECH_QUANTIZATION=int8/int4` runs Fish-Speech's `tools/llama/quantize.py` and writes a new directory next to the original checkpoint (`<model>-int8/` or `<model>-int4-g<size>-q/`). Subsequent startups reuse it. Mount `/checkpoints` to persist. `int4` packing hard-codes `.to("cuda")`, so **the quantize step needs a GPU** even though runtime can be CPU afterward; when no CUDA is available the service logs a warning and falls back to `none`. Only the LLaMA backbone is quantized — the DAC decoder (`codec.pth`) is always loaded from the original un-quantized directory.
- **First-request warm-up**: Fish-Speech performs a warm-up run at startup, so the container is "ready" only after the health check flips to `ok`.
- **License**: Fish-Speech / OpenAudio checkpoints ship under the [Fish Audio Research License](https://github.com/fishaudio/fish-speech/blob/main/LICENSE). Review it before commercial use.
- **No built-in auth** — deploy behind a reverse proxy (Nginx, Cloudflare, etc.) if you need token-based access control.
- **Model**: `fishaudio/s2-pro` — 4B params, ~9 GB sharded safetensors + standard HF tokenizer. Needs ≥16 GB VRAM in bf16 or ≥10 GB with `FISHSPEECH_QUANTIZATION=int8`. You can point `FISHSPEECH_MODEL` at any other Fish-Speech-compatible HuggingFace repo that uses the same S2 code path.

## Project layout

```
.
├── fish-speech/                # read-only submodule, never modified
├── app/                        # FastAPI application
│   ├── server.py
│   ├── engine.py               # model loading + inference
│   ├── voices.py               # voices directory scanner
│   ├── audio.py                # multi-format encoder
│   ├── config.py
│   └── schemas.py
├── docker/
│   ├── Dockerfile.cuda
│   ├── Dockerfile.cpu
│   ├── requirements.api.txt
│   ├── entrypoint.sh
│   └── docker-compose.example.yml
├── .github/workflows/
│   └── build-images.yml        # cuda + cpu matrix build
└── README.md
```

## Acknowledgements

Built on top of [fishaudio/fish-speech](https://github.com/fishaudio/fish-speech) (Fish Audio Research License).
