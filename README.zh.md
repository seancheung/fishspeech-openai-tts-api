# Fish-Speech OpenAI-TTS API

[English](./README.md) · **中文**

一个 [OpenAI TTS](https://platform.openai.com/docs/api-reference/audio/createSpeech) 兼容的 HTTP 服务，对 [Fish-Speech](https://github.com/fishaudio/fish-speech)（Fish Audio 开源的 TTS 系统，OpenAudio S1 系列）进行封装，支持从挂载目录零样本克隆音色。

## 特性

- **OpenAI TTS 兼容**：`POST /v1/audio/speech`，请求体格式与 OpenAI SDK 一致
- **音色克隆**：挂载 `voices/` 目录下的 `xxx.wav` + `xxx.txt` 对，Fish-Speech 会续写参考说话人以还原音色与韵律
- **直接合成**：额外提供 `POST /v1/audio/direct`，无需参考音频，使用默认音色
- **2 个镜像**：`cuda` 与 `cpu`
- **模型运行时下载**：不打包进镜像，`/checkpoints` 与 HuggingFace 缓存目录挂载后可复用
- **多种输出格式**：`mp3`、`opus`、`aac`、`flac`、`wav`、`pcm`

## 可用镜像

| 镜像 | 设备 |
|---|---|
| `ghcr.io/seancheung/fishspeech-openai-tts-api:cuda-latest` | CUDA 12.8 |
| `ghcr.io/seancheung/fishspeech-openai-tts-api:latest`      | CPU |

镜像仅构建 `linux/amd64`。

## 快速开始

### 1. 准备音色目录

```
voices/
├── alice.wav     # 参考音频，单声道，16kHz 以上，推荐 3-20 秒
├── alice.txt     # UTF-8 纯文本，内容为 alice.wav 中说出的原文
├── bob.wav
└── bob.txt
```

**规则**：必须同时存在同名的 `.wav` 与 `.txt` 才会被识别为有效音色；文件名（不含后缀）即音色 id；多余或缺对的文件会被忽略。音色仅由 `/v1/audio/speech` 使用；`/v1/audio/direct` 端点不需要 `voices/`。

### 2. 运行容器

GPU 版本（推荐）：

```bash
docker run --rm -p 8000:8000 --gpus all \
  -v $PWD/voices:/voices:ro \
  -v $PWD/checkpoints:/checkpoints \
  -v $PWD/cache:/root/.cache \
  ghcr.io/seancheung/fishspeech-openai-tts-api:cuda-latest
```

CPU 版本：

```bash
docker run --rm -p 8000:8000 \
  -v $PWD/voices:/voices:ro \
  -v $PWD/checkpoints:/checkpoints \
  -v $PWD/cache:/root/.cache \
  ghcr.io/seancheung/fishspeech-openai-tts-api:latest
```

首次启动会从 HuggingFace 下载模型权重到 `/checkpoints/<模型名>/`。挂载 `/checkpoints` 可让权重在容器重启后复用。

> **Gated 模型**：`fishaudio/s2-pro` 需要在 HuggingFace 上接受许可证。接受后把 token 传入容器让 `huggingface_hub` 下载：
>
> ```bash
> -e HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
> ```
>
> 或者在宿主机预先下载，然后 `-v ./checkpoints:/checkpoints` 挂进来 —— 服务检测到 `codec.pth` 已存在就跳过下载。

> **GPU 要求**：宿主机需安装 NVIDIA 驱动与 [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)。Windows 需 Docker Desktop + WSL2 + NVIDIA Windows 驱动。`s2-pro`（4B 参数，~9 GB safetensors）bf16 下需要 ≥16 GB 显存，`FISHSPEECH_QUANTIZATION=int8` 后 ≥10 GB。**12 GB 显卡（RTX 4070 Ti 等）必须开 int8。**

### 3. docker-compose

参考 [`docker/docker-compose.example.yml`](./docker/docker-compose.example.yml)。

## API 用法

服务默认监听 `8000` 端口。

### GET `/v1/audio/voices`

列出所有可用音色。

```bash
curl -s http://localhost:8000/v1/audio/voices | jq
```

返回：

```json
{
  "object": "list",
  "data": [
    {
      "id": "alice",
      "preview_url": "http://localhost:8000/v1/audio/voices/preview?id=alice",
      "prompt_text": "你好，这是一段参考音频。"
    }
  ]
}
```

### GET `/v1/audio/voices/preview?id={id}`

返回参考音频本体（`audio/wav`），可用于浏览器 `<audio>` 试听。

### POST `/v1/audio/speech`

OpenAI TTS 兼容接口——音色克隆模式。音色的 `.wav` 与 `.txt` 会一并作为 prompt 传给 Fish-Speech，由模型续写参考说话人，还原音色、韵律与情绪。

```bash
curl -s http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "fish-speech",
    "input": "你好世界，这是一段测试语音。",
    "voice": "alice",
    "response_format": "mp3"
  }' \
  -o out.mp3
```

请求字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `model` | string | 接受但忽略（为了与 OpenAI SDK 兼容） |
| `input` | string | 要合成的文本，最长 8000 字符 |
| `voice` | string | 音色 id，必须匹配 `/v1/audio/voices` 中的某一项 |
| `response_format` | string | `mp3`（默认） / `opus` / `aac` / `flac` / `wav` / `pcm` |
| `speed` | float | 为 OpenAI SDK 兼容而保留，**实际忽略**——Fish-Speech 无语速控制 |
| `temperature` | float | 可选采样温度（`0.1 - 1.0`，默认 `0.8`） |
| `top_p` | float | 可选 nucleus 采样（`0.1 - 1.0`，默认 `0.8`） |
| `repetition_penalty` | float | 可选重复惩罚（`0.9 - 2.0`，默认 `1.1`） |
| `max_new_tokens` | int | 可选 LLaMA 生成长度上限（默认 `1024`） |
| `chunk_length` | int | 可选长文本分块长度（`100 - 1000`，默认 `200`） |
| `normalize` | bool | 可选中英文文本规范化（默认 `true`） |
| `seed` | int | 可选随机种子，用于可复现生成 |

输出为单声道音频，采样率由解码器决定（`s2-pro` 通常为 44.1 kHz）；`pcm` 为裸 s16le 数据。

### POST `/v1/audio/direct`

非标准端点，无参考音频的合成模式，Fish-Speech 使用自带的默认音色；在不需要克隆时比较方便。

```bash
curl -s http://localhost:8000/v1/audio/direct \
  -H 'Content-Type: application/json' \
  -d '{
    "input": "你好，这是 Fish-Speech 的默认音色。",
    "response_format": "mp3"
  }' \
  -o out_direct.mp3
```

请求字段：同 `/v1/audio/speech`，去掉 `model`、`voice` 与 `speed`。

### 使用 OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="sk-noop")

with client.audio.speech.with_streaming_response.create(
    model="fish-speech",
    voice="alice",
    input="你好世界",
    response_format="mp3",
) as resp:
    resp.stream_to_file("out.mp3")
```

`temperature`、`top_p`、`repetition_penalty`、`chunk_length` 等扩展字段可通过 `extra_body={...}` 传入。

### GET `/healthz`

返回模型名、设备、采样率与状态，用于健康检查。

## 环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `FISHSPEECH_MODEL` | `fishaudio/s2-pro` | HuggingFace 仓库 id，或 `/checkpoints/` 下的本地目录名 |
| `FISHSPEECH_CHECKPOINTS_DIR` | `/checkpoints` | 权重目录（也是首次启动的下载目标） |
| `FISHSPEECH_DECODER_CONFIG_NAME` | `modded_dac_vq` | 解码器配置名，需与模型自带的 `codec.pth` 匹配 |
| `FISHSPEECH_DEVICE` | `auto` | `auto` 按 CUDA > MPS > CPU 优先级；也可强制 `cuda` / `mps` / `cpu` |
| `FISHSPEECH_CUDA_INDEX` | `0` | `cuda` / `auto` 时选择的 `cuda:N` |
| `FISHSPEECH_HALF` | `false` | `true` → `torch.half`（fp16）；`false` → `torch.bfloat16` |
| `FISHSPEECH_COMPILE` | `false` | 启用 `torch.compile` —— 首次请求慢，后续更快 |
| `FISHSPEECH_QUANTIZATION` | `none` | `none` / `int8` / `int4`，仅对 LLaMA 主干做权重量化。首次启动时量化一次，结果缓存为原 checkpoint 相邻的 `<model>-int8` 或 `<model>-int4-g<size>-q/` 目录。`int4` 在量化阶段需要 CUDA；非 CUDA 设备会自动降级为 `none`。**⚠️ `int4` 在 torch ≥ 2.4 下已损坏**（fish-speech 上游没跟上新的 `_convert_weight_to_int4pack` API），请用 `int8`。 |
| `FISHSPEECH_INT4_GROUPSIZE` | `128` | `int4` 量化的 group size（可选 `32` / `64` / `128` / `256`）。 |
| `FISHSPEECH_CACHE_DIR` | — | 加载模型前写入 `HF_HOME` |
| `FISHSPEECH_VOICES_DIR` | `/voices` | 音色目录 |
| `FISHSPEECH_TEMPERATURE` | `0.8` | 默认采样温度（`0.1 - 1.0`） |
| `FISHSPEECH_TOP_P` | `0.8` | 默认 `top_p`（`0.1 - 1.0`） |
| `FISHSPEECH_REPETITION_PENALTY` | `1.1` | 默认 `repetition_penalty`（`0.9 - 2.0`） |
| `FISHSPEECH_MAX_NEW_TOKENS` | `1024` | 默认生成长度上限 |
| `FISHSPEECH_CHUNK_LENGTH` | `200` | 默认长文本分块长度（`100 - 1000`） |
| `FISHSPEECH_NORMALIZE` | `true` | 请求 `normalize` 字段的默认值 |
| `FISHSPEECH_USE_MEMORY_CACHE` | `true` | 在同一进程内缓存参考音频的 VQ 编码 |
| `FISHSPEECH_WARMUP_TOKENS` | `64` | 启动预热时生成的 token 数。上游硬编码 1024，大模型（s2-pro）上会阻塞容器 ready 30-90 秒。设为 `0` 可完全跳过——代价是首请求更慢（开 `FISHSPEECH_COMPILE=true` 时尤甚）。 |
| `FISHSPEECH_MAX_SEQ_LEN` | `4096` | 覆盖 `DualARTransformer` 的 `max_seq_len`（上游默认 32768），限制 KV cache 与 causal mask 的预分配。32768 会在推理前就占 3-4 GB 显存，12 GB 卡会溢出到共享显存导致吞吐暴跌。**最小值 2560**——上游硬编码保留 2048 token 给生成，允许的 prompt 长度 = `max_seq_len - 2048`，所以设 ≤2048 会立刻报 `Prompt is too long`。 |
| `MAX_INPUT_CHARS` | `8000` | `input` 字段上限 |
| `DEFAULT_RESPONSE_FORMAT` | `mp3` | |
| `HOST` | `0.0.0.0` | |
| `PORT` | `8000` | |
| `LOG_LEVEL` | `info` | |

## 本地构建镜像

构建前需先初始化 submodule（workflow 已处理）。

```bash
git submodule update --init --recursive

# CUDA 镜像
docker buildx build -f docker/Dockerfile.cuda \
  -t fishspeech-openai-tts-api:cuda .

# CPU 镜像
docker buildx build -f docker/Dockerfile.cpu \
  -t fishspeech-openai-tts-api:cpu .
```

## 局限 / 注意事项

- **`speed` 字段是 no-op**：Fish-Speech 无原生语速控制；保留该字段只为让 OpenAI Python SDK 的默认请求体（总会带 `speed=1.0`）不被 422。若需调速请对返回音频做后处理。
- **不做 OpenAI 固定音色名映射**（`alloy`、`echo`、`fable` 等）。Fish-Speech 本身是零样本，没有内置音色；若想通过这些名字调用稳定的声音，直接在 `voices/` 放同名 `.wav` + `.txt` 即可。
- **并发**：Fish-Speech 单实例非线程安全，服务内部用 asyncio Lock 串行化。并发请求依赖横向扩容（多容器 + 负载均衡）。
- **长文本**：超过 `MAX_INPUT_CHARS`（默认 8000）返回 413。Fish-Speech 自身会按 `chunk_length` 对长文本做分句处理。
- **不支持 HTTP 层流式返回**：生成完成后一次性返回。（Fish-Speech 本身支持流式，服务层目前未暴露。）
- **`FISHSPEECH_COMPILE=true`** 借助 `torch.compile` 可加速后续请求，但首次推理会显著变慢并额外占显存；低延迟首请求场景建议关闭。
- **量化是一次性且缓存的**。首次以 `FISHSPEECH_QUANTIZATION=int8/int4` 启动时，服务会跑一遍 Fish-Speech 自带的 `tools/llama/quantize.py`，在原 checkpoint 旁写出 `<model>-int8/` 或 `<model>-int4-g<size>-q/` 目录；后续启动直接复用。挂载 `/checkpoints` 可跨容器复用。`int4` 的打包阶段硬编码了 `.to("cuda")`，**因此量化阶段必须有 GPU**（但运行时之后可迁回 CPU）。无 CUDA 时服务会警告并降级为 `none`。只有 LLaMA 主干会被量化，DAC 解码器 (`codec.pth`) 始终从原始未量化目录加载。
- **首次启动预热**：Fish-Speech 启动时会做一次预热推理，容器的健康检查通过后才真正 ready。
- **许可证**：Fish-Speech / OpenAudio 权重以 [Fish Audio Research License](https://github.com/fishaudio/fish-speech/blob/main/LICENSE) 发布，商用前请自行确认条款。
- **无内置鉴权**：如需 token 访问控制，请在反向代理层（Nginx、Cloudflare 等）做。
- **模型**：`fishaudio/s2-pro`——4B 参数，~9 GB 分片 safetensors + 标准 HF tokenizer。bf16 ≥16 GB 显存；开 `FISHSPEECH_QUANTIZATION=int8` 后 ≥10 GB。`FISHSPEECH_MODEL` 也可指向其他使用同一 S2 代码路径的 Fish-Speech 兼容仓库。

## 目录结构

```
.
├── fish-speech/                # 只读 submodule，不修改
├── app/                        # FastAPI 应用
│   ├── server.py
│   ├── engine.py               # 模型加载 + 推理
│   ├── voices.py               # 音色扫描
│   ├── audio.py                # 多格式编码
│   ├── config.py
│   └── schemas.py
├── docker/
│   ├── Dockerfile.cuda
│   ├── Dockerfile.cpu
│   ├── requirements.api.txt
│   ├── entrypoint.sh
│   └── docker-compose.example.yml
├── .github/workflows/
│   └── build-images.yml        # cuda + cpu 矩阵构建
└── README.md
```

## 致谢

基于 [fishaudio/fish-speech](https://github.com/fishaudio/fish-speech)（Fish Audio Research License）。
