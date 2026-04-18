#!/usr/bin/env bash
set -euo pipefail

: "${FISHSPEECH_MODEL:=fishaudio/openaudio-s1-mini}"
: "${FISHSPEECH_CHECKPOINTS_DIR:=/checkpoints}"
: "${FISHSPEECH_VOICES_DIR:=/voices}"
: "${FISHSPEECH_DEVICE:=auto}"
: "${HOST:=0.0.0.0}"
: "${PORT:=8000}"
: "${LOG_LEVEL:=info}"

# Map a custom cache dir to HF_HOME before transformers imports.
if [ -n "${FISHSPEECH_CACHE_DIR:-}" ]; then
  export HF_HOME="$FISHSPEECH_CACHE_DIR"
fi

export FISHSPEECH_MODEL FISHSPEECH_CHECKPOINTS_DIR FISHSPEECH_VOICES_DIR \
       FISHSPEECH_DEVICE HOST PORT LOG_LEVEL

if [ "$#" -eq 0 ]; then
  exec uvicorn app.server:app --host "$HOST" --port "$PORT" --log-level "$LOG_LEVEL"
fi
exec "$@"
