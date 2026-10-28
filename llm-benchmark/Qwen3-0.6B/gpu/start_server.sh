#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

# 가상환경 활성화
VENV_PATH="/home/modu1/npu_juwon_tmp/.venv"
if [ -f "$VENV_PATH/bin/activate" ]; then
    source "$VENV_PATH/bin/activate"
elif [ -f "$DIR/.venv/bin/activate" ]; then
    source "$DIR/.venv/bin/activate"
elif [ -f "$DIR/../../../../../.venv/bin/activate" ]; then
    source "$DIR/../../../../../.venv/bin/activate"
else
    echo "[!] Virtual environment (.venv) not found!"
    exit 1
fi

MODEL="${1:-Qwen/Qwen3-0.6B}"
PORT="${PORT:-8001}"
HOST="${HOST:-0.0.0.0}"
DTYPE="float16"

echo "=================================================="
echo " Starting vLLM Server"
echo " Model : $MODEL"
echo " Dtype : $DTYPE"
echo " Host  : $HOST"
echo " Port  : $PORT"
echo "=================================================="

# vllm serve 실행
exec python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --dtype "$DTYPE" \
    --host "$HOST" \
    --port "$PORT" \
    --gpu-memory-utilization 0.85 \
    --max-model-len 40960 \
    --trust-remote-code
