#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

# 가상환경 활성화 (NPU 전용 rbln-env 우선)
VENV_PATH="/home/juwon/rbln-env"
if [ -f "$VENV_PATH/bin/activate" ]; then
    source "$VENV_PATH/bin/activate"
elif [ -f "$DIR/.venv/bin/activate" ]; then
    source "$DIR/.venv/bin/activate"
elif [ -f "$DIR/../../../../../rbln-env/bin/activate" ]; then
    source "$DIR/../../../../../rbln-env/bin/activate"
else
    echo "[!] Virtual environment (rbln-env) not found!"
    exit 1
fi

MODEL="${1:-/home/juwon/Qwen3-0.6B-fp16-dynamic}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"

# RBLN Sampler 동적 컴파일(torch.compile FailOnRecompileLimitHit) 에러 방지
export VLLM_RBLN_SAMPLER=0
export TORCHDYNAMO_RECOMPILE_LIMIT=128

echo "=================================================="
echo " Starting vLLM Server on NPU (Rebellions ATOM)"
echo " Model : $MODEL"
echo " Host  : $HOST"
echo " Port  : $PORT"
echo "=================================================="

# vllm serve 실행 (RBLN 백엔드 자동 활성화)
exec python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --trust-remote-code

