#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

# 가상환경 활성화 (NPU 전용 rbln-env)
VENV_PATH="/home/juwon/rbln-env"
if [ -f "$VENV_PATH/bin/activate" ]; then
    source "$VENV_PATH/bin/activate"
elif [ -f "$DIR/.venv/bin/activate" ]; then
    source "$DIR/.venv/bin/activate"
else
    echo "[!] Virtual environment (rbln-env) not found!"
    exit 1
fi

PORT="${PORT:-8000}"
HOST="${HOST:-localhost}"
HEALTH_URL="http://${HOST}:${PORT}/health"
MODELS_URL="http://${HOST}:${PORT}/v1/models"

# 마크다운 요약 결과 파일 경로
SUMMARY_MD="${SUMMARY_MD:-$DIR/benchmark_summary.md}"

echo "=================================================="
echo " vLLM Prefill Benchmark Runner (NPU - Rebellions ATOM)"
echo " Server     : http://${HOST}:${PORT}"
echo " Summary MD : $SUMMARY_MD"
echo "=================================================="

# 1. 서버 헬스체크
echo -n "Checking if vLLM server is alive at ${HEALTH_URL}... "
if ! curl -s -f "$HEALTH_URL" > /dev/null; then
    echo "FAILED"
    echo ""
    echo "[!] NPU vLLM 서버가 실행되고 있지 않습니다."
    echo "    먼저 다른 터미널에서 다음 명령으로 서버를 기동해 주세요:"
    echo "    ./start_server.sh"
    echo ""
    exit 1
fi
echo "OK!"

# 2. 서버에서 가용한 모델 확인 및 매칭
echo -n "Checking served models from ${MODELS_URL}... "
SERVER_MODELS_JSON=$(curl -s "$MODELS_URL" || true)

AVAILABLE_MODELS=$(python3 -c "
import sys, json
try:
    data = json.loads('''$SERVER_MODELS_JSON''').get('data', [])
    for m in data:
        if 'id' in m:
            print(m['id'])
except Exception:
    pass
" 2>/dev/null || true)

if [ -z "$AVAILABLE_MODELS" ]; then
    echo "FAILED"
    echo "[!] vLLM 서버에서 제공 중인 모델 목록을 가져올 수 없습니다. (${MODELS_URL})"
    exit 1
fi

FIRST_MODEL=$(echo "$AVAILABLE_MODELS" | head -n 1)

# 사용자가 명시한 MODEL이 없을 때 서버의 첫 번째 모델을 기본 선택
if [ -z "$MODEL" ]; then
    MODEL="$FIRST_MODEL"
    echo "OK! (Auto-detected model: $MODEL)"
else
    # 사용자가 MODEL을 지정했을 때 일치 여부 확인
    if echo "$AVAILABLE_MODELS" | grep -Fxq "$MODEL"; then
        echo "OK! (Using specified model: $MODEL)"
    else
        echo "MISMATCH"
        echo "    [!] 지정한 MODEL('$MODEL')이 서버 모델 목록에 없습니다."
        echo "        서버 제공 모델 목록: [$(echo $AVAILABLE_MODELS | tr '\n' ' ')]"
        echo "        가용한 첫 번째 모델('$FIRST_MODEL')로 자동 전환합니다."
        MODEL="$FIRST_MODEL"
    fi
fi

NUM_PROMPTS="${NUM_PROMPTS:-100}"

echo ""
echo "=================================================="
echo " [*] Running Case: PREFILL (NPU)"
echo "     Model         : ${MODEL}"
echo "     Input  Tokens : 1024"
echo "     Output Tokens : 32"
echo "     Concurrency   : 1"
echo "     Num Requests  : ${NUM_PROMPTS}"
echo "=================================================="

python "$DIR/benchmark_cases.py" \
    --url "http://${HOST}:${PORT}/v1/completions" \
    --metrics-url "http://${HOST}:${PORT}/metrics" \
    --model "$MODEL" \
    --requests "$NUM_PROMPTS" \
    --output-md "$SUMMARY_MD"

echo ""
echo "=================================================="
echo " Done! Benchmark summary updated: $SUMMARY_MD"
echo "=================================================="
