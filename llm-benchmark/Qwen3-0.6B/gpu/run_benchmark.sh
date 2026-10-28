#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

# 가상환경 활성화 (고정 경로 우선, 없으면 상위 탐색)
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

PORT="${PORT:-8001}"
HOST="${HOST:-localhost}"
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
HEALTH_URL="http://${HOST}:${PORT}/health"

# 결과 저장 디렉토리 (현재 gpu 디렉토리 내의 raw/)
RAW_DIR="${RAW_DIR:-$DIR/raw}"
mkdir -p "$RAW_DIR"

echo "=================================================="
echo " vLLM Benchmark Runner"
echo " Model      : $MODEL"
echo " Server     : http://${HOST}:${PORT}"
echo " Result Dir : $RAW_DIR"
echo "=================================================="

# 서버 헬스체크
echo -n "Checking if vLLM server is alive at ${HEALTH_URL}... "
if ! curl -s -f "$HEALTH_URL" > /dev/null; then
    echo "FAILED"
    echo ""
    echo "[!] vLLM 서버가 실행되고 있지 않습니다."
    echo "    먼저 다른 터미널에서 다음 명령으로 서버를 기동해 주세요:"
    echo "    ./start_server.sh"
    echo ""
    exit 1
fi
echo "OK!"

run_case() {
    local CASE_NAME="$1"
    local IN_LEN="$2"
    local OUT_LEN="$3"
    local CONCURRENCY="${4:-1}"
    local NUM_PROMPTS="${5:-100}"
    
    local TIMESTAMP
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    local OUT_FILE="${CASE_NAME}_${TIMESTAMP}.json"

    echo ""
    echo "=================================================="
    echo " [*] Running Case: ${CASE_NAME}"
    echo "     Input  Tokens : ${IN_LEN}"
    echo "     Output Tokens : ${OUT_LEN}"
    echo "     Concurrency   : ${CONCURRENCY}"
    echo "     Num Requests  : ${NUM_PROMPTS}"
    echo "     Output File   : ${OUT_FILE}"
    echo "=================================================="

    vllm bench serve \
        --backend openai-chat \
        --endpoint /v1/chat/completions \
        --model "$MODEL" \
        --host "$HOST" \
        --port "$PORT" \
        --dataset-name random \
        --random-input-len "$IN_LEN" \
        --random-output-len "$OUT_LEN" \
        --random-range-ratio 0.0 \
        --num-prompts "$NUM_PROMPTS" \
        --request-rate inf \
        --max-concurrency "$CONCURRENCY" \
        --seed 42 \
        --ignore-eos \
        --percentile-metrics "ttft,tpot,itl,e2el" \
        --save-result \
        --result-dir "$RAW_DIR" \
        --result-filename "$OUT_FILE"

    echo "[+] Saved result to: ${RAW_DIR}/${OUT_FILE}"
}

TARGET="${1:-all}"

case "$TARGET" in
    short)
        run_case "short" 128 128 1 100
        ;;
    prefill)
        run_case "prefill" 1024 32 1 100
        ;;
    decode)
        run_case "decode" 128 512 1 100
        ;;
    mixed)
        run_case "mixed" 1024 256 1 100
        ;;
    all)
        run_case "short" 128 128 1 100
        run_case "prefill" 1024 32 1 100
        run_case "decode" 128 512 1 100
        run_case "mixed" 1024 256 1 100
        ;;
    *)
        echo "Unknown case: $TARGET"
        echo "Available options: all, short, prefill, decode, mixed"
        exit 1
        ;;
esac

# 요약 마크다운 생성기 호출 (현재 gpu 디렉토리 내의 benchmark_summary.md)
SUMMARY_SCRIPT="$DIR/generate_summary.py"
if [ ! -f "$SUMMARY_SCRIPT" ]; then
    SUMMARY_SCRIPT="/home/modu1/npu_juwon_tmp/generate_summary.py"
fi

if [ -f "$SUMMARY_SCRIPT" ]; then
    python "$SUMMARY_SCRIPT" \
        --raw-dir "$RAW_DIR" \
        --output-md "$DIR/benchmark_summary.md"
fi
