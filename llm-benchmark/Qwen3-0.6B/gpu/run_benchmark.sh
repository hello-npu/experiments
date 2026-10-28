#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

# 가상환경 활성화
source "$DIR/.venv/bin/activate"

PORT="${PORT:-8001}"
HOST="${HOST:-localhost}"
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
HEALTH_URL="http://${HOST}:${PORT}/health"

# 마크다운 요약 결과 파일 경로
SUMMARY_MD="${SUMMARY_MD:-$DIR/git/hello-npu-experiments/llm-benchmark/Qwen3-0.6B/gpu/benchmark_summary.md}"

echo "=================================================="
echo " vLLM Prefill Benchmark Runner"
echo " Model      : $MODEL"
echo " Server     : http://${HOST}:${PORT}"
echo " Summary MD : $SUMMARY_MD"
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

NUM_PROMPTS="${NUM_PROMPTS:-100}"

echo ""
echo "=================================================="
echo " [*] Running Case: PREFILL"
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
