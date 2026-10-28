#!/usr/bin/env python3
"""
vLLM Prefill Benchmark Script for Qwen3-0.6B on NPU (Rebellions ATOM)
Case:
- prefill: input 1024 tokens, output 32 tokens, concurrency 1, requests 100

Prefill 시간 측정 방법:
  vLLM Prometheus 메트릭 (request_prefill_time_seconds_sum/count) 을 이용해
  벤치마크 실행 전/후 스냅샷을 비교합니다.
  → avg_prefill = (delta_sum / delta_count) * 1000 (ms)
"""

import argparse
import asyncio
import json
import os
import re
import statistics
import time
from typing import Dict, Any, List, Optional, Tuple
import aiohttp
from transformers import AutoTokenizer

BENCHMARK_CASE = {
    "description": "Prefill Heavy",
    "input_tokens": 1024,
    "output_tokens": 32,
    "concurrency": 1,
    "num_requests": 100,
}

# ---------------------------------------------------------------------------
# Prometheus 메트릭 헬퍼
# ---------------------------------------------------------------------------

def _parse_prometheus_value(text: str, metric_name: str) -> Optional[float]:
    """
    /metrics 응답에서 특정 메트릭의 합산(sum) 값을 파싱합니다.
    """
    total = 0.0
    found = False
    pattern = re.compile(
        rf'^{re.escape(metric_name)}(?:\{{[^}}]*\}})?\s+([\d.eE+\-]+)',
        re.MULTILINE,
    )
    for m in pattern.finditer(text):
        total += float(m.group(1))
        found = True
    return total if found else None


async def fetch_prefill_snapshot(metrics_url: str) -> Tuple[Optional[float], Optional[float]]:
    """
    (sum, count) 스냅샷을 반환합니다. 실패 시 (None, None).
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(metrics_url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    return None, None
                text = await resp.text()
        s = _parse_prometheus_value(text, "vllm:request_prefill_time_seconds_sum")
        c = _parse_prometheus_value(text, "vllm:request_prefill_time_seconds_count")
        return s, c
    except Exception as e:
        print(f"  [!] 메트릭 수집 실패: {e}")
        return None, None


# ---------------------------------------------------------------------------
# 프롬프트 생성
# ---------------------------------------------------------------------------

def generate_prompt_for_tokens(tokenizer: Any, target_tokens: int) -> str:
    """Generate exact token-length prompt using repetitive natural text."""
    seed_phrase = "Artificial intelligence and machine learning technologies are rapidly transforming our modern digital world. "
    seed_ids = tokenizer.encode(seed_phrase, add_special_tokens=False)

    repeated_ids = (seed_ids * ((target_tokens // len(seed_ids)) + 2))[:target_tokens]
    prompt = tokenizer.decode(repeated_ids)

    actual_len = len(tokenizer.encode(prompt, add_special_tokens=False))
    if actual_len != target_tokens:
        prompt = tokenizer.decode(tokenizer.encode(prompt, add_special_tokens=False)[:target_tokens])
    return prompt


# ---------------------------------------------------------------------------
# 요청 전송
# ---------------------------------------------------------------------------

async def send_streaming_request(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    prompt: str,
    output_tokens: int,
    request_id: int,
    ignore_eos: bool = True,
) -> Dict[str, Any]:
    """Send streaming request and measure TTFT / ITL / latency."""
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": output_tokens,
        "min_tokens": output_tokens,
        "temperature": 0.0,
        "stream": True,
        "ignore_eos": ignore_eos,
    }

    start_time = time.perf_counter()
    first_token_time = None
    last_token_time = start_time
    token_latencies = []
    generated_tokens = 0

    try:
        async with session.post(url, json=payload) as response:
            if response.status != 200:
                text = await response.text()
                return {
                    "request_id": request_id,
                    "success": False,
                    "error": f"HTTP {response.status}: {text[:100]}",
                }

            async for line in response.content:
                now = time.perf_counter()
                line = line.decode("utf-8").strip()
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break

                try:
                    data = json.loads(data_str)
                    choices = data.get("choices", [])
                    if choices and choices[0].get("text"):
                        if first_token_time is None:
                            first_token_time = now
                        token_latencies.append(now - last_token_time)
                        last_token_time = now
                        generated_tokens += 1
                except Exception:
                    pass

    except Exception as e:
        return {"request_id": request_id, "success": False, "error": str(e)}

    end_time = time.perf_counter()
    total_latency = end_time - start_time
    ttft = (first_token_time - start_time) if first_token_time is not None else total_latency

    itls = token_latencies[1:] if len(token_latencies) > 1 else []
    avg_itl = statistics.mean(itls) if itls else 0.0

    return {
        "request_id": request_id,
        "success": True,
        "total_latency": total_latency,
        "ttft": ttft,
        "avg_itl": avg_itl,
        "generated_tokens": generated_tokens or output_tokens,
    }


# ---------------------------------------------------------------------------
# 마크다운 요약 생성기
# ---------------------------------------------------------------------------

def write_markdown_summary(summary: Dict[str, Any], output_path: str):
    metric_proc = summary.get("metric_process", {})
    avg_prefill_val = summary.get("avg_prefill_metric_ms")
    delta_sum = metric_proc.get("delta_sum") or 0.0
    delta_count = metric_proc.get("delta_count") or 0.0

    if avg_prefill_val is not None and delta_count > 0:
        calc_formula_md = f"""### 2. 평균 Prefill 시간 계산

$$\\text{{Avg Prefill Time}} = \\frac{{\\Delta \\text{{request\\_prefill\\_time\\_seconds\\_sum}}}}{{\\Delta \\text{{request\\_prefill\\_time\\_seconds\\_count}}}} \\times 1000\\text{{ (ms)}}$$

$$\\text{{Avg Prefill Time}} = \\frac{{{delta_sum:.6f}}}{{{delta_count:.0f}}} \\times 1000 = \\mathbf{{{avg_prefill_val:.3f}\\text{{ ms}}}}$$
"""
        avg_prefill_table_val = f"**{avg_prefill_val:.3f} ms**"
    else:
        calc_formula_md = """### 2. 평균 Prefill 시간 계산

> [!WARNING]
> Prometheus 메트릭(`request_prefill_time_seconds`)의 변화량을 감지하지 못하여 순수 Prefill 시간을 계산할 수 없습니다.
"""
        avg_prefill_table_val = "N/A (메트릭 수집 불가)"

    md_content = f"""# Qwen3-0.6B NPU Prefill 벤치마크 결과

vLLM의 Prometheus 메트릭(`vllm:request_prefill_time_seconds_sum`, `count`)을 이용하여 클라이언트 측정(TTFT)의 오버헤드를 배제하고 순수 서버 측 prefill 지연시간을 측정한 결과입니다.

## 공통 설정

- 엔드포인트: `{summary.get('endpoint', 'v1/completions')}`
- 모델: `{summary.get('model', '/home/juwon/Qwen3-0.6B-fp16-dynamic')}`
- 테스트 요청 수: {summary.get('num_requests', 100)}건 (성공 {summary.get('successful_requests', 0)}건, 실패 {summary.get('failed_requests', 0)}건)
- 입력 토큰: {summary.get('input_tokens', 1024):,} tokens
- 출력 토큰: {summary.get('output_tokens', 32):,} tokens
- 동시성: {summary.get('concurrency', 1)}

## 하드웨어 및 소프트웨어 환경

### 하드웨어 환경 (Rebellions ATOM - RBLN-CA22)

| 항목 | 정보 | 비고 |
|---|---|---|
| **NPU 모델** | RBLN-CA22 (ATOM) | 디바이스 ID: `rbln0` (NPU 0) |
| **PCI BUS ID** | `0000:24:00.0` | PCIe Gen5 호환 |
| **NPU 메모리** | 15.7 GiB (16 GiB GDDR6) | 대역폭 288 GB/s |
| **FP16 연산 성능** | 64 TFLOPS | FP16 / BF16 연산 |
| **INT8 연산 성능** | 128 TOPS | 8-bit 정밀도 연산 |
| **소비 전력 (TDP)** | 약 60 ~ 75 W | 초저전력 설계 |
| **KMD 드라이버 버전** | 3.0.0 | Rebellions Kernel Module Driver |

### 소프트웨어 환경

| 구분 | 패키지 | 버전 | 비고 |
|---|---|---|---|
| **RBLN SDK** | `rebel-compiler` | `0.10.1` | Rebellions NPU 컴파일러 |
| | `optimum-rbln` | `0.10.1` | Hugging Face Optimum 연동 |
| | `torch-rbln` | `0.1.7` | PyTorch RBLN 백엔드 |
| | `vllm_rbln` | `0.10.1.post2` | vLLM RBLN 백엔드 |
| **vLLM** | `vllm` | `0.13.0` | 고성능 LLM 서빙 프레임워크 |
| **PyTorch** | `torch` | `2.9.0+cpu` | PyTorch 호스트 백엔드 (Python 3.12.3) |
| | `torchvision` | `0.24.0+cpu` | Vision 모듈 |
| | `torchaudio` | `2.9.0+cpu` | Audio 모듈 |

> 시스템 드라이버 및 도구(`rebellions-dkms`, `rebellions-tool`, `librbln`): 버전 `3.0.0`

---

## Prefill 메트릭 도출 과정

TTFT(Time To First Token)에는 HTTP 네트워크 왕복 시간(RTT), 웹 서버 요청 파싱, 스케줄링 큐 대기 시간 등이 포함됩니다.  
반면 vLLM 내부 Prometheus 메트릭은 NPU 가속기에서 수행한 순수 prefill(Prompt processing) 시간만을 엔진 레벨에서 집계하므로 신뢰도 높은 실제 연산 지연시간을 제공합니다.

### 1. Prometheus 메트릭 전/후 스냅샷

| 시점 | `request_prefill_time_seconds_sum` | `request_prefill_time_seconds_count` | 비고 |
|---|---:|---:|---|
| **벤치마크 실행 전 (Pre)** | {metric_proc.get('pre_sum', 0.0):.6f} s | {metric_proc.get('pre_count', 0.0):.0f} | 워밍업 이후 기준선 |
| **벤치마크 실행 후 (Post)** | {metric_proc.get('post_sum', 0.0):.6f} s | {metric_proc.get('post_count', 0.0):.0f} | {summary.get('num_requests', 100)}건 요청 완료 시점 |
| **변화량 (Delta, $\\Delta$)** | **{delta_sum:.6f} s** | **{delta_count:.0f} 회** | 테스트 구간 누적 prefill |

{calc_formula_md}
---

## 결과 비교: TTFT vs 순수 Prefill 지연시간

| 지표 | 지연시간 (ms) | 설명 |
|---|---:|---|
| **순수 Prefill 시간 (Prometheus 메트릭)** | {avg_prefill_table_val} | **NPU 가속기 엔진 내부 순수 Prompt Processing 시간** |
| **클라이언트 측정 TTFT (평균)** | {summary.get('ttft_mean_ms', 0.0):.2f} ms | HTTP 전송 + 스케줄링 대기 + Prefill 합산 |
| **클라이언트 측정 TTFT (중앙값 / P50)** | {summary.get('ttft_p50_ms', 0.0):.2f} ms | 중간값 기준 TTFT |
| **클라이언트 측정 TTFT (P95)** | {summary.get('ttft_p95_ms', 0.0):.2f} ms | 95분위수 TTFT |
| **클라이언트 측정 TTFT (P99)** | {summary.get('ttft_p99_ms', 0.0):.2f} ms | 99분위수 TTFT |
| **토큰 간 생성 시간 (ITL 평균)** | {summary.get('itl_mean_ms', 0.0):.2f} ms | Decode 단계의 평균 1토큰 생성 시간 |
| **요청당 전체 지연시간 (E2E 평균)** | {summary.get('latency_mean_s', 0.0) * 1000:.2f} ms | 요청 완료까지의 평균 시간 |

### 처리량 (Throughput)

| 항목 | 수치 |
|---|---:|
| **실행 시간** | {summary.get('wall_time_sec', 0.0):.2f} s |
| **요청 처리량 (Req Throughput)** | {summary.get('req_throughput_rps', 0.0):.2f} req/s |
| **출력 토큰 처리량 (Output Throughput)** | {summary.get('out_throughput_tok_s', 0.0):.2f} tok/s |
| **전체 토큰 처리량 (Total Throughput)** | {summary.get('total_throughput_tok_s', 0.0):.2f} tok/s |
"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_content.strip() + "\n")
    print(f"\n[+] Benchmark summary recorded at: {output_path}")


# ---------------------------------------------------------------------------
# 벤치마크 실행
# ---------------------------------------------------------------------------

async def run_benchmark(
    config: Dict[str, Any],
    api_url: str,
    metrics_url: str,
    model: str,
    tokenizer: Any,
    output_md: Optional[str] = None,
) -> Dict[str, Any]:
    print(f"\n{'=' * 65}")
    print(f" [*] Running Case: [PREFILL - NPU] - {config['description']}")
    print(f"     Input tokens : {config['input_tokens']}")
    print(f"     Output tokens: {config['output_tokens']}")
    print(f"     Concurrency  : {config['concurrency']}")
    print(f"     Total reqs   : {config['num_requests']}")
    print(f"{'=' * 65}")

    prompt = generate_prompt_for_tokens(tokenizer, config["input_tokens"])

    # 워밍업
    print(" -> Warming up server with 1 request...")
    timeout = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        await send_streaming_request(
            session=session, url=api_url, model=model,
            prompt=prompt, output_tokens=16, request_id=-1,
        )

    # ── 벤치마크 전 메트릭 스냅샷 ──────────────────────────────
    print("\n[Step 1] 벤치마크 실행 전 Prometheus 메트릭 수집 (Before)...")
    pre_sum, pre_count = await fetch_prefill_snapshot(metrics_url)
    if pre_sum is not None:
        print(f"  - vllm:request_prefill_time_seconds_sum   : {pre_sum:.6f} s")
        print(f"  - vllm:request_prefill_time_seconds_count : {pre_count:.0f}")
    else:
        print("  [!] 메트릭 수집 실패 — Prometheus 엔드포인트를 확인하세요.")

    # ── 벤치마크 실행 ──────────────────────────────────────────
    print(f"\n[Step 2] Executing {config['num_requests']} benchmark requests...")
    semaphore = asyncio.Semaphore(config["concurrency"])
    results: List[Dict[str, Any]] = []

    bench_start = time.perf_counter()

    async def worker(req_id: int):
        async with semaphore:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                res = await send_streaming_request(
                    session=session, url=api_url, model=model,
                    prompt=prompt, output_tokens=config["output_tokens"],
                    request_id=req_id,
                )
                results.append(res)
                if (req_id + 1) % 20 == 0 or (req_id + 1) == config["num_requests"]:
                    print(f"    Progress: {req_id + 1}/{config['num_requests']} requests done")

    await asyncio.gather(*[worker(i) for i in range(config["num_requests"])])
    bench_end = time.perf_counter()
    wall_time = bench_end - bench_start

    # 서버의 Prometheus 메트릭 비동기 flush 대기
    await asyncio.sleep(0.5)

    # ── 벤치마크 후 메트릭 스냅샷 ──────────────────────────────
    print("\n[Step 3] 벤치마크 실행 후 Prometheus 메트릭 수집 (After)...")
    post_sum, post_count = await fetch_prefill_snapshot(metrics_url)
    if post_sum is not None:
        print(f"  - vllm:request_prefill_time_seconds_sum   : {post_sum:.6f} s")
        print(f"  - vllm:request_prefill_time_seconds_count : {post_count:.0f}")

    # ── 평균 prefill 시간 계산 과정 ───────────────────────────
    avg_prefill_ms: Optional[float] = None
    delta_sum = 0.0
    delta_count = 0.0

    print(f"\n{'=' * 65}")
    print(" [Step 4] Avg Prefill Metric 도출 및 계산 과정")
    print(f"{'=' * 65}")

    if (
        pre_sum is not None and post_sum is not None
        and pre_count is not None and post_count is not None
    ):
        delta_sum = post_sum - pre_sum
        delta_count = post_count - pre_count
        print(f"  1) Delta Sum   (post - pre)   : {post_sum:.6f} - {pre_sum:.6f} = {delta_sum:.6f} s")
        print(f"  2) Delta Count (post - pre)   : {post_count:.0f} - {pre_count:.0f} = {delta_count:.0f} requests")

        if delta_count > 0:
            avg_prefill_ms = (delta_sum / delta_count) * 1000.0
            print(f"  3) Calculation Formula        : (Delta Sum / Delta Count) * 1000")
            print(f"                                : ({delta_sum:.6f} / {delta_count:.0f}) * 1000")
            print(f"  >>> Avg Prefill Time (Metric) : {avg_prefill_ms:.3f} ms")
        else:
            print("  [!] delta_count <= 0 — 메트릭 변화량이 없습니다.")
    else:
        print("  [!] Prometheus 메트릭 수집 실패로 계산 불가")

    # ── TTFT 기반 지표 계산 ────────────────────────────────────
    successful = [r for r in results if r["success"]]
    failed = len(results) - len(successful)

    if not successful:
        print(" [!] All requests failed.")
        return {"case": "prefill", "success": False, "failed": failed}

    latencies = [r["total_latency"] for r in successful]
    ttfts     = [r["ttft"] for r in successful]
    itls      = [r["avg_itl"] for r in successful if r["avg_itl"] > 0]
    total_out = sum(r["generated_tokens"] for r in successful)
    total_in  = config["input_tokens"] * len(successful)

    def pct(data, p):
        if not data:
            return 0.0
        s = sorted(data)
        return s[min(int(len(s) * p / 100), len(s) - 1)]

    req_tp   = len(successful) / wall_time
    out_tp   = total_out / wall_time
    total_tp = (total_in + total_out) / wall_time

    summary = {
        "case": "prefill",
        "endpoint": api_url,
        "model": model,
        "input_tokens": config["input_tokens"],
        "output_tokens": config["output_tokens"],
        "concurrency": config["concurrency"],
        "num_requests": config["num_requests"],
        "successful_requests": len(successful),
        "failed_requests": failed,
        "wall_time_sec": wall_time,
        "req_throughput_rps": req_tp,
        "out_throughput_tok_s": out_tp,
        "total_throughput_tok_s": total_tp,
        "ttft_mean_ms":  statistics.mean(ttfts) * 1000,
        "ttft_p50_ms":   pct(ttfts, 50) * 1000,
        "ttft_p95_ms":   pct(ttfts, 95) * 1000,
        "ttft_p99_ms":   pct(ttfts, 99) * 1000,
        "itl_mean_ms":   statistics.mean(itls) * 1000 if itls else 0.0,
        "latency_mean_s": statistics.mean(latencies),
        "latency_p50_s":  pct(latencies, 50),
        "latency_p95_s":  pct(latencies, 95),
        "avg_prefill_metric_ms": avg_prefill_ms,
        "metric_process": {
            "pre_sum": pre_sum or 0.0,
            "pre_count": pre_count or 0.0,
            "post_sum": post_sum or 0.0,
            "post_count": post_count or 0.0,
            "delta_sum": delta_sum,
            "delta_count": delta_count,
        }
    }

    print(f"\n{'=' * 65}")
    print(" [Step 5] 종합 비교 요약 (Console Echo)")
    print(f"{'=' * 65}")
    print(f"  Wall Time              : {wall_time:.2f} s")
    print(f"  Req Throughput         : {req_tp:.2f} req/s")
    print(f"  Output Throughput      : {out_tp:.2f} tok/s")
    print(f"  Total Throughput       : {total_tp:.2f} tok/s")
    print("  -------------------------------------------------------------")
    if avg_prefill_ms is not None:
        print(f"  * Avg Prefill [Metric] : {avg_prefill_ms:.3f} ms   <-- 순수 NPU prefill 연산 시간")
    print(f"  * TTFT Mean   [Client] : {summary['ttft_mean_ms']:.2f} ms   <-- 네트워크/큐잉 포함")
    print(f"  * TTFT P50    [Client] : {summary['ttft_p50_ms']:.2f} ms")
    print(f"  * TTFT P95    [Client] : {summary['ttft_p95_ms']:.2f} ms")
    print(f"  * ITL Mean    [Client] : {summary['itl_mean_ms']:.2f} ms")
    print(f"  * Latency Mean[Client] : {summary['latency_mean_s']:.3f} s")
    print(f"{'=' * 65}\n")

    if output_md:
        write_markdown_summary(summary, output_md)

    return summary


async def get_server_models(base_url: str) -> List[str]:
    """vLLM 서버의 /v1/models 엔드포인트를 호출하여 현재 서빙 중인 모델 id 목록을 반환합니다."""
    models_url = f"{base_url.rstrip('/')}/v1/models"
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
            async with session.get(models_url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return [m["id"] for m in data.get("data", []) if "id" in m]
    except Exception as e:
        print(f"  [!] 서버 모델 목록 조회 실패 ({models_url}): {e}")
    return []


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(description="vLLM Prefill Benchmark for Qwen3-0.6B on NPU")
    parser.add_argument("--url",         default="http://localhost:8000/v1/completions",
                        help="vLLM completions endpoint")
    parser.add_argument("--metrics-url", default="http://localhost:8000/metrics",
                        help="vLLM Prometheus metrics endpoint")
    parser.add_argument("--model",       default=None,
                        help="Model path or name (None = auto-detect from server)")
    parser.add_argument("--requests",    type=int, default=None,
                        help="Override number of requests (default: 100)")
    parser.add_argument("--output-md",   default=None,
                        help="Path to save Markdown summary result (e.g. benchmark_summary.md)")
    parser.add_argument("--output-json", default=None,
                        help="Optional: Path to save JSON results")
    args = parser.parse_args()

    # 서버 base_url 추출 (e.g., http://localhost:8000)
    base_url = args.url.split("/v1/")[0] if "/v1/" in args.url else "http://localhost:8000"
    server_models = await get_server_models(base_url)

    target_model = args.model
    if server_models:
        print(f"[*] Server served models: {server_models}")
        if not target_model:
            target_model = server_models[0]
            print(f"[*] Auto-selected model: {target_model}")
        elif target_model not in server_models:
            # 혹시 슬래시 차이 등이 있는지 확인
            matched = [m for m in server_models if m.rstrip("/") == target_model.rstrip("/")]
            if matched:
                target_model = matched[0]
                print(f"[*] Matched model from server: {target_model}")
            else:
                print(f"  [!] Warning: '{target_model}' not in server models {server_models}. Using server default '{server_models[0]}'")
                target_model = server_models[0]
    else:
        target_model = target_model or "/home/juwon/Qwen3-0.6B-fp16-dynamic"

    print(f"[*] Initializing Tokenizer for: {target_model}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(target_model, trust_remote_code=True)
    except Exception as e:
        print(f"  [!] Failed to load tokenizer from '{target_model}': {e}")
        print("  [*] Falling back to 'Qwen/Qwen3-0.6B' tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)

    cfg = dict(BENCHMARK_CASE)
    if args.requests is not None:
        cfg["num_requests"] = args.requests

    summary = await run_benchmark(
        config=cfg,
        api_url=args.url,
        metrics_url=args.metrics_url,
        model=target_model,
        tokenizer=tokenizer,
        output_md=args.output_md,
    )

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"[*] Raw JSON results saved to: {args.output_json}")


if __name__ == "__main__":
    asyncio.run(main())

