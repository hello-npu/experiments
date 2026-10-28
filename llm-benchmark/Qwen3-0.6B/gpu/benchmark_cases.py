#!/usr/bin/env python3
"""
vLLM Benchmark Script for Qwen3-0.6B (fp16)
Cases:
- short:   input 128 tokens,  output 128 tokens, concurrency 1, requests 100
- prefill: input 1024 tokens, output 32 tokens,  concurrency 1, requests 100
- decode:  input 128 tokens,  output 512 tokens, concurrency 1, requests 100
- mixed:   input 1024 tokens, output 256 tokens, concurrency 1, requests 100
"""

import argparse
import asyncio
import json
import statistics
import time
from typing import List, Dict, Any
import aiohttp
from transformers import AutoTokenizer

BENCHMARK_CASES = {
    "short": {
        "description": "Short Prompt / Short Gen",
        "input_tokens": 128,
        "output_tokens": 128,
        "concurrency": 1,
        "num_requests": 100,
    },
    "prefill": {
        "description": "Prefill Heavy",
        "input_tokens": 1024,
        "output_tokens": 32,
        "concurrency": 1,
        "num_requests": 100,
    },
    "decode": {
        "description": "Decode Heavy",
        "input_tokens": 128,
        "output_tokens": 512,
        "concurrency": 1,
        "num_requests": 100,
    },
    "mixed": {
        "description": "Balanced / Mixed",
        "input_tokens": 1024,
        "output_tokens": 256,
        "concurrency": 1,
        "num_requests": 100,
    },
}

def generate_prompt_for_tokens(tokenizer: Any, target_tokens: int) -> str:
    """Generate exact token-length prompt using repetitive natural text."""
    seed_phrase = "Artificial intelligence and machine learning technologies are rapidly transforming our modern digital world. "
    seed_ids = tokenizer.encode(seed_phrase, add_special_tokens=False)
    
    repeated_ids = (seed_ids * ((target_tokens // len(seed_ids)) + 2))[:target_tokens]
    prompt = tokenizer.decode(repeated_ids)
    
    # Verify exact length
    actual_len = len(tokenizer.encode(prompt, add_special_tokens=False))
    if actual_len != target_tokens:
        # fine-tune
        prompt = tokenizer.decode(tokenizer.encode(prompt, add_special_tokens=False)[:target_tokens])
    return prompt


async def send_streaming_request(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    prompt: str,
    output_tokens: int,
    request_id: int,
    ignore_eos: bool = True
) -> Dict[str, Any]:
    """Send streaming request to measure TTFT, ITL and latency."""
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
    error = None

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
        error = str(e)
        return {
            "request_id": request_id,
            "success": False,
            "error": error,
        }

    end_time = time.perf_counter()
    total_latency = end_time - start_time
    ttft = (first_token_time - start_time) if first_token_time is not None else total_latency

    # Inter-token latency (excluding TTFT)
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


async def run_case(
    case_name: str,
    config: Dict[str, Any],
    api_url: str,
    model: str,
    tokenizer: Any
) -> Dict[str, Any]:
    print(f"\n=======================================================")
    print(f" [*] Running Case: [{case_name.upper()}] - {config['description']}")
    print(f"     Input tokens : {config['input_tokens']}")
    print(f"     Output tokens: {config['output_tokens']}")
    print(f"     Concurrency  : {config['concurrency']}")
    print(f"     Total reqs   : {config['num_requests']}")
    print(f"=======================================================")

    prompt = generate_prompt_for_tokens(tokenizer, config["input_tokens"])
    
    # Warmup request
    print(" -> Warming up server with 1 request...")
    timeout = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        await send_streaming_request(
            session=session,
            url=api_url,
            model=model,
            prompt=prompt,
            output_tokens=16,
            request_id=-1,
        )

    # Benchmark loop
    print(f" -> Executing {config['num_requests']} benchmark requests...")
    semaphore = asyncio.Semaphore(config["concurrency"])
    results: List[Dict[str, Any]] = []

    bench_start = time.perf_counter()

    async def worker(req_id: int):
        async with semaphore:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                res = await send_streaming_request(
                    session=session,
                    url=api_url,
                    model=model,
                    prompt=prompt,
                    output_tokens=config["output_tokens"],
                    request_id=req_id,
                )
                results.append(res)
                if (req_id + 1) % 20 == 0 or (req_id + 1) == config["num_requests"]:
                    print(f"    Progress: {req_id + 1}/{config['num_requests']} requests done")

    tasks = [worker(i) for i in range(config["num_requests"])]
    await asyncio.gather(*tasks)

    bench_end = time.perf_counter()
    wall_time = bench_end - bench_start

    # Metrics calculation
    successful = [r for r in results if r["success"]]
    failed = len(results) - len(successful)

    if not successful:
        print(f" [!] All requests failed for {case_name}")
        return {"case": case_name, "success": False, "failed": failed}

    latencies = [r["total_latency"] for r in successful]
    ttfts = [r["ttft"] for r in successful]
    itls = [r["avg_itl"] for r in successful if r["avg_itl"] > 0]
    total_out_tokens = sum(r["generated_tokens"] for r in successful)
    total_in_tokens = config["input_tokens"] * len(successful)

    req_throughput = len(successful) / wall_time
    in_token_throughput = total_in_tokens / wall_time
    out_token_throughput = total_out_tokens / wall_time
    total_token_throughput = (total_in_tokens + total_out_tokens) / wall_time

    def pct(data, p):
        if not data:
            return 0.0
        s = sorted(data)
        idx = int(len(s) * (p / 100.0))
        return s[min(idx, len(s) - 1)]

    summary = {
        "case": case_name,
        "input_tokens": config["input_tokens"],
        "output_tokens": config["output_tokens"],
        "concurrency": config["concurrency"],
        "num_requests": config["num_requests"],
        "successful_requests": len(successful),
        "failed_requests": failed,
        "wall_time_sec": wall_time,
        "req_throughput_rps": req_throughput,
        "out_throughput_tok_s": out_token_throughput,
        "total_throughput_tok_s": total_token_throughput,
        "ttft_mean_ms": statistics.mean(ttfts) * 1000,
        "ttft_p50_ms": pct(ttfts, 50) * 1000,
        "ttft_p95_ms": pct(ttfts, 95) * 1000,
        "ttft_p99_ms": pct(ttfts, 99) * 1000,
        "itl_mean_ms": statistics.mean(itls) * 1000 if itls else 0.0,
        "latency_mean_s": statistics.mean(latencies),
        "latency_p50_s": pct(latencies, 50),
        "latency_p95_s": pct(latencies, 95),
    }

    print(f"\n --- [{case_name}] Results ---")
    print(f"  Wall Time         : {wall_time:.2f} s")
    print(f"  Req Throughput    : {req_throughput:.2f} req/s")
    print(f"  Output Throughput : {out_token_throughput:.2f} tok/s")
    print(f"  Total Throughput  : {total_token_throughput:.2f} tok/s")
    print(f"  TTFT Mean (ms)    : {summary['ttft_mean_ms']:.2f} ms (P95: {summary['ttft_p95_ms']:.2f} ms)")
    print(f"  ITL Mean (ms)     : {summary['itl_mean_ms']:.2f} ms")
    print(f"  Latency Mean (s)  : {summary['latency_mean_s']:.3f} s (P95: {summary['latency_p95_s']:.3f} s)")

    return summary


def print_overall_table(all_summaries: List[Dict[str, Any]]):
    print("\n" + "=" * 90)
    print("                     BENCHMARK SUMMARY (Qwen3-0.6B fp16)")
    print("=" * 90)
    header = f"{'Case':<9} | {'In Tok':<6} | {'Out Tok':<7} | {'Req/s':<7} | {'Out tok/s':<10} | {'TTFT(ms)':<9} | {'ITL(ms)':<8} | {'Lat(s)':<7}"
    print(header)
    print("-" * 90)
    for s in all_summaries:
        if not s.get("successful_requests"):
            continue
        line = (
            f"{s['case']:<9} | "
            f"{s['input_tokens']:<6} | "
            f"{s['output_tokens']:<7} | "
            f"{s['req_throughput_rps']:<7.2f} | "
            f"{s['out_throughput_tok_s']:<10.2f} | "
            f"{s['ttft_mean_ms']:<9.2f} | "
            f"{s['itl_mean_ms']:<8.2f} | "
            f"{s['latency_mean_s']:<7.3f}"
        )
        print(line)
    print("=" * 90 + "\n")


async def main():
    parser = argparse.ArgumentParser(description="vLLM Benchmark for Qwen3-0.6B")
    parser.add_argument("--url", default="http://localhost:8000/v1/completions", help="vLLM completions endpoint")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="Model name")
    parser.add_argument("--case", choices=["all", "short", "prefill", "decode", "mixed"], default="all", help="Target benchmark case")
    parser.add_argument("--requests", type=int, default=None, help="Override number of requests")
    parser.add_argument("--output-json", default="benchmark_results.json", help="Path to save JSON results")
    args = parser.parse_args()

    print(f"[*] Initializing Tokenizer for: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    target_cases = list(BENCHMARK_CASES.keys()) if args.case == "all" else [args.case]
    
    all_summaries = []
    for c in target_cases:
        cfg = dict(BENCHMARK_CASES[c])
        if args.requests is not None:
            cfg["num_requests"] = args.requests
        res = await run_case(c, cfg, args.url, args.model, tokenizer)
        all_summaries.append(res)

    print_overall_table(all_summaries)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)
    print(f"[*] Results saved to: {args.output_json}")


if __name__ == "__main__":
    asyncio.run(main())
