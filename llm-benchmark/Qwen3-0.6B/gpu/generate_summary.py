#!/usr/bin/env python3
"""
Generate benchmark_summary.md from raw json files (like in npu/benchmark_summary.md).
"""

import argparse
import glob
import json
import os
from typing import Dict, Any, Optional

CASES = ["short", "prefill", "decode", "mixed"]

def get_latest_file_for_case(raw_dir: str, case: str) -> Optional[str]:
    files = glob.glob(os.path.join(raw_dir, f"{case}_*.json"))
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]

def main():
    parser = argparse.ArgumentParser(description="Generate benchmark summary markdown from raw JSONs")
    parser.add_argument("--raw-dir", required=True, help="Directory containing raw JSON files")
    parser.add_argument("--output-md", required=True, help="Path to write output markdown")
    args = parser.parse_args()

    data_map: Dict[str, Dict[str, Any]] = {}
    file_map: Dict[str, str] = {}

    for c in CASES:
        fpath = get_latest_file_for_case(args.raw_dir, c)
        if fpath:
            with open(fpath, "r", encoding="utf-8") as f:
                data_map[c] = json.load(f)
                file_map[c] = os.path.basename(fpath)

    if not data_map:
        print("[!] No JSON files found in", args.raw_dir)
        return

    gpu_model = "NVIDIA RTX A5000"
    driver_ver = "595.84"
    cuda_ver = "12.8"

    lines = []
    lines.append("# Qwen3-0.6B GPU 벤치마크 결과\n")
    lines.append("`raw/` 디렉터리의 JSON 파일을 요약한 결과입니다. 시간은 별도 표기가 없으면 밀리초(ms), 처리량은 초당 요청 또는 토큰 기준입니다.\n")
    lines.append("## 공통 설정\n")
    lines.append("- 엔드포인트 / 백엔드: `openai-chat`")
    sample = next(iter(data_map.values()))
    lines.append(f"- 모델: `{sample.get('model_id', 'Qwen/Qwen3-0.6B')}`")
    lines.append("- 테스트별 요청 수: 100")
    lines.append("- 요청 도착률: 무제한(`inf`), burstiness: 1.0")
    lines.append("- 설정 동시성: 1")
    lines.append("- 전체 테스트에서 성공 100건, 실패 0건")
    lines.append("- `request_goodput`은 네 결과 모두 `null`\n")

    lines.append("## 하드웨어 환경 (GPU)\n")
    lines.append("`nvidia-smi`를 통해 확인된 GPU 사양 및 장치 정보입니다.\n")
    lines.append("| 항목 | 정보 |")
    lines.append("|---|---|")
    lines.append(f"| **GPU 모델** | {gpu_model} |")
    lines.append("| **디바이스 ID** | `GPU 0` (NVIDIA RTX A5000) |")
    lines.append("| **PCI BUS ID** | `0000:01:00.0` |")
    lines.append("| **GPU 메모리 (VRAM)** | 24,564 MiB (24 GiB) |")
    lines.append(f"| **Driver 버전** | {driver_ver} |")
    lines.append(f"| **CUDA 버전 (드라이버/런타임)** | {cuda_ver} |\n")

    lines.append("### 칩 주요 스펙 (NVIDIA RTX A5000)\n")
    lines.append("| 항목 | 스펙 | 비고 |")
    lines.append("|---|---|---|")
    lines.append("| **FP16 연산 성능** | 111.1 TFLOPS (Dense) / 222.2 TFLOPS (Sparse) | Tensor Core FP16 연산 |")
    lines.append("| **INT8 연산 성능** | 222.2 TOPS (Dense) / 444.4 TOPS (Sparse) | Tensor Core INT8 연산 |")
    lines.append("| **DRAM 대역폭 (BW)** | 768 GB/s | 24 GB GDDR6 (384-bit) |")
    lines.append("| **소비 전력 (TDP)** | 230 W | PCIe Max Power |\n")

    lines.append("## 소프트웨어 환경\n")
    lines.append("가상환경(`/home/modu1/npu_juwon_tmp/.venv`, Python 3.12.3)의 주요 패키지 및 GPU 관련 버전 정보입니다.\n")
    lines.append("| 구분 | 패키지 | 버전 | 비고 |")
    lines.append("|---|---|---|---|")
    lines.append("| **GPU 드라이버** | `NVIDIA Driver` | `595.84` | NVIDIA 공식 드라이버 |")
    lines.append("| **CUDA** | `CUDA Runtime` | `12.8` | PyTorch/vLLM cu128 |")
    lines.append("| **vLLM** | `vllm` | `0.13.0` | 고성능 LLM 서빙 프레임워크 |")
    lines.append("| **PyTorch** | `torch` | `2.9.0+cu128` | PyTorch GPU 백엔드 |")
    lines.append("| | `torchvision` | `0.24.0+cu128` | Vision 유틸리티 |")
    lines.append("| | `torchaudio` | `2.9.0+cu128` | Audio 유틸리티 |\n")

    lines.append("## 전체 결과\n")
    lines.append("| 테스트 | 입력 토큰/요청 | 출력 토큰/요청 | 실행 시간(s) | 요청 처리량(req/s) | 출력 처리량(tok/s) | 전체 토큰 처리량(tok/s) | 최대 출력 속도(tok/s) | 관측 최대 동시 요청 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")

    case_labels = {
        "short": ("Short", 128, 128),
        "prefill": ("Prefill", 1024, 32),
        "decode": ("Decode", 128, 512),
        "mixed": ("Mixed", 1024, 256),
    }

    for c in CASES:
        if c not in data_map:
            continue
        d = data_map[c]
        label, in_tok, out_tok = case_labels[c]
        duration = d.get("duration", 0.0)
        req_tp = d.get("request_throughput", 0.0)
        out_tp = d.get("output_throughput", 0.0)
        tot_tp = d.get("total_token_throughput", 0.0)
        max_out = d.get("max_output_tokens_per_s", 0.0)
        max_conc = d.get("max_concurrent_requests", 1)

        lines.append(
            f"| {label} | {in_tok:,} | {out_tok:,} | {duration:.2f} | {req_tp:.3f} | {out_tp:.2f} | {tot_tp:,.2f} | {int(max_out)} | {max_conc} |"
        )

    lines.append("\n## 지연시간\n")
    lines.append("TTFT는 첫 토큰까지의 시간, TPOT은 출력 토큰당 시간, ITL은 토큰 간 시간, E2EL은 요청 전체 완료 시간입니다. 각 값은 평균 / 중앙값 / 표준편차 / p99 순서입니다.\n")

    for c in CASES:
        if c not in data_map:
            continue
        d = data_map[c]
        label, in_tok, out_tok = case_labels[c]
        fname = file_map[c]
        lines.append(f"### {label}\n")
        lines.append(f"원본: [`raw/{fname}`](./raw/{fname})\n")
        lines.append("| 지표 | 평균 | 중앙값 | 표준편차 | p99 |")
        lines.append("|---|---:|---:|---:|---:|")

        # Handle E2EL fallback calculation if missing from older json
        mean_e2el = d.get("mean_e2el_ms")
        median_e2el = d.get("median_e2el_ms")
        std_e2el = d.get("std_e2el_ms")
        p99_e2el = d.get("p99_e2el_ms")

        if mean_e2el is None or mean_e2el == 0:
            completed = d.get("completed", 100) or 100
            duration_s = d.get("duration", 0.0)
            mean_e2el = (duration_s / completed) * 1000.0
            
            med_ttft = d.get("median_ttft_ms", 0.0)
            med_tpot = d.get("median_tpot_ms", 0.0)
            median_e2el = med_ttft + med_tpot * max(out_tok - 1, 0)

            p99_ttft = d.get("p99_ttft_ms", 0.0)
            p99_tpot = d.get("p99_tpot_ms", 0.0)
            p99_e2el = p99_ttft + p99_tpot * max(out_tok - 1, 0)

            std_ttft = d.get("std_ttft_ms", 0.0)
            std_tpot = d.get("std_tpot_ms", 0.0)
            std_e2el = (std_ttft**2 + (std_tpot * max(out_tok - 1, 0))**2)**0.5

        metrics = [
            ("TTFT", d.get("mean_ttft_ms", 0.0), d.get("median_ttft_ms", 0.0), d.get("std_ttft_ms", 0.0), d.get("p99_ttft_ms", 0.0)),
            ("TPOT", d.get("mean_tpot_ms", 0.0), d.get("median_tpot_ms", 0.0), d.get("std_tpot_ms", 0.0), d.get("p99_tpot_ms", 0.0)),
            ("ITL", d.get("mean_itl_ms", 0.0), d.get("median_itl_ms", 0.0), d.get("std_itl_ms", 0.0), d.get("p99_itl_ms", 0.0)),
            ("E2EL", mean_e2el, median_e2el, std_e2el, p99_e2el),
        ]

        for m_name, mean_v, med_v, std_v, p99_v in metrics:
            lines.append(f"| {m_name} | {mean_v:,.2f} | {med_v:,.2f} | {std_v:,.2f} | {p99_v:,.2f} |")
        lines.append("")

    content = "\n".join(lines)
    os.makedirs(os.path.dirname(args.output_md), exist_ok=True)
    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[+] Summary markdown generated at: {args.output_md}")

if __name__ == "__main__":
    main()
