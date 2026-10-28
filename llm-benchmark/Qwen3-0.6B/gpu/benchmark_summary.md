# Qwen3-0.6B GPU Prefill 벤치마크 결과

vLLM의 Prometheus 메트릭(`vllm:request_prefill_time_seconds_sum`, `count`)을 이용하여 클라이언트 측정(TTFT)의 오버헤드를 배제하고 순수 서버 측 prefill 지연시간을 측정한 결과입니다.

## 공통 설정

- 엔드포인트: `http://localhost:8001/v1/completions`
- 모델: `Qwen/Qwen3-0.6B`
- 테스트 요청 수: 100건 (성공 100건, 실패 0건)
- 입력 토큰: 1,024 tokens
- 출력 토큰: 32 tokens
- 동시성: 1

## 하드웨어 및 소프트웨어 환경

### 하드웨어 환경 (NVIDIA RTX A5000)

| 항목 | 정보 | 비고 |
|---|---|---|
| **GPU 모델** | NVIDIA RTX A5000 | 디바이스 ID: `GPU 0` |
| **GPU 메모리** | 24,564 MiB (24 GiB) | GDDR6 (384-bit, 대역폭 768 GB/s) |
| **Driver / CUDA 버전** | 595.84 / 12.8 | PCIe Gen4 x16 |
| **FP16 연산 성능** | 111.1 TFLOPS (Dense) / 222.2 TFLOPS (Sparse) | Tensor Core FP16 연산 |
| **INT8 연산 성능** | 222.2 TOPS (Dense) / 444.4 TOPS (Sparse) | Tensor Core INT8 연산 |
| **소비 전력 (TDP)** | 230 W | PCIe Max Power |

### 소프트웨어 환경

| 구분 | 패키지 | 버전 | 비고 |
|---|---|---|---|
| **vLLM** | `vllm` | `0.13.0` | 고성능 LLM 서빙 프레임워크 |
| **PyTorch** | `torch` | `2.9.0+cu128` | PyTorch GPU 백엔드 (Python 3.12.3) |

---

## Prefill 메트릭 도출 과정

TTFT(Time To First Token)에는 HTTP 네트워크 왕복 시간(RTT), 웹 서버 파싱, 요청 스케줄링 큐 대기 시간 등이 포함됩니다.  
반면 vLLM 내부 Prometheus 메트릭은 모델 실행 엔진이 실제로 수행한 순수 prefill(Prompt processing) 시간만을 집계하므로 정확한 연산 시간을 제공합니다.

### 1. Prometheus 메트릭 전/후 스냅샷

| 시점 | `request_prefill_time_seconds_sum` | `request_prefill_time_seconds_count` | 비고 |
|---|---:|---:|---|
| **벤치마크 실행 전 (Pre)** | 0.477312 s | 102 | 워밍업 이후 기준선 |
| **벤치마크 실행 후 (Post)** | 0.892326 s | 202 | 100건 요청 완료 시점 |
| **변화량 (Delta, $\Delta$)** | **0.415014 s** | **100 회** | 테스트 구간 누적 prefill |

### 2. 평균 Prefill 시간 계산

$$\text{Avg Prefill Time} = \frac{\Delta \text{request\_prefill\_time\_seconds\_sum}}{\Delta \text{request\_prefill\_time\_seconds\_count}} \times 1000\text{ (ms)}$$

$$\text{Avg Prefill Time} = \frac{0.415014}{100} \times 1000 = \mathbf{4.150\text{ ms}}$$

---

## 결과 비교: TTFT vs 순수 Prefill 지연시간

| 지표 | 지연시간 (ms) | 설명 |
|---|---:|---|
| **순수 Prefill 시간 (Prometheus 메트릭)** | **4.150 ms** | **vLLM 엔진 내부 순수 Prompt Processing 시간** |
| **클라이언트 측정 TTFT (평균)** | 9.21 ms | HTTP 전송 + 스케줄링 대기 + Prefill 합산 |
| **클라이언트 측정 TTFT (중앙값 / P50)** | 9.18 ms | 중간값 기준 TTFT |
| **클라이언트 측정 TTFT (P95)** | 10.04 ms | 95분위수 TTFT |
| **클라이언트 측정 TTFT (P99)** | 10.30 ms | 99분위수 TTFT |
| **토큰 간 생성 시간 (ITL 평균)** | 3.15 ms | Decode 단계의 평균 1토큰 생성 시간 |
| **요청당 전체 지연시간 (E2E 평균)** | 107.05 ms | 요청 완료까지의 평균 시간 |

### 처리량 (Throughput)

| 항목 | 수치 |
|---|---:|
| **실행 시간** | 10.72 s |
| **요청 처리량 (Req Throughput)** | 9.33 req/s |
| **출력 토큰 처리량 (Output Throughput)** | 298.47 tok/s |
| **전체 토큰 처리량 (Total Throughput)** | 9849.55 tok/s |
