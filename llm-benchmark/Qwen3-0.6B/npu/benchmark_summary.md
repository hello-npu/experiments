# Qwen3-0.6B NPU Prefill 벤치마크 결과

vLLM의 Prometheus 메트릭(`vllm:request_prefill_time_seconds_sum`, `count`)을 이용하여 클라이언트 측정(TTFT)의 오버헤드를 배제하고 순수 서버 측 prefill 지연시간을 측정한 결과입니다.

## 공통 설정

- 엔드포인트: `http://localhost:8000/v1/completions`
- 모델: `/home/juwon/Qwen3-0.6B-fp16-dynamic`
- 테스트 요청 수: 100건 (성공 100건, 실패 0건)
- 입력 토큰: 1,024 tokens
- 출력 토큰: 32 tokens
- 동시성: 1

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
| **벤치마크 실행 전 (Pre)** | 0.112585 s | 1 | 워밍업 이후 기준선 |
| **벤치마크 실행 후 (Post)** | 1.959103 s | 101 | 100건 요청 완료 시점 |
| **변화량 (Delta, $\Delta$)** | **1.846518 s** | **100 회** | 테스트 구간 누적 prefill |

### 2. 평균 Prefill 시간 계산

$$\text{Avg Prefill Time} = \frac{\Delta \text{request\_prefill\_time\_seconds\_sum}}{\Delta \text{request\_prefill\_time\_seconds\_count}} \times 1000\text{ (ms)}$$

$$\text{Avg Prefill Time} = \frac{1.846518}{100} \times 1000 = \mathbf{18.465\text{ ms}}$$

---

## 결과 비교: TTFT vs 순수 Prefill 지연시간

| 지표 | 지연시간 (ms) | 설명 |
|---|---:|---|
| **순수 Prefill 시간 (Prometheus 메트릭)** | **18.465 ms** | **NPU 가속기 엔진 내부 순수 Prompt Processing 시간** |
| **클라이언트 측정 TTFT (평균)** | 29.25 ms | HTTP 전송 + 스케줄링 대기 + Prefill 합산 |
| **클라이언트 측정 TTFT (중앙값 / P50)** | 29.28 ms | 중간값 기준 TTFT |
| **클라이언트 측정 TTFT (P95)** | 32.51 ms | 95분위수 TTFT |
| **클라이언트 측정 TTFT (P99)** | 34.86 ms | 99분위수 TTFT |
| **토큰 간 생성 시간 (ITL 평균)** | 8.01 ms | Decode 단계의 평균 1토큰 생성 시간 |
| **요청당 전체 지연시간 (E2E 평균)** | 277.67 ms | 요청 완료까지의 평균 시간 |

### 처리량 (Throughput)

| 항목 | 수치 |
|---|---:|
| **실행 시간** | 27.81 s |
| **요청 처리량 (Req Throughput)** | 3.60 req/s |
| **출력 토큰 처리량 (Output Throughput)** | 115.06 tok/s |
| **전체 토큰 처리량 (Total Throughput)** | 3797.00 tok/s |
