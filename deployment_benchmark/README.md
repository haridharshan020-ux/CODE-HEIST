# Rural Deployment Benchmark Module

**SIH 2026 — Objective 10: Rural Deployment Feasibility**

Python-based deployment benchmarking and simulation for the DR Screening System.
MATLAB/Simulink was not used because it is currently not accessible.

---

## Module Structure

```
deployment_benchmark/
├── __init__.py
├── benchmark.py            # Main orchestrator — runs all scenarios
├── scenarios.py            # Scenario A/B/C definitions and CPU affinity helpers
├── metrics.py              # ImageMetrics and ScenarioResult data classes
├── system_monitor.py       # Background CPU/RAM poller (psutil)
├── offline_test.py         # Standalone offline validation (checkpoint + inference)
├── connectivity_simulation.py  # Socket-level network blocking utility
├── report.py               # Text report + matplotlib chart generator
├── results/                # Output directory (created automatically)
│   ├── charts/             # PNG charts
│   ├── benchmark_raw_<ts>.json
│   ├── benchmark_report_<ts>.txt
│   └── offline_test_result.json
└── README.md               # This file

pages/
└── Rural_Deployment_Simulation.py   # Streamlit page (separate from app.py)
```

---

## Benchmark Scenarios

| Scenario | Description | Constraint |
|----------|-------------|------------|
| **A — Normal Local** | Baseline on current dev machine | None |
| **B — Simulated Low-Resource** | 1-core CPU affinity | `psutil.cpu_affinity([0])` |
| **C — Poor-Connectivity Sim** | Network blocked during inference | `socket.setdefaulttimeout(0.001)` |

> **IMPORTANT DISCLAIMER**
>
> Scenario B and C results are **SOFTWARE SIMULATIONS** on the current development machine.
> They are **NOT** measurements on actual rural or low-resource hardware.
> All results are labelled "Prototype Benchmark Results" accordingly.

---

## Prerequisites

```powershell
C:\Python314\python.exe -m pip install psutil
```

All other dependencies (torch, PIL, streamlit, matplotlib, pandas) are already installed.

---

## How to Run

### Option 1: Streamlit Page (recommended)

```powershell
cd C:\Users\hariharan\.gemini\antigravity\scratch\diabetic-retinopathy-sih2026
C:\Python314\python.exe -m streamlit run app.py
```

Then navigate to **Rural Deployment Simulation** in the Streamlit sidebar.

### Option 2: CLI Benchmark (all scenarios)

```powershell
cd C:\Users\hariharan\.gemini\antigravity\scratch\diabetic-retinopathy-sih2026
C:\Python314\python.exe -m deployment_benchmark.benchmark --scenario all --report
```

### Option 3: Single scenario

```powershell
C:\Python314\python.exe -m deployment_benchmark.benchmark --scenario A --report
C:\Python314\python.exe -m deployment_benchmark.benchmark --scenario B
C:\Python314\python.exe -m deployment_benchmark.benchmark --scenario C
```

### Option 4: Offline validation only

```powershell
C:\Python314\python.exe deployment_benchmark/offline_test.py
```

---

## Benchmark Images

- **20 fixed images** from `dataset/images/` (see `BENCHMARK_IMAGE_IDS` in `scenarios.py`)
- **5 warmup images** (not measured) — used to warm up the JIT/model cache
- **20 measured images** — timing is recorded for all
- These images are used for **timing only** — no labels are used, no retraining occurs

---

## Methodology

| Aspect | Implementation |
|--------|---------------|
| Timing | `time.perf_counter()` wall clock around `pipeline.process()` |
| CPU%   | `psutil.cpu_percent()` snapshot per image + background thread poll |
| RAM    | `psutil.Process().memory_info().rss` per image |
| Warmup | 5 images run but not included in statistics |
| Pipeline | `DRScreeningPipeline` imported as black box — no model code duplicated |
| Network | `socket.setdefaulttimeout(0.001)` during Scenario C inference |
| Affinity | `psutil.Process().cpu_affinity([0])` for Scenario B |

---

## Output Files

After a benchmark run, results are saved to `deployment_benchmark/results/`:

- `benchmark_raw_<timestamp>.json` — full per-image measurements (all scenarios)
- `benchmark_report_<timestamp>.txt` — human-readable text summary
- `offline_test_result.json` — offline validation pass/fail
- `charts/latency_comparison_<ts>.png`
- `charts/throughput_comparison_<ts>.png`
- `charts/per_image_latency_A_<ts>.png`
- `charts/resource_usage_<ts>.png`

---

## Safety Guarantees

This module makes **zero modifications** to the existing screening system:

- Does NOT modify `app.py`, `pipeline.py`, `gradcam.py`, `referral_engine.py`, or any model file
- Does NOT modify `dataset/test.csv` (frozen test set)
- Does NOT retrain, fine-tune, or overwrite any model checkpoint
- Does NOT duplicate model architecture or preprocessing code
- Imports `DRScreeningPipeline` as a pure black box

The existing Streamlit app continues to work exactly as before.

---

## Known Limitations

1. Scenario B uses CPU affinity, not actual lower-frequency or lower-core hardware — results will differ from a real Raspberry Pi or similar device.
2. Scenario C blocks Python-level TCP sockets; OS-level or hardware-level network isolation is not performed.
3. All timing is on CPU only (no CUDA available in the current environment).
4. First-image latency is higher due to PyTorch operator caching — warmup images mitigate this.

---

## Checkpoint Integrity

The benchmark verifies `best_model_combined_v1.pth` SHA-256 before running:

```
Known SHA-256: 3FA407E505F8653F00BD2CFB997224247E76FAC93C18DDEA7F35A973B0B85D05
```

If the hash mismatches, the benchmark prints a warning. **It does NOT stop or alter the checkpoint.**
