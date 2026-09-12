"""
scenarios.py - Benchmark scenario definitions.

Three scenarios representing different deployment conditions:
  A - Normal local desktop (current machine, baseline)
  B - Simulated low-resource configuration (CPU throttle via process affinity)
  C - Poor-connectivity simulation (network blocked during inference)

All scenario B/C labels clearly state "Simulated" because no actual
rural hardware is being used. Results must be labelled accordingly.
"""

import os
import sys
import time
import psutil
from dataclasses import dataclass
from typing import List


# Fixed benchmark image list — 20 verified images from dataset/images/ (test.csv IDs)
# All confirmed to exist as PNG files. Used for TIMING ONLY — no retraining.
BENCHMARK_IMAGE_IDS = [
    "4f0866b90c27",   # test row 1  (diagnosis 3)
    "fe674c2f73f5",   # test row 2  (diagnosis 1)
    "165634a6167e",   # test row 3  (diagnosis 0)
    "b16dd4483ca5",   # test row 4  (diagnosis 0)
    "1a7e3356b39c",   # test row 5
    "521d3e264d71",   # test row 6
    "9782c0489eca",   # test row 7
    "49a4765f8822",   # test row 8
    "58184d6fd087",   # test row 9
    "5c7ab966a3ee",   # test row 10
    "54bdcdecd8f3",   # test row 11
    "d6e26fe51dce",   # test row 12
    "a4d41c495666",   # test row 13
    "0cb14014117d",   # test row 14
    "6d454444f17c",   # test row 15
    "3ac3fbfca7d4",   # test row 16
    "080f66eedfb9",   # test row 17
    "3461dc601cc2",   # test row 18
    "78b3f819dcc5",   # test row 19
    "d473f6fafba0",   # test row 20
]

WARMUP_COUNT = 5   # images used only for JIT/cache warmup, not measured
MEASURED_COUNT = 20  # images to measure (BENCHMARK_IMAGE_IDS)


@dataclass
class ScenarioConfig:
    name: str
    label: str
    description: str
    hardware_note: str
    affinity_cores: int    # 0 = no restriction (use all); N = restrict to N cores
    inter_image_sleep_sec: float  # additional delay between images (simulated IO)
    block_network: bool    # whether to block outbound sockets during inference


SCENARIO_A = ScenarioConfig(
    name="Scenario_A",
    label="Prototype Benchmark Results - Scenario A (Normal Local)",
    description=(
        "Baseline benchmark on the current development machine using all available CPU cores. "
        "Represents best-case local desktop deployment. No artificial resource restrictions applied."
    ),
    hardware_note=(
        "Measured on current development hardware. "
        "Not representative of actual rural deployment hardware."
    ),
    affinity_cores=0,
    inter_image_sleep_sec=0.0,
    block_network=False,
)

SCENARIO_B = ScenarioConfig(
    name="Scenario_B",
    label="Prototype Benchmark Results - Scenario B (Simulated Low-Resource)",
    description=(
        "Simulated low-resource configuration: CPU affinity restricted to 1 logical core "
        "to emulate constrained rural hardware (e.g. Raspberry Pi class device). "
        "DISCLAIMER: This is a SOFTWARE simulation on the current machine, NOT a measurement "
        "on actual low-resource hardware. Results are indicative only."
    ),
    hardware_note=(
        "SIMULATED low-resource configuration (1-core affinity on current development hardware). "
        "Actual rural device performance may differ significantly."
    ),
    affinity_cores=1,
    inter_image_sleep_sec=0.0,
    block_network=False,
)

SCENARIO_C = ScenarioConfig(
    name="Scenario_C",
    label="Prototype Benchmark Results - Scenario C (Poor-Connectivity Simulation)",
    description=(
        "Poor-connectivity simulation: outbound TCP is blocked via a near-zero socket timeout "
        "during inference to demonstrate that local inference requires no internet connection. "
        "Verifies that model predictions are unchanged when network is unavailable. "
        "DISCLAIMER: This is a software-level connectivity simulation, not a real rural network test."
    ),
    hardware_note=(
        "SIMULATED poor-connectivity (socket timeout=0.001s during inference). "
        "Core local inference proved network-independent."
    ),
    affinity_cores=0,
    inter_image_sleep_sec=0.0,
    block_network=True,
)

ALL_SCENARIOS = [SCENARIO_A, SCENARIO_B, SCENARIO_C]


def apply_cpu_affinity(n_cores: int):
    """
    Restrict process to n_cores logical CPUs via psutil affinity.
    n_cores=0 means restore all CPUs (no restriction).
    Returns the previous affinity list for restoration.
    """
    proc = psutil.Process()
    try:
        current = proc.cpu_affinity()
    except AttributeError:
        # cpu_affinity not supported on this platform
        return None
    if n_cores == 0:
        all_cpus = list(range(psutil.cpu_count(logical=True)))
        proc.cpu_affinity(all_cpus)
        return current
    restricted = list(range(psutil.cpu_count(logical=True)))[:n_cores]
    proc.cpu_affinity(restricted)
    return current


def restore_cpu_affinity(previous):
    """Restore CPU affinity to previously saved value."""
    if previous is None:
        return
    try:
        psutil.Process().cpu_affinity(previous)
    except Exception:
        pass
