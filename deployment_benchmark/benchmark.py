"""
benchmark.py - Main benchmark orchestrator for SIH Objective 10.

Runs three deployment scenarios (A, B, C) using the existing DRScreeningPipeline
as a black box. Records per-image timing, CPU%, and RAM. Generates aggregate stats.

Usage:
    python -m deployment_benchmark.benchmark [--scenario A|B|C|all]

The existing pipeline.py, gradcam.py, and model checkpoint are NEVER modified.
This file ONLY imports and calls the existing pipeline.
"""

import os
import sys
import time
import json
import argparse
import socket
import datetime
import hashlib
import psutil

# Allow running from project root
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from deployment_benchmark.metrics import ImageMetrics, ScenarioResult
from deployment_benchmark.scenarios import (
    SCENARIO_A, SCENARIO_B, SCENARIO_C, ALL_SCENARIOS, ScenarioConfig,
    BENCHMARK_IMAGE_IDS, WARMUP_COUNT, MEASURED_COUNT,
    apply_cpu_affinity, restore_cpu_affinity,
)
from deployment_benchmark.system_monitor import SystemMonitor, snapshot_system

IMAGES_DIR = os.path.join(_PROJECT_ROOT, "dataset", "images")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
CHECKPOINT_PATH = os.path.join(_PROJECT_ROOT, "checkpoints", "best_model_combined_v1.pth")


def get_image_path(img_id: str) -> str:
    return os.path.join(IMAGES_DIR, img_id + ".png")


def _screen_one(pipeline, img_path: str, img_id: str,
                 block_network: bool = False) -> ImageMetrics:
    """
    Run pipeline.process() on one image and record timing + resource usage.
    Returns an ImageMetrics instance.

    pipeline.process() return dict keys (from pipeline.py):
      overall_status, prediction.{class, severity, confidence, probabilities},
      quality.{quality_ok, ...}, processing_time_sec
    """
    original_timeout = None
    if block_network:
        original_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(0.001)

    proc = psutil.Process()
    psutil.cpu_percent(interval=None)  # prime CPU reading (first call is always 0)

    try:
        t_total_start = time.perf_counter()
        result = pipeline.process(img_path, generate_gradcam=False, save_visualizations=False)
        t_total_end = time.perf_counter()

        total_ms = (t_total_end - t_total_start) * 1000.0
        cpu_pct = psutil.cpu_percent(interval=None)
        ram_mb = proc.memory_info().rss / (1024 * 1024)

        if result is None:
            return ImageMetrics(
                image_id=img_id, image_path=img_path,
                preprocessing_ms=0.0, inference_ms=0.0,
                total_pipeline_ms=total_ms, cpu_percent=cpu_pct, ram_used_mb=ram_mb,
                error="None result from pipeline",
            )

        status = result.get("overall_status", "")
        if status == "ERROR":
            return ImageMetrics(
                image_id=img_id, image_path=img_path,
                preprocessing_ms=0.0, inference_ms=0.0,
                total_pipeline_ms=total_ms, cpu_percent=cpu_pct, ram_used_mb=ram_mb,
                error=result.get("message", "Pipeline error"),
            )

        # Extract prediction from nested dict
        pred_dict = result.get("prediction") or {}
        pred_label = pred_dict.get("severity", None)  # e.g. "Moderate NPDR"
        confidence = pred_dict.get("confidence", None)

        # Quality gate passed?
        quality_ok = True
        if status == "RECAPTURE_REQUIRED":
            quality_ok = False

        # pipeline provides processing_time_sec as float
        # We measured wall time ourselves; use pipeline's value for infer_ms fallback
        pipeline_reported_sec = result.get("processing_time_sec", 0.0)
        infer_ms = pipeline_reported_sec * 1000.0  # pipeline total (no sub-timing exposed)

        return ImageMetrics(
            image_id=img_id,
            image_path=img_path,
            preprocessing_ms=0.0,          # pipeline does not expose sub-timing
            inference_ms=infer_ms,
            total_pipeline_ms=total_ms,     # wall-clock measured by benchmark
            cpu_percent=cpu_pct,
            ram_used_mb=ram_mb,
            prediction=pred_label,
            confidence=confidence,
            quality_passed=quality_ok,
        )

    except Exception as exc:
        return ImageMetrics(
            image_id=img_id,
            image_path=img_path,
            preprocessing_ms=0.0,
            inference_ms=0.0,
            total_pipeline_ms=0.0,
            cpu_percent=0.0,
            ram_used_mb=0.0,
            error=str(exc),
        )
    finally:
        if block_network and original_timeout is not None:
            socket.setdefaulttimeout(original_timeout)


def run_scenario(scenario: ScenarioConfig, pipeline, verbose: bool = True) -> ScenarioResult:
    """
    Execute one benchmark scenario. Returns a populated ScenarioResult.
    """
    print("\n" + "=" * 60)
    print("RUNNING: {}".format(scenario.label))
    print("=" * 60)
    print(scenario.description)
    print()

    # Collect valid image paths
    image_paths = []
    for img_id in BENCHMARK_IMAGE_IDS:
        p = get_image_path(img_id)
        if os.path.isfile(p):
            image_paths.append((img_id, p))
        else:
            print("  WARN: Image not found, skipping: {}".format(img_id))

    if len(image_paths) < 10:
        print("  ERROR: Fewer than 10 benchmark images found. Check IMAGES_DIR.")

    # Apply CPU affinity restriction for Scenario B
    saved_affinity = None
    if scenario.affinity_cores > 0:
        print("  Restricting CPU affinity to {} logical core(s) (SIMULATED)...".format(
            scenario.affinity_cores
        ))
        saved_affinity = apply_cpu_affinity(scenario.affinity_cores)

    system_info = snapshot_system()
    result = ScenarioResult(
        scenario_name=scenario.name,
        scenario_label=scenario.label,
        description=scenario.description,
        hardware_note=scenario.hardware_note,
        system_info=system_info,
        scenario_start_ts=datetime.datetime.now().isoformat(),
        warmup_count=0,
    )

    monitor = SystemMonitor(interval_sec=0.5)
    monitor.start()

    try:
        # ── Warmup pass (not measured) ─────────────────────────────
        warmup_images = image_paths[:WARMUP_COUNT]
        print("  Warming up ({} images, not measured)...".format(len(warmup_images)))
        for img_id, img_path in warmup_images:
            _screen_one(pipeline, img_path, img_id, block_network=scenario.block_network)
        result.warmup_count = len(warmup_images)
        print("  Warmup complete.")

        # ── Measured pass ──────────────────────────────────────────
        print("  Measuring {} images...".format(len(image_paths)))
        for i, (img_id, img_path) in enumerate(image_paths):
            m = _screen_one(pipeline, img_path, img_id, block_network=scenario.block_network)
            result.image_metrics.append(m)
            status = "OK" if m.error is None else "ERR"
            net_tag = "[NET-BLOCKED]" if scenario.block_network else ""
            print("  [{:2d}/{}] {} {} {:.0f}ms {}".format(
                i + 1, len(image_paths), img_id, status, m.total_pipeline_ms, net_tag
            ))
            if scenario.inter_image_sleep_sec > 0:
                time.sleep(scenario.inter_image_sleep_sec)

    finally:
        monitor.stop()
        if saved_affinity is not None:
            restore_cpu_affinity(saved_affinity)
            print("  CPU affinity restored.")

    result.scenario_end_ts = datetime.datetime.now().isoformat()
    result.compute_aggregates()

    # Attach monitor summary
    result.system_info["monitor_mean_cpu"] = round(monitor.mean_cpu(), 1)
    result.system_info["monitor_max_cpu"] = round(monitor.max_cpu(), 1)
    result.system_info["monitor_mean_ram_mb"] = round(monitor.mean_ram_mb(), 1)
    result.system_info["monitor_max_ram_mb"] = round(monitor.max_ram_mb(), 1)

    print("\n  --- {} RESULTS ---".format(scenario.name))
    print("  Mean latency  : {:.1f} ms".format(result.mean_total_ms))
    print("  Median latency: {:.1f} ms".format(result.median_total_ms))
    print("  Min / Max     : {:.1f} / {:.1f} ms".format(result.min_total_ms, result.max_total_ms))
    print("  Throughput    : {:.4f} img/sec".format(result.throughput_img_per_sec))
    print("  Mean CPU      : {:.1f}%".format(result.mean_cpu_percent))
    print("  Mean RAM      : {:.1f} MB".format(result.mean_ram_mb))
    print("  NOTE: {}".format(scenario.hardware_note))

    return result


def run_all_scenarios(scenarios=None, verbose: bool = True) -> list:
    """
    Run all specified scenarios sequentially. Returns list of ScenarioResult.
    Imports DRScreeningPipeline once as a black box.
    """
    if scenarios is None:
        scenarios = ALL_SCENARIOS

    print("=" * 60)
    print("DR SCREENING DEPLOYMENT BENCHMARK")
    print("SIH Objective 10 - Rural Deployment Feasibility")
    print("=" * 60)
    print("IMPORTANT: Results are PROTOTYPE BENCHMARK RESULTS.")
    print("Scenarios B and C use SIMULATED constraints on current hardware.")
    print("These do NOT represent actual rural device measurements.")
    print("=" * 60)

    # Verify checkpoint integrity before running
    print("\nVerifying checkpoint integrity...")
    from deployment_benchmark.offline_test import sha256_file, KNOWN_CHECKPOINT_SHA256
    if os.path.isfile(CHECKPOINT_PATH):
        actual = sha256_file(CHECKPOINT_PATH)
        if actual == KNOWN_CHECKPOINT_SHA256:
            print("  Checkpoint SHA-256: MATCH ({}...)".format(actual[:16]))
        else:
            print("  WARNING: Checkpoint SHA-256 MISMATCH!")
            print("  Expected: {}".format(KNOWN_CHECKPOINT_SHA256))
            print("  Actual:   {}".format(actual))
    else:
        print("  ERROR: Checkpoint not found!")
        return []

    # Load pipeline once (black box import)
    print("\nLoading DRScreeningPipeline (black box)...")
    from pipeline import DRScreeningPipeline
    pipeline = DRScreeningPipeline()
    print("  Pipeline ready.")

    all_results = []
    for scenario in scenarios:
        sr = run_scenario(scenario, pipeline, verbose=verbose)
        all_results.append(sr)

    return all_results


def save_results(all_results: list) -> str:
    """
    Save all scenario results to a JSON file in results/.
    Returns the output file path.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(RESULTS_DIR, "benchmark_raw_{}.json".format(ts))
    data = [r.to_dict() for r in all_results]
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print("\nRaw results saved to: {}".format(out_path))
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DR Screening Deployment Benchmark")
    parser.add_argument(
        "--scenario", choices=["A", "B", "C", "all"], default="all",
        help="Which scenario(s) to run (default: all)"
    )
    parser.add_argument("--report", action="store_true", help="Generate report after benchmark")
    args = parser.parse_args()

    scenario_map = {"A": [SCENARIO_A], "B": [SCENARIO_B], "C": [SCENARIO_C], "all": ALL_SCENARIOS}
    chosen = scenario_map[args.scenario]

    results = run_all_scenarios(scenarios=chosen)
    if results:
        raw_path = save_results(results)

        if args.report:
            from deployment_benchmark.report import generate_report
            generate_report(results, raw_path)
