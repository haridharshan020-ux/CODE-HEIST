"""
report.py - Generate text reports and matplotlib charts from benchmark results.

Outputs:
  results/benchmark_report_<datetime>.txt   -- human-readable text report
  results/charts/latency_comparison.png     -- bar chart of mean latency per scenario
  results/charts/throughput_comparison.png  -- bar chart of throughput per scenario
  results/charts/per_image_latency_A.png    -- per-image latency for Scenario A
  results/charts/resource_usage.png         -- CPU/RAM comparison

IMPORTANT: This module does NOT modify any existing pipeline, model, or dataset files.
"""

import os
import sys
import json
import datetime

import matplotlib
matplotlib.use("Agg")   # non-interactive backend (safe for Streamlit/no display)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
CHARTS_DIR = os.path.join(RESULTS_DIR, "charts")


def _ensure_dirs():
    os.makedirs(CHARTS_DIR, exist_ok=True)


def generate_report(all_results: list, raw_json_path: str = None) -> str:
    """
    Generate text report + charts from a list of ScenarioResult objects.
    Returns path to the generated text report.
    """
    _ensure_dirs()
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(RESULTS_DIR, "benchmark_report_{}.txt".format(ts))

    lines = []

    def w(s=""):
        lines.append(s)

    w("=" * 70)
    w("DR SCREENING SYSTEM - RURAL DEPLOYMENT BENCHMARK REPORT")
    w("SIH 2026 - Objective 10: Rural Deployment Feasibility")
    w("Generated: {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    w("=" * 70)
    w()
    w("METHODOLOGY")
    w("-" * 40)
    w("Model Checkpoint : best_model_combined_v1.pth (EfficientNet-B0)")
    w("Training Data    : APTOS 2019 + IDRiD Disease Grading (combined)")
    w("Benchmark Images : 20 images from dataset/images/ (timing only, no retraining)")
    w("Warmup Images    : 5 images (excluded from measurements)")
    w("Timing Method    : time.perf_counter() wall time for full pipeline call")
    w("Resource Monitor : psutil CPU% + process RSS RAM, polled every 0.5s")
    w()
    w("IMPORTANT DISCLAIMER")
    w("-" * 40)
    w("Scenario A: Measured on current development hardware (no restrictions).")
    w("Scenario B: SIMULATED low-resource via 1-core CPU affinity on current hardware.")
    w("            NOT a measurement on actual rural/low-resource hardware.")
    w("Scenario C: SIMULATED poor-connectivity via blocked outbound TCP (timeout=0.001s).")
    w("            NOT a real rural network test.")
    w("All results labelled 'Prototype Benchmark Results' as required.")
    w()
    w("Rationale: Python-based deployment benchmarking was used because MATLAB/Simulink")
    w("           is currently not accessible.")
    w()

    for sr in all_results:
        w("=" * 70)
        w(sr.scenario_label)
        w("=" * 70)
        w("Description  : " + sr.description[:100])
        w("Hardware Note: " + sr.hardware_note)
        w()
        w("  Warmup images    : {}".format(sr.warmup_count))
        w("  Measured images  : {}".format(sr.measured_count))
        w("  Start timestamp  : {}".format(sr.scenario_start_ts))
        w("  End timestamp    : {}".format(sr.scenario_end_ts))
        w()
        w("  LATENCY STATISTICS (ms)")
        w("    Mean   : {:.2f} ms".format(sr.mean_total_ms))
        w("    Median : {:.2f} ms".format(sr.median_total_ms))
        w("    Min    : {:.2f} ms".format(sr.min_total_ms))
        w("    Max    : {:.2f} ms".format(sr.max_total_ms))
        w("    StdDev : {:.2f} ms".format(sr.std_total_ms))
        w()
        w("  THROUGHPUT")
        w("    Images/second : {:.4f}".format(sr.throughput_img_per_sec))
        w()
        w("  RESOURCE USAGE")
        w("    Mean CPU%  : {:.1f}%".format(sr.mean_cpu_percent))
        w("    Mean RAM   : {:.1f} MB".format(sr.mean_ram_mb))
        sinfo = sr.system_info
        w("    Monitor Mean CPU% : {:.1f}%".format(sinfo.get("monitor_mean_cpu", 0)))
        w("    Monitor Max CPU%  : {:.1f}%".format(sinfo.get("monitor_max_cpu", 0)))
        w("    Monitor Mean RAM  : {:.1f} MB".format(sinfo.get("monitor_mean_ram_mb", 0)))
        w("    Monitor Max RAM   : {:.1f} MB".format(sinfo.get("monitor_max_ram_mb", 0)))
        w()
        w("  SYSTEM INFO")
        w("    Platform        : {}".format(sinfo.get("platform", "N/A")))
        w("    CPU cores (L/P) : {} / {}".format(
            sinfo.get("cpu_count_logical", "?"), sinfo.get("cpu_count_physical", "?")
        ))
        w("    Total RAM       : {:.0f} MB".format(sinfo.get("total_ram_mb", 0)))
        w()

        # Per-image table
        w("  PER-IMAGE RESULTS")
        w("  {:<16} {:>10} {:>10} {:>10} {:>8} {:>30}".format(
            "Image ID", "Total(ms)", "CPU%", "RAM(MB)", "Quality", "Prediction"
        ))
        w("  " + "-" * 90)
        for m in sr.image_metrics:
            if m.error:
                w("  {:<16} {:>10} {:>10} {:>10} {:>8} {:>30}".format(
                    m.image_id, "ERROR", "-", "-", "-", m.error[:30]
                ))
            else:
                pred_str = "{} ({:.0f}%)".format(
                    m.prediction or "-", (m.confidence or 0) * 100
                )
                w("  {:<16} {:>10.1f} {:>10.1f} {:>10.1f} {:>8} {:>30}".format(
                    m.image_id,
                    m.total_pipeline_ms,
                    m.cpu_percent,
                    m.ram_used_mb,
                    "PASS" if m.quality_passed else "FAIL",
                    pred_str[:30],
                ))
        w()

    w("=" * 70)
    w("END OF REPORT")
    w("=" * 70)

    report_text = "\n".join(lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print("Text report saved: {}".format(report_path))

    # ── Generate charts ────────────────────────────────────────────────────────
    _generate_charts(all_results, ts)

    return report_path


def _generate_charts(all_results: list, ts: str):
    _ensure_dirs()

    names = [sr.scenario_name.replace("_", " ") for sr in all_results]
    means = [sr.mean_total_ms for sr in all_results]
    medians = [sr.median_total_ms for sr in all_results]
    mins = [sr.min_total_ms for sr in all_results]
    maxs = [sr.max_total_ms for sr in all_results]
    throughputs = [sr.throughput_img_per_sec for sr in all_results]
    cpus = [sr.mean_cpu_percent for sr in all_results]
    rams = [sr.mean_ram_mb for sr in all_results]

    colors = ["#2196F3", "#FF9800", "#4CAF50"][:len(all_results)]

    # ── Chart 1: Latency comparison ────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(names))
    bars = ax.bar(x, means, color=colors, alpha=0.85, label="Mean latency")
    ax.errorbar(x, means, yerr=[
        [m - mn for m, mn in zip(means, mins)],
        [mx - m for m, mx in zip(means, maxs)]
    ], fmt="none", color="black", capsize=5, linewidth=1.5)
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel("Latency (ms)", fontsize=11)
    ax.set_title("Mean Pipeline Latency per Scenario\n[Prototype Benchmark Results]", fontsize=12)
    for bar, v in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                "{:.0f} ms".format(v), ha="center", fontsize=10)
    ax.set_ylim(0, max(maxs) * 1.25 if maxs else 100)
    fig.text(0.5, 0.01, "Note: B=Simulated 1-core; C=Simulated no-network. NOT actual rural hardware.",
             ha="center", fontsize=8, color="gray")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    out = os.path.join(CHARTS_DIR, "latency_comparison_{}.png".format(ts))
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print("Chart saved: {}".format(out))

    # ── Chart 2: Throughput comparison ────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(names, throughputs, color=colors, alpha=0.85)
    ax.set_ylabel("Images / second", fontsize=11)
    ax.set_title("Throughput per Scenario\n[Prototype Benchmark Results]", fontsize=12)
    for bar, v in zip(bars, throughputs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0005,
                "{:.4f}".format(v), ha="center", fontsize=10)
    fig.text(0.5, 0.01, "Note: B=Simulated 1-core; C=Simulated no-network.",
             ha="center", fontsize=8, color="gray")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    out = os.path.join(CHARTS_DIR, "throughput_comparison_{}.png".format(ts))
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print("Chart saved: {}".format(out))

    # ── Chart 3: Per-image latency (Scenario A) ──────────────────
    scenario_a = next((r for r in all_results if "Scenario_A" in r.scenario_name), None)
    if scenario_a:
        valid = [m for m in scenario_a.image_metrics if m.error is None]
        if valid:
            ids = [m.image_id[:8] for m in valid]
            lat = [m.total_pipeline_ms for m in valid]
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.bar(ids, lat, color="#2196F3", alpha=0.8)
            ax.axhline(scenario_a.mean_total_ms, color="red", linestyle="--",
                       linewidth=1.5, label="Mean: {:.0f}ms".format(scenario_a.mean_total_ms))
            ax.set_xlabel("Image ID (first 8 chars)", fontsize=10)
            ax.set_ylabel("Latency (ms)", fontsize=10)
            ax.set_title("Per-Image Latency - Scenario A (Normal Local)\n[Prototype Benchmark Results]",
                         fontsize=11)
            ax.legend(fontsize=10)
            plt.xticks(rotation=45, ha="right", fontsize=8)
            fig.tight_layout()
            out = os.path.join(CHARTS_DIR, "per_image_latency_A_{}.png".format(ts))
            fig.savefig(out, dpi=120)
            plt.close(fig)
            print("Chart saved: {}".format(out))

    # ── Chart 4: CPU/RAM resource usage ──────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(names, cpus, color=colors, alpha=0.85)
    axes[0].set_ylabel("Mean CPU %", fontsize=11)
    axes[0].set_title("CPU Usage per Scenario", fontsize=11)
    for ax_ in axes:
        ax_.set_ylim(bottom=0)
    for i, v in enumerate(cpus):
        axes[0].text(i, v + 0.5, "{:.1f}%".format(v), ha="center", fontsize=10)

    axes[1].bar(names, rams, color=colors, alpha=0.85)
    axes[1].set_ylabel("Mean RAM (MB)", fontsize=11)
    axes[1].set_title("RAM Usage per Scenario", fontsize=11)
    for i, v in enumerate(rams):
        axes[1].text(i, v + 5, "{:.0f}MB".format(v), ha="center", fontsize=10)

    fig.suptitle("Resource Usage [Prototype Benchmark Results]", fontsize=12, y=1.02)
    fig.text(0.5, -0.04, "Note: B=Simulated 1-core; C=Simulated no-network.",
             ha="center", fontsize=8, color="gray")
    fig.tight_layout()
    out = os.path.join(CHARTS_DIR, "resource_usage_{}.png".format(ts))
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print("Chart saved: {}".format(out))


def load_and_report_from_json(json_path: str) -> str:
    """
    Load raw JSON benchmark results and generate a report.
    Useful for regenerating reports from saved results.
    """
    with open(json_path) as f:
        data = json.load(f)

    # Reconstruct ScenarioResult-like objects from dict
    from deployment_benchmark.metrics import ScenarioResult, ImageMetrics
    results = []
    for d in data:
        sr = ScenarioResult(
            scenario_name=d["scenario_name"],
            scenario_label=d["scenario_label"],
            description=d["description"],
            hardware_note=d["hardware_note"],
            warmup_count=d.get("warmup_count", 0),
            measured_count=d.get("measured_count", 0),
            scenario_start_ts=d.get("scenario_start_ts", ""),
            scenario_end_ts=d.get("scenario_end_ts", ""),
            system_info=d.get("system_info", {}),
            mean_total_ms=d.get("mean_total_ms", 0),
            median_total_ms=d.get("median_total_ms", 0),
            min_total_ms=d.get("min_total_ms", 0),
            max_total_ms=d.get("max_total_ms", 0),
            std_total_ms=d.get("std_total_ms", 0),
            throughput_img_per_sec=d.get("throughput_img_per_sec", 0),
            mean_cpu_percent=d.get("mean_cpu_percent", 0),
            mean_ram_mb=d.get("mean_ram_mb", 0),
        )
        for im in d.get("per_image", []):
            sr.image_metrics.append(ImageMetrics(
                image_id=im["image_id"],
                image_path="",
                preprocessing_ms=im.get("preprocessing_ms", 0),
                inference_ms=im.get("inference_ms", 0),
                total_pipeline_ms=im.get("total_pipeline_ms", 0),
                cpu_percent=im.get("cpu_percent", 0),
                ram_used_mb=im.get("ram_used_mb", 0),
                prediction=im.get("prediction"),
                confidence=im.get("confidence"),
                quality_passed=im.get("quality_passed", True),
                error=im.get("error"),
            ))
        results.append(sr)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return generate_report(results, json_path)
