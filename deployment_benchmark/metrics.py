"""
metrics.py - Data structures for deployment benchmark measurements.

Records per-image timing, system resource usage, and aggregate statistics.
All timing uses time.perf_counter() for highest resolution.
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional
import statistics


@dataclass
class ImageMetrics:
    """Per-image benchmark measurement."""
    image_id: str
    image_path: str
    preprocessing_ms: float       # pipeline quality assessment + image load time
    inference_ms: float           # model forward pass time (from pipeline)
    total_pipeline_ms: float      # full pipeline.screen_image() wall time
    cpu_percent: float            # CPU % snapshot at inference time
    ram_used_mb: float            # RAM used (MB) at inference time
    prediction: Optional[str] = None
    confidence: Optional[float] = None
    quality_passed: bool = True
    error: Optional[str] = None


@dataclass
class ScenarioResult:
    """Aggregate results for one benchmark scenario."""
    scenario_name: str
    scenario_label: str           # e.g. "Prototype Benchmark Results - Scenario A"
    description: str
    hardware_note: str            # clarifies simulated vs real
    image_metrics: List[ImageMetrics] = field(default_factory=list)
    warmup_count: int = 0
    measured_count: int = 0
    scenario_start_ts: str = ""
    scenario_end_ts: str = ""
    system_info: dict = field(default_factory=dict)

    # Aggregate timing (computed after run)
    mean_total_ms: float = 0.0
    median_total_ms: float = 0.0
    min_total_ms: float = 0.0
    max_total_ms: float = 0.0
    std_total_ms: float = 0.0
    throughput_img_per_sec: float = 0.0
    mean_cpu_percent: float = 0.0
    mean_ram_mb: float = 0.0

    def compute_aggregates(self):
        """Compute summary statistics from measured image_metrics."""
        valid = [m for m in self.image_metrics if m.error is None]
        if not valid:
            return
        totals = [m.total_pipeline_ms for m in valid]
        self.mean_total_ms = statistics.mean(totals)
        self.median_total_ms = statistics.median(totals)
        self.min_total_ms = min(totals)
        self.max_total_ms = max(totals)
        self.std_total_ms = statistics.stdev(totals) if len(totals) > 1 else 0.0
        total_seconds = sum(totals) / 1000.0
        self.throughput_img_per_sec = len(valid) / total_seconds if total_seconds > 0 else 0.0
        cpus = [m.cpu_percent for m in valid]
        rams = [m.ram_used_mb for m in valid]
        self.mean_cpu_percent = statistics.mean(cpus)
        self.mean_ram_mb = statistics.mean(rams)
        self.measured_count = len(valid)

    def to_dict(self) -> dict:
        """Serialize to plain dict for reporting."""
        return {
            "scenario_name": self.scenario_name,
            "scenario_label": self.scenario_label,
            "description": self.description,
            "hardware_note": self.hardware_note,
            "warmup_count": self.warmup_count,
            "measured_count": self.measured_count,
            "scenario_start_ts": self.scenario_start_ts,
            "scenario_end_ts": self.scenario_end_ts,
            "mean_total_ms": round(self.mean_total_ms, 2),
            "median_total_ms": round(self.median_total_ms, 2),
            "min_total_ms": round(self.min_total_ms, 2),
            "max_total_ms": round(self.max_total_ms, 2),
            "std_total_ms": round(self.std_total_ms, 2),
            "throughput_img_per_sec": round(self.throughput_img_per_sec, 4),
            "mean_cpu_percent": round(self.mean_cpu_percent, 1),
            "mean_ram_mb": round(self.mean_ram_mb, 1),
            "system_info": self.system_info,
            "per_image": [
                {
                    "image_id": m.image_id,
                    "preprocessing_ms": round(m.preprocessing_ms, 2),
                    "inference_ms": round(m.inference_ms, 2),
                    "total_pipeline_ms": round(m.total_pipeline_ms, 2),
                    "cpu_percent": m.cpu_percent,
                    "ram_used_mb": round(m.ram_used_mb, 1),
                    "prediction": m.prediction,
                    "confidence": round(m.confidence, 4) if m.confidence else None,
                    "quality_passed": m.quality_passed,
                    "error": m.error,
                }
                for m in self.image_metrics
            ],
        }
