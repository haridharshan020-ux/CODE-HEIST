"""
system_monitor.py - Background CPU/RAM poller during benchmark runs.

Runs in a daemon thread; sampling interval is configurable.
Caller controls start/stop via context manager or explicit start()/stop().
"""

import threading
import time
import psutil
from typing import List, Tuple


class SystemMonitor:
    """
    Background thread that polls CPU and RAM usage at a fixed interval.

    Usage:
        monitor = SystemMonitor(interval_sec=0.5)
        monitor.start()
        # ... run workload ...
        monitor.stop()
        samples = monitor.get_samples()   # List[(cpu_pct, ram_mb)]
    """

    def __init__(self, interval_sec: float = 0.5):
        self._interval = interval_sec
        self._samples: List[Tuple[float, float]] = []
        self._running = False
        self._thread: threading.Thread = None

    def start(self):
        """Start background polling thread."""
        self._samples = []
        self._running = True
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self):
        """Signal polling thread to stop and wait for it."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def _poll(self):
        process = psutil.Process()
        while self._running:
            try:
                cpu = psutil.cpu_percent(interval=None)
                ram_mb = process.memory_info().rss / (1024 * 1024)
                self._samples.append((cpu, ram_mb))
            except Exception:
                pass
            time.sleep(self._interval)

    def get_samples(self) -> List[Tuple[float, float]]:
        """Return list of (cpu_percent, ram_mb) tuples collected so far."""
        return list(self._samples)

    def mean_cpu(self) -> float:
        if not self._samples:
            return 0.0
        return sum(s[0] for s in self._samples) / len(self._samples)

    def mean_ram_mb(self) -> float:
        if not self._samples:
            return 0.0
        return sum(s[1] for s in self._samples) / len(self._samples)

    def max_cpu(self) -> float:
        if not self._samples:
            return 0.0
        return max(s[0] for s in self._samples)

    def max_ram_mb(self) -> float:
        if not self._samples:
            return 0.0
        return max(s[1] for s in self._samples)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


def snapshot_system() -> dict:
    """
    Take an instantaneous snapshot of system resource state.
    Returns dict suitable for ScenarioResult.system_info.
    """
    import platform
    try:
        process = psutil.Process()
        vm = psutil.virtual_memory()
        cpu_count = psutil.cpu_count(logical=True)
        cpu_count_phys = psutil.cpu_count(logical=False)
        return {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count_logical": cpu_count,
            "cpu_count_physical": cpu_count_phys,
            "total_ram_mb": round(vm.total / (1024 * 1024), 1),
            "available_ram_mb": round(vm.available / (1024 * 1024), 1),
            "ram_used_percent": vm.percent,
            "process_ram_mb": round(process.memory_info().rss / (1024 * 1024), 1),
        }
    except Exception as e:
        return {"error": str(e)}
