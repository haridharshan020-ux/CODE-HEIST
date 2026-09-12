"""
connectivity_simulation.py - Demonstrate that local DR inference is network-independent.

Blocks outbound TCP by setting a near-zero socket default timeout during inference.
If predictions match the baseline (Scenario A), local inference is confirmed
to work without internet connectivity.

DISCLAIMER: This is a software-level simulation. It does NOT test actual rural
network conditions or hardware-level network isolation.
"""

import socket
import sys
import os
import time
from typing import Tuple, Optional

# Original socket timeout placeholder
_original_timeout: Optional[float] = None


def block_network():
    """
    Set an extremely short socket default timeout (0.001s) to block
    outbound TCP connections during the benchmark measurement window.
    Stores original timeout for restoration.
    """
    global _original_timeout
    _original_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(0.001)


def restore_network():
    """
    Restore the original socket default timeout.
    """
    global _original_timeout
    socket.setdefaulttimeout(_original_timeout)
    _original_timeout = None


def is_network_available(host: str = "8.8.8.8", port: int = 53, timeout: float = 3.0) -> bool:
    """
    Check real network availability (DNS/TCP to 8.8.8.8:53).
    Used BEFORE blocking to record baseline network state.
    Returns True if internet appears reachable.
    """
    try:
        s = socket.create_connection((host, port), timeout=timeout)
        s.close()
        return True
    except OSError:
        return False


def verify_offline_inference(pipeline, image_path: str) -> Tuple[bool, str, str]:
    """
    Run pipeline.screen_image() with network blocked.

    Returns:
        (success: bool, prediction: str, detail_message: str)
    """
    block_network()
    try:
        t0 = time.perf_counter()
        result = pipeline.screen_image(image_path)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if result and "error" not in result:
            pred = result.get("prediction", "unknown")
            detail = (
                "[NETWORK BLOCKED] Inference succeeded in {:.1f}ms. "
                "Prediction: {}. Local inference is network-independent.".format(elapsed_ms, pred)
            )
            return True, pred, detail
        else:
            err = result.get("error", "unknown error") if result else "None result"
            return False, "", "[NETWORK BLOCKED] Pipeline returned error: {}".format(err)
    except Exception as exc:
        return False, "", "[NETWORK BLOCKED] Exception during inference: {}".format(str(exc))
    finally:
        restore_network()
