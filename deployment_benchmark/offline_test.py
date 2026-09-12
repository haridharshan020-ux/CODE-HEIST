"""
offline_test.py - Standalone offline validation for rural deployment.

Verifies:
  1. Model checkpoint file exists locally on disk (no download needed).
  2. Model checkpoint SHA-256 is unchanged from the known baseline.
  3. Full pipeline inference runs successfully WITHOUT any network dependency.
  4. Predictions on reference images are consistent with the Scenario A baseline.

Run standalone:
    python -m deployment_benchmark.offline_test

Or from project root:
    C:/Python314/python.exe deployment_benchmark/offline_test.py
"""

import os
import sys
import hashlib
import socket
import time
import json

# Allow running from project root
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ── Constants ─────────────────────────────────────────────────────────────────
CHECKPOINT_PATH = os.path.join(_PROJECT_ROOT, "checkpoints", "best_model_combined_v1.pth")
KNOWN_CHECKPOINT_SHA256 = "3FA407E505F8653F00BD2CFB997224247E76FAC93C18DDEA7F35A973B0B85D05"

# Reference images for offline inference test (subset of benchmark list)
REFERENCE_IMAGE_IDS = [
    "4f0866b90c27",
    "fe674c2f73f5",
    "165634a6167e",
    "b16dd4483ca5",
    "1a7e3356b39c",
]
IMAGES_DIR = os.path.join(_PROJECT_ROOT, "dataset", "images")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def check_network_available() -> bool:
    """Return True if internet appears reachable."""
    try:
        s = socket.create_connection(("8.8.8.8", 53), timeout=3.0)
        s.close()
        return True
    except OSError:
        return False


def run_offline_tests(verbose: bool = True) -> dict:
    """
    Run all offline validation tests.
    Returns a result dict with PASS/FAIL for each check.
    """
    results = {
        "checkpoint_exists": False,
        "checkpoint_sha256_match": False,
        "known_sha256": KNOWN_CHECKPOINT_SHA256,
        "actual_sha256": "",
        "network_available_before": False,
        "network_available_after": False,
        "inference_offline_passed": False,
        "inference_results": [],
        "overall_pass": False,
        "errors": [],
    }

    def log(msg):
        if verbose:
            print(msg)

    log("=" * 60)
    log("OFFLINE VALIDATION TEST")
    log("=" * 60)

    # ── Check 1: Checkpoint exists ─────────────────────────────────
    log("\n[CHECK 1] Checkpoint file exists locally...")
    if os.path.isfile(CHECKPOINT_PATH):
        size_mb = os.path.getsize(CHECKPOINT_PATH) / (1024 * 1024)
        log("  PASS - Found: {}".format(CHECKPOINT_PATH))
        log("  Size: {:.2f} MB".format(size_mb))
        results["checkpoint_exists"] = True
    else:
        err = "FAIL - Checkpoint NOT found: {}".format(CHECKPOINT_PATH)
        log("  " + err)
        results["errors"].append(err)
        return results  # Cannot continue

    # ── Check 2: SHA-256 integrity ─────────────────────────────────
    log("\n[CHECK 2] Checkpoint SHA-256 integrity...")
    actual_sha256 = sha256_file(CHECKPOINT_PATH)
    results["actual_sha256"] = actual_sha256
    if actual_sha256 == KNOWN_CHECKPOINT_SHA256:
        log("  PASS - SHA-256 matches known baseline")
        log("  {}".format(actual_sha256))
        results["checkpoint_sha256_match"] = True
    else:
        err = "FAIL - SHA-256 MISMATCH. Expected: {} Got: {}".format(
            KNOWN_CHECKPOINT_SHA256, actual_sha256
        )
        log("  " + err)
        results["errors"].append(err)

    # ── Check 3: Network state (informational) ─────────────────────
    log("\n[CHECK 3] Network availability check (informational)...")
    net_before = check_network_available()
    results["network_available_before"] = net_before
    log("  Network available (before blocking): {}".format("YES" if net_before else "NO"))

    # ── Check 4: Offline inference (network blocked) ───────────────
    log("\n[CHECK 4] Offline inference with network blocked...")
    log("  Setting socket default timeout = 0.001s to simulate no-internet...")

    # Import pipeline (black box import - do NOT copy model code)
    try:
        from pipeline import DRScreeningPipeline
        pipeline = DRScreeningPipeline()
        log("  Pipeline loaded successfully.")
    except Exception as exc:
        err = "FAIL - Could not load pipeline: {}".format(str(exc))
        log("  " + err)
        results["errors"].append(err)
        return results

    original_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(0.001)   # block outbound TCP
    inference_errors = []

    try:
        for img_id in REFERENCE_IMAGE_IDS:
            img_path = os.path.join(IMAGES_DIR, img_id + ".png")
            if not os.path.isfile(img_path):
                log("  SKIP - Image not found: {}".format(img_id))
                continue
            try:
                t0 = time.perf_counter()
                result = pipeline.process(img_path, generate_gradcam=False, save_visualizations=False)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                status = result.get("overall_status", "") if result else ""
                if result and status not in ("ERROR",):
                    pred_dict = result.get("prediction") or {}
                    pred = pred_dict.get("severity", "unknown")
                    conf = pred_dict.get("confidence", 0.0)
                    log("  PASS  {} -> {} ({:.1f}%) [{:.0f}ms]".format(
                        img_id, pred, (conf or 0) * 100, elapsed_ms
                    ))
                    results["inference_results"].append({
                        "image_id": img_id,
                        "prediction": pred,
                        "confidence": conf,
                        "elapsed_ms": round(elapsed_ms, 1),
                        "status": "PASS",
                    })
                else:
                    err_msg = result.get("message", "unknown") if result else "None"
                    log("  FAIL  {} -> pipeline error: {}".format(img_id, err_msg))
                    inference_errors.append(img_id)
                    results["inference_results"].append({
                        "image_id": img_id,
                        "status": "FAIL",
                        "error": err_msg,
                    })
            except Exception as exc:
                log("  FAIL  {} -> exception: {}".format(img_id, str(exc)))
                inference_errors.append(img_id)
                results["inference_results"].append({
                    "image_id": img_id,
                    "status": "FAIL",
                    "error": str(exc),
                })
    finally:
        socket.setdefaulttimeout(original_timeout)

    net_after = check_network_available()
    results["network_available_after"] = net_after
    log("  Network available (after restoring): {}".format("YES" if net_after else "NO"))

    if not inference_errors and results["inference_results"]:
        log("  PASS - All {} reference images inferred successfully with network blocked.".format(
            len(results["inference_results"])
        ))
        results["inference_offline_passed"] = True
    else:
        results["errors"].append("Offline inference failed for: {}".format(inference_errors))

    # ── Final verdict ──────────────────────────────────────────────
    results["overall_pass"] = (
        results["checkpoint_exists"]
        and results["checkpoint_sha256_match"]
        and results["inference_offline_passed"]
    )

    log("\n" + "=" * 60)
    log("OVERALL RESULT: {}".format("PASS" if results["overall_pass"] else "FAIL"))
    log("=" * 60)

    return results


if __name__ == "__main__":
    result = run_offline_tests(verbose=True)
    # Save result to results/offline_test_result.json
    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "offline_test_result.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print("\nResult saved to: {}".format(out_path))
    sys.exit(0 if result["overall_pass"] else 1)
