"""
test_quality_gate_phase1.py
----------------------------
Phase 1 Quality Gate Strengthening — Comprehensive Test Suite.

Tests:
  1.  Good retinal image                   → accepted (GOOD_QUALITY)
  2.  Artificially blurred image            → rejected (RECAPTURE_REQUIRED)
  3.  Very dark / underexposed image        → rejected
  4.  Very bright / washed-out image        → rejected
  5.  Insufficient retinal FOV              → rejected
  6.  Normal image with variation           → not incorrectly rejected
  7.  Borderline blur (mild)                → deterministic result (passes)
  8.  Quality failure blocks downstream     → pipeline returns RECAPTURE_REQUIRED
  9.  Representative image prediction reg.  → class 0 99.89%, class 1 97.89%, class 4 90.07%
  10. 10/10 consistency pass                → verify_consistency still passes

All thresholds are prototype engineering parameters, not clinically validated.
"""

import sys
import time
import json
import hashlib
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).parent.resolve()))

from quality_assessment import RetinalQualityAssessor, QualityConfig, OUTCOME_GOOD, OUTCOME_RECAPTURE

ROOT = Path(__file__).parent.resolve()
IMAGES_DIR = ROOT / "dataset" / "images"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest().upper()


def separator(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def check(condition, label):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}")
    if not condition:
        raise AssertionError(f"Test FAILED: {label}")


def run_tests():
    assessor = RetinalQualityAssessor()

    # ── 0. PRE-FLIGHT: Hash verification ─────────────────────────────────────
    separator("0. PRE-FLIGHT HASH VERIFICATION")
    model_sha = sha256_file(ROOT / "checkpoints" / "best_model_combined_v1.pth")
    csv_sha = sha256_file(ROOT / "dataset" / "test.csv")
    print(f"  Model SHA-256 : {model_sha}")
    print(f"  test.csv SHA  : {csv_sha}")
    check(model_sha == "3FA407E505F8653F00BD2CFB997224247E76FAC93C18DDEA7F35A973B0B85D05",
          "V1 model SHA unchanged")
    check(csv_sha == "DC053778AC39365696DD9836FC22F396942FCC56245EE088C69ECCBF8BDFEC76",
          "Frozen test.csv SHA unchanged")

    # Reference image paths
    img0 = IMAGES_DIR / "165634a6167e.png"  # Class 0
    img1 = IMAGES_DIR / "fe674c2f73f5.png"  # Class 1
    img4 = IMAGES_DIR / "1a7e3356b39c.png"  # Class 4
    orig_img = Image.open(img0)

    # ── 1. Good retinal image → accepted ─────────────────────────────────────
    separator("TEST 1: Good retinal image → GOOD_QUALITY")
    r = assessor.assess_image(img0)
    print(f"  outcome={r['outcome']}, score={r['quality_score']}, blur={r['blur_score']:.2f}, "
          f"brightness={r['brightness_score']:.2f}")
    check(r["quality_ok"], "Good fundus image passes quality gate")
    check(r["outcome"] == OUTCOME_GOOD, "outcome == GOOD_QUALITY")
    check(r["primary_failure_reason"] is None, "primary_failure_reason is None when passing")

    # ── 2. Artificially blurred image → rejected ─────────────────────────────
    separator("TEST 2: Heavily blurred image → RECAPTURE_REQUIRED")
    blurred = orig_img.filter(ImageFilter.GaussianBlur(radius=15))
    r = assessor.assess_image(blurred)
    print(f"  outcome={r['outcome']}, score={r['quality_score']}, blur={r['blur_score']:.2f}, "
          f"reason={r['primary_failure_reason']}")
    check(not r["quality_ok"], "Heavily blurred image rejected")
    check(r["outcome"] == OUTCOME_RECAPTURE, "outcome == RECAPTURE_REQUIRED")
    check(r["primary_failure_reason"] is not None, "primary_failure_reason populated on failure")

    # ── 3. Very dark / underexposed image → rejected ─────────────────────────
    separator("TEST 3: Very dark image → RECAPTURE_REQUIRED")
    dark = Image.fromarray((np.array(orig_img) * 0.08).astype(np.uint8))
    r = assessor.assess_image(dark)
    print(f"  outcome={r['outcome']}, score={r['quality_score']}, brightness={r['brightness_score']:.2f}, "
          f"reason={r['primary_failure_reason']}")
    check(not r["quality_ok"], "Very dark image rejected")
    check(r["outcome"] == OUTCOME_RECAPTURE, "outcome == RECAPTURE_REQUIRED")

    # ── 4. Very bright / washed-out image → rejected ─────────────────────────
    separator("TEST 4: Washed-out / overexposed image → RECAPTURE_REQUIRED")
    bright_arr = np.clip(np.array(orig_img, dtype=np.float32) * 5.0, 0, 255).astype(np.uint8)
    bright = Image.fromarray(bright_arr)
    r = assessor.assess_image(bright)
    print(f"  outcome={r['outcome']}, score={r['quality_score']}, brightness={r['brightness_score']:.2f}, "
          f"reason={r['primary_failure_reason']}")
    check(not r["quality_ok"], "Washed-out image rejected (brightness > max_foreground_brightness=200)")
    check(r["outcome"] == OUTCOME_RECAPTURE, "outcome == RECAPTURE_REQUIRED")

    # ── 5. Insufficient FOV → rejected ───────────────────────────────────────
    separator("TEST 5: Insufficient retinal FOV → RECAPTURE_REQUIRED")
    black_img = Image.new("RGB", (1024, 1024), color=(0, 0, 0))
    r = assessor.assess_image(black_img)
    print(f"  outcome={r['outcome']}, score={r['quality_score']}, fov={r['field_of_view_score']}%")
    check(not r["quality_ok"], "All-black / no-FOV image rejected")
    check(r["outcome"] == OUTCOME_RECAPTURE, "outcome == RECAPTURE_REQUIRED")

    # Small FOV: tiny bright spot on black background
    small_fov = Image.new("RGB", (1024, 1024), color=(0, 0, 0))
    import PIL.ImageDraw as ImageDraw
    draw = ImageDraw.Draw(small_fov)
    draw.ellipse([462, 462, 562, 562], fill=(120, 80, 60))  # ~1% of area
    r2 = assessor.assess_image(small_fov)
    print(f"  Small FOV: outcome={r2['outcome']}, fov={r2['field_of_view_score']}%, reason={r2['primary_failure_reason']}")
    check(not r2["quality_ok"], "Image with <5% fundus content rejected")

    # ── 6. Normal image with reasonable variation → not rejected ─────────────
    separator("TEST 6: Normal images with variation → not incorrectly rejected")
    for img_path, label in [(img0, "Class 0"), (img1, "Class 1"), (img4, "Class 4")]:
        r = assessor.assess_image(img_path)
        print(f"  {label}: outcome={r['outcome']}, score={r['quality_score']}")
        check(r["quality_ok"], f"{label} reference image not incorrectly rejected")

    # ── 7. Borderline blur (mild radius=2) → deterministic pass ──────────────
    separator("TEST 7: Borderline blur (mild radius=2) → deterministic result")
    mild_blur = orig_img.filter(ImageFilter.GaussianBlur(radius=2))
    r = assessor.assess_image(mild_blur)
    print(f"  outcome={r['outcome']}, blur_score={r['blur_score']:.2f}, score={r['quality_score']}")
    # Mild blur should pass (blur_score well above threshold 5.0)
    check(r["quality_ok"], "Mildly blurred image (radius=2) accepted — not over-rejected")
    # Result must be deterministic: run twice and compare
    r2 = assessor.assess_image(mild_blur)
    check(r["quality_score"] == r2["quality_score"], "Borderline result is deterministic")

    # Moderate blur radius=5 should now be caught (new threshold 5.0)
    mod_blur = orig_img.filter(ImageFilter.GaussianBlur(radius=5))
    r3 = assessor.assess_image(mod_blur)
    print(f"  Moderate blur (radius=5): outcome={r3['outcome']}, blur_score={r3['blur_score']:.2f}")
    check(not r3["quality_ok"], "Moderately blurred image (radius=5) rejected by strengthened threshold")

    # ── 8. Quality failure blocks downstream pipeline inference ───────────────
    separator("TEST 8: Quality failure blocks downstream inference")
    from pipeline import DRScreeningPipeline
    pipeline = DRScreeningPipeline()
    heavy_blur = orig_img.filter(ImageFilter.GaussianBlur(radius=15))
    res = pipeline.process(heavy_blur)
    print(f"  overall_status={res['overall_status']}")
    print(f"  prediction={res['prediction']}")
    print(f"  referral={res['referral']}")
    check(res["overall_status"] == "RECAPTURE_REQUIRED", "Pipeline status is RECAPTURE_REQUIRED")
    check(res["prediction"] is None, "Prediction is None when quality fails")
    check(res["referral"] is None, "Referral is None when quality fails")
    check(not res["explainability"]["gradcam_generated"], "Grad-CAM not generated when quality fails")

    # ── 9. Representative prediction regression ───────────────────────────────
    separator("TEST 9: Representative prediction regression (3 images)")
    from gradcam import GradCAM
    checkpoint = ROOT / "checkpoints" / "best_model_combined_v1.pth"
    gradcam = GradCAM(checkpoint_path=checkpoint)

    regressions = [
        (img0, 0, 99.89),
        (img1, 1, 97.89),
        (img4, 4, 90.07),
    ]
    for img_path, expected_class, expected_conf in regressions:
        # Verify quality passes first
        qr = assessor.assess_image(img_path)
        check(qr["quality_ok"], f"{img_path.name}: quality passes")
        # Run inference
        cam_res = gradcam.generate(img_path)
        pred_class = cam_res["predicted_class"]
        confidence = round(cam_res["confidence"], 2)
        print(f"  {img_path.name}: predicted class {pred_class} ({confidence}%) — "
              f"expected class {expected_class} ({expected_conf}%)")
        check(pred_class == expected_class,
              f"{img_path.name}: predicted class {pred_class} == expected {expected_class}")
        check(abs(confidence - expected_conf) < 0.05,
              f"{img_path.name}: confidence {confidence}% ≈ expected {expected_conf}%")

    # ── 10. 10/10 consistency test ────────────────────────────────────────────
    separator("TEST 10: 10/10 consistency (verify_consistency.py)")
    import subprocess
    result = subprocess.run(
        [sys.executable, "verify_consistency.py"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=300
    )
    print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    if result.returncode != 0:
        print(result.stderr[-1000:])
    check(result.returncode == 0, "verify_consistency.py exits with code 0")
    check("10/10" in result.stdout or "PASS" in result.stdout,
          "10/10 consistency result in output")

    # ── POST-FLIGHT HASH RE-CHECK ─────────────────────────────────────────────
    separator("POST-FLIGHT HASH RE-VERIFICATION")
    model_sha2 = sha256_file(ROOT / "checkpoints" / "best_model_combined_v1.pth")
    csv_sha2 = sha256_file(ROOT / "dataset" / "test.csv")
    check(model_sha2 == "3FA407E505F8653F00BD2CFB997224247E76FAC93C18DDEA7F35A973B0B85D05",
          "V1 model SHA still unchanged after tests")
    check(csv_sha2 == "DC053778AC39365696DD9836FC22F396942FCC56245EE088C69ECCBF8BDFEC76",
          "Frozen test.csv SHA still unchanged after tests")

    separator("ALL TESTS PASSED")


if __name__ == "__main__":
    run_tests()
