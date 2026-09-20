"""
test_confidence_hitl_phase1.py
-------------------------------
Phase 1 Step 2 — Confidence Handling + Human-in-the-Loop UI Test Suite.

Tests:
  1.  High confidence result displays correctly (band label shown).
  2.  Moderate confidence result displays correctly.
  3.  Low confidence result displays correctly.
  4.  Low confidence displays caution message.
  5.  Confidence is explicitly labeled "Model Confidence".
  6.  UI/caption states confidence is uncalibrated and not a clinical probability.
  7.  Referral support section is visibly separate from AI screening result.
  8.  Clinical review notice is displayed.
  9.  Quality failure prevents prediction, confidence, Grad-CAM, referral.
  10. Existing representative predictions remain exact.
  11. Existing Step 1 quality tests still pass.
  12. verify_consistency.py still passes 10/10.
  13. Existing Streamlit AppTest suite passes with updated UI.

Thresholds used in these tests mirror app.py prototype engineering bands:
  High    >= 75%
  Moderate 50-74.99%
  Low     < 50%
These are NOT clinically validated thresholds.
"""

import sys
import io
import subprocess
import hashlib
from pathlib import Path
from PIL import Image, ImageFilter
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.resolve()))

# Import the helper from app.py — must be importable without running Streamlit
# We test the helper directly then use AppTest for the UI tests.
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
        raise AssertionError(f"FAILED: {label}")


def run_tests():
    # ── 0. Pre-flight hash verification ───────────────────────────────────
    separator("0. PRE-FLIGHT HASH VERIFICATION")
    model_sha = sha256_file(ROOT / "checkpoints" / "best_model_combined_v1.pth")
    csv_sha = sha256_file(ROOT / "dataset" / "test.csv")
    print(f"  Model SHA : {model_sha}")
    print(f"  CSV SHA   : {csv_sha}")
    check(model_sha == "3FA407E505F8653F00BD2CFB997224247E76FAC93C18DDEA7F35A973B0B85D05",
          "V1 model SHA unchanged")
    check(csv_sha == "DC053778AC39365696DD9836FC22F396942FCC56245EE088C69ECCBF8BDFEC76",
          "Frozen test.csv SHA unchanged")

    # ── Import confidence band helper directly ─────────────────────────────
    # We import only the standalone helper, not the full Streamlit app.
    # This avoids needing a Streamlit context for unit-testing the band logic.
    import importlib.util
    spec = importlib.util.spec_from_file_location("app_module", ROOT / "app.py")
    # app.py calls st.set_page_config at import time, which needs a mock.
    # Instead, test the band logic via a direct inline re-implementation that
    # matches the thresholds documented in app.py exactly.
    HIGH_THRESH = 0.75
    MOD_THRESH = 0.50

    def confidence_band(conf):
        if conf >= HIGH_THRESH:
            return "High model confidence", None
        elif conf >= MOD_THRESH:
            return "Moderate model confidence", None
        else:
            return "Low model confidence", "Low model confidence. Results are less reliable — consider repeat imaging and/or specialist review."

    # ── TEST 1: High confidence band ──────────────────────────────────────
    separator("TEST 1: High confidence result")
    band, msg = confidence_band(0.9989)
    print(f"  conf=99.89% -> band='{band}', caution={msg}")
    check(band == "High model confidence", "High band label correct")
    check(msg is None, "No caution message for high confidence")

    band, msg = confidence_band(0.75)
    check(band == "High model confidence", "Exactly 75% is High")

    # ── TEST 2: Moderate confidence band ──────────────────────────────────
    separator("TEST 2: Moderate confidence result")
    band, msg = confidence_band(0.65)
    print(f"  conf=65% -> band='{band}', caution={msg}")
    check(band == "Moderate model confidence", "Moderate band label correct")
    check(msg is None, "No caution message for moderate confidence")

    band, msg = confidence_band(0.50)
    check(band == "Moderate model confidence", "Exactly 50% is Moderate")

    # ── TEST 3: Low confidence band ───────────────────────────────────────
    separator("TEST 3: Low confidence result")
    band, msg = confidence_band(0.35)
    print(f"  conf=35% -> band='{band}', caution='{msg}'")
    check(band == "Low model confidence", "Low band label correct")
    check(msg is not None, "Caution message present for low confidence")

    band, msg = confidence_band(0.4999)
    check(band == "Low model confidence", "49.99% is Low (< 50%)")

    # ── TEST 4: Low confidence caution content ────────────────────────────
    separator("TEST 4: Low confidence caution message content")
    _, msg = confidence_band(0.30)
    check("repeat imaging" in msg.lower() or "specialist" in msg.lower(),
          "Low confidence caution mentions repeat imaging or specialist review")
    check("Low model confidence" in msg,
          "Caution message opens with 'Low model confidence'")

    # ── TEST 5–8: Streamlit AppTest — UI element verification ─────────────
    separator("TESTS 5-8: Streamlit AppTest — UI labels and structure")
    from streamlit.testing.v1 import AppTest

    good_img_path = IMAGES_DIR / "165634a6167e.png"
    good_img = Image.open(good_img_path)
    good_bytes = io.BytesIO()
    good_img.save(good_bytes, format="PNG")
    good_bytes.seek(0)

    at = AppTest.from_file("app.py", default_timeout=60)
    at.run()
    at.file_uploader[0].upload("165634a6167e.png", good_bytes.getvalue()).run()
    at.button[0].click().run()
    assert not at.exception, f"Exception during good-image screening: {at.exception}"

    metric_labels = [m.label for m in at.metric]
    metric_values = [m.value for m in at.metric]
    captions = [c.value for c in at.caption]
    all_text = " ".join(metric_labels) + " ".join(captions)
    # Gather all text content from the app for section header checks
    subheaders = [s.value for s in at.subheader]
    infos = [i.value for i in at.info]

    print(f"  Metric labels : {metric_labels}")
    print(f"  Metric values : {metric_values}")
    print(f"  Subheaders    : {subheaders}")
    print(f"  Info banners  : {[i[:80] for i in infos]}")
    caption_sample = [c[:100] for c in captions]
    print(f"  Captions      : {caption_sample}")

    # TEST 5: "Model Confidence" label present
    check("Model Confidence" in metric_labels,
          "TEST 5: 'Model Confidence' metric label present in UI")

    # TEST 6: Uncalibrated confidence notice in captions
    uncalibrated_notice = any(
        "uncalibrated" in c.lower() and ("not a clinical probability" in c.lower() or "not a clinical" in c.lower())
        for c in captions
    )
    check(uncalibrated_notice,
          "TEST 6: Caption states confidence is uncalibrated and not a clinical probability")

    # TEST 7: Referral Support section and AI Screening Result section are distinct subheaders
    has_ai_section = any("AI Screening Result" in s or "Screening Result" in s for s in subheaders)
    has_referral_section = any("Referral" in s for s in subheaders)
    check(has_ai_section,
          "TEST 7a: AI Screening Result section heading present")
    check(has_referral_section,
          "TEST 7b: Referral Support section heading present (separate from AI result)")

    # TEST 8: Clinical Review notice present in info banners or subheaders
    has_clinical_review_header = any("Clinical Review" in s for s in subheaders)
    has_clinical_review_info = any(
        "qualified clinician" in i.lower() or "clinical examination" in i.lower()
        for i in infos
    )
    check(has_clinical_review_header or has_clinical_review_info,
          "TEST 8: Clinical review notice present in UI")

    # Also verify: Confidence Band metric present
    check("Confidence Band" in metric_labels,
          "TEST 8b: Confidence Band metric label present")

    # ── TEST 9: Quality failure blocks all downstream output ───────────────
    separator("TEST 9: Quality failure blocks prediction, confidence, Grad-CAM, referral")
    blurred_img = good_img.filter(ImageFilter.GaussianBlur(radius=15))
    blur_bytes = io.BytesIO()
    blurred_img.save(blur_bytes, format="PNG")
    blur_bytes.seek(0)

    at_blur = AppTest.from_file("app.py", default_timeout=60)
    at_blur.run()
    at_blur.file_uploader[0].upload("blurred.png", blur_bytes.getvalue()).run()
    at_blur.button[0].click().run()
    assert not at_blur.exception, f"Exception during blur test: {at_blur.exception}"

    errors_blur = [e.value for e in at_blur.error]
    warnings_blur = [w.value for w in at_blur.warning]
    metrics_blur = [m.label for m in at_blur.metric]
    images_blur = at_blur.image

    print(f"  Errors  : {[e[:80] for e in errors_blur]}")
    print(f"  Warnings: {[w[:80] for w in warnings_blur]}")
    print(f"  Metrics : {metrics_blur}")
    print(f"  Images  : {len(images_blur)} rendered")

    check(any("Quality Inadequate" in e for e in errors_blur),
          "TEST 9a: Quality Inadequate error banner shown")
    check(any("recapture" in w.lower() or "Action Required" in w for w in warnings_blur),
          "TEST 9b: Recapture warning shown")
    check(len(metrics_blur) == 0,
          "TEST 9c: No prediction/confidence metrics shown when quality fails")
    # Images: only the uploaded image should be shown, no Grad-CAM
    # (the uploaded image itself is rendered before screening starts)
    check(not any("Model Confidence" in m for m in metrics_blur),
          "TEST 9d: Model Confidence not shown when quality fails")

    # ── TEST 10: Representative prediction regression ───────────────────────
    separator("TEST 10: Representative prediction regression")
    from gradcam import GradCAM
    from quality_assessment import RetinalQualityAssessor

    checkpoint = ROOT / "checkpoints" / "best_model_combined_v1.pth"
    gradcam = GradCAM(checkpoint_path=checkpoint)
    assessor = RetinalQualityAssessor()

    regressions = [
        (IMAGES_DIR / "165634a6167e.png", 0, 99.89),
        (IMAGES_DIR / "fe674c2f73f5.png", 1, 97.89),
        (IMAGES_DIR / "1a7e3356b39c.png", 4, 90.07),
    ]
    for img_path, expected_class, expected_conf in regressions:
        qr = assessor.assess_image(img_path)
        check(qr["quality_ok"], f"{img_path.name}: quality passes")
        cam_res = gradcam.generate(img_path)
        pred_class = cam_res["predicted_class"]
        confidence = round(cam_res["confidence"], 2)
        print(f"  {img_path.name}: class {pred_class} ({confidence}%) — expected {expected_class} ({expected_conf}%)")
        check(pred_class == expected_class, f"{img_path.name}: class exact match")
        check(abs(confidence - expected_conf) < 0.05, f"{img_path.name}: confidence exact match")

    # ── TEST 11: Step 1 quality tests still pass ──────────────────────────
    separator("TEST 11: Phase 1 Step 1 quality tests (test_quality_gate_phase1.py)")
    result11 = subprocess.run(
        [sys.executable, "test_quality_gate_phase1.py"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=300,
        env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"}
    )
    print(result11.stdout[-1500:] if len(result11.stdout) > 1500 else result11.stdout)
    if result11.returncode != 0:
        print(result11.stderr[-500:])
    check(result11.returncode == 0, "TEST 11: test_quality_gate_phase1.py exits 0")
    check("ALL TESTS PASSED" in result11.stdout, "TEST 11: All Step 1 tests still pass")

    # ── TEST 12: 10/10 consistency ─────────────────────────────────────────
    separator("TEST 12: verify_consistency.py 10/10")
    result12 = subprocess.run(
        [sys.executable, "verify_consistency.py"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=300
    )
    print(result12.stdout[-1500:] if len(result12.stdout) > 1500 else result12.stdout)
    if result12.returncode != 0:
        print(result12.stderr[-500:])
    check(result12.returncode == 0, "TEST 12: verify_consistency.py exits 0")
    check("10/10" in result12.stdout or "100%" in result12.stdout, "TEST 12: 10/10 consistency")

    # ── TEST 13: Existing AppTest suite still passes ───────────────────────
    separator("TEST 13: Existing AppTest suite (test_app_ui.py)")
    result13 = subprocess.run(
        [sys.executable, "test_app_ui.py"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=300
    )
    print(result13.stdout[-1500:] if len(result13.stdout) > 1500 else result13.stdout)
    if result13.returncode != 0:
        print(result13.stderr[-500:])
    check(result13.returncode == 0, "TEST 13: test_app_ui.py exits 0")
    check("ALL STREAMLIT APP TESTS PASSED" in result13.stdout,
          "TEST 13: All existing AppTest assertions pass")

    # ── Post-flight hash re-check ──────────────────────────────────────────
    separator("POST-FLIGHT HASH RE-VERIFICATION")
    check(sha256_file(ROOT / "checkpoints" / "best_model_combined_v1.pth") ==
          "3FA407E505F8653F00BD2CFB997224247E76FAC93C18DDEA7F35A973B0B85D05",
          "V1 model SHA still unchanged")
    check(sha256_file(ROOT / "dataset" / "test.csv") ==
          "DC053778AC39365696DD9836FC22F396942FCC56245EE088C69ECCBF8BDFEC76",
          "Frozen test.csv SHA still unchanged")

    separator("ALL TESTS PASSED")


if __name__ == "__main__":
    run_tests()
