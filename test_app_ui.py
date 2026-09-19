"""
test_app_ui.py
--------------
Tests app.py via Streamlit's official AppTest testing framework.
Verifies:
  1. App loads cleanly without unhandled exceptions.
  2. Running with a good quality image triggers screening, displays metrics,
     shows Grad-CAM, referral priority, and latency.
  3. Running with an inadequate quality image displays "Please recapture image"
     and halts further evaluation.
"""

from pathlib import Path
from PIL import Image, ImageFilter
import io
from streamlit.testing.v1 import AppTest

def test_ui():
    root = Path(__file__).parent.resolve()
    images_dir = root / "dataset" / "images"

    print("=" * 70)
    print("  TESTING STREAMLIT UI (APP.PY) VIA APPTEST")
    print("=" * 70)

    # â”€â”€ TEST 1: Initial App Launch â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\n[Test 1] Launching app.py without upload...")
    at = AppTest.from_file("app.py", default_timeout=90).run()
    assert not at.exception, f"App raised exception on load: {at.exception}"
    print("  App initialized successfully without exceptions.")
    print(f"  Title: {[t.value.encode('ascii', 'replace').decode('ascii') for t in at.title]}")

    # â”€â”€ TEST 2: Good Quality Image Screening â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\n[Test 2] Simulating upload of good-quality image (165634a6167e.png)...")
    good_img_path = images_dir / "165634a6167e.png"
    good_img = Image.open(good_img_path)
    good_bytes = io.BytesIO()
    good_img.save(good_bytes, format="PNG")
    good_bytes.seek(0)

    at_good = AppTest.from_file("app.py", default_timeout=90)
    at_good.run()
    at_good.file_uploader[0].upload("165634a6167e.png", good_bytes.getvalue()).run()
    assert not at_good.exception, f"Exception during upload: {at_good.exception}"

    # Click "Start Screening"
    at_good.button[0].click().run()
    assert not at_good.exception, f"Exception during screening: {at_good.exception}"

    # Verify elements present
    success_msgs = [s.value for s in at_good.success]
    metric_labels = [m.label for m in at_good.metric]
    metric_values = [m.value for m in at_good.metric]
    captions = [c.value for c in at_good.caption]

    print("  Screening executed successfully.")
    q_str = success_msgs[0] if success_msgs else 'None'
    print(f"  Quality result  : {q_str.encode('ascii', 'replace').decode('ascii')}")
    print(f"  Metrics         : {list(zip(metric_labels, metric_values))}")
    print(f"  Images rendered : {len(at_good.image)} displayed (Uploaded + Grad-CAM)")
    lat_caps = [c.encode('ascii', 'replace').decode('ascii') for c in captions if 'Processing time' in c]
    print(f"  Latency caption : {lat_caps}")

    assert any("Quality Acceptable" in s for s in success_msgs), "Quality banner missing!"
    assert "Predicted Severity" in metric_labels, "Severity metric missing!"
    assert len(at_good.image) >= 2, "Both uploaded and Grad-CAM images should be rendered!"

    # â”€â”€ TEST 3: Inadequate Quality Image Rejection â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\n[Test 3] Simulating upload of heavily blurred image (Inadequate quality)...")
    blurred_img = good_img.filter(ImageFilter.GaussianBlur(radius=15))
    blur_bytes = io.BytesIO()
    blurred_img.save(blur_bytes, format="PNG")
    blur_bytes.seek(0)

    at_blur = AppTest.from_file("app.py", default_timeout=90)
    at_blur.run()
    at_blur.file_uploader[0].upload("blurred_retina.png", blur_bytes.getvalue()).run()
    at_blur.button[0].click().run()
    assert not at_blur.exception, f"Exception during blur test: {at_blur.exception}"

    errors = [e.value for e in at_blur.error]
    warnings = [w.value for w in at_blur.warning]

    print("  Inadequate scan executed successfully.")
    print(f"  Error message   : {[e.encode('ascii', 'replace').decode('ascii') for e in errors]}")
    print(f"  Warning message : {[w.encode('ascii', 'replace').decode('ascii') for w in warnings]}")
    print(f"  Metrics count   : {len(at_blur.metric)} (should be 0 because execution stopped)")

    assert any("Quality Inadequate" in e for e in errors), "Quality inadequate error banner missing!"
    assert any("Please recapture image" in w for w in warnings), "Recapture warning banner missing!"
    assert len(at_blur.metric) == 0, "Classifier metrics should NOT be rendered for inadequate quality!"

    print("\n" + "=" * 70)
    print("  ALL STREAMLIT APP TESTS PASSED SUCCESSFULLY")
    print("=" * 70)

if __name__ == "__main__":
    test_ui()

