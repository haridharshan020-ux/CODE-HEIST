"""
test_referral_tracker.py
-------------------------
Referral Follow-up Tracker â€” Comprehensive Test Suite.

Tests:
  1.  init_db() creates schema with correct columns.
  2.  save_record() succeeds and returns a valid UUID hex.
  3.  load_records() returns the saved row with correct field values.
  4.  update_status() changes followup_status and sets updated_at.
  5.  update_status() with invalid status raises ValueError.
  6.  load_records(status_filter="Pending") returns only Pending rows.
  7.  export_csv() produces a valid CSV with correct headers and row count.
  8.  save_record() with empty patient_ref raises ValueError.
  9.  AppTest: Save to Tracker form absent before screening starts.
  10. AppTest: Save to Tracker subheader present after good-quality screening.
  11. AppTest: Save to Tracker NOT shown after quality failure (st.stop() fires).
  12. Referral_Tracker page loads without error.
  13. Regression: 3-image representative predictions unchanged.
  14. Regression: verify_consistency.py 10/10.
  15. Regression: test_quality_gate_phase1.py all pass.
  16. Regression: test_app_ui.py all pass.

All tracker thresholds/statuses are prototype administrative values.
Protected hashes verified at start and end.
"""

import sys
import io
import csv
import gc
import subprocess
import hashlib
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))

ROOT = Path(__file__).parent.resolve()
IMAGES_DIR = ROOT / "dataset" / "images"

V1_SHA = "3FA407E505F8653F00BD2CFB997224247E76FAC93C18DDEA7F35A973B0B85D05"
CSV_SHA = "DC053778AC39365696DD9836FC22F396942FCC56245EE088C69ECCBF8BDFEC76"


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


def make_temp_db():
    """Return a Path to a temporary SQLite file."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    return Path(tmp.name)


def safe_unlink(path: Path):
    """
    Unlink a file safely on Windows where SQLite may hold a handle briefly.
    Forces garbage collection and retries a few times before giving up.
    """
    gc.collect()
    for _ in range(5):
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError:
            time.sleep(0.1)
    # Last attempt â€” ignore error to not fail test cleanup
    try:
        path.unlink(missing_ok=True)
    except PermissionError:
        pass


def run_tests():
    from referral_tracker import (
        init_db,
        save_record,
        load_records,
        update_status,
        export_csv,
        VALID_STATUSES,
        TrackerRecord,
    )

    # â”€â”€ 0. Pre-flight hash verification â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    separator("0. PRE-FLIGHT HASH VERIFICATION")
    model_sha = sha256_file(ROOT / "checkpoints" / "best_model_combined_v1.pth")
    csv_sha = sha256_file(ROOT / "dataset" / "test.csv")
    print(f"  Model SHA : {model_sha}")
    print(f"  CSV SHA   : {csv_sha}")
    check(model_sha == V1_SHA, "V1 model SHA unchanged")
    check(csv_sha == CSV_SHA, "Frozen test.csv SHA unchanged")

    # â”€â”€ TEST 1: init_db creates schema â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    separator("TEST 1: init_db() creates schema with correct columns")
    db = make_temp_db()
    try:
        init_db(db)
        import sqlite3
        conn = sqlite3.connect(str(db))
        cursor = conn.execute("PRAGMA table_info(referral_records);")
        cols = [row[1] for row in cursor.fetchall()]
        conn.close()
        expected_cols = [
            "record_id", "created_at", "patient_ref",
            "predicted_class", "severity", "confidence_pct",
            "confidence_band", "quality_score",
            "referral_priority", "referral_due", "recommendation",
            "followup_status", "followup_notes", "updated_at",
        ]
        print(f"  Columns found: {cols}")
        check(set(cols) == set(expected_cols),
              "All 14 expected columns present")
        check("referral_due" in cols,
              "referral_due column present")
        # Calling init_db again must be idempotent
        init_db(db)
        check(True, "init_db() is idempotent (no error on second call)")
    finally:
        safe_unlink(db)

    # â”€â”€ TEST 2: save_record() succeeds and returns UUID â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    separator("TEST 2: save_record() returns valid UUID hex")
    db = make_temp_db()
    try:
        rec_id = save_record(
            patient_ref="PHC-TEST-001",
            predicted_class=2,
            severity="Moderate NPDR",
            confidence_pct=61.5,
            confidence_band="Moderate",
            quality_score=88.0,
            referral_priority="Semi-urgent",
            referral_due="Ophthalmic referral for comprehensive retinal evaluation",
            recommendation="Scheduled ophthalmic consultation within 3-6 months.",
            db_path=db,
        )
        print(f"  record_id: {rec_id}")
        check(isinstance(rec_id, str) and len(rec_id) == 32,
              "record_id is 32-char hex string")
        # Verify row was actually inserted
        import sqlite3
        conn = sqlite3.connect(str(db))
        count = conn.execute("SELECT COUNT(*) FROM referral_records").fetchone()[0]
        conn.close()
        check(count == 1, "Exactly 1 row in DB after save")
    finally:
        safe_unlink(db)

    # â”€â”€ TEST 3: load_records() returns saved row with correct fields â”€â”€â”€â”€â”€â”€â”€â”€
    separator("TEST 3: load_records() returns saved row with correct field values")
    db = make_temp_db()
    try:
        rec_id = save_record(
            patient_ref=" PHC-TEST-002 ",   # leading/trailing spaces stripped
            predicted_class=0,
            severity="No DR",
            confidence_pct=99.89,
            confidence_band="High",
            quality_score=99.8,
            referral_priority="Routine",
            referral_due="Routine annual diabetic eye screening",
            recommendation="No features of DR detected.",
            db_path=db,
        )
        records = load_records(db_path=db)
        check(len(records) == 1, "One record returned")
        r = records[0]
        check(r.record_id == rec_id, "record_id matches")
        check(r.patient_ref == "PHC-TEST-002", "patient_ref stripped of whitespace")
        check(r.predicted_class == 0, "predicted_class correct")
        check(r.severity == "No DR", "severity correct")
        check(abs(r.confidence_pct - 99.89) < 0.001, "confidence_pct correct")
        check(r.confidence_band == "High", "confidence_band correct")
        check(r.referral_priority == "Routine", "referral_priority correct")
        check(r.referral_due == "Routine annual diabetic eye screening",
              "referral_due captured from action_pathway")
        check(r.followup_status == "Pending", "initial followup_status is Pending")
        check(r.followup_notes == "", "initial followup_notes is empty")
        check(r.updated_at == "", "updated_at is empty on initial save")
    finally:
        safe_unlink(db)

    # â”€â”€ TEST 4: update_status() changes status and sets updated_at â”€â”€â”€â”€â”€â”€â”€â”€â”€
    separator("TEST 4: update_status() changes status and sets updated_at")
    db = make_temp_db()
    try:
        rec_id = save_record(
            patient_ref="PHC-TEST-003",
            predicted_class=3,
            severity="Severe NPDR",
            confidence_pct=93.3,
            confidence_band="High",
            quality_score=94.0,
            referral_priority="Urgent",
            referral_due="Prompt ophthalmic referral",
            recommendation="Urgent ophthalmic evaluation.",
            db_path=db,
        )
        update_status(rec_id, "Referred", notes="Referred to city hospital.", db_path=db)
        records = load_records(db_path=db)
        r = records[0]
        print(f"  Status after update: {r.followup_status}, updated_at: {r.updated_at}")
        check(r.followup_status == "Referred", "followup_status updated to Referred")
        check("city hospital" in r.followup_notes, "followup_notes saved")
        check(r.updated_at != "", "updated_at set after update")

        # Update again to Attended
        update_status(rec_id, "Attended", notes="Attended on 2026-10-01.", db_path=db)
        records2 = load_records(db_path=db)
        check(records2[0].followup_status == "Attended", "Second update to Attended")
    finally:
        safe_unlink(db)

    # â”€â”€ TEST 5: invalid status raises ValueError â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    separator("TEST 5: update_status() with invalid status raises ValueError")
    db = make_temp_db()
    try:
        rec_id = save_record(
            patient_ref="PHC-TEST-004",
            predicted_class=1,
            severity="Mild NPDR",
            confidence_pct=75.0,
            confidence_band="High",
            quality_score=88.0,
            referral_priority="Routine / Clinical Review",
            referral_due="Primary care review",
            recommendation="Mild signs detected.",
            db_path=db,
        )
        raised = False
        try:
            update_status(rec_id, "InvalidStatus", db_path=db)
        except ValueError:
            raised = True
        check(raised, "ValueError raised for invalid status string")

        # Also check that load_records rejects invalid filter
        raised2 = False
        try:
            load_records(status_filter="NotAStatus", db_path=db)
        except ValueError:
            raised2 = True
        check(raised2, "ValueError raised for invalid status_filter")
    finally:
        safe_unlink(db)

    # â”€â”€ TEST 6: status_filter returns only matching rows â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    separator("TEST 6: load_records(status_filter='Pending') returns only Pending rows")
    db = make_temp_db()
    try:
        id1 = save_record(
            patient_ref="PHC-A", predicted_class=0, severity="No DR",
            confidence_pct=99.0, confidence_band="High", quality_score=99.0,
            referral_priority="Routine", referral_due="Annual check",
            recommendation="No DR.", db_path=db,
        )
        id2 = save_record(
            patient_ref="PHC-B", predicted_class=4, severity="PDR",
            confidence_pct=90.0, confidence_band="High", quality_score=94.0,
            referral_priority="Urgent", referral_due="Immediate care",
            recommendation="Urgent.", db_path=db,
        )
        update_status(id2, "Referred", db_path=db)

        pending = load_records(status_filter="Pending", db_path=db)
        all_recs = load_records(db_path=db)
        print(f"  All records: {len(all_recs)}, Pending only: {len(pending)}")
        check(len(all_recs) == 2, "Total 2 records in DB")
        check(len(pending) == 1, "Only 1 Pending record returned")
        check(pending[0].patient_ref == "PHC-A", "Correct record filtered")

        # All valid statuses filter without error
        for s in VALID_STATUSES:
            load_records(status_filter=s, db_path=db)
        check(True, "All valid status filters work without error")
    finally:
        safe_unlink(db)

    # â”€â”€ TEST 7: export_csv() produces valid CSV â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    separator("TEST 7: export_csv() produces valid CSV with correct headers and row count")
    db = make_temp_db()
    csv_out = Path(tempfile.mktemp(suffix=".csv"))
    try:
        save_record(
            patient_ref="PHC-EXPORT-1", predicted_class=2, severity="Moderate NPDR",
            confidence_pct=65.0, confidence_band="Moderate", quality_score=90.0,
            referral_priority="Semi-urgent", referral_due="Ophthalmic referral",
            recommendation="Scheduled consult.", db_path=db,
        )
        save_record(
            patient_ref="PHC-EXPORT-2", predicted_class=0, severity="No DR",
            confidence_pct=99.9, confidence_band="High", quality_score=99.0,
            referral_priority="Routine", referral_due="Annual check",
            recommendation="No DR.", db_path=db,
        )
        n = export_csv(out_path=csv_out, db_path=db)
        check(n == 2, "export_csv returns correct row count (2)")
        check(csv_out.exists(), "CSV file created")

        with open(csv_out, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames
            rows = list(reader)

        expected_cols = [
            "record_id", "created_at", "patient_ref",
            "predicted_class", "severity", "confidence_pct",
            "confidence_band", "quality_score",
            "referral_priority", "referral_due", "recommendation",
            "followup_status", "followup_notes", "updated_at",
        ]
        check(set(header) == set(expected_cols), "CSV has all 14 expected columns")
        check("referral_due" in header, "referral_due column in CSV")
        check(len(rows) == 2, "CSV has 2 data rows")
        check(
            any(r["patient_ref"] == "PHC-EXPORT-1" for r in rows),
            "PHC-EXPORT-1 in CSV rows"
        )
    finally:
        safe_unlink(db)
        safe_unlink(csv_out)

    # â”€â”€ TEST 8: empty patient_ref raises ValueError â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    separator("TEST 8: save_record() with empty patient_ref raises ValueError")
    db = make_temp_db()
    try:
        for bad_ref in ["", "   ", "\t"]:
            raised = False
            try:
                save_record(
                    patient_ref=bad_ref, predicted_class=0, severity="No DR",
                    confidence_pct=99.0, confidence_band="High", quality_score=99.0,
                    referral_priority="Routine", referral_due="Annual check",
                    recommendation="No DR.", db_path=db,
                )
            except ValueError:
                raised = True
            check(raised, f"ValueError raised for patient_ref={repr(bad_ref)}")

        # invalid predicted_class
        raised3 = False
        try:
            save_record(
                patient_ref="PHC-X", predicted_class=5, severity="Unknown",
                confidence_pct=50.0, confidence_band="Moderate", quality_score=80.0,
                referral_priority="Routine", referral_due="check",
                recommendation="N/A", db_path=db,
            )
        except ValueError:
            raised3 = True
        check(raised3, "ValueError raised for predicted_class=5 (out of range)")
    finally:
        safe_unlink(db)

    # â”€â”€ TESTS 9â€“11: Streamlit AppTest â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    separator("TESTS 9-11: Streamlit AppTest â€” tracker form and quality gate")
    from PIL import Image, ImageFilter
    from streamlit.testing.v1 import AppTest

    good_img_path = IMAGES_DIR / "165634a6167e.png"
    good_img = Image.open(good_img_path)
    good_bytes = io.BytesIO()
    good_img.save(good_bytes, format="PNG")
    good_bytes.seek(0)

    # TEST 9: No save button before screening starts
    print("\n  [Test 9] App loads â€” no save form before screening...")
    at_init = AppTest.from_file("app.py", default_timeout=90).run()
    assert not at_init.exception, f"Exception on load: {at_init.exception}"
    subheaders_init = [s.value for s in at_init.subheader]
    check(
        not any("Save to Referral Tracker" in s for s in subheaders_init),
        "TEST 9: 'Save to Referral Tracker' not shown before screening"
    )

    # TEST 10: Save section present after good-quality screening
    print("\n  [Test 10] Good image â†’ Save to Tracker section appears...")
    at_good = AppTest.from_file("app.py", default_timeout=60)
    at_good.run()
    at_good.file_uploader[0].upload("165634a6167e.png", good_bytes.getvalue()).run()
    at_good.button[0].click().run()
    assert not at_good.exception, f"Exception during screening: {at_good.exception}"
    subheaders_good = [s.value for s in at_good.subheader]
    print(f"  Subheaders after screening: {subheaders_good}")
    check(
        any("Save to Referral Tracker" in s for s in subheaders_good),
        "TEST 10: 'Save to Referral Tracker' subheader present after screening"
    )
    # All previous sections still present
    check(any("AI Screening Result" in s for s in subheaders_good),
          "TEST 10: AI Screening Result section still present")
    check(any("Clinical Review" in s for s in subheaders_good),
          "TEST 10: Clinical Review section still present")

    # TEST 11: Quality failure â€” no tracker section (st.stop fires)
    print("\n  [Test 11] Quality failure â†’ Tracker section NOT shown...")
    blurred_img = good_img.filter(ImageFilter.GaussianBlur(radius=15))
    blur_bytes = io.BytesIO()
    blurred_img.save(blur_bytes, format="PNG")
    blur_bytes.seek(0)

    at_blur = AppTest.from_file("app.py", default_timeout=60)
    at_blur.run()
    at_blur.file_uploader[0].upload("blurred.png", blur_bytes.getvalue()).run()
    at_blur.button[0].click().run()
    assert not at_blur.exception, f"Exception: {at_blur.exception}"
    subheaders_blur = [s.value for s in at_blur.subheader]
    check(
        not any("Save to Referral Tracker" in s for s in subheaders_blur),
        "TEST 11: Save to Referral Tracker NOT shown after quality failure"
    )
    check(
        any("Quality Inadequate" in e.value for e in at_blur.error),
        "TEST 11: Quality Inadequate error still shown"
    )
    check(len(at_blur.metric) == 0,
          "TEST 11: No metrics shown after quality failure")

    # â”€â”€ TEST 12: Tracker page loads without error â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    separator("TEST 12: Referral_Tracker.py page loads without error")
    at_tracker = AppTest.from_file(
        str(ROOT / "pages" / "Referral_Tracker.py"), default_timeout=60
    ).run()
    assert not at_tracker.exception, f"Tracker page raised: {at_tracker.exception}"
    tracker_title = [t.value for t in at_tracker.title]
    print(f"  Tracker page title: {tracker_title}")
    check(
        any("Referral" in t for t in tracker_title),
        "TEST 12: Referral Tracker page renders title without error"
    )

    # â”€â”€ TEST 13: Representative prediction regression â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    separator("TEST 13: Representative prediction regression (3 images)")
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
        print(f"  {img_path.name}: class {pred_class} ({confidence}%) â€” expected {expected_class} ({expected_conf}%)")
        check(pred_class == expected_class, f"{img_path.name}: class exact match")
        check(abs(confidence - expected_conf) < 0.05, f"{img_path.name}: confidence exact match")

    # â”€â”€ TEST 14: verify_consistency.py 10/10 â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    separator("TEST 14: verify_consistency.py 10/10")
    r14 = subprocess.run(
        [sys.executable, "verify_consistency.py"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=300
    )
    print(r14.stdout[-1500:] if len(r14.stdout) > 1500 else r14.stdout)
    if r14.returncode != 0:
        print(r14.stderr[-500:])
    check(r14.returncode == 0, "TEST 14: verify_consistency.py exits 0")
    check("10/10" in r14.stdout or "100%" in r14.stdout, "TEST 14: 10/10 consistency")

    # â”€â”€ TEST 15: Step 1 quality gate tests â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    separator("TEST 15: test_quality_gate_phase1.py all pass")
    r15 = subprocess.run(
        [sys.executable, "test_quality_gate_phase1.py"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=300,
        env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"}
    )
    print(r15.stdout[-1000:] if len(r15.stdout) > 1000 else r15.stdout)
    if r15.returncode != 0:
        print(r15.stderr[-500:])
    check(r15.returncode == 0, "TEST 15: test_quality_gate_phase1.py exits 0")
    check("ALL TESTS PASSED" in r15.stdout, "TEST 15: All Step 1 tests pass")

    # â”€â”€ TEST 16: Existing AppTest suite â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    separator("TEST 16: test_app_ui.py â€” existing AppTest suite still passes")
    r16 = subprocess.run(
        [sys.executable, "test_app_ui.py"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=300
    )
    print(r16.stdout[-1000:] if len(r16.stdout) > 1000 else r16.stdout)
    if r16.returncode != 0:
        print(r16.stderr[-500:])
    check(r16.returncode == 0, "TEST 16: test_app_ui.py exits 0")
    check("ALL STREAMLIT APP TESTS PASSED" in r16.stdout,
          "TEST 16: All existing AppTest assertions pass")

    # â”€â”€ Post-flight hash re-check â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    separator("POST-FLIGHT HASH RE-VERIFICATION")
    check(sha256_file(ROOT / "checkpoints" / "best_model_combined_v1.pth") == V1_SHA,
          "V1 model SHA still unchanged")
    check(sha256_file(ROOT / "dataset" / "test.csv") == CSV_SHA,
          "Frozen test.csv SHA still unchanged")

    separator("ALL 16 TESTS PASSED")


if __name__ == "__main__":
    run_tests()

