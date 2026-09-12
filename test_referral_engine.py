"""
test_referral_engine.py
-----------------------
Test suite for referral_engine.py verifying:
1. Class 0, high confidence, good quality
2. Class 1, high confidence, good quality
3. Class 2, moderate confidence, good quality
4. Class 3, moderate confidence, good quality
5. Class 4, high confidence, good quality
6. Low-confidence prediction
7. Inadequate image quality
8. Invalid class
9. Invalid confidence
10. Missing quality information
11. End-to-end evaluation using actual APTOS test set image predictions
"""

import json
from pathlib import Path
import torch
from referral_engine import DRReferralEngine, ReferralConfig, ReferralResult
from quality_assessment import RetinalQualityAssessor
from gradcam import load_model, infer_transforms, CLASS_NAMES
from PIL import Image

def run_tests():
    engine = DRReferralEngine()
    print("=" * 70)
    print("  TESTING CLINICAL REFERRAL-SUPPORT ENGINE")
    print("=" * 70)

    # ── Test 1: Class 0, High Confidence, Good Quality ─────────────
    t1 = engine.evaluate(predicted_class=0, confidence=0.96, quality_status=True)
    print("\n[Test 1] Class 0, High Confidence, Good Quality")
    print(f"  Priority   : {t1.referral_priority}")
    print(f"  Confidence : {t1.confidence_level} ({t1.confidence:.2f})")
    print(f"  Action     : {t1.action_pathway}")
    assert t1.referral_priority == "Routine"
    assert t1.confidence_level == "High"
    assert len(t1.warnings) == 0

    # ── Test 2: Class 1, High Confidence, Good Quality ─────────────
    t2 = engine.evaluate(predicted_class=1, confidence=0.98, quality_status="Adequate")
    print("\n[Test 2] Class 1, High Confidence, Good Quality")
    print(f"  Priority   : {t2.referral_priority}")
    print(f"  Action     : {t2.action_pathway}")
    assert "Routine" in t2.referral_priority or "Clinical Review" in t2.referral_priority
    assert t2.confidence_level == "High"

    # ── Test 3: Class 2, Moderate Confidence, Good Quality ─────────
    t3 = engine.evaluate(predicted_class=2, confidence=0.62, quality_status=True)
    print("\n[Test 3] Class 2, Moderate Confidence, Good Quality")
    print(f"  Priority   : {t3.referral_priority}")
    print(f"  Action     : {t3.action_pathway}")
    assert t3.referral_priority == "Semi-urgent"
    assert t3.confidence_level == "Moderate"

    # ── Test 4: Class 3, Moderate Confidence, Good Quality ─────────
    t4 = engine.evaluate(predicted_class=3, confidence=0.55, quality_status=True)
    print("\n[Test 4] Class 3, Moderate Confidence, Good Quality")
    print(f"  Priority   : {t4.referral_priority}")
    print(f"  Action     : {t4.action_pathway}")
    assert t4.referral_priority == "Urgent"
    assert t4.confidence_level == "Moderate"

    # ── Test 5: Class 4, High Confidence, Good Quality ─────────────
    t5 = engine.evaluate(predicted_class=4, confidence=0.88, quality_status=True)
    print("\n[Test 5] Class 4, High Confidence, Good Quality")
    print(f"  Priority   : {t5.referral_priority}")
    print(f"  Action     : {t5.action_pathway}")
    assert t5.referral_priority == "Urgent"
    assert t5.confidence_level == "High"

    # ── Test 6: Low-confidence prediction (< 0.50) ─────────────────
    t6 = engine.evaluate(predicted_class=2, confidence=0.38, quality_status=True)
    print("\n[Test 6] Low Confidence Prediction")
    print(f"  Priority   : {t6.referral_priority}")
    print(f"  Conf Level : {t6.confidence_level}")
    print(f"  Warnings   : {t6.warnings}")
    assert t6.confidence_level == "Low"
    assert any("Low model confidence" in w for w in t6.warnings)

    # ── Test 7: Inadequate Image Quality ───────────────────────────
    t7 = engine.evaluate(
        predicted_class=4,
        confidence=0.92,
        quality_status=False,
        quality_warnings=["Severe blur detected (score: 1.2)."]
    )
    print("\n[Test 7] Inadequate Image Quality Interlock")
    print(f"  Quality Status   : {t7.quality_status}")
    print(f"  Priority         : {t7.referral_priority}")
    print(f"  Recommendation   : {t7.recommendation}")
    print(f"  Warnings         : {t7.warnings}")
    assert t7.quality_status == "Inadequate"
    assert t7.referral_priority == "Recapture Required"
    assert "Recapture" in t7.recommendation

    # ── Test 8: Invalid Class ──────────────────────────────────────
    t8 = engine.evaluate(predicted_class=99, confidence=0.85, quality_status=True)
    print("\n[Test 8] Invalid Class Input")
    print(f"  Priority   : {t8.referral_priority}")
    print(f"  Warnings   : {t8.warnings}")
    assert t8.predicted_class is None
    assert "Invalid" in t8.warnings[0]

    # ── Test 9: Invalid Confidence Value ───────────────────────────
    t9 = engine.evaluate(predicted_class=0, confidence="not_a_number", quality_status=True)
    print("\n[Test 9] Invalid Confidence Input")
    print(f"  Confidence : {t9.confidence}")
    print(f"  Conf Level : {t9.confidence_level}")
    print(f"  Warnings   : {t9.warnings}")
    assert t9.confidence == 0.0
    assert t9.confidence_level == "Low"

    # ── Test 10: Missing Quality Information ───────────────────────
    t10 = engine.evaluate(predicted_class=1, confidence=0.80, quality_status=None)
    print("\n[Test 10] Missing Quality Information Fallback")
    print(f"  Quality Status   : {t10.quality_status}")
    print(f"  Priority         : {t10.referral_priority}")
    print(f"  Warnings         : {t10.warnings}")
    assert t10.quality_status == "Inadequate"
    assert t10.referral_priority == "Recapture Required"

    # ── Test 11: Real APTOS Test Images with Quality + Model ───────
    print("\n[Test 11] Running Referral Engine on Real APTOS Test Images...")
    root = Path(__file__).parent.resolve()
    images_dir = root / "dataset" / "images"
    ckpt_path = root / "checkpoints" / "best_model.pth"
    
    assessor = RetinalQualityAssessor()
    model = load_model(ckpt_path, torch.device("cpu"))
    
    sample_images = [
        ("4f0866b90c27.png", 3),  # Severe NPDR
        ("fe674c2f73f5.png", 1),  # Mild NPDR
        ("165634a6167e.png", 0),  # No DR
        ("1a7e3356b39c.png", 4),  # Proliferative DR
    ]
    
    for filename, expected_class in sample_images:
        path = images_dir / filename
        q_res = assessor.assess_image(path)
        
        # Inference
        img = Image.open(path).convert("RGB")
        tensor = infer_transforms(img).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(model(tensor), dim=1).squeeze(0)
            pred_cls = int(torch.argmax(probs).item())
            conf = float(probs[pred_cls].item())

        ref_res = engine.evaluate(
            predicted_class=pred_cls,
            confidence=conf,
            quality_status=q_res
        )

        print(f"\nImage: {filename}")
        print(f"  Quality Score    : {q_res['quality_score']}/100 ({q_res['quality_ok']})")
        print(f"  Model Prediction : Class {pred_cls} ({ref_res.severity})")
        print(f"  Model Confidence : {conf*100:.1f}% ({ref_res.confidence_level})")
        print(f"  Referral Priority: {ref_res.referral_priority}")
        print(f"  Pathway          : {ref_res.action_pathway}")
        print(f"  Summary          : {ref_res.human_readable_summary}")
        if ref_res.warnings:
            print(f"  Warnings         : {ref_res.warnings}")

    print("\n" + "=" * 70)
    print("  ALL 11 REFERRAL ENGINE TESTS COMPLETED SUCCESSFULLY")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()
