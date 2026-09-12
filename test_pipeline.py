"""
test_pipeline.py
----------------
Verification suite for pipeline.py covering:
  TEST 1: Good-quality No DR image
  TEST 2: Good-quality DR image
  TEST 3: Severe/PDR image
  TEST 4: Artificially blurred image (Quality rejection gate)
  TEST 5: Missing image (Graceful error handling)
  TEST 6: Consistency verification against predict.py on multiple images
  TEST 7: Determinism test (Repeat evaluation on same image)
  TEST 8: Verify Grad-CAM output exists and is non-empty for accepted images
  TEST 9: Verify no referral recommendation exists for rejected-quality images
  TEST 10: 10-image real dataset benchmark (metrics, latency, distribution)
"""

from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
import torch

from pipeline import DRScreeningPipeline
from predict import build_model, infer_transforms, BEST_CKPT

def run_tests():
    root = Path(__file__).parent.resolve()
    images_dir = root / "dataset" / "images"
    test_csv = root / "dataset" / "test.csv"
    outputs_dir = root / "outputs" / "pipeline"

    print("=" * 75)
    print("  VERIFYING UNIFIED OFFLINE DIABETIC RETINOPATHY SCREENING PIPELINE")
    print("=" * 75)

    pipeline = DRScreeningPipeline()
    df_test = pd.read_csv(test_csv)

    # ── TEST 1: Good-quality No DR image ─────────────────────────────
    print("\n--- TEST 1: Good-Quality No DR Image (165634a6167e.png) ---")
    img_no_dr = images_dir / "165634a6167e.png"
    res1 = pipeline.process(img_no_dr)
    print(f"  Overall Status : {res1['overall_status']}")
    print(f"  Quality OK     : {res1['quality']['quality_ok']} (Score: {res1['quality']['quality_score']})")
    print(f"  Predicted Class: {res1['prediction']['class']} ({res1['prediction']['severity']})")
    print(f"  Confidence     : {res1['prediction']['confidence']*100:.1f}%")
    print(f"  Grad-CAM Path  : {res1['explainability']['output_path']}")
    print(f"  Referral Pri   : {res1['referral']['referral_priority']}")
    print(f"  Pathway        : {res1['referral']['action_pathway']}")

    assert res1['overall_status'] == "SCREENING_COMPLETE"
    assert res1['quality']['quality_ok'] is True
    assert res1['prediction'] is not None
    assert res1['explainability']['gradcam_generated'] is True
    assert res1['referral'] is not None
    assert Path(res1['explainability']['output_path']).exists()

    # ── TEST 2: Good-quality DR image ────────────────────────────────
    print("\n--- TEST 2: Good-Quality Mild DR Image (fe674c2f73f5.png) ---")
    img_mild = images_dir / "fe674c2f73f5.png"
    res2 = pipeline.process(img_mild)
    print(f"  Overall Status : {res2['overall_status']}")
    print(f"  Predicted Class: {res2['prediction']['class']} ({res2['prediction']['severity']})")
    print(f"  Referral Pri   : {res2['referral']['referral_priority']}")

    assert res2['overall_status'] == "SCREENING_COMPLETE"
    assert res2['prediction']['class'] == 1
    assert res2['referral']['referral_priority'] == "Routine / Clinical Review"

    # ── TEST 3: Severe/PDR image ─────────────────────────────────────
    print("\n--- TEST 3: Severe/PDR Image (4f0866b90c27.png) ---")
    img_severe = images_dir / "4f0866b90c27.png"
    res3 = pipeline.process(img_severe)
    print(f"  Overall Status : {res3['overall_status']}")
    print(f"  Predicted Class: {res3['prediction']['class']} ({res3['prediction']['severity']})")
    print(f"  Referral Pri   : {res3['referral']['referral_priority']}")
    print(f"  Pathway        : {res3['referral']['action_pathway']}")

    assert res3['overall_status'] == "SCREENING_COMPLETE"
    assert res3['prediction']['class'] in (3, 4)
    assert res3['referral']['referral_priority'] == "Urgent"

    # ── TEST 4: Artificially Blurred Image (Quality Gate Interlock) ──
    print("\n--- TEST 4: Artificially Blurred Image Quality Rejection ---")
    orig_img = Image.open(img_no_dr)
    blurred_img = orig_img.filter(ImageFilter.GaussianBlur(radius=15))
    res4 = pipeline.process(blurred_img)

    print(f"  Overall Status : {res4['overall_status']}")
    print(f"  Message        : {res4['message']}")
    print(f"  Quality OK     : {res4['quality']['quality_ok']}")
    print(f"  Prediction     : {res4['prediction']}")
    print(f"  Explainability : {res4['explainability']}")
    print(f"  Referral       : {res4['referral']}")

    assert res4['overall_status'] == "RECAPTURE_REQUIRED"
    assert res4['quality']['quality_ok'] is False
    assert res4['prediction'] is None
    assert res4['explainability']['gradcam_generated'] is False
    assert res4['referral'] is None
    assert "recapture required" in res4['message'].lower()

    # ── TEST 5: Missing Image (Graceful Error Handling) ──────────────
    print("\n--- TEST 5: Missing Image File Handling ---")
    missing_path = images_dir / "non_existent_retina_image_99999.png"
    res5 = pipeline.process(missing_path)

    print(f"  Overall Status : {res5['overall_status']}")
    print(f"  Error Type     : {res5.get('error_type')}")
    print(f"  Message        : {res5['message']}")
    assert res5['overall_status'] == "ERROR"
    assert res5['error_type'] == "FileNotFoundError"

    # ── TEST 6: Consistency Against predict.py Logic ────────────────
    print("\n--- TEST 6: Consistency Verification Against predict.py ---")
    model_predict = build_model()
    ckpt = torch.load(BEST_CKPT, map_location="cpu")
    model_predict.load_state_dict(ckpt["model_state"])
    model_predict.eval()

    sample_test_ids = ["4f0866b90c27", "fe674c2f73f5", "165634a6167e", "1a7e3356b39c", "521d3e264d71"]
    for id_code in sample_test_ids:
        path = images_dir / f"{id_code}.png"
        # 1. Pipeline result
        res_p = pipeline.process(path, generate_gradcam=False)
        pipe_cls = res_p["prediction"]["class"]
        pipe_conf = res_p["prediction"]["confidence"]

        # 2. Direct predict.py logic
        pil_img = Image.open(path).convert("RGB")
        tensor = infer_transforms(pil_img).unsqueeze(0)
        with torch.no_grad():
            logits = model_predict(tensor)
            probs = torch.softmax(logits, dim=1).squeeze().numpy()
        pred_cls = int(probs.argmax())
        pred_conf = float(probs[pred_cls])

        print(f"  Image: {id_code}.png")
        print(f"    Pipeline   : Class {pipe_cls}, Conf {pipe_conf*100:.2f}%")
        print(f"    predict.py : Class {pred_cls}, Conf {pred_conf*100:.2f}%")

        assert pipe_cls == pred_cls, f"Class mismatch on {id_code}!"
        assert abs(pipe_conf - pred_conf) < 1e-4, f"Confidence mismatch on {id_code}!"

    # ── TEST 7: Determinism on Repeated Execution ────────────────────
    print("\n--- TEST 7: Determinism on Repeated Execution ---")
    res7_a = pipeline.process(img_no_dr, generate_gradcam=False)
    res7_b = pipeline.process(img_no_dr, generate_gradcam=False)
    assert res7_a["prediction"]["class"] == res7_b["prediction"]["class"]
    assert res7_a["prediction"]["confidence"] == res7_b["prediction"]["confidence"]
    assert res7_a["quality"]["quality_score"] == res7_b["quality"]["quality_score"]
    print("  Deterministic execution confirmed across repeated runs.")

    # ── TEST 8: Grad-CAM Output Exists and Non-Empty ─────────────────
    print("\n--- TEST 8: Verify Grad-CAM Output Exists and Non-Empty ---")
    g_path = Path(res1['explainability']['output_path'])
    assert g_path.exists(), "Grad-CAM file was not saved!"
    assert g_path.stat().st_size > 10000, "Grad-CAM file appears truncated or empty!"
    print(f"  Grad-CAM file verified: {g_path.name} ({g_path.stat().st_size:,} bytes)")

    # ── TEST 9: Verify No Referral for Inadequate Quality ────────────
    print("\n--- TEST 9: Verify No Referral for Rejected Image ---")
    assert res4["referral"] is None
    print("  Confirmed: Referral recommendation is strictly suppressed for inadequate quality.")

    # ── TEST 10: 10-Image Real Dataset Benchmark ─────────────────────
    print("\n--- TEST 10: 10-Image Real Dataset Engineering Benchmark ---")
    ten_rows = df_test.iloc[:10]
    accepted_count = 0
    rejected_count = 0
    latencies = []
    class_dist = {}

    for idx, row in ten_rows.iterrows():
        p = images_dir / f"{row['id_code']}.png"
        res_bench = pipeline.process(p, generate_gradcam=True, save_visualizations=True)
        t_sec = res_bench["processing_time_sec"]

        if res_bench["overall_status"] == "SCREENING_COMPLETE":
            accepted_count += 1
            latencies.append(t_sec)
            c = res_bench["prediction"]["class"]
            class_dist[c] = class_dist.get(c, 0) + 1
            print(f"  [{idx+1}/10] {row['id_code']}.png -> Class {c} ({res_bench['prediction']['severity']}) in {t_sec:.2f}s")
        else:
            rejected_count += 1
            print(f"  [{idx+1}/10] {row['id_code']}.png -> REJECTED (Reason: {res_bench['message']}) in {t_sec:.2f}s")

    avg_time = np.mean(latencies) if latencies else 0.0
    print("\n  Benchmark Summary (10 Images):")
    print(f"    Accepted Images     : {accepted_count} / 10")
    print(f"    Rejected Images     : {rejected_count} / 10")
    print(f"    Avg Processing Time : {avg_time:.2f} seconds/image (CPU)")
    print(f"    Class Distribution  : {sorted(class_dist.items())}")

    print("\n" + "=" * 75)
    print("  ALL 10 PIPELINE INTEGRATION TESTS COMPLETED SUCCESSFULLY")
    print("=" * 75)

if __name__ == "__main__":
    run_tests()
