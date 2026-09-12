"""
test_quality_assessment.py
--------------------------
Tests the RetinalQualityAssessor on a sample of images from dataset/test.csv,
as well as edge cases (synthetic blurred, synthetic dark, synthetic blank, corrupted path).
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image, ImageFilter
from quality_assessment import RetinalQualityAssessor, QualityConfig

def run_tests():
    assessor = RetinalQualityAssessor()
    root = Path(__file__).parent.resolve()
    test_csv = root / "dataset" / "test.csv"
    images_dir = root / "dataset" / "images"

    print("=" * 70)
    print("  TESTING RETINAL IMAGE QUALITY ASSESSMENT MODULE")
    print("=" * 70)

    # 1. Test on Real Dataset Images (5 samples from test set across classes)
    df = pd.read_csv(test_csv)
    sample_rows = df.iloc[:8]

    print("\n--- 1. Evaluating Real Dataset Images ---")
    for idx, row in sample_rows.iterrows():
        img_path = images_dir / f"{row['id_code']}.png"
        res = assessor.assess_image(img_path)
        status = "PASSED [OK]" if res["quality_ok"] else "REJECTED [FAIL]"
        print(f"\nImage: {row['id_code']}.png (Class: {row['diagnosis']})")
        print(f"  Status          : {status}")
        print(f"  Quality Score   : {res['quality_score']}/100")
        print(f"  Resolution      : {res['resolution'][0]}x{res['resolution'][1]} px")
        print(f"  Sharpness / Blur: {res['blur_score']} (subscore: {res['subscores']['sharpness']})")
        print(f"  Brightness      : {res['brightness_score']} (subscore: {res['subscores']['illumination']})")
        print(f"  Contrast        : {res['contrast_score']} (subscore: {res['subscores']['contrast']})")
        print(f"  Field of View   : {res['field_of_view_score']}% (subscore: {res['subscores']['field_of_view']})")
        print(f"  Recommendation  : {res['recommendation']}")
        if res["warnings"]:
            print(f"  Warnings        : {res['warnings']}")

    # 2. Test Robustness / Edge Cases:
    print("\n--- 2. Evaluating Edge Cases (Blur, Low Light, Non-Existent, Corrupt) ---")

    # A. Non-existent file
    bad_path = root / "dataset" / "images" / "non_existent_image_123.png"
    res_bad = assessor.assess_image(bad_path)
    print(f"\nNon-existent File:")
    print(f"  Quality OK : {res_bad['quality_ok']}")
    print(f"  Warnings   : {res_bad['warnings']}")
    print(f"  Rec        : {res_bad['recommendation']}")
    assert not res_bad["quality_ok"], "Non-existent file should fail!"

    # B. Artificially Heavily Blurred Image
    sample_img_path = images_dir / f"{df.iloc[0]['id_code']}.png"
    orig_img = Image.open(sample_img_path)
    blurred_img = orig_img.filter(ImageFilter.GaussianBlur(radius=15))
    res_blur = assessor.assess_image(blurred_img)
    print(f"\nArtificially Heavy Blurred Image:")
    print(f"  Quality OK : {res_blur['quality_ok']}")
    print(f"  Score      : {res_blur['quality_score']}")
    print(f"  Blur Score : {res_blur['blur_score']}")
    print(f"  Warnings   : {res_blur['warnings']}")
    print(f"  Rec        : {res_blur['recommendation']}")

    # C. Artificially Dark / Underexposed Image
    dark_img = Image.fromarray((np.array(orig_img) * 0.1).astype(np.uint8))
    res_dark = assessor.assess_image(dark_img)
    print(f"\nArtificially Dark / Underexposed Image:")
    print(f"  Quality OK : {res_dark['quality_ok']}")
    print(f"  Score      : {res_dark['quality_score']}")
    print(f"  Brightness : {res_dark['brightness_score']}")
    print(f"  Warnings   : {res_dark['warnings']}")
    print(f"  Rec        : {res_dark['recommendation']}")

    # D. Blank Black Image
    blank_img = Image.new("RGB", (1024, 1024), color=(0, 0, 0))
    res_blank = assessor.assess_image(blank_img)
    print(f"\nBlank Image:")
    print(f"  Quality OK : {res_blank['quality_ok']}")
    print(f"  Score      : {res_blank['quality_score']}")
    print(f"  Warnings   : {res_blank['warnings']}")
    print(f"  Rec        : {res_blank['recommendation']}")

    print("\n" + "=" * 70)
    print("  ALL QUALITY ASSESSMENT TESTS COMPLETED SUCCESSFULLY")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()
