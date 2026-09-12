"""
test_gradcam.py
---------------
Verifies the Grad-CAM explainability module on real APTOS images.
Integrates with the Quality Gate (quality_assessment.py).
Saves side-by-side visualizations to outputs/gradcam/.
"""

import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter
from quality_assessment import RetinalQualityAssessor
from gradcam import GradCAM, CLASS_NAMES

def main():
    root = Path(__file__).parent.resolve()
    images_dir = root / "dataset" / "images"
    ckpt_path = root / "checkpoints" / "best_model_combined_v1.pth"
    output_dir = root / "outputs" / "gradcam"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  TESTING GRAD-CAM EXPLAINABILITY PIPELINE (EFFICIENTNET-B0)")
    print("=" * 70)

    # 1. Initialize Modules
    print("\n[1] Initializing Quality Assessor and Grad-CAM Engine...")
    quality_assessor = RetinalQualityAssessor()
    gradcam = GradCAM(checkpoint_path=ckpt_path)
    print(f"    Checkpoint loaded: {ckpt_path.name}")
    print(f"    Target layer     : {gradcam.target_layer.__class__.__name__} (features[8])")

    # 2. Test Images list from requirements
    test_filenames = [
        "4f0866b90c27.png",
        "fe674c2f73f5.png",
        "165634a6167e.png",
        "b16dd4483ca5.png",
        "1a7e3356b39c.png",
        "521d3e264d71.png",
    ]

    print("\n[2] Processing Real Dataset Images...")
    all_passed = True

    for filename in test_filenames:
        img_path = images_dir / filename
        print(f"\nEvaluating: {filename}")

        if not img_path.exists():
            print(f"  WARNING: File not found: {img_path}")
            all_passed = False
            continue

        # Step A: Quality Gate Check
        q_res = quality_assessor.assess_image(img_path)
        if not q_res["quality_ok"]:
            print(f"  Quality Status   : REJECTED (Score: {q_res['quality_score']})")
            print(f"  Grad-CAM skipped because image quality is inadequate.")
            print(f"  Reason           : {q_res['warnings']}")
            continue

        print(f"  Quality Status   : ACCEPTED (Score: {q_res['quality_score']})")

        # Step B: Generate Grad-CAM
        try:
            cam_res = gradcam.generate(img_path)
            pred_class = cam_res["predicted_class"]
            pred_name = cam_res["class_name"]
            confidence = cam_res["confidence"]
            h_w, h_h = cam_res["heatmap_size"]
            raw_hm = cam_res["raw_heatmap"]

            # Sanity checks on heatmap
            is_non_trivial = (raw_hm.max() > raw_hm.min()) and (not np.isnan(raw_hm).any())
            print(f"  Predicted Class  : Class {pred_class} ({pred_name})")
            print(f"  Model Confidence : {confidence:.2f}%")
            print(f"  Heatmap Size     : {h_w} x {h_h} px")
            print(f"  Heatmap Dynamic  : min={raw_hm.min():.3f}, max={raw_hm.max():.3f}, non-trivial={is_non_trivial}")

            # Save Side-by-side visualization
            out_filename = f"gradcam_{Path(filename).stem}_class{pred_class}.png"
            out_path = output_dir / out_filename
            gradcam.save_visualization(cam_res, out_path, mode="side_by_side")
            print(f"  Saved Visual     : {out_path.relative_to(root)}")

            if not is_non_trivial:
                print("  ERROR: Generated heatmap is trivial (constant/NaN)!")
                all_passed = False

        except Exception as e:
            print(f"  ERROR generating Grad-CAM: {e}")
            all_passed = False

    # 3. Test Quality-Gate Fallback on an Inadequate Image (e.g., Heavy Blur)
    print("\n[3] Testing Quality Gate Rejection Scenario...")
    sample_img_path = images_dir / test_filenames[0]
    orig_img = Image.open(sample_img_path)
    blurred_img = orig_img.filter(ImageFilter.GaussianBlur(radius=15))

    q_blur = quality_assessor.assess_image(blurred_img)
    print(f"Inadequate Image Quality Check (Artificially Blurred):")
    if not q_blur["quality_ok"]:
        print(f"  Quality Status   : REJECTED (Score: {q_blur['quality_score']})")
        print(f"  Grad-CAM skipped because image quality is inadequate.")
        print(f"  Quality Gate Integration verified: Inadequate image correctly prevented from Grad-CAM execution.")
    else:
        print("  ERROR: Quality gate failed to block blurred image!")
        all_passed = False

    # 4. Cleanup hooks
    gradcam.remove_hooks()

    print("\n" + "=" * 70)
    if all_passed:
        print("  ALL GRAD-CAM TESTS AND INTEGRATION CHECKS PASSED SUCCESSFULLY")
    else:
        print("  SOME GRAD-CAM CHECKS ENCOUNTERED ISSUES")
    print("=" * 70)

if __name__ == "__main__":
    main()
