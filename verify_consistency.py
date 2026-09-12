"""
verify_consistency.py
---------------------
Rigorous verification that:
  1. Standalone evaluation script logic
  2. DRScreeningPipeline (pipeline.py)
  3. Streamlit UI AppTest (app.py)
produce the exact same predictions and probabilities for test images from frozen APTOS test set.
"""

from pathlib import Path
import io
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
from streamlit.testing.v1 import AppTest

from pipeline import DRScreeningPipeline

ROOT = Path(__file__).parent.resolve()
CKPT_PATH = ROOT / "checkpoints" / "best_model_combined_v1.pth"
TEST_CSV = ROOT / "dataset" / "test.csv"
IMAGES_DIR = ROOT / "dataset" / "images"

CLASS_NAMES = {
    0: "No DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "PDR",
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

infer_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

def load_standalone_model():
    model = models.efficientnet_b0(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(1280, 5),
    )
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model

def standalone_inference(model, img_path):
    img = Image.open(img_path).convert("RGB")
    tensor = infer_transforms(img).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    pred_class = int(probs.argmax())
    confidence = float(probs[pred_class])
    probabilities = [round(float(p), 4) for p in probs]
    return pred_class, confidence, probabilities

def main():
    print("=" * 80)
    print("  CONSISTENCY VERIFICATION: STANDALONE EVALUATION vs PIPELINE vs UI")
    print("  Checkpoint: checkpoints/best_model_combined_v1.pth")
    print("=" * 80)

    # 1. Select representative test images across diverse ground-truth classes
    df_test = pd.read_csv(TEST_CSV)
    
    # Pick at least 2 images for each class (0 to 4) from the frozen test set
    selected_samples = []
    for c in range(5):
        subset = df_test[df_test["diagnosis"] == c]
        for _, row in subset.head(2).iterrows():
            selected_samples.append((row["id_code"], int(row["diagnosis"])))

    print(f"\nSelected {len(selected_samples)} test images across classes 0-4 for comparison:\n")

    standalone_model = load_standalone_model()
    pipeline = DRScreeningPipeline(checkpoint_path=CKPT_PATH)

    all_matched = True

    print(f"{'Image ID':<15} {'GT':<4} {'Standalone Pred (Conf)':<24} {'Pipeline Pred (Conf)':<24} {'Match?':<8}")
    print("-" * 80)

    for id_code, gt_class in selected_samples:
        img_path = IMAGES_DIR / f"{id_code}.png"
        assert img_path.exists(), f"Missing test image: {img_path}"

        # 1. Standalone evaluation prediction
        s_class, s_conf, s_probs = standalone_inference(standalone_model, img_path)
        s_name = CLASS_NAMES[s_class]

        # 2. Pipeline prediction
        pipe_res = pipeline.process(img_path, generate_gradcam=True, save_visualizations=False)
        p_pred = pipe_res["prediction"]
        assert p_pred is not None, f"Quality failed unexpectedly on valid test image {id_code}"
        p_class = p_pred["class"]
        p_name = p_pred["severity"]
        p_conf = p_pred["confidence"]
        p_probs = p_pred["probabilities"]

        # Comparison checks
        class_match = (s_class == p_class)
        conf_match = abs(s_conf - p_conf) < 1e-4
        probs_match = all(abs(sp - pp) < 1e-4 for sp, pp in zip(s_probs, p_probs))
        name_match = (s_name == p_name)

        overall_match = class_match and conf_match and probs_match and name_match
        if not overall_match:
            all_matched = False

        status_str = "PASS" if overall_match else "FAIL"
        s_summary = f"Class {s_class} ({s_conf*100:.2f}%)"
        p_summary = f"Class {p_class} ({p_conf*100:.2f}%)"
        print(f"{id_code:<15} {gt_class:<4} {s_summary:<24} {p_summary:<24} {status_str:<8}")

        if not overall_match:
            print(f"  [MISMATCH DETAIL]")
            print(f"    Standalone: class={s_class}, conf={s_conf:.6f}, probs={s_probs}")
            print(f"    Pipeline  : class={p_class}, conf={p_conf:.6f}, probs={p_probs}")

    assert all_matched, "Standalone and Pipeline predictions differed!"
    print("\n" + "=" * 80)
    print("  ALL STANDALONE EVALUATION vs PIPELINE COMPARISONS MATCHED 100%!")
    print("=" * 80)

    # 3. Test through Streamlit AppTest on a sample image to verify UI display
    print("\nVerifying Streamlit UI rendering on test image 165634a6167e.png...")
    sample_path = IMAGES_DIR / "165634a6167e.png"
    sample_img = Image.open(sample_path)
    b = io.BytesIO()
    sample_img.save(b, format="PNG")
    b.seek(0)

    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()
    at.file_uploader[0].upload("165634a6167e.png", b.getvalue()).run()
    at.button[0].click().run()

    metric_dict = {m.label: m.value for m in at.metric}
    print(f"  UI Rendered Metrics: {metric_dict}")

    s_class, s_conf, _ = standalone_inference(standalone_model, sample_path)
    expected_severity = f"Class {s_class} ({CLASS_NAMES[s_class]})"
    expected_conf = f"{s_conf*100:.2f}%"

    assert metric_dict.get("Predicted Severity") == expected_severity, (
        f"UI Severity mismatch! Expected: {expected_severity}, Got: {metric_dict.get('Predicted Severity')}"
    )
    assert metric_dict.get("Model Confidence") == expected_conf, (
        f"UI Confidence mismatch! Expected: {expected_conf}, Got: {metric_dict.get('Model Confidence')}"
    )

    print("  UI metrics match standalone evaluation exactly!")
    print("=" * 80)

if __name__ == "__main__":
    main()
