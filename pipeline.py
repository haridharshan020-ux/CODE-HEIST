"""
pipeline.py
-----------
Unified, Offline-Capable AI Screening Pipeline for Diabetic Retinopathy.

Architecture Flow:
  INPUT FUNDUS IMAGE
          |
  IMAGE QUALITY ASSESSMENT (quality_assessment.py)
          |
  Quality Inadequate?
     YES -> Halt, set status RECAPTURE_REQUIRED, return recapture guidance.
            (NO classification, NO Grad-CAM, NO referral recommendation generated)
     NO  -> Proceed to Screening:
              1. EfficientNet-B0 Inference (same preprocessing as predict.py)
              2. Grad-CAM Explainable Attention Map (gradcam.py)
              3. Clinical Referral-Support Decision Engine (referral_engine.py)
              4. Return Complete Structured Result

Strictly Offline: No internet calls or external network dependencies.
Research/Prototype: Decision support prototype, not an autonomous diagnostic replacement.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Union
import time
import torch
from PIL import Image

from quality_assessment import RetinalQualityAssessor, QualityConfig
from gradcam import GradCAM, CLASS_NAMES
from referral_engine import DRReferralEngine, ReferralConfig, ReferralResult


class DRScreeningPipeline:
    """
    Integrates Quality Gate, EfficientNet-B0 classifier, Grad-CAM, and Referral Engine.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        device: Optional[torch.device] = None,
        quality_config: Optional[QualityConfig] = None,
        referral_config: Optional[ReferralConfig] = None
    ):
        root = Path(__file__).parent.resolve()
        self.checkpoint_path = Path(checkpoint_path or (root / "checkpoints" / "best_model_combined_v1.pth"))
        self.output_dir = Path(output_dir or (root / "outputs" / "pipeline"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize modular components
        self.quality_assessor = RetinalQualityAssessor(config=quality_config)
        self.referral_engine = DRReferralEngine(config=referral_config)

        # GradCAM initializes and wraps the exact model architecture & checkpoint
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found at: {self.checkpoint_path}")
        self.gradcam = GradCAM(checkpoint_path=self.checkpoint_path, device=self.device)

    def process(
        self,
        image_input: Union[str, Path, Image.Image],
        generate_gradcam: bool = True,
        save_visualizations: bool = True
    ) -> Dict[str, Any]:
        """
        Executes end-to-end screening evaluation on a single fundus image.

        Parameters:
            image_input: File path or PIL Image object
            generate_gradcam: Whether to create Grad-CAM heatmap if quality passes
            save_visualizations: Whether to save visualization artifact to outputs/pipeline/

        Returns:
            Structured dictionary matching screening requirements.
        """
        t_start = time.perf_counter()
        image_name = "in_memory_image"
        if isinstance(image_input, (str, Path)):
            image_path = Path(image_input)
            image_name = image_path.name
            if not image_path.exists():
                return self._error_response(
                    f"Image file not found: {image_path}",
                    image_path=str(image_path),
                    error_type="FileNotFoundError"
                )
        elif isinstance(image_input, Image.Image):
            image_path = None
        else:
            return self._error_response(
                f"Unsupported image input type: {type(image_input)}",
                image_path=None,
                error_type="TypeError"
            )

        # ── STEP 1: IMAGE QUALITY ASSESSMENT ──────────────────────────
        try:
            quality_res = self.quality_assessor.assess_image(image_input)
        except Exception as e:
            return self._error_response(
                f"Quality assessment failed with error: {str(e)}",
                image_path=str(image_path) if image_path else None,
                error_type="QualityAssessmentError"
            )

        # ── QUALITY GATE INTERLOCK ────────────────────────────────────
        if not quality_res["quality_ok"]:
            elapsed = time.perf_counter() - t_start
            return {
                "image_path": str(image_path) if image_path else None,
                "overall_status": "RECAPTURE_REQUIRED",
                "message": "Image quality inadequate — recapture required before screening.",
                "quality": quality_res,
                "prediction": None,
                "explainability": {
                    "gradcam_generated": False,
                    "output_path": None,
                    "reason": "Grad-CAM skipped because image quality is inadequate."
                },
                "referral": None,
                "processing_time_sec": round(elapsed, 4)
            }

        # ── STEP 2: CLASSIFICATION & GRAD-CAM ─────────────────────────
        try:
            cam_res = self.gradcam.generate(image_input)
            pred_class = cam_res["predicted_class"]
            severity_name = cam_res["class_name"]
            confidence = cam_res["confidence"] / 100.0  # normalize to 0-1
            if "raw_probabilities" in cam_res:
                probabilities = [round(float(p), 4) for p in cam_res["raw_probabilities"]]
            else:
                probabilities = [round(p / 100.0, 4) for p in cam_res["probabilities"]]

            gradcam_saved_path = None
            if generate_gradcam and save_visualizations:
                stem = Path(image_name).stem
                out_name = f"pipeline_{stem}_class{pred_class}.png"
                out_file = self.output_dir / out_name
                self.gradcam.save_visualization(cam_res, out_file, mode="side_by_side")
                gradcam_saved_path = str(out_file)

        except Exception as e:
            return self._error_response(
                f"Model inference / Grad-CAM failed: {str(e)}",
                image_path=str(image_path) if image_path else None,
                error_type="InferenceError"
            )

        # ── STEP 3: CLINICAL REFERRAL-SUPPORT ENGINE ──────────────────
        try:
            referral_res = self.referral_engine.evaluate(
                predicted_class=pred_class,
                confidence=confidence,
                quality_status=quality_res
            )
            referral_dict = referral_res.to_dict()
        except Exception as e:
            return self._error_response(
                f"Referral evaluation failed: {str(e)}",
                image_path=str(image_path) if image_path else None,
                error_type="ReferralEngineError"
            )

        # ── STEP 4: ASSEMBLE FINAL STRUCTURED RESULT ──────────────────
        elapsed = time.perf_counter() - t_start
        return {
            "image_path": str(image_path) if image_path else None,
            "overall_status": "SCREENING_COMPLETE",
            "message": "Screening completed successfully.",
            "quality": quality_res,
            "prediction": {
                "class": pred_class,
                "severity": severity_name,
                "confidence": round(confidence, 4),
                "probabilities": probabilities
            },
            "explainability": {
                "gradcam_generated": bool(generate_gradcam),
                "output_path": gradcam_saved_path,
                "heatmap_size": cam_res["heatmap_size"] if generate_gradcam else None
            },
            "referral": referral_dict,
            "processing_time_sec": round(elapsed, 4)
        }

    def _error_response(self, message: str, image_path: Optional[str], error_type: str) -> Dict[str, Any]:
        return {
            "image_path": image_path,
            "overall_status": "ERROR",
            "error_type": error_type,
            "message": message,
            "quality": None,
            "prediction": None,
            "explainability": {
                "gradcam_generated": False,
                "output_path": None,
                "reason": message
            },
            "referral": None,
            "processing_time_sec": 0.0
        }


if __name__ == "__main__":
    import sys
    import json
    if len(sys.argv) > 1:
        pipeline = DRScreeningPipeline()
        res = pipeline.process(sys.argv[1])
        # Format output safely without crashing on complex objects
        res_copy = dict(res)
        print(json.dumps(res_copy, indent=2))
    else:
        print("Usage: python pipeline.py <path_to_fundus_image.png>")
