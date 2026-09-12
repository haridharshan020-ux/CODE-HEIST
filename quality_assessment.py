"""
quality_assessment.py
---------------------
Heuristic Image Quality Assessment for Fundus Images (Research/Prototype).

IMPORTANT NOTICE:
This module provides a heuristic quality gate for screening workflows.
It is NOT a clinically validated medical diagnostic quality model.
APTOS 2019 dataset does not contain ground-truth quality annotations;
assessments are derived from computer vision signal properties:
  1. Resolution / Pixel count
  2. Sharpness / Focus (Laplacian edge variance)
  3. Illumination / Brightness (Forefront retinal mean intensity)
  4. Contrast (Forefront intensity standard deviation)
  5. Field of View (Retinal disc foreground coverage ratio)

Design:
  - Completely independent from classifier architecture.
  - Fully configurable thresholds via QualityConfig dataclass.
  - Graceful handling of corrupted/missing/invalid images.
  - Standard library, PIL, numpy, and scipy only (no heavy/unnecessary dependencies).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Union
import numpy as np
from PIL import Image
from scipy.ndimage import laplace


@dataclass
class QualityConfig:
    """Configurable thresholds for retinal image quality assessment."""
    min_width: int = 512
    min_height: int = 512
    
    # Sharpness / Blur: Laplacian variance on standard resized 512x512 image
    min_blur_score: float = 3.0       # below this is severely blurred / out-of-focus
    optimal_blur_score: float = 20.0   # score giving 100% blur subscore
    
    # Illumination / Brightness (foreground retinal region 0-255 scale)
    min_foreground_brightness: float = 30.0   # underexposed / dark
    max_foreground_brightness: float = 215.0  # overexposed / washed out
    optimal_brightness_low: float = 55.0
    optimal_brightness_high: float = 145.0
    
    # Contrast (standard deviation in foreground region)
    min_contrast: float = 8.0          # flat, uninformative contrast
    optimal_contrast: float = 22.0
    
    # Field of View (FOV: fraction of image area occupied by fundus foreground)
    min_fov_ratio: float = 0.20        # less than 20% indicates off-center or non-retinal image
    optimal_fov_ratio: float = 0.60
    
    # Minimum overall score (0-100) to pass quality check
    pass_threshold_score: float = 60.0


class RetinalQualityAssessor:
    """
    Assesses whether a fundus image is suitable for AI-based DR screening.
    """

    def __init__(self, config: QualityConfig = None):
        self.config = config or QualityConfig()

    def assess_image(self, image_input: Union[str, Path, Image.Image]) -> Dict[str, Any]:
        """
        Assess image quality from file path or PIL Image object.

        Returns structured dict:
        {
            "quality_ok": bool,
            "quality_score": float (0-100),
            "resolution": (w, h),
            "blur_score": float,
            "brightness_score": float,
            "contrast_score": float,
            "field_of_view_score": float,
            "subscores": dict,
            "warnings": list of str,
            "recommendation": str
        }
        """
        warnings: List[str] = []

        # 1. Load image gracefully
        img = None
        try:
            if isinstance(image_input, (str, Path)):
                img_path = Path(image_input)
                if not img_path.exists():
                    return self._failure_result("Image file not found.", warnings=["File not found."])
                img = Image.open(img_path)
            elif isinstance(image_input, Image.Image):
                img = image_input
            else:
                return self._failure_result("Invalid image input type.")
        except Exception as e:
            return self._failure_result(f"Could not open image: {str(e)}", warnings=["Unreadable or corrupted image."])

        # Verify channels & resolution
        w, h = img.size
        if w < self.config.min_width or h < self.config.min_height:
            warnings.append(f"Low resolution ({w}x{h} px). Minimum recommended: {self.config.min_width}x{self.config.min_height} px.")

        # Convert to grayscale and standardized working dimension for consistent metrics
        img_gray = img.convert("L")
        working_size = (512, 512)
        img_std = img_gray.resize(working_size, Image.Resampling.BILINEAR)
        arr = np.array(img_std, dtype=np.float32)

        # 2. Retinal Field of View (FOV) segmentation
        # Fundus images typically have black borders (< 15 intensity)
        foreground_mask = arr > 15.0
        fg_pixel_count = np.sum(foreground_mask)
        total_pixels = arr.size
        fov_ratio = float(fg_pixel_count / total_pixels)

        if fov_ratio < self.config.min_fov_ratio:
            warnings.append(f"Insufficient retinal field of view ({fov_ratio*100:.1f}%). Fundus disc may be missing or cropped.")
            fov_subscore = 0.0
        else:
            fov_subscore = min(100.0, (fov_ratio / self.config.optimal_fov_ratio) * 100.0)

        # If no retinal foreground is detected, abort further signal processing
        if fg_pixel_count < 100:
            warnings.append("No retinal tissue detected in image.")
            return self._failure_result("No retinal tissue detected.", warnings=warnings)

        fg_pixels = arr[foreground_mask]

        # 3. Brightness / Illumination Analysis
        fg_mean_brightness = float(np.mean(fg_pixels))
        if fg_mean_brightness < self.config.min_foreground_brightness:
            warnings.append(f"Image severely underexposed/dark (brightness: {fg_mean_brightness:.1f}/255).")
            brightness_subscore = max(0.0, (fg_mean_brightness / self.config.min_foreground_brightness) * 50.0)
        elif fg_mean_brightness > self.config.max_foreground_brightness:
            warnings.append(f"Image overexposed/washed out (brightness: {fg_mean_brightness:.1f}/255).")
            brightness_subscore = max(0.0, ((255.0 - fg_mean_brightness) / (255.0 - self.config.max_foreground_brightness)) * 50.0)
        else:
            # Within acceptable range, calculate score based on proximity to optimal center
            if fg_mean_brightness < self.config.optimal_brightness_low:
                brightness_subscore = 70.0 + 30.0 * ((fg_mean_brightness - self.config.min_foreground_brightness) / (self.config.optimal_brightness_low - self.config.min_foreground_brightness))
            elif fg_mean_brightness > self.config.optimal_brightness_high:
                brightness_subscore = 70.0 + 30.0 * ((self.config.max_foreground_brightness - fg_mean_brightness) / (self.config.max_foreground_brightness - self.config.optimal_brightness_high))
            else:
                brightness_subscore = 100.0
        brightness_subscore = float(np.clip(brightness_subscore, 0.0, 100.0))

        # 4. Contrast Analysis
        fg_contrast = float(np.std(fg_pixels))
        if fg_contrast < self.config.min_contrast:
            warnings.append(f"Low contrast ({fg_contrast:.1f}). Vascular details may be obscured.")
            contrast_subscore = max(0.0, (fg_contrast / self.config.min_contrast) * 50.0)
        else:
            contrast_subscore = min(100.0, (fg_contrast / self.config.optimal_contrast) * 100.0)
        contrast_subscore = float(np.clip(contrast_subscore, 0.0, 100.0))

        # 5. Blur / Focus Analysis (Laplacian Variance)
        # We apply Laplacian filter to the standardized image to gauge edge richness
        lap = laplace(arr)
        # Compute variance on foreground to prevent black edge ring from artificially inflating sharpness
        fg_lap_var = float(np.var(lap[foreground_mask]))

        if fg_lap_var < self.config.min_blur_score:
            warnings.append(f"Image appears blurry / out of focus (sharpness metric: {fg_lap_var:.1f}).")
            blur_subscore = max(0.0, (fg_lap_var / self.config.min_blur_score) * 50.0)
        else:
            blur_subscore = min(100.0, (fg_lap_var / self.config.optimal_blur_score) * 100.0)
        blur_subscore = float(np.clip(blur_subscore, 0.0, 100.0))

        # 6. Resolution subscore
        res_factor = min(1.0, (w * h) / (1024 * 1024))
        res_subscore = float(res_factor * 100.0)

        # 7. Aggregate Quality Score (Weighted combination)
        # Weights prioritize Focus & Illumination which most directly impact microaneurysm / exudate detection
        weights = {
            "blur": 0.35,
            "brightness": 0.25,
            "contrast": 0.20,
            "fov": 0.15,
            "resolution": 0.05
        }
        total_score = (
            weights["blur"] * blur_subscore +
            weights["brightness"] * brightness_subscore +
            weights["contrast"] * contrast_subscore +
            weights["fov"] * fov_subscore +
            weights["resolution"] * res_subscore
        )
        total_score = round(float(np.clip(total_score, 0.0, 100.0)), 1)

        # Determine pass/fail
        # Must exceed threshold score AND have no critical failure
        critical_failure = (
            fg_mean_brightness < self.config.min_foreground_brightness or
            fg_mean_brightness > self.config.max_foreground_brightness or
            fg_lap_var < self.config.min_blur_score or
            fov_ratio < self.config.min_fov_ratio
        )
        quality_ok = (total_score >= self.config.pass_threshold_score) and not critical_failure

        if quality_ok:
            recommendation = "Image quality acceptable for screening."
        else:
            recommendation = "Please recapture the retinal image with better focus/illumination/field of view."

        return {
            "quality_ok": bool(quality_ok),
            "quality_score": total_score,
            "resolution": (w, h),
            "blur_score": round(fg_lap_var, 2),
            "brightness_score": round(fg_mean_brightness, 2),
            "contrast_score": round(fg_contrast, 2),
            "field_of_view_score": round(fov_ratio * 100.0, 2),
            "subscores": {
                "sharpness": round(blur_subscore, 1),
                "illumination": round(brightness_subscore, 1),
                "contrast": round(contrast_subscore, 1),
                "field_of_view": round(fov_subscore, 1),
                "resolution": round(res_subscore, 1)
            },
            "warnings": warnings,
            "recommendation": recommendation
        }

    def _failure_result(self, reason: str, warnings: List[str] = None) -> Dict[str, Any]:
        return {
            "quality_ok": False,
            "quality_score": 0.0,
            "resolution": (0, 0),
            "blur_score": 0.0,
            "brightness_score": 0.0,
            "contrast_score": 0.0,
            "field_of_view_score": 0.0,
            "subscores": {
                "sharpness": 0.0,
                "illumination": 0.0,
                "contrast": 0.0,
                "field_of_view": 0.0,
                "resolution": 0.0
            },
            "warnings": warnings or [reason],
            "recommendation": "Please recapture the retinal image with better focus/illumination/field of view."
        }


if __name__ == "__main__":
    import sys
    import json
    if len(sys.argv) > 1:
        assessor = RetinalQualityAssessor()
        result = assessor.assess_image(sys.argv[1])
        print(json.dumps(result, indent=2))
    else:
        print("Usage: python quality_assessment.py <path_to_image.png>")
