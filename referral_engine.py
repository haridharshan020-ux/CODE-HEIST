"""
referral_engine.py
------------------
Clinical Referral-Support Engine for Diabetic Retinopathy Screening (Research/Prototype).

IMPORTANT NOTICE:
This module provides automated screening decision-support intended to assist
healthcare workers in rural outreach settings. It is NOT an autonomous medical
device, does not provide definitive clinical diagnoses, and does not replace an
ophthalmologist. All recommendations must be reviewed alongside local clinical
guidelines and supervising clinician oversight.

Three Primary Inputs:
  1. Quality-Gate Status (from quality_assessment.py)
  2. Predicted DR Severity Class (0 to 4 from EfficientNet-B0)
  3. Model Prediction Confidence (0.0 to 1.0)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union


@dataclass
class ReferralConfig:
    """Configurable thresholds for prototype decision-support logic."""
    # Confidence tiers (prototype values, not clinically validated cutoffs)
    high_confidence_threshold: float = 0.75     # >= 75%
    moderate_confidence_threshold: float = 0.50 # 50% to 75%
    # Below moderate_confidence_threshold (<50%) is flagged as low confidence


@dataclass
class ReferralResult:
    """Structured machine- and human-readable referral support output."""
    predicted_class: Optional[int]
    severity: str
    confidence: float
    confidence_level: str
    quality_status: str
    referral_priority: str
    action_pathway: str
    recommendation: str
    warnings: List[str]
    disclaimer: str
    human_readable_summary: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "predicted_class": self.predicted_class,
            "severity": self.severity,
            "confidence": round(self.confidence, 4),
            "confidence_level": self.confidence_level,
            "quality_status": self.quality_status,
            "referral_priority": self.referral_priority,
            "action_pathway": self.action_pathway,
            "recommendation": self.recommendation,
            "warnings": self.warnings,
            "disclaimer": self.disclaimer,
            "human_readable_summary": self.human_readable_summary,
        }


class DRReferralEngine:
    """
    Evaluates quality gate status, classification severity, and prediction confidence
    to generate stratified referral support pathways for rural screening prototypes.
    """

    STANDARD_DISCLAIMER = (
        "AI screening result — not a clinical diagnosis. This decision-support "
        "recommendation must be interpreted by a qualified clinician in accordance "
        "with local healthcare protocols."
    )

    SEVERITY_MAPPINGS = {
        0: {
            "name": "No DR",
            "interpretation": "No Diabetic Retinopathy detected by model",
            "priority": "Routine",
            "action": "Routine annual diabetic eye screening",
            "rec": (
                "No features of diabetic retinopathy were detected by the screening model. "
                "Recommend continuing scheduled annual diabetic eye examination in accordance "
                "with local protocol and primary diabetic management."
            )
        },
        1: {
            "name": "Mild NPDR",
            "interpretation": "Mild Non-Proliferative Diabetic Retinopathy features detected",
            "priority": "Routine / Clinical Review",
            "action": "Primary care / optometry review and follow-up as per local protocol",
            "rec": (
                "Mild microvascular signs detected. Clinical review recommended to ensure optimal "
                "glycemic/blood pressure control and repeat retinal screening within 6–12 months "
                "or per supervising clinician guidance."
            )
        },
        2: {
            "name": "Moderate NPDR",
            "interpretation": "Moderate Non-Proliferative Diabetic Retinopathy features detected",
            "priority": "Semi-urgent",
            "action": "Ophthalmic referral for comprehensive retinal evaluation",
            "rec": (
                "Features consistent with Moderate NPDR observed. Recommend scheduled ophthalmic "
                "consultation for full dilated fundus evaluation within 3–6 months or per local protocol."
            )
        },
        3: {
            "name": "Severe NPDR",
            "interpretation": "Severe Non-Proliferative Diabetic Retinopathy features detected",
            "priority": "Urgent",
            "action": "Prompt ophthalmic referral (high risk of progression to PDR)",
            "rec": (
                "Urgent ophthalmic evaluation recommended. High risk of vision impairment; "
                "patient requires specialized assessment for timely management."
            )
        },
        4: {
            "name": "PDR",
            "interpretation": "Proliferative Diabetic Retinopathy (PDR) features detected",
            "priority": "Urgent",
            "action": "Immediate / Urgent specialized ophthalmic care",
            "rec": (
                "Urgent ophthalmic evaluation recommended. Neovascular or proliferative features "
                "indicated; prompt specialist intervention required to safeguard vision."
            )
        }
    }

    def __init__(self, config: Optional[ReferralConfig] = None):
        self.config = config or ReferralConfig()

    def evaluate(
        self,
        predicted_class: Optional[int],
        confidence: Union[float, int],
        quality_status: Union[bool, str, Dict[str, Any]],
        quality_warnings: Optional[List[str]] = None
    ) -> ReferralResult:
        """
        Synthesize screening input into structured referral guidance.

        Parameters:
            predicted_class: Integer 0 to 4, or None
            confidence: Float between 0.0 and 1.0 (or percentage 0.0-100.0)
            quality_status: bool (True=adequate), str ('adequate'/'inadequate'), or quality dict
            quality_warnings: Optional list of warnings from quality assessment module
        """
        warnings: List[str] = []

        # ── 1. Resolve Quality-Gate Status ─────────────────────────────
        is_quality_ok = False
        if isinstance(quality_status, bool):
            is_quality_ok = quality_status
        elif isinstance(quality_status, str):
            is_quality_ok = (quality_status.strip().lower() in ("adequate", "ok", "true", "pass", "passed"))
        elif isinstance(quality_status, dict):
            is_quality_ok = bool(quality_status.get("quality_ok", False))
            if "warnings" in quality_status and isinstance(quality_status["warnings"], list):
                warnings.extend(quality_status["warnings"])
        else:
            warnings.append("Missing or unreadable quality-gate status; treated as inadequate for patient safety.")
            is_quality_ok = False

        if quality_warnings:
            for qw in quality_warnings:
                if qw not in warnings:
                    warnings.append(qw)

        # Normalize confidence to [0.0, 1.0]
        try:
            conf_val = float(confidence)
            if conf_val > 1.0 and conf_val <= 100.0:
                conf_val = conf_val / 100.0
            if conf_val < 0.0 or conf_val > 1.0:
                warnings.append(f"Confidence value {confidence} outside expected range [0, 1]; clamped.")
                conf_val = max(0.0, min(1.0, conf_val))
        except (ValueError, TypeError):
            warnings.append(f"Invalid confidence '{confidence}' provided; set to 0.0.")
            conf_val = 0.0

        # Confidence level designation
        if conf_val >= self.config.high_confidence_threshold:
            conf_level = "High"
        elif conf_val >= self.config.moderate_confidence_threshold:
            conf_level = "Moderate"
        else:
            conf_level = "Low"
            warnings.append("Low model confidence — clinical review is especially important.")

        # ── 2. Quality Interlock Check ─────────────────────────────────
        if not is_quality_ok:
            return ReferralResult(
                predicted_class=predicted_class if isinstance(predicted_class, int) else None,
                severity="Undetermined (Inadequate Image Quality)",
                confidence=conf_val,
                confidence_level=conf_level,
                quality_status="Inadequate",
                referral_priority="Recapture Required",
                action_pathway="Recapture retinal fundus photograph before screening evaluation",
                recommendation="Recapture image before screening.",
                warnings=warnings,
                disclaimer=self.STANDARD_DISCLAIMER,
                human_readable_summary=(
                    "IMAGE QUALITY INADEQUATE: The retinal image does not meet quality criteria for reliable "
                    "screening. Action required: Recapture image before attempting DR referral assessment."
                )
            )

        # ── 3. Validate Severity Class ─────────────────────────────────
        if predicted_class is None or not isinstance(predicted_class, int) or predicted_class not in self.SEVERITY_MAPPINGS:
            warnings.append(f"Invalid or unrecognized predicted class '{predicted_class}'.")
            return ReferralResult(
                predicted_class=None,
                severity="Unrecognized Class",
                confidence=conf_val,
                confidence_level=conf_level,
                quality_status="Adequate",
                referral_priority="Clinical Review Required",
                action_pathway="Manual ophthalmic evaluation",
                recommendation="Classification output invalid. Refer patient to attending clinician for direct examination.",
                warnings=warnings,
                disclaimer=self.STANDARD_DISCLAIMER,
                human_readable_summary=(
                    "CLASSIFICATION ERROR: The model returned an invalid class identifier. "
                    "Action required: Clinician manual review."
                )
            )

        # ── 4. Build Structured Referral Pathway ───────────────────────
        mapping = self.SEVERITY_MAPPINGS[predicted_class]
        severity_name = mapping["name"]
        priority = mapping["priority"]
        action = mapping["action"]
        recommendation = mapping["rec"]

        summary = (
            f"Screening Result: {severity_name} (Class {predicted_class}) | "
            f"Confidence: {conf_val*100:.1f}% ({conf_level}) | "
            f"Referral Priority: {priority} | "
            f"Recommended Pathway: {action}"
        )

        return ReferralResult(
            predicted_class=predicted_class,
            severity=severity_name,
            confidence=conf_val,
            confidence_level=conf_level,
            quality_status="Adequate",
            referral_priority=priority,
            action_pathway=action,
            recommendation=recommendation,
            warnings=warnings,
            disclaimer=self.STANDARD_DISCLAIMER,
            human_readable_summary=summary
        )
