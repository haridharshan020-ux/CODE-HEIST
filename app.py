"""
app.py
------
Minimal, Offline Healthcare-Worker UI for Diabetic Retinopathy Screening.
Smart India Hackathon 2026.

Design Principles:
  - Simple, uncluttered interface suitable for rural health workers (ASHA/PHC).
  - Strictly wraps and reuses the existing pipeline.py without duplicating logic.
  - Quality-gate first: Immediately shows "Please recapture image" if quality fails and stops.
  - Transparent explainability: Displays Grad-CAM attention heatmap overlay.
  - Mandatory disclaimer: "AI screening result — not a clinical diagnosis."
  - Displays execution processing latency.
  - Mobile-responsive: works on phones (320px+), tablets, and desktop.

UI Change Log:
  - v1.1 (2026-09-12): Mobile-responsive CSS added via st.markdown().
    Only presentation layer changed. Zero AI/pipeline/model changes.
    CSS uses only: standard HTML element selectors, Streamlit data-testid
    attributes confirmed present in Streamlit 1.63.0 AppTest API, and
    @media queries. No internal st-emotion-cache-* class names used.
  - v1.2 (2026-09-15): Phase 1 Step 2 — Confidence handling + Human-in-the-Loop UI.
    - Explicit AI SCREENING RESULT / EXPLAINABILITY / REFERRAL SUPPORT / CLINICAL REVIEW sections.
    - Model confidence band (High/Moderate/Low) displayed as prototype engineering bands.
    - Uncalibrated confidence notice added inline.
    - Low-confidence caution message added.
    - Grad-CAM explainability notice clarified.
    - Referral support clearly separated from AI prediction.
    - Clinical review notice added as a mandatory final section.
    - Zero changes to pipeline.py, model, quality gate, referral engine, or Grad-CAM.
  - v1.3 (2026-09-19): Referral Follow-up Tracker integration.
    - Added "Save to Tracker" section after Clinical Review (additive only).
    - Health worker enters patient_ref (free text) and saves to local SQLite DB.
    - No patient images stored. No data leaves the device.
    - referral_due captures existing action_pathway (not a new clinical rule).
    - Zero changes to pipeline, model, quality gate, referral engine, or Grad-CAM.
"""

from pathlib import Path
from PIL import Image
import streamlit as st
from pipeline import DRScreeningPipeline
from referral_tracker import (
    default_db_path,
    init_db,
    save_record,
    VALID_STATUSES,
)

# ── Page Configuration ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="DR Screening Decision Support",
    page_icon="👁️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Mobile-Responsive CSS ──────────────────────────────────────────────────
# Targets:
#   - Standard HTML elements (img, button, h1-h3): universally stable.
#   - [data-testid="stAppViewContainer"]: stable since Streamlit 1.x, confirmed
#     in AppTest element tree for Streamlit 1.63.0.
#   - [data-testid="stMetric"]: confirmed stable in AppTest Metric class.
#   - [data-testid="stFileUploader"]: confirmed stable in AppTest FileUploader.
#   - .main > .block-container: documented in Streamlit theming guide; stable
#     across 1.x. Controls page padding.
#   - @media max-width queries: pure CSS standard, no Streamlit dependency.
# No st-emotion-cache-* class names are used anywhere in this block.
st.markdown("""
<style>
/* ── 1. Base: ensure images never overflow their container ── */
img {
    max-width: 100% !important;
    height: auto !important;
}

/* ── 2. Reduce default page padding on narrow screens ── */
.main > .block-container {
    padding-left: 1rem;
    padding-right: 1rem;
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 100%;
}

/* ── 3. On screens wider than 768px restore comfortable desktop padding ── */
@media (min-width: 768px) {
    .main > .block-container {
        padding-left: 2.5rem;
        padding-right: 2.5rem;
        padding-top: 2rem;
        max-width: 860px;
    }
}

/* ── 4. Prevent horizontal overflow at the root ── */
[data-testid="stAppViewContainer"] {
    overflow-x: hidden;
}

/* ── 5. App title: scale down on narrow screens ── */
@media (max-width: 480px) {
    h1 { font-size: 1.45rem !important; line-height: 1.3 !important; }
    h2 { font-size: 1.15rem !important; }
    h3 { font-size: 1.05rem !important; }
}

/* ── 6. Metrics: stack naturally on any width, add breathing room ── */
[data-testid="stMetric"] {
    background: rgba(0, 0, 0, 0.03);
    border-radius: 8px;
    padding: 0.75rem 1rem !important;
}

/* ── 7. Buttons: comfortable tap target on mobile ── */
[data-testid="stButton"] > button {
    min-height: 2.75rem;
    font-size: 1rem;
    width: 100%;
}

/* ── 8. File uploader: prevent overflow on small screens ── */
[data-testid="stFileUploader"] {
    max-width: 100%;
}

/* ── 9. Alert/info/warning/error boxes: better mobile wrap ── */
[data-testid="stAlert"] {
    word-break: break-word;
}

/* ── 10. Captions and small text: readable minimum size ── */
.stCaption, [data-testid="stCaptionContainer"] {
    font-size: 0.8rem !important;
    line-height: 1.5 !important;
}

/* ══════════════════════════════════════════════════════
   PRINT CSS — @media print only.
   Zero effect on screen layout (desktop or mobile).
   UI Change Log v1.2 (2026-09-12): added for browser print support.
   All data-testid values confirmed in Streamlit 1.63.0 AppTest API.
   ══════════════════════════════════════════════════════ */
@media print {

    /* ── P1. Expand all scroll containers so content is not clipped ── */
    html, body {
        height: auto !important;
        overflow: visible !important;
    }
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stMainBlockContainer"],
    .main,
    .main > .block-container {
        height: auto !important;
        max-height: none !important;
        overflow: visible !important;
        position: static !important;
    }

    /* ── P2. Hide screen-only interactive chrome ── */
    [data-testid="stHeader"],
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    [data-testid="stSidebar"],
    [data-testid="stButton"],
    [data-testid="stFileUploader"],
    [data-testid="stStatusWidget"] {
        display: none !important;
    }

    /* ── P3. Prevent images from being split across print pages ── */
    img {
        page-break-inside: avoid;
        break-inside: avoid;
        max-width: 100% !important;
    }

    /* ── P4. Keep metric cards, alerts, and images whole on a page ── */
    [data-testid="stMetric"],
    [data-testid="stAlert"],
    [data-testid="stImage"] {
        page-break-inside: avoid;
        break-inside: avoid;
    }

    /* ── P5. Keep section headings attached to their following content ── */
    h1, h2, h3 {
        page-break-after: avoid;
        break-after: avoid;
    }
}
</style>
""", unsafe_allow_html=True)


# ── Confidence Band Helper ─────────────────────────────────────────────────
# PROTOTYPE ENGINEERING BANDS — NOT CLINICALLY VALIDATED.
# These bands (High >= 75%, Moderate 50-74.99%, Low < 50%) are engineering
# thresholds for UI display purposes only. They are NOT clinical confidence
# thresholds, have not been validated against expert graders, and MUST NOT
# be described as clinically validated certainty levels.
_CONF_HIGH_THRESHOLD = 0.75   # prototype engineering threshold
_CONF_MOD_THRESHOLD = 0.50    # prototype engineering threshold


def _confidence_band(confidence_0_to_1: float) -> tuple:
    """
    Returns (band_label, caution_message_or_None) for a given confidence in [0,1].

    PROTOTYPE ENGINEERING BANDS — NOT CLINICALLY VALIDATED.
    Returns:
        band_label: "High model confidence" | "Moderate model confidence" | "Low model confidence"
        caution_msg: None for High/Moderate, caution string for Low.
    """
    if confidence_0_to_1 >= _CONF_HIGH_THRESHOLD:
        return "High model confidence", None
    elif confidence_0_to_1 >= _CONF_MOD_THRESHOLD:
        return "Moderate model confidence", None
    else:
        return (
            "Low model confidence",
            "Low model confidence. Results are less reliable — consider repeat imaging and/or specialist review."
        )


# ── Header & Mandatory Clinical Disclaimer ─────────────────────────────────
st.title("👁️ Retinal DR Screening System")
st.caption("AI-assisted Decision Support for Rural Health Centers | SIH 2026")

st.info(
    "**Clinical Disclaimer:** This tool provides AI-assisted decision support only. "
    "It is NOT a clinical diagnosis. All outputs require review by a qualified clinician."
)

# ── Pipeline Caching ───────────────────────────────────────────────────────
DEFAULT_CHECKPOINT = Path(__file__).parent.resolve() / "checkpoints" / "best_model_combined_v1.pth"

@st.cache_resource(show_spinner="Loading screening models...")
def get_pipeline():
    return DRScreeningPipeline(checkpoint_path=DEFAULT_CHECKPOINT)

pipeline = get_pipeline()

# ── Image Upload ───────────────────────────────────────────────────────────
st.subheader("1. Retinal Fundus Photograph")
uploaded_file = st.file_uploader(
    "Choose a retinal fundus image (PNG, JPG, JPEG)",
    type=["png", "jpg", "jpeg"],
    help="Upload an uncompressed or standard fundus photograph.",
)

if uploaded_file is not None:
    try:
        pil_image = Image.open(uploaded_file).convert("RGB")
        st.image(pil_image, caption=f"Uploaded: {uploaded_file.name}", use_container_width=True)
    except Exception as e:
        st.error(f"Error reading image file: {e}")
        st.stop()

    # ── Screening Trigger ──────────────────────────────────────────────────
    st.write("")
    start_button = st.button("🔍 Start Screening", type="primary", use_container_width=True)

    if start_button:
        with st.spinner("Processing retinal scan..."):
            result = pipeline.process(pil_image, generate_gradcam=True, save_visualizations=True)

        st.divider()

        # ── 2. Image Quality Assessment ────────────────────────────────────
        st.subheader("2. Image Quality Assessment")
        quality = result["quality"]

        if quality and quality["quality_ok"]:
            st.success(
                f"Quality Acceptable (Score: {quality['quality_score']}/100) — "
                f"{quality['recommendation']}"
            )
        else:
            q_score = quality["quality_score"] if quality else 0.0
            st.error(f"Quality Inadequate (Score: {q_score}/100)")
            # Use primary_failure_reason from Step 1 if available
            failure_reason = (quality or {}).get("primary_failure_reason", "")
            recapture_msg = (
                "Action Required: Please recapture image with better "
                "focus, illumination, or field of view."
            )
            if failure_reason:
                recapture_msg = f"Action Required: {failure_reason} Please recapture image."
            st.warning(recapture_msg)
            if quality and quality["warnings"]:
                st.write("**Detected Issues:**")
                for w in quality["warnings"]:
                    st.write(f"- {w}")
            # Quality gate interlock: stop all downstream output
            st.stop()

        # ══════════════════════════════════════════════════════════════════
        # ── 3. AI SCREENING RESULT ────────────────────────────────────────
        # ══════════════════════════════════════════════════════════════════
        st.divider()
        st.subheader("3. AI Screening Result")

        pred = result["prediction"]
        conf_pct = pred["confidence"] * 100.0
        band_label, caution_msg = _confidence_band(pred["confidence"])

        # Three metrics: severity / model confidence / confidence band
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Predicted Severity", f"Class {pred['class']} ({pred['severity']})")
        with col2:
            st.metric("Model Confidence", f"{conf_pct:.2f}%")
        with col3:
            st.metric("Confidence Band", band_label)

        # Mandatory uncalibrated-confidence notice
        st.caption(
            "Model confidence is an uncalibrated model output and is not a clinical probability. "
            "It reflects the model's output score for the predicted class, not a medically validated "
            "likelihood of disease. Confidence band thresholds are prototype engineering values only."
        )

        # Low-confidence caution (only shown when applicable)
        if caution_msg:
            st.warning(f"Low model confidence: {caution_msg}")

        # ══════════════════════════════════════════════════════════════════
        # ── 4. Explainability (Grad-CAM) ─────────────────────────────────
        # ══════════════════════════════════════════════════════════════════
        st.divider()
        st.subheader("4. Explainability (Grad-CAM)")
        st.caption(
            "Grad-CAM highlights image regions that influenced the model's prediction for the predicted class. "
            "This is a visual explanation of model attention only — it does NOT prove the presence of a lesion "
            "or constitute clinical evidence of disease."
        )

        exp = result["explainability"]
        if exp["gradcam_generated"] and exp["output_path"]:
            cam_path = Path(exp["output_path"])
            if cam_path.exists():
                cam_img = Image.open(cam_path)
                st.image(
                    cam_img,
                    caption="Grad-CAM Visualization [Original | Model Attention Overlay]",
                    use_container_width=True,
                )
            else:
                st.info("Grad-CAM visualization artifact not found on disk.")
        else:
            st.info("Grad-CAM explanation was not generated.")

        # ══════════════════════════════════════════════════════════════════
        # ── 5. Referral Support ───────────────────────────────────────────
        # ══════════════════════════════════════════════════════════════════
        st.divider()
        st.subheader("5. Referral Support")
        st.caption(
            "The referral support output below is decision-support information intended to assist "
            "healthcare workers. It is NOT an autonomous clinical decision, a confirmed diagnosis, "
            "or a treatment plan."
        )

        ref = result["referral"]
        priority = ref["referral_priority"]

        if priority == "Routine":
            st.success(f"**Referral Priority:** {priority}")
        elif "Clinical Review" in priority or "Semi-urgent" in priority:
            st.warning(f"**Referral Priority:** {priority}")
        else:
            st.error(f"**Referral Priority:** {priority}")

        st.write(f"**Action Pathway:** {ref['action_pathway']}")
        st.write(f"**Referral Recommendation:** {ref['recommendation']}")

        if ref["warnings"]:
            for w in ref["warnings"]:
                st.caption(f"Note: {w}")

        st.caption(f"Disclaimer: {ref['disclaimer']}")

        # ══════════════════════════════════════════════════════════════════
        # ── 6. Clinical Review Notice ─────────────────────────────────────
        # ══════════════════════════════════════════════════════════════════
        st.divider()
        st.subheader("6. Clinical Review")
        st.info(
            "Final interpretation of this screening result must be made by an appropriately "
            "trained clinician. This tool provides AI-assisted decision support only. "
            "It is not a substitute for clinical examination, expert ophthalmological assessment, "
            "or locally approved diagnostic protocols."
        )

        # ══════════════════════════════════════════════════════════════════
        # ── 7. Save to Referral Tracker ───────────────────────────────────
        # ══════════════════════════════════════════════════════════════════
        st.divider()
        st.subheader("7. Save to Referral Tracker")
        st.caption(
            "Save this screening result to the local offline referral tracker. "
            "No patient image is stored. All data stays on this device only."
        )

        # Initialise tracker DB (idempotent)
        _tracker_db = default_db_path()
        try:
            init_db(_tracker_db)
        except Exception as _db_err:
            st.warning(f"Tracker DB initialisation failed: {_db_err}")
            _tracker_db = None

        if _tracker_db is not None:
            with st.form("save_to_tracker_form", clear_on_submit=True):
                patient_ref_input = st.text_input(
                    "Patient Reference ID",
                    placeholder="e.g. PHC-2026-001 (free text, your local identifier)",
                    help=(
                        "Enter a local patient reference for follow-up coordination. "
                        "Do not enter biometric or national ID numbers."
                    ),
                    key="tracker_patient_ref",
                )
                save_btn = st.form_submit_button(
                    "Save Screening to Tracker", type="secondary"
                )

            if save_btn:
                if not patient_ref_input.strip():
                    st.error("Please enter a Patient Reference ID before saving.")
                else:
                    try:
                        # Capture referral_due from the existing action_pathway field.
                        # This is NOT a new clinical rule — it records the existing
                        # referral engine output verbatim.
                        _referral_due = ref.get("action_pathway", "")
                        _band, _ = _confidence_band(pred["confidence"])
                        _band_short = _band.replace(" model confidence", "").strip()
                        _rec_id = save_record(
                            patient_ref=patient_ref_input.strip(),
                            predicted_class=int(pred["class"]),
                            severity=pred["severity"],
                            confidence_pct=round(pred["confidence"] * 100.0, 4),
                            confidence_band=_band_short,
                            quality_score=float(quality.get("quality_score", 0.0)),
                            referral_priority=ref.get("referral_priority", ""),
                            referral_due=_referral_due,
                            recommendation=ref.get("recommendation", ""),
                            db_path=_tracker_db,
                        )
                        st.success(
                            f"Saved to tracker. Record ID: `{_rec_id}` — "
                            "View in the Referral Tracker page."
                        )
                    except Exception as _save_err:
                        st.error(f"Save failed: {_save_err}")

        # ── 8. Latency & Metadata ──────────────────────────────────────────
        st.divider()
        st.caption(
            f"Processing time: {result['processing_time_sec']:.2f}s "
            f"| Device: {pipeline.device} | Fully Offline"
        )
        st.caption("Checkpoint: best_model_combined_v1.pth")
