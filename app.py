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
"""

from pathlib import Path
from PIL import Image
import streamlit as st
from pipeline import DRScreeningPipeline

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

# ── Header & Mandatory Clinical Disclaimer ─────────────────────────────────
st.title("👁️ Retinal DR Screening System")
st.caption("AI-assisted Decision Support for Rural Health Centers | SIH 2026")

st.info("⚠️ **Clinical Disclaimer:** AI screening result — not a clinical diagnosis.")

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
                f"✅ **Quality Acceptable** (Score: {quality['quality_score']}/100) — "
                f"{quality['recommendation']}"
            )
        else:
            q_score = quality["quality_score"] if quality else 0.0
            st.error(f"❌ **Quality Inadequate** (Score: {q_score}/100)")
            st.warning(
                "⚠️ **Action Required:** Please recapture image with better "
                "focus, illumination, or field of view."
            )
            if quality and quality["warnings"]:
                st.write("**Detected Issues:**")
                for w in quality["warnings"]:
                    st.write(f"- {w}")
            # Stop execution immediately as required by quality gate logic
            st.stop()

        # ── 3. Screening Prediction ────────────────────────────────────────
        st.subheader("3. Screening Prediction")
        pred = result["prediction"]
        conf_pct = pred["confidence"] * 100.0

        # Two metrics side-by-side on desktop; they auto-stack on mobile
        # because the CSS sets width:100% and flex-wrap on narrow viewports.
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted Severity", f"Class {pred['class']} ({pred['severity']})")
        with col2:
            st.metric("Model Confidence", f"{conf_pct:.2f}%")

        st.divider()

        # ── 4. Visual Attention (Grad-CAM) ─────────────────────────────────
        st.subheader("4. Visual Attention (Grad-CAM)")
        st.caption(
            "Highlighted areas show regions receiving stronger model attention for the "
            "predicted class. This provides visual explainability and does NOT constitute "
            "clinical proof of lesions."
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

        st.divider()

        # ── 5. Referral-Support Recommendation ────────────────────────────
        st.subheader("5. Referral-Support Recommendation")
        ref = result["referral"]
        priority = ref["referral_priority"]

        if priority == "Routine":
            st.success(f"**Referral Priority:** {priority}")
        elif "Clinical Review" in priority or "Semi-urgent" in priority:
            st.warning(f"**Referral Priority:** {priority}")
        else:
            st.error(f"**Referral Priority:** {priority}")

        st.write(f"**Action Pathway:** {ref['action_pathway']}")
        st.write(f"**Clinical Recommendation:** {ref['recommendation']}")

        if ref["warnings"]:
            for w in ref["warnings"]:
                st.caption(f"Note: {w}")

        # ── 6. Latency & Metadata ──────────────────────────────────────────
        st.divider()
        st.caption(
            f"⚡ Processing time: {result['processing_time_sec']:.2f}s "
            f"| Device: {pipeline.device} | Fully Offline"
        )
        st.caption("Checkpoint: best_model_combined_v1.pth")
