"""
pages/Referral_Tracker.py
--------------------------
Local Referral Follow-up Tracker — Streamlit Page.
Smart India Hackathon 2026 — Research/Prototype.

IMPORTANT NOTICE:
This is a prototype administrative coordination tool for rural health workers.
It is NOT a certified clinical record system, EHR, or medical device.
No patient images are stored. All data remains on this device.

This page is intentionally excluded from Streamlit Cloud deployment
via the /pages/ rule in .gitignore. It is a local-only tool.
"""

import io
import sys
from pathlib import Path

# Ensure project root is in sys.path so referral_tracker can always be imported
_ROOT_DIR = Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

import streamlit as st

from referral_tracker import (
    VALID_STATUSES,
    EXPORT_COLUMNS,
    default_db_path,
    init_db,
    load_records,
    update_status,
    export_csv,
    format_record_for_export,
)

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Referral Tracker — CODE HEIST",
    page_icon="📋",
    layout="centered",
    initial_sidebar_state="collapsed",
)

DB_PATH = default_db_path()
init_db(DB_PATH)

st.title("📋 Referral Follow-up Tracker")
st.caption("Local offline tool for rural screening coordination | SIH 2026")

st.info(
    "**Prototype Administrative Tool Only.**  \n"
    "This tracker supports referral continuity in rural screening centres that may not have an onsite ophthalmologist. "
    "**It does NOT claim to automatically know whether a patient visited the eye hospital.** "
    "All follow-up statuses are entered manually by an authorised healthcare worker based on available follow-up information. "
    "Records are stored locally on this device in an offline SQLite database. No patient images are stored."
)

st.divider()

# ── Load records ───────────────────────────────────────────────────────────
STATUS_OPTIONS = ["All"] + sorted(VALID_STATUSES)

col_f1, col_f2 = st.columns([2, 1])
with col_f1:
    st.subheader("Screening Records")
with col_f2:
    status_filter = st.selectbox(
        "Filter by Status",
        options=STATUS_OPTIONS,
        key="tracker_status_filter",
        label_visibility="collapsed",
    )

try:
    records = load_records(
        status_filter=(status_filter if status_filter != "All" else None),
        db_path=DB_PATH,
    )
except Exception as exc:
    st.error(f"Error loading tracker records: {exc}")
    records = []

if not records:
    st.info("No records found. Run a screening on the main page and save the result to the tracker.")
else:
    # Build a display-friendly list using only stdlib
    rows = []
    for r in records:
        date_str = r.created_at[:10] if r.created_at else ""
        rows.append({
            "Record ID": r.record_id[:8] + "…",
            "Date": date_str,
            "Patient Ref": r.patient_ref,
            "Severity": r.severity,
            "Confidence": f"{r.confidence_pct:.1f}%",
            "Priority": r.referral_priority,
            "Status": r.followup_status,
        })

    # Render as a simple table using st.dataframe via dict list
    # Using st.table for offline compat (no pandas required)
    st.write(f"**{len(records)} record(s) found.**")

    # Render each record as an expander for detail view
    for r in records:
        date_str = r.created_at[:10] if r.created_at else "Unknown"
        expander_label = (
            f"[{r.followup_status}]  {date_str}  |  {r.patient_ref}  |  "
            f"{r.severity}  |  {r.referral_priority}"
        )
        with st.expander(expander_label, expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                st.write(f"**Record ID:** `{r.record_id}`")
                st.write(f"**Screened:** {r.created_at}")
                st.write(f"**Patient Ref:** {r.patient_ref}")
                st.write(f"**Severity:** {r.severity} (Class {r.predicted_class})")
                st.write(f"**Model Confidence:** {r.confidence_pct:.2f}% ({r.confidence_band})")
                st.caption(
                    "Model confidence is an uncalibrated model output, "
                    "not a clinical probability."
                )
                st.write(f"**Quality Score:** {r.quality_score}/100")
            with c2:
                st.write(f"**Referral Priority:** {r.referral_priority}")
                st.write(f"**Referral Due (action pathway):** {r.referral_due}")
                st.caption(
                    "Referral due reflects the existing action pathway from the "
                    "referral engine. It is NOT a new clinical timeline."
                )
                st.write(f"**Follow-up Status:** {r.followup_status}")
                if r.updated_at:
                    st.write(f"**Last Updated:** {r.updated_at}")
                if r.followup_notes:
                    st.write(f"**Notes:** {r.followup_notes}")
            st.write(f"**Recommendation:** {r.recommendation}")

st.divider()

# ── Update Record ──────────────────────────────────────────────────────────
st.subheader("Update Follow-up Status")
st.caption(
    "Enter the full Record ID from the record you want to update. "
    "Statuses are updated manually based on follow-up communication:\n\n"
    "- **Pending**: referral is recommended but the worker has not yet initiated/given the referral.\n"
    "- **Referred**: referral information has been given/initiated by the worker.\n"
    "- **Attended**: worker has received information that the patient attended the referral facility.\n"
    "- **Did Not Attend**: worker has received information that the patient did not attend.\n"
    "- **Completed**: the referral/follow-up process has been completed according to the information available to the worker.\n\n"
    "*(Statuses must not be automatically advanced.)*"
)

with st.form("update_status_form", clear_on_submit=True):
    upd_record_id = st.text_input(
        "Record ID (full 32-character hex)",
        placeholder="e.g. 4a1b2c3d4e5f...",
        key="upd_record_id",
    )
    upd_status = st.selectbox(
        "New Status",
        options=sorted(VALID_STATUSES),
        key="upd_status",
    )
    upd_notes = st.text_area(
        "Notes (optional)",
        placeholder="e.g. Patient attended clinic on 2026-10-01. Referred to ophthalmologist.",
        key="upd_notes",
        max_chars=1000,
    )
    submitted = st.form_submit_button("Update Status", type="primary")

if submitted:
    if not upd_record_id.strip():
        st.error("Please enter a Record ID.")
    else:
        try:
            update_status(
                record_id=upd_record_id.strip(),
                new_status=upd_status,
                notes=upd_notes,
                db_path=DB_PATH,
            )
            st.success(f"Record `{upd_record_id[:8]}...` updated to **{upd_status}**.")
            st.rerun()
        except ValueError as ve:
            st.error(f"Update failed: {ve}")
        except Exception as exc:
            st.error(f"Unexpected error: {exc}")

st.divider()

# ── Export ─────────────────────────────────────────────────────────────────
st.subheader("Export Records")
st.caption(
    "Export all records to CSV for local audit or handover. "
    "No data is transmitted externally."
)

if st.button("Export to CSV", key="export_btn"):
    try:
        all_records = load_records(db_path=DB_PATH)
        if not all_records:
            st.warning("No records to export.")
        else:
            # Build CSV in memory using stdlib
            import csv as _csv
            buf = io.StringIO()
            writer = _csv.DictWriter(buf, fieldnames=EXPORT_COLUMNS)
            writer.writeheader()
            for rec in all_records:
                writer.writerow(format_record_for_export(rec))
            csv_bytes = buf.getvalue().encode("utf-8")
            st.download_button(
                label=f"Download referral_tracker_export.csv ({len(all_records)} records)",
                data=csv_bytes,
                file_name="referral_tracker_export.csv",
                mime="text/csv",
            )
    except Exception as exc:
        st.error(f"Export failed: {exc}")

st.divider()

# ── Disclaimer ─────────────────────────────────────────────────────────────
st.caption(
    "Disclaimer: This tracker is a prototype administrative tool for rural "
    "screening coordination. It does NOT constitute a clinical record system, "
    "EHR, or patient management system. All AI results are decision-support "
    "outputs and require review by a qualified clinician. No patient images "
    "are stored. Data is kept locally on this device only."
)
