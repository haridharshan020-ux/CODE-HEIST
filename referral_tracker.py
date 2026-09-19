"""
referral_tracker.py
--------------------
Minimal Offline Referral Follow-up Tracker for Diabetic Retinopathy Screening.
Smart India Hackathon 2026 — Research/Prototype.

IMPORTANT NOTICE:
This is a prototype administrative coordination tool for rural health workers.
It is NOT a certified clinical record system, electronic health record (EHR),
or medical device. No patient images are stored. All data is kept locally on
the device and is never transmitted externally.

Storage:
  - Local SQLite database (stdlib sqlite3 — zero new dependencies).
  - Database file: referral_tracker.db (in the project root by default).
  - Excluded from Git via .gitignore — never committed.

Design:
  - Append/update only (no deletion) for auditability.
  - referral_due field records the existing action_pathway from referral engine.
    It does NOT represent a new or validated clinical timeline.
  - All AI result fields are captured read-only from pipeline output.
  - followup_status is the only field the health worker updates post-save.
"""

import sqlite3
import uuid
import csv
from dataclasses import dataclass, fields, astuple
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional


# ── Constants ──────────────────────────────────────────────────────────────

DB_FILENAME = "referral_tracker.db"

# Valid follow-up status values (append-only, no deletion).
VALID_STATUSES = frozenset({"Pending", "Referred", "Attended", "Did Not Attend", "Completed"})

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS referral_records (
    -- Record identity
    record_id           TEXT PRIMARY KEY,
    created_at          TEXT NOT NULL,

    -- Patient identifier (free-text, entered by health worker)
    patient_ref         TEXT NOT NULL,

    -- AI screening result (captured read-only from pipeline output)
    predicted_class     INTEGER NOT NULL,
    severity            TEXT NOT NULL,
    confidence_pct      REAL NOT NULL,
    confidence_band     TEXT NOT NULL,
    quality_score       REAL NOT NULL,

    -- Referral support output (captured read-only from referral engine)
    referral_priority   TEXT NOT NULL,
    referral_due        TEXT NOT NULL,
    recommendation      TEXT NOT NULL,

    -- Follow-up status (updated by health worker post-save)
    followup_status     TEXT NOT NULL DEFAULT 'Pending',
    followup_notes      TEXT NOT NULL DEFAULT '',
    updated_at          TEXT NOT NULL DEFAULT ''
);
"""


# ── Data Class ─────────────────────────────────────────────────────────────

@dataclass
class TrackerRecord:
    """
    One referral tracking record.

    Fields are ordered to match the SQLite column order exactly —
    do not reorder without updating _INSERT_SQL and _ROW_TO_RECORD.
    """
    record_id: str
    created_at: str         # ISO-8601 UTC e.g. "2026-09-19T05:30:00Z"
    patient_ref: str        # Free-text local patient identifier

    predicted_class: int    # 0–4 from V1 model
    severity: str           # "No DR" / "Mild NPDR" / etc.
    confidence_pct: float   # 0.0–100.0 (uncalibrated model output)
    confidence_band: str    # "High" / "Moderate" / "Low" (prototype engineering bands)
    quality_score: float    # 0.0–100.0

    referral_priority: str  # From referral engine (e.g. "Routine", "Urgent")
    referral_due: str       # action_pathway from referral engine (NOT a new clinical rule)
    recommendation: str     # Full text recommendation from referral engine

    followup_status: str    # "Pending" | "Referred" | "Attended" | "Did Not Attend" | "Completed"
    followup_notes: str     # Health-worker free-text notes
    updated_at: str         # ISO-8601 UTC of last status change (empty string if never updated)


_COLUMNS = [f.name for f in fields(TrackerRecord)]

_INSERT_SQL = f"""
INSERT INTO referral_records ({', '.join(_COLUMNS)})
VALUES ({', '.join('?' for _ in _COLUMNS)});
"""

_SELECT_ALL_SQL = "SELECT * FROM referral_records ORDER BY created_at DESC;"
_SELECT_FILTER_SQL = (
    "SELECT * FROM referral_records WHERE followup_status = ? ORDER BY created_at DESC;"
)
_UPDATE_SQL = """
UPDATE referral_records
SET followup_status = ?, followup_notes = ?, updated_at = ?
WHERE record_id = ?;
"""
_SELECT_BY_ID_SQL = "SELECT * FROM referral_records WHERE record_id = ?;"


# ── Internal helpers ────────────────────────────────────────────────────────

def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _row_to_record(row: tuple) -> TrackerRecord:
    return TrackerRecord(*row)


def _get_connection(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")   # safer concurrent writes
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


# ── Public API ──────────────────────────────────────────────────────────────

def default_db_path() -> Path:
    """Returns the default local database path (project root)."""
    return Path(__file__).parent.resolve() / DB_FILENAME


def init_db(db_path: Optional[Path] = None) -> None:
    """
    Create the referral_records table if it does not already exist.
    Safe to call multiple times (idempotent).
    """
    db_path = db_path or default_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with _get_connection(db_path) as conn:
        conn.execute(_CREATE_TABLE_SQL)
        conn.commit()


def save_record(
    patient_ref: str,
    predicted_class: int,
    severity: str,
    confidence_pct: float,
    confidence_band: str,
    quality_score: float,
    referral_priority: str,
    referral_due: str,
    recommendation: str,
    db_path: Optional[Path] = None,
) -> str:
    """
    Insert a new referral record and return its record_id (UUID4 hex).

    referral_due captures the action_pathway from the existing referral engine.
    It does NOT represent a new or validated clinical timeline.

    Raises:
        ValueError: if patient_ref is empty or predicted_class is out of range.
    """
    db_path = db_path or default_db_path()

    if not patient_ref or not patient_ref.strip():
        raise ValueError("patient_ref must not be empty.")
    if not isinstance(predicted_class, int) or predicted_class not in range(5):
        raise ValueError(f"predicted_class must be 0–4, got: {predicted_class!r}")

    record_id = uuid.uuid4().hex
    now = _utc_now()

    record = TrackerRecord(
        record_id=record_id,
        created_at=now,
        patient_ref=patient_ref.strip(),
        predicted_class=predicted_class,
        severity=severity,
        confidence_pct=round(float(confidence_pct), 4),
        confidence_band=confidence_band,
        quality_score=round(float(quality_score), 2),
        referral_priority=referral_priority,
        referral_due=referral_due,
        recommendation=recommendation,
        followup_status="Pending",
        followup_notes="",
        updated_at="",
    )

    init_db(db_path)
    with _get_connection(db_path) as conn:
        conn.execute(_INSERT_SQL, astuple(record))
        conn.commit()

    return record_id


def load_records(
    status_filter: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> List[TrackerRecord]:
    """
    Return all records, newest first.
    If status_filter is provided, return only records with that followup_status.

    Raises:
        ValueError: if status_filter is not a valid status string.
    """
    db_path = db_path or default_db_path()
    if status_filter is not None and status_filter not in VALID_STATUSES:
        raise ValueError(
            f"Invalid status_filter '{status_filter}'. "
            f"Valid values: {sorted(VALID_STATUSES)}"
        )

    init_db(db_path)
    with _get_connection(db_path) as conn:
        if status_filter:
            rows = conn.execute(_SELECT_FILTER_SQL, (status_filter,)).fetchall()
        else:
            rows = conn.execute(_SELECT_ALL_SQL).fetchall()

    return [_row_to_record(row) for row in rows]


def update_status(
    record_id: str,
    new_status: str,
    notes: str = "",
    db_path: Optional[Path] = None,
) -> None:
    """
    Update followup_status and followup_notes for a given record.
    Sets updated_at to the current UTC time.

    Raises:
        ValueError: if new_status is not a valid status string or record not found.
    """
    db_path = db_path or default_db_path()

    if new_status not in VALID_STATUSES:
        raise ValueError(
            f"Invalid status '{new_status}'. "
            f"Valid values: {sorted(VALID_STATUSES)}"
        )

    init_db(db_path)
    with _get_connection(db_path) as conn:
        row = conn.execute(_SELECT_BY_ID_SQL, (record_id,)).fetchone()
        if row is None:
            raise ValueError(f"No record found with record_id='{record_id}'.")
        conn.execute(_UPDATE_SQL, (new_status, notes.strip(), _utc_now(), record_id))
        conn.commit()


def export_csv(
    out_path: Path,
    db_path: Optional[Path] = None,
) -> int:
    """
    Export all records to a CSV file at out_path.
    Returns the number of rows exported.
    All data stays local — no network operations.
    """
    db_path = db_path or default_db_path()
    records = load_records(db_path=db_path)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_COLUMNS)
        writer.writeheader()
        for rec in records:
            writer.writerow({
                "record_id": rec.record_id,
                "created_at": rec.created_at,
                "patient_ref": rec.patient_ref,
                "predicted_class": rec.predicted_class,
                "severity": rec.severity,
                "confidence_pct": rec.confidence_pct,
                "confidence_band": rec.confidence_band,
                "quality_score": rec.quality_score,
                "referral_priority": rec.referral_priority,
                "referral_due": rec.referral_due,
                "recommendation": rec.recommendation,
                "followup_status": rec.followup_status,
                "followup_notes": rec.followup_notes,
                "updated_at": rec.updated_at,
            })

    return len(records)
