"""SQLite scan history for 4-class brain tumor detection."""

import os
import sqlite3
import json
from datetime import datetime

DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scans.db"
)


def _conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def init_db():
    with _conn() as c:

        c.execute("""
            CREATE TABLE IF NOT EXISTS scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                stored_name TEXT NOT NULL,
                prediction TEXT NOT NULL,
                tumor_probability REAL NOT NULL,
                confidence REAL NOT NULL,
                created_at TEXT NOT NULL,
                tumor_type TEXT DEFAULT 'Unknown',
                probabilities TEXT DEFAULT '{}'
            )
        """)

        # Migration for existing databases
        columns = [
            row["name"]
            for row in c.execute("PRAGMA table_info(scans)").fetchall()
        ]

        if "tumor_type" not in columns:
            c.execute("""
                ALTER TABLE scans
                ADD COLUMN tumor_type TEXT DEFAULT 'Unknown'
            """)

        if "probabilities" not in columns:
            c.execute("""
                ALTER TABLE scans
                ADD COLUMN probabilities TEXT DEFAULT '{}'
            """)


def scan_code(row_id: int) -> str:
    return f"SCAN{row_id:03d}"


def _fmt(r):

    try:
        probabilities = json.loads(r["probabilities"] or "{}")
    except (json.JSONDecodeError, TypeError):
        probabilities = {}

    return {
        "id": r["id"],
        "scan_id": scan_code(r["id"]),
        "filename": r["filename"],
        "prediction": r["prediction"],
        "tumor_type": r["tumor_type"],
        "tumor_probability": r["tumor_probability"],
        "confidence": r["confidence"],
        "probabilities": probabilities,
        "created_at": r["created_at"],
        "image_url": f"/api/uploads/{r['stored_name']}",
    }


def add_scan(
    filename,
    stored_name,
    prediction,
    tumor_prob,
    confidence,
    tumor_type="Unknown",
    probabilities=None
):

    probabilities_json = json.dumps(probabilities or {})

    with _conn() as c:

        cur = c.execute(
            """
            INSERT INTO scans (
                filename,
                stored_name,
                prediction,
                tumor_probability,
                confidence,
                created_at,
                tumor_type,
                probabilities
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                filename,
                stored_name,
                prediction,
                tumor_prob,
                confidence,
                datetime.now().isoformat(timespec="seconds"),
                tumor_type,
                probabilities_json
            )
        )

        new_id = cur.lastrowid

    return get_scan(new_id)


def get_scan(row_id):

    with _conn() as c:
        r = c.execute(
            "SELECT * FROM scans WHERE id=?",
            (row_id,)
        ).fetchone()

    return _fmt(r) if r else None


def list_scans(limit=100):

    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM scans ORDER BY id DESC LIMIT ?",
            (limit,)
        ).fetchall()

    return [_fmt(r) for r in rows]


def stats():

    with _conn() as c:

        total = c.execute(
            "SELECT COUNT(*) FROM scans"
        ).fetchone()[0]

        tumor = c.execute("""
            SELECT COUNT(*)
            FROM scans
            WHERE prediction != 'No Tumor Detected'
        """).fetchone()[0]

    return {
        "total": total,
        "tumor": tumor,
        "normal": total - tumor
    }