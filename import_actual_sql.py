# import_actual_sql.py
"""Streaming import of foot‑fall data from a large MySQL dump.

The original script expected a table named ``actual_footfall``. The dump
actually contains the table ``app_agehourlyfootfall`` which stores foot‑fall
per store, date, gate and age‑group, together with hourly columns. This updated
script extracts the relevant fields (date, store_id, gate_id) and computes the
total foot‑fall as the sum of all hourly columns.

Usage (from the project root)::
    python import_actual_sql.py

Make sure the SQLite database path matches the one used by the Django project.
"""

import os
import re
import sqlite3
import sys
from pathlib import Path
from typing import Generator, List, Tuple

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Path to the Django project's SQLite database (relative to this script).
DB_PATH = Path(__file__).resolve().parent / "django_project" / "db.sqlite3"

# Path to the MySQL dump containing the `app_agehourlyfootfall` INSERT statements.
SQL_DUMP_PATH = Path(__file__).resolve().parent / "shoppersstop_backup.sql"

# Number of rows to insert per batch. 1000 is a sensible default.
BATCH_SIZE = 1000

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def stream_insert_statements(file_path: Path) -> Generator[str, None, None]:
    """Yield complete INSERT statements for ``app_agehourlyfootfall``.

    The function reads the file line‑by‑line, concatenating lines until a line
    ends with a semicolon (the end of an INSERT). Only statements whose table
    name is exactly ``app_agehourlyfootfall`` are yielded.
    """
    insert_prefix = "INSERT INTO `app_agehourlyfootfall`"
    buffer: List[str] = []
    with file_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not buffer and insert_prefix not in line:
                continue
            buffer.append(line.rstrip())
            if line.rstrip().endswith(";"):
                stmt = " ".join(buffer)
                if insert_prefix in stmt:
                    yield stmt
                buffer.clear()
        # In case the file ends without a trailing semicolon
        if buffer:
            stmt = " ".join(buffer)
            if insert_prefix in stmt:
                yield stmt


def parse_values_from_insert(stmt: str) -> List[Tuple]:
    """Parse the VALUES part of a MySQL INSERT for ``app_agehourlyfootfall``.

    Expected column order (based on the dump) is:
        id, created_at, modified_at, store_id, date, gate_id, age_grp,
        t7_00_8_00, t8_00_9_00, ..., t23_00_23_59
    The script extracts ``date``, ``store_id`` and ``gate_id`` and computes the
    total foot‑fall by summing all hourly columns (the numeric values after the
    ``age_grp`` field).
    """
    match = re.search(r"VALUES\s*(.*);$", stmt, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return []
    values_blob = match.group(1).strip()
    # Normalise whitespace and remove surrounding parentheses for split
    values_blob = values_blob.replace("\n", " ")
    if values_blob.startswith("(") and values_blob.endswith(")"):
        values_blob = values_blob[1:-1]
    # Split rows on '),(' while handling possible spaces
    raw_rows = re.split(r"\),\s*\(", values_blob)
    rows: List[Tuple] = []
    for raw in raw_rows:
        raw = f"({raw})"  # re‑add parentheses for CSV parsing
        # Replace escaped single quotes to avoid CSV confusion
        raw_clean = raw.replace("\\'", "__SINGLE_QUOTE__")
        inner = raw_clean[1:-1]  # strip outer parentheses
        import csv
        from io import StringIO
        csv_reader = csv.reader(StringIO(inner), delimiter=",", quotechar="'", escapechar="\\")
        parts = next(csv_reader)
        parts = [p.replace("__SINGLE_QUOTE__", "'") for p in parts]
        try:
            # Expected indices based on dump sample:
            # 0:id, 1:created_at, 2:modified_at, 3:store_id,
            # 4:date, 5:gate_id, 6:age_grp, 7+:hourly values
            store_id = int(parts[3])
            date_str = parts[4].strip("'")
            gate_id = int(parts[5])
            hourly_vals = parts[7:]  # everything after age_grp
            total_footfall = sum(int(v) for v in hourly_vals if v)
            rows.append((date_str, store_id, gate_id, total_footfall))
        except Exception:
            # Skip malformed rows silently
            continue
    return rows


def batch_insert(cursor: sqlite3.Cursor, rows: List[Tuple]):
    """Insert a batch of rows into ``actual_footfall``.

    The ``actual_footfall`` schema (created by migration) is:
        date TEXT, store_id INTEGER, gate_id INTEGER, actual INTEGER
    """
    insert_sql = (
        "INSERT OR REPLACE INTO actual_footfall (date, store_id, gate_id, actual) "
        "VALUES (?, ?, ?, ?)"
    )
    cursor.executemany(insert_sql, rows)


def main() -> None:
    if not DB_PATH.exists():
        print(f"[ERROR] SQLite database not found at {DB_PATH}")
        sys.exit(1)
    if not SQL_DUMP_PATH.exists():
        print(f"[ERROR] SQL dump not found at {SQL_DUMP_PATH}")
        sys.exit(1)

    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()

    total_inserted = 0
    batch: List[Tuple] = []
    processed_statements = 0

    print("[INFO] Starting streaming import …")
    for stmt in stream_insert_statements(SQL_DUMP_PATH):
        processed_statements += 1
        rows = parse_values_from_insert(stmt)
        for r in rows:
            batch.append(r)
            if len(batch) >= BATCH_SIZE:
                batch_insert(cur, batch)
                conn.commit()
                total_inserted += len(batch)
                print(f"[INFO] Inserted {total_inserted:,} rows (batch of {len(batch)})")
                batch.clear()
    # Insert any remaining rows
    if batch:
        batch_insert(cur, batch)
        conn.commit()
        total_inserted += len(batch)
        print(f"[INFO] Inserted final {len(batch)} rows (total {total_inserted:,})")

    # Final verification
    cur.execute("SELECT COUNT(*), MIN(date), MAX(date) FROM actual_footfall")
    cnt, min_date, max_date = cur.fetchone()
    print("\n[RESULT] Import complete")
    print(f"INSERT statements processed : {processed_statements}")
    print(f"Rows inserted               : {total_inserted}")
    print(f"Rows in DB                  : {cnt}")
    print(f"Date range in DB            : {min_date} – {max_date}")

    cur.close()
    conn.close()

if __name__ == "__main__":
    main()