import sqlite3
import re
from pathlib import Path

# Paths (adjust if needed)
BASE_DIR = Path(__file__).resolve().parent
SQL_FILE = BASE_DIR.parent / "shoppersstop_backup.sql"
DB_PATH = BASE_DIR / "db.sqlite3"

def load_actual_data():
    """Parse INSERT statements for `actual_footfall` from the SQL dump and insert them into the SQLite DB.
    Expected columns: id, date, store_id, gate_id, actual, created_at
    """
    if not SQL_FILE.exists():
        print(f"SQL dump not found at {SQL_FILE}")
        return

    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()

    # Capture INSERT ... VALUES statements for the actual_footfall table
    insert_regex = re.compile(r"INSERT INTO `actual_footfall`.*?VALUES\s*\((.*?)\);", re.DOTALL | re.IGNORECASE)
    with open(SQL_FILE, "r", encoding="utf-8") as f:
        sql_content = f.read()

    for match in insert_regex.finditer(sql_content):
        values_block = match.group(1)
        # Split multi‑row INSERTs into individual rows
        row_strings = re.findall(r"\(([^)]+)\)", f"({values_block})")
        for row in row_strings:
            # Split on commas that are not inside single quotes
            cols = []
            current = ""
            in_quotes = False
            for ch in row:
                if ch == "'":
                    in_quotes = not in_quotes
                if ch == "," and not in_quotes:
                    cols.append(current.strip().strip("'"))
                    current = ""
                else:
                    current += ch
            cols.append(current.strip().strip("'"))
            if len(cols) != 6:
                continue
            cur.execute(
                "INSERT OR REPLACE INTO actual_footfall (id, date, store_id, gate_id, actual, created_at) VALUES (?,?,?,?,?,?)",
                cols,
            )
    conn.commit()
    conn.close()
    print("Actual footfall data imported successfully.")

if __name__ == "__main__":
    load_actual_data()
