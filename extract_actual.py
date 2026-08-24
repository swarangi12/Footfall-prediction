import re
import csv
from dotenv import load_dotenv
import os

load_dotenv()

print("================================")
print("DATABASE CONNECTION")
print("================================")
print("HOST:", os.getenv("DB_HOST"))
print("PORT:", os.getenv("DB_PORT"))
print("NAME:", os.getenv("DB_NAME"))
print("USER:", os.getenv("DB_USER"))
print("================================")

INPUT_FILE = "shoppersstop_backup.sql"
OUTPUT_FILE = "actual_footfall.csv"

print("Opening SQL file...")

# ---------------------------------------------------------
# We ONLY need these columns:
#
# values[3]  = date
# values[4]  = gate_id
#
# Last columns from your sample:
#
# ..., total_footfall, min_footfall, max_footfall,
#     store_id, 0
#
# Therefore:
#
# values[-5] = total_foo
# values[-2] = store_id
# ---------------------------------------------------------

# This regex finds complete SQL tuples.
# It does NOT load the entire SQL file into memory.
tuple_pattern = re.compile(
    rb"\(([^()]*)\)"
)

rows = 0
skipped = 0

with open(
    INPUT_FILE,
    "rb",
    buffering=1024 * 1024
) as infile, open(
    OUTPUT_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as outfile:

    writer = csv.writer(outfile)

    writer.writerow([
        "date",
        "store_id",
        "gate_id",
        "total_footfall"
    ])

    print("Extraction started...")

    # Process file in chunks
    buffer = b""

    while True:

        chunk = infile.read(10 * 1024 * 1024)

        if not chunk:
            break

        buffer += chunk

        # Keep last part in buffer in case a tuple
        # is split between two chunks
        last_close = buffer.rfind(b")")

        if last_close == -1:
            continue

        process = buffer[:last_close + 1]

        buffer = buffer[last_close + 1:]

        matches = tuple_pattern.finditer(process)

        for match in matches:

            try:

                row = match.group(1)

                # Split by comma
                values = row.split(b",")

                if len(values) < 10:
                    skipped += 1
                    continue

                # Decode required fields
                date = values[3].strip()
                gate_id = values[4].strip()

                total_footfall = values[-5].strip()
                store_id = values[-2].strip()

                # Remove quotes from date
                date = date.strip(b"'\"")

                # Decode
                date = date.decode(
                    "utf-8",
                    errors="ignore"
                )

                gate_id = gate_id.decode(
                    "utf-8",
                    errors="ignore"
                )

                total_footfall = total_footfall.decode(
                    "utf-8",
                    errors="ignore"
                )

                store_id = store_id.decode(
                    "utf-8",
                    errors="ignore"
                )

                # Validate
                int(float(total_footfall))
                int(float(store_id))
                int(float(gate_id))

                writer.writerow([
                    date,
                    store_id,
                    gate_id,
                    total_footfall
                ])

                rows += 1

                if rows % 10000 == 0:

                    outfile.flush()

                    print(
                        f"Extracted: {rows:,} rows",
                        flush=True
                    )

            except Exception:
                skipped += 1

print()
print("=" * 60)
print("DONE")
print("=" * 60)
print(f"Rows extracted : {rows:,}")
print(f"Rows skipped   : {skipped:,}")
print(f"Output         : {OUTPUT_FILE}")
print("=" * 60)