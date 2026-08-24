import os
import sys
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv


# ============================================================
# LOAD .ENV
# ============================================================

# .env is in the parent folder:
# C:\Users\swara\Downloads\footfall\.env

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(os.path.dirname(BASE_DIR), ".env")

print("==============================================")
print("LOADING DATABASE CONFIGURATION")
print("==============================================")
print("ENV FILE:", ENV_PATH)

load_dotenv(ENV_PATH)

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")


print("HOST:", DB_HOST)
print("PORT:", DB_PORT)
print("NAME:", DB_NAME)
print("USER:", DB_USER)
print("==============================================")


# ============================================================
# CHECK DATABASE VARIABLES
# ============================================================

if not DB_HOST:
    print("\nERROR: DB_HOST is None")
    print("Check your .env file.")
    sys.exit(1)

if not DB_NAME:
    print("\nERROR: DB_NAME is None")
    print("Check your .env file.")
    sys.exit(1)

if not DB_USER:
    print("\nERROR: DB_USER is None")
    print("Check your .env file.")
    sys.exit(1)

if not DB_PASSWORD:
    print("\nERROR: DB_PASSWORD is None")
    print("Check your .env file.")
    sys.exit(1)


# ============================================================
# CSV PATH
# ============================================================

CSV_PATH = os.path.join(
    os.path.dirname(BASE_DIR),
    "actual_footfall.csv"
)

print("\nCSV FILE:")
print(CSV_PATH)


if not os.path.exists(CSV_PATH):
    print("\nERROR: actual_footfall.csv not found.")
    sys.exit(1)


# ============================================================
# CONNECT TO POSTGRESQL
# ============================================================

print("\n==============================================")
print("CONNECTING TO POSTGRESQL")
print("==============================================")

try:

    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        connect_timeout=30
    )

    cursor = conn.cursor()

    print("Connected successfully.")

except Exception as e:

    print("\nDATABASE CONNECTION ERROR:")
    print(e)

    sys.exit(1)


# ============================================================
# CLEAR OLD DATA
# ============================================================

print("\n==============================================")
print("CLEARING OLD ACTUAL DATA")
print("==============================================")

try:

    cursor.execute("TRUNCATE TABLE actual_footfall RESTART IDENTITY")

    conn.commit()

    print("Old actual_footfall data cleared.")

except Exception as e:

    conn.rollback()

    print("\nERROR CLEARING TABLE:")
    print(e)

    cursor.close()
    conn.close()

    sys.exit(1)


# ============================================================
# READ CSV
# ============================================================

print("\n==============================================")
print("READING CSV")
print("==============================================")

try:

    df = pd.read_csv(
        CSV_PATH,
        usecols=[
            "date",
            "store_id",
            "gate_id",
            "total_footfall"
        ]
    )

except Exception as e:

    print("\nERROR READING CSV:")
    print(e)

    cursor.close()
    conn.close()

    sys.exit(1)


print("CSV rows:", len(df))
print("CSV columns:", list(df.columns))


# ============================================================
# CLEAN DATE
# ============================================================

print("\n==============================================")
print("CLEANING DATA")
print("==============================================")


original_rows = len(df)


df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)


invalid_dates = df["date"].isna().sum()

print("Invalid dates:", invalid_dates)


df = df.dropna(
    subset=["date"]
)


# Convert datetime to date only

df["date"] = df["date"].dt.date


# ============================================================
# CLEAN NUMERIC COLUMNS
# ============================================================

df["store_id"] = pd.to_numeric(
    df["store_id"],
    errors="coerce"
)

df["gate_id"] = pd.to_numeric(
    df["gate_id"],
    errors="coerce"
)

df["total_footfall"] = pd.to_numeric(
    df["total_footfall"],
    errors="coerce"
)


# Remove invalid numeric rows

before_numeric_clean = len(df)

df = df.dropna(
    subset=[
        "store_id",
        "gate_id",
        "total_footfall"
    ]
)

removed_numeric = (
    before_numeric_clean - len(df)
)

print("Invalid numeric rows removed:", removed_numeric)


# ============================================================
# CONVERT INTEGER COLUMNS
# ============================================================

df["store_id"] = df["store_id"].astype(int)

df["gate_id"] = df["gate_id"].astype(int)

df["total_footfall"] = df["total_footfall"].astype(int)


# ============================================================
# REMOVE NEGATIVE FOOTFALL
# ============================================================

negative_rows = (
    df["total_footfall"] < 0
).sum()

print("Negative footfall rows:", negative_rows)


df = df[
    df["total_footfall"] >= 0
]


# ============================================================
# HANDLE DUPLICATES
# ============================================================

print("\n==============================================")
print("CHECKING DUPLICATES")
print("==============================================")


duplicate_count = df.duplicated(
    subset=[
        "date",
        "store_id",
        "gate_id"
    ]
).sum()


print(
    "Duplicate date/store/gate rows:",
    duplicate_count
)


# ------------------------------------------------------------
# IMPORTANT:
#
# Your PostgreSQL table has:
#
# UNIQUE(date, store_id, gate_id)
#
# Therefore only one row can exist for each combination.
#
# We keep the LAST occurrence from the CSV.
# ------------------------------------------------------------

df = df.drop_duplicates(
    subset=[
        "date",
        "store_id",
        "gate_id"
    ],
    keep="last"
)


print(
    "Rows after duplicate removal:",
    len(df)
)


# ============================================================
# SORT DATA
# ============================================================

df = df.sort_values(
    by=[
        "date",
        "store_id",
        "gate_id"
    ]
)


df = df.reset_index(drop=True)


# ============================================================
# SHOW DATA RANGE
# ============================================================

print("\n==============================================")
print("DATA RANGE")
print("==============================================")

print("MIN DATE:", df["date"].min())
print("MAX DATE:", df["date"].max())


# ============================================================
# CHECK 2026 DATA
# ============================================================

start_date = pd.Timestamp("2026-07-06").date()

end_date = pd.Timestamp("2026-07-12").date()


week_df = df[
    (df["date"] >= start_date)
    &
    (df["date"] <= end_date)
]


print("\n==============================================")
print("2026-07-06 TO 2026-07-12")
print("==============================================")

print(
    "Rows for dashboard week:",
    len(week_df)
)


if len(week_df) > 0:

    print("\nSample actual data:")

    print(
        week_df.head(10).to_string(
            index=False
        )
    )

else:

    print(
        "WARNING: No rows found for "
        "2026-07-06 to 2026-07-12"
    )


# ============================================================
# PREPARE INSERT
# ============================================================

insert_query = """
    INSERT INTO actual_footfall
    (
        date,
        store_id,
        gate_id,
        actual
    )
    VALUES %s
    ON CONFLICT
    (
        date,
        store_id,
        gate_id
    )
    DO UPDATE SET
        actual = EXCLUDED.actual
"""


# ============================================================
# INSERT DATA IN BATCHES
# ============================================================

print("\n==============================================")
print("STARTING POSTGRESQL IMPORT")
print("==============================================")


BATCH_SIZE = 5000

total_rows = len(df)

inserted_rows = 0


for start in range(
    0,
    total_rows,
    BATCH_SIZE
):

    batch = df.iloc[
        start:start + BATCH_SIZE
    ]


    values = []

    for _, row in batch.iterrows():

        values.append(
            (
                row["date"],
                int(row["store_id"]),
                int(row["gate_id"]),
                int(row["total_footfall"])
            )
        )


    try:

        execute_values(
            cursor,
            insert_query,
            values,
            page_size=BATCH_SIZE
        )

        conn.commit()


        inserted_rows += len(values)


        print(
            f"Imported "
            f"{inserted_rows:,} / "
            f"{total_rows:,}"
        )


    except Exception as e:

        conn.rollback()

        print("\n==============================================")
        print("IMPORT ERROR")
        print("==============================================")

        print(e)

        cursor.close()
        conn.close()

        sys.exit(1)


# ============================================================
# VERIFY TOTAL ROWS
# ============================================================

print("\n==============================================")
print("VERIFYING DATABASE")
print("==============================================")


cursor.execute(
    """
    SELECT
        COUNT(*),
        MIN(date),
        MAX(date)
    FROM actual_footfall
    """
)


result = cursor.fetchone()


print("Database rows:", result[0])
print("Database MIN date:", result[1])
print("Database MAX date:", result[2])


# ============================================================
# VERIFY DASHBOARD WEEK
# ============================================================

cursor.execute(
    """
    SELECT
        COUNT(*),
        MIN(date),
        MAX(date)
    FROM actual_footfall
    WHERE date BETWEEN
        '2026-07-06'
        AND
        '2026-07-12'
    """
)


week_result = cursor.fetchone()


print("\n==============================================")
print("DASHBOARD WEEK CHECK")
print("==============================================")


print(
    "Rows:",
    week_result[0]
)

print(
    "MIN:",
    week_result[1]
)

print(
    "MAX:",
    week_result[2]
)


# ============================================================
# SHOW SAMPLE ACTUAL VALUES
# ============================================================

cursor.execute(
    """
    SELECT
        date,
        store_id,
        gate_id,
        actual
    FROM actual_footfall
    WHERE date BETWEEN
        '2026-07-06'
        AND
        '2026-07-12'
    ORDER BY
        date,
        store_id,
        gate_id
    LIMIT 20
    """
)


rows = cursor.fetchall()


print("\n==============================================")
print("SAMPLE ACTUAL VALUES")
print("==============================================")


for row in rows:

    print(row)


# ============================================================
# CLOSE CONNECTION
# ============================================================

cursor.close()

conn.close()


# ============================================================
# FINAL RESULT
# ============================================================

print("\n")
print("============================================================")
print("IMPORT COMPLETE")
print("============================================================")
print(
    "CSV rows originally      :",
    f"{original_rows:,}"
)
print(
    "Rows imported/processed  :",
    f"{inserted_rows:,}"
)
print(
    "Database rows            :",
    f"{result[0]:,}"
)
print(
    "Dashboard week rows      :",
    f"{week_result[0]:,}"
)
print("============================================================")