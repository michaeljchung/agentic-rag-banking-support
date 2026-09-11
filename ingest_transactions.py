"""
Loads the transaction CSV into a SQLite database.
"""

import sqlite3
from pathlib import Path

import pandas as pd

RAW_CSV_PATH = Path("data/raw/bank_transactions.csv")
DB_PATH = Path("data/processed/transactions.db")
TABLE_NAME = "transactions"


def ingest_transactions(csv_path: Path, db_path: Path, table_name: str) -> None:
    df = pd.read_csv(csv_path)

    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"Columns: {list(df.columns)}")

    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    try:
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        print(f"Wrote {row_count} rows to '{table_name}' in {db_path}")
    finally:
        conn.close()


if __name__ == "__main__":
    ingest_transactions(RAW_CSV_PATH, DB_PATH, TABLE_NAME)