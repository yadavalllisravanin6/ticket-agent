"""
db.py - sets up a tiny local SQLite database standing in for a real CRM.

Run this once before starting the agent:
    python db.py
"""

import sqlite3

DB_PATH = "crm.db"


def setup_database():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS customers")
    cur.execute("""
        CREATE TABLE customers (
            customer_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            account_status TEXT NOT NULL,  -- ACTIVE, DELINQUENT, SUSPENDED
            plan_tier TEXT NOT NULL,       -- FREE, PRO, ENTERPRISE
            past_tickets INTEGER NOT NULL
        )
    """)

    # Synthetic customers only - fine to invent freely
    fake_customers = [
        ("C-1001", "Acme Corp", "ACTIVE", "ENTERPRISE", 3),
        ("C-1002", "Globex Inc", "DELINQUENT", "PRO", 12),
        ("C-1003", "Initech", "ACTIVE", "FREE", 0),
        ("C-1004", "Umbrella LLC", "SUSPENDED", "PRO", 8),
        ("C-1005", "Wayne Enterprises", "ACTIVE", "ENTERPRISE", 1),
    ]

    cur.executemany("INSERT INTO customers VALUES (?, ?, ?, ?, ?)", fake_customers)

    conn.commit()
    conn.close()
    print(f"Seeded {len(fake_customers)} customers into {DB_PATH}")


if __name__ == "__main__":
    setup_database()
