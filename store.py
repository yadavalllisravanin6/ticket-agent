"""
store.py - saves every processed ticket so a human can approve or reject it later.

Uses its own SQLite file (tickets.db), separate from the mock CRM (crm.db),
so re-running `python db.py` never wipes the ticket history.

Status lifecycle:
    AWAITING_APPROVAL --approve--> APPROVED
    AWAITING_APPROVAL --reject---> REJECTED
A ticket can be decided once. Deciding it again is refused.
"""

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Optional

DB_PATH = "tickets.db"

PENDING = "AWAITING_APPROVAL"
APPROVED = "APPROVED"
REJECTED = "REJECTED"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_store() -> None:
    """Creates the tickets table if it does not exist yet. Safe to call often."""
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tickets (
                ticket_id     TEXT PRIMARY KEY,
                created_at    TEXT NOT NULL,
                raw_text      TEXT NOT NULL,
                agent_result  TEXT NOT NULL,   -- JSON of the graph's output
                status        TEXT NOT NULL,
                reviewer_note TEXT,
                decided_at    TEXT
            )
        """)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _row_to_dict(row: sqlite3.Row) -> dict:
    return {
        "ticket_id": row["ticket_id"],
        "created_at": row["created_at"],
        "raw_text": row["raw_text"],
        "agent_result": json.loads(row["agent_result"]),
        "status": row["status"],
        "reviewer_note": row["reviewer_note"],
        "decided_at": row["decided_at"],
    }


def save_ticket(raw_text: str, agent_result: dict) -> dict:
    """Stores a new ticket in AWAITING_APPROVAL and returns it."""
    ticket_id = "T-" + uuid.uuid4().hex[:8].upper()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO tickets (ticket_id, created_at, raw_text, agent_result, status) "
            "VALUES (?, ?, ?, ?, ?)",
            (ticket_id, _now(), raw_text, json.dumps(agent_result), PENDING),
        )
    return get_ticket(ticket_id)


def get_ticket(ticket_id: str) -> Optional[dict]:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM tickets WHERE ticket_id = ?", (ticket_id,)
        ).fetchone()
    return _row_to_dict(row) if row else None


def list_tickets(status: Optional[str] = None) -> list[dict]:
    with _connect() as conn:
        if status:
            rows = conn.execute(
                "SELECT * FROM tickets WHERE status = ? ORDER BY created_at DESC", (status,)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM tickets ORDER BY created_at DESC").fetchall()
    return [_row_to_dict(r) for r in rows]


class AlreadyDecided(Exception):
    """Raised when someone tries to approve/reject a ticket a second time."""


def decide_ticket(ticket_id: str, decision: str, note: Optional[str]) -> Optional[dict]:
    """Moves a pending ticket to APPROVED or REJECTED.

    Returns None if the ticket does not exist. Raises AlreadyDecided if it was
    already approved or rejected. The UPDATE only matches pending rows, so two
    reviewers clicking at once cannot both succeed.
    """
    if decision not in (APPROVED, REJECTED):
        raise ValueError(f"Invalid decision: {decision}")

    with _connect() as conn:
        cur = conn.execute(
            "UPDATE tickets SET status = ?, reviewer_note = ?, decided_at = ? "
            "WHERE ticket_id = ? AND status = ?",
            (decision, note, _now(), ticket_id, PENDING),
        )
        changed = cur.rowcount

    if changed:
        return get_ticket(ticket_id)

    existing = get_ticket(ticket_id)
    if existing is None:
        return None
    raise AlreadyDecided(f"Ticket {ticket_id} is already {existing['status']}")
