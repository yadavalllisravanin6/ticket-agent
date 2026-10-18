"""
graph.py - the LangGraph agent (first version: 2 nodes).

    parse_ticket    -> LLM reads raw ticket text, extracts structured fields
    lookup_customer -> looks up that customer in the local mock CRM (SQLite)

State flows through both nodes, accumulating fields as it goes. That shared,
growing state object is the core LangGraph idea.

Next steps (not built yet): a rule_check node, and a conditional edge that
routes to either a draft_resolution node or an escalate node.
"""

import json
import os
import sqlite3
from typing import Optional, TypedDict

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

DB_PATH = "crm.db"


class TicketState(TypedDict):
    raw_text: str
    customer_id: Optional[str]
    intent: Optional[str]
    urgency: Optional[str]
    account_status: Optional[str]
    plan_tier: Optional[str]
    past_tickets: Optional[int]


def parse_ticket(state: TicketState) -> dict:
    """Calls the LLM to extract structured fields from the raw ticket text."""
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.environ["GROQ_API_KEY"],
        temperature=0,
    )

    prompt = f"""You are a support ticket parser. Given the ticket text below,
respond with ONLY a JSON object, no other text, in exactly this format:
{{"customer_id": "...", "intent": "...", "urgency": "LOW|MEDIUM|HIGH"}}

customer_id must be extracted exactly as written in the ticket (e.g. "C-1001").
intent should be a short phrase like "refund request" or "login issue".

Ticket: "{state['raw_text']}"
"""

    response = llm.invoke(prompt)
    data = json.loads(response.content)

    # Return only the new fields - LangGraph merges this into the existing state
    return {
        "customer_id": data.get("customer_id"),
        "intent": data.get("intent"),
        "urgency": data.get("urgency"),
    }


def lookup_customer(state: TicketState) -> dict:
    """Looks up the customer_id extracted in node 1 against the mock CRM."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT account_status, plan_tier, past_tickets FROM customers WHERE customer_id = ?",
        (state["customer_id"],),
    )
    row = cur.fetchone()
    conn.close()

    if row:
        return {"account_status": row[0], "plan_tier": row[1], "past_tickets": row[2]}

    # Customer not found - fail safe with an explicit UNKNOWN, not a guess
    return {"account_status": "UNKNOWN", "plan_tier": "UNKNOWN", "past_tickets": 0}


builder = StateGraph(TicketState)
builder.add_node("parse_ticket", parse_ticket)
builder.add_node("lookup_customer", lookup_customer)
builder.set_entry_point("parse_ticket")
builder.add_edge("parse_ticket", "lookup_customer")
builder.add_edge("lookup_customer", END)

graph = builder.compile()


if __name__ == "__main__":
    sample_ticket = (
        "Hi, this is customer C-1002. I was charged twice for my subscription "
        "this month and need a refund for the duplicate charge."
    )

    result = graph.invoke({"raw_text": sample_ticket})

    print("\n--- Final state after both nodes ran ---")
    for key, value in result.items():
        print(f"{key}: {value}")
