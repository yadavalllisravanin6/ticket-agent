"""
graph.py - the LangGraph agent (full version: 5 nodes, one conditional edge).

    parse_ticket      -> LLM reads raw ticket text, extracts structured fields
    lookup_customer   -> looks up that customer in the local mock CRM (SQLite)
    rule_check        -> plain Python checks business rules (no LLM)
        |-- rules pass -> draft_resolution
        |-- rules fail -> escalate
    both end with status AWAITING_APPROVAL - a human signs off on everything.

The LLM only touches parse_ticket. Every decision that matters (lookup, rules,
routing, the drafted action) is deterministic code.
"""
from dotenv import load_dotenv
load_dotenv()
import json
import os
import sqlite3 
from typing import Optional, TypedDict

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

DB_PATH = "crm.db"

# Groq changes which models are available. Run `python check_models.py` to see
# what your key can use, then set GROQ_MODEL (no code edit needed).
DEFAULT_MODEL = "llama-3.3-70b-versatile"

# Business rules - kept as constants so they are easy to find and change
BLOCKED_STATUSES = {"DELINQUENT", "SUSPENDED", "UNKNOWN"}
MAX_PAST_TICKETS_FOR_AUTO = 10


class TicketState(TypedDict, total=False):
    # Input
    raw_text: str
    # Added by parse_ticket
    customer_id: Optional[str]
    intent: Optional[str]
    urgency: Optional[str]
    # Added by lookup_customer
    account_status: Optional[str]
    plan_tier: Optional[str]
    past_tickets: Optional[int]
    # Added by rule_check
    rules_passed: bool
    rule_violations: list[str]
    # Added by draft_resolution or escalate
    route: str
    draft_action: str
    reasoning: str
    status: str


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model reply.

    Models sometimes wrap JSON in markdown fences or add a sentence around it,
    which makes a plain json.loads() fail. Taking the text between the first
    '{' and the last '}' handles both cases.
    """
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON object found in model reply: {text!r}")
    return json.loads(text[start:end + 1])


# ---------------------------------------------------------------- node 1 (LLM)
def parse_ticket(state: TicketState) -> dict:
    """Calls the LLM to extract structured fields from the raw ticket text."""
    llm = ChatGroq(
        model=os.environ.get("GROQ_MODEL", DEFAULT_MODEL),
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
    data = extract_json(response.content)

    # Return only the new fields - LangGraph merges this into the existing state
    return {
        "customer_id": data.get("customer_id"),
        "intent": data.get("intent"),
        "urgency": (data.get("urgency") or "").upper() or None,
    }


# ---------------------------------------------------------------- node 2 (code)
def lookup_customer(state: TicketState) -> dict:
    """Looks up the customer_id extracted in node 1 against the mock CRM."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT account_status, plan_tier, past_tickets FROM customers WHERE customer_id = ?",
        (state.get("customer_id"),),
    )
    row = cur.fetchone()
    conn.close()

    if row:
        return {"account_status": row[0], "plan_tier": row[1], "past_tickets": row[2]}

    # Customer not found - fail safe with an explicit UNKNOWN, not a guess
    return {"account_status": "UNKNOWN", "plan_tier": "UNKNOWN", "past_tickets": 0}


# ---------------------------------------------------------------- node 3 (code)
def rule_check(state: TicketState) -> dict:
    """Deterministic business rules. Collects every violation, not just the first,
    so the human reviewer sees the full reasoning."""
    violations = []

    status = state.get("account_status")
    if status == "UNKNOWN":
        violations.append(f"Customer {state.get('customer_id')!r} not found in CRM")
    elif status in BLOCKED_STATUSES:
        violations.append(f"Account is {status}")

    if state.get("urgency") not in {"LOW", "MEDIUM", "HIGH"}:
        violations.append(f"Urgency could not be determined ({state.get('urgency')!r})")
    elif state.get("urgency") == "HIGH":
        violations.append("HIGH urgency tickets go to a human")

    past = state.get("past_tickets") or 0
    if past > MAX_PAST_TICKETS_FOR_AUTO:
        violations.append(f"{past} past tickets (auto-resolve limit is {MAX_PAST_TICKETS_FOR_AUTO})")

    intent = (state.get("intent") or "").lower()
    if "refund" in intent and state.get("plan_tier") == "FREE":
        violations.append("Refund requested on a FREE plan (no charges to refund)")

    return {"rules_passed": not violations, "rule_violations": violations}


def route_after_rules(state: TicketState) -> str:
    """The conditional edge: picks the next node based on rule_check's result."""
    return "draft_resolution" if state["rules_passed"] else "escalate"


# ---------------------------------------------------------------- node 4a (code)
def draft_resolution(state: TicketState) -> dict:
    """Drafts an action for a clean ticket. Template-based, not LLM - the action
    itself should be predictable."""
    return {
        "route": "draft_resolution",
        "draft_action": (
            f"Process '{state.get('intent')}' for {state['customer_id']} "
            f"({state['plan_tier']} plan)"
        ),
        "reasoning": (
            f"All rules passed: account {state['account_status']}, "
            f"urgency {state['urgency']}, {state['past_tickets']} past tickets."
        ),
        "status": "AWAITING_APPROVAL",
    }


# ---------------------------------------------------------------- node 4b (code)
def escalate(state: TicketState) -> dict:
    """Hands the ticket to a human with the reasons it was not auto-drafted."""
    return {
        "route": "escalate",
        "draft_action": f"Escalate {state.get('customer_id') or 'UNKNOWN'} to a human agent",
        "reasoning": "Rules failed: " + "; ".join(state["rule_violations"]),
        "status": "AWAITING_APPROVAL",
    }


# ---------------------------------------------------------------- wiring
def build_graph(parse_node=parse_ticket):
    """Wires the nodes together. `parse_node` can be swapped for a stub in
    tests, so the whole graph runs without calling Groq."""
    builder = StateGraph(TicketState)
    builder.add_node("parse_ticket", parse_node)
    builder.add_node("lookup_customer", lookup_customer)
    builder.add_node("rule_check", rule_check)
    builder.add_node("draft_resolution", draft_resolution)
    builder.add_node("escalate", escalate)

    builder.set_entry_point("parse_ticket")
    builder.add_edge("parse_ticket", "lookup_customer")
    builder.add_edge("lookup_customer", "rule_check")
    builder.add_conditional_edges(
        "rule_check",
        route_after_rules,
        {"draft_resolution": "draft_resolution", "escalate": "escalate"},
    )
    builder.add_edge("draft_resolution", END)
    builder.add_edge("escalate", END)
    return builder.compile()


graph = build_graph()


if __name__ == "__main__":
    # One ticket per path, plus edge cases, so you can see the branching work
    sample_tickets = [
        # DELINQUENT + 12 past tickets -> escalate
        "Hi, this is customer C-1002. I was charged twice for my subscription "
        "this month and need a refund for the duplicate charge.",
        # ACTIVE, ENTERPRISE, 3 past tickets -> draft_resolution
        "Hello, customer C-1001 here. I can't log in to the dashboard since "
        "yesterday, could you reset my access when you get a chance?",
        # ACTIVE but FREE plan asking for refund -> escalate
        "This is C-1003, please refund my last payment.",
        # Customer not in CRM -> escalate
        "Customer C-9999 here, my invoice looks wrong.",
    ]

    for ticket in sample_tickets:
        result = graph.invoke({"raw_text": ticket})
        print("\n" + "=" * 70)
        for key, value in result.items():
            print(f"{key}: {value}")
