"""
app.py - exposes the ticket agent as an HTTP API with a human approval step.

Run (from the project folder, with GROQ_API_KEY set and crm.db created):
    uvicorn app:app --reload

Then open http://127.0.0.1:8000 (redirects to the /docs test page).

Endpoints
    POST /ticket                   run a ticket through the agent, save it as AWAITING_APPROVAL
    GET  /tickets?status=...       list saved tickets (optionally filtered)
    GET  /ticket/{id}              one ticket
    POST /ticket/{id}/approve      human approves the agent's proposed action
    POST /ticket/{id}/reject       human rejects it
    GET  /health                   server check
"""

from dotenv import load_dotenv
load_dotenv()  # reads GROQ_API_KEY and GROQ_MODEL from .env

from contextlib import asynccontextmanager
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

import store
from graph import graph


@asynccontextmanager
async def lifespan(app: FastAPI):
    store.init_store()  # make sure tickets.db and its table exist
    yield


app = FastAPI(
    title="Ticket Triage Agent",
    description=(
        "LLM parses the ticket, plain code checks the CRM and business rules, "
        "and every proposed action waits for a human to approve or reject it."
    ),
    version="2.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------- models
class TicketIn(BaseModel):
    text: str = Field(
        ...,
        min_length=5,
        examples=[
            "Hi, this is customer C-1002. I was charged twice this month "
            "and need a refund for the duplicate charge."
        ],
    )


class AgentResult(BaseModel):
    customer_id: Optional[str]
    intent: Optional[str]
    urgency: Optional[str]
    account_status: Optional[str]
    plan_tier: Optional[str]
    past_tickets: Optional[int]
    rules_passed: bool
    rule_violations: list[str]
    route: str
    draft_action: str
    reasoning: str


class TicketRecord(BaseModel):
    ticket_id: str
    created_at: str
    raw_text: str
    agent_result: AgentResult
    status: str
    reviewer_note: Optional[str]
    decided_at: Optional[str]


class Decision(BaseModel):
    note: Optional[str] = Field(None, examples=["Checked billing, duplicate charge confirmed."])


# ---------------------------------------------------------------- routes
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ticket", response_model=TicketRecord, status_code=201)
def process_ticket(ticket: TicketIn):
    """Runs one ticket through the full graph and saves the result for review.

    Plain `def` (not async) because graph.invoke is blocking; FastAPI runs it
    in a worker thread so the server stays responsive.
    """
    try:
        result = graph.invoke({"raw_text": ticket.text})
    except KeyError as e:
        raise HTTPException(status_code=500, detail=f"Missing setting: {e}")
    except ValueError as e:
        raise HTTPException(status_code=502, detail=f"Could not parse ticket: {e}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Agent failed: {e}")

    agent_result = {k: result.get(k) for k in AgentResult.model_fields}
    return store.save_ticket(ticket.text, agent_result)


@app.get("/tickets", response_model=list[TicketRecord])
def list_tickets(
    status: Optional[Literal["AWAITING_APPROVAL", "APPROVED", "REJECTED"]] = Query(None),
):
    """The review queue. Use status=AWAITING_APPROVAL to see what needs a human."""
    return store.list_tickets(status)


@app.get("/ticket/{ticket_id}", response_model=TicketRecord)
def get_ticket(ticket_id: str):
    ticket = store.get_ticket(ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail=f"Ticket {ticket_id} not found")
    return ticket


def _decide(ticket_id: str, decision: str, body: Optional[Decision]) -> dict:
    note = body.note if body else None
    try:
        ticket = store.decide_ticket(ticket_id, decision, note)
    except store.AlreadyDecided as e:
        raise HTTPException(status_code=409, detail=str(e))
    if ticket is None:
        raise HTTPException(status_code=404, detail=f"Ticket {ticket_id} not found")
    return ticket


@app.post("/ticket/{ticket_id}/approve", response_model=TicketRecord)
def approve_ticket(ticket_id: str, body: Optional[Decision] = None):
    """A human signs off on the agent's proposed action."""
    return _decide(ticket_id, store.APPROVED, body)


@app.post("/ticket/{ticket_id}/reject", response_model=TicketRecord)
def reject_ticket(ticket_id: str, body: Optional[Decision] = None):
    """A human turns down the agent's proposed action."""
    return _decide(ticket_id, store.REJECTED, body)