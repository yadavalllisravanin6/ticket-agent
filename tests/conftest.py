"""
Shared test setup. Nothing here calls Groq: the LLM parse step is replaced by a
stub that reads fixed answers, so tests are fast, free and repeatable.
"""

import os
import sys

import pytest

# Make the project folder importable when running `pytest` from it
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import db  # noqa: E402
import graph as graph_module  # noqa: E402
import store  # noqa: E402

# What the "LLM" would return for each customer in these tests
STUB_PARSES = {
    "C-1001": {"intent": "login issue", "urgency": "LOW"},
    "C-1002": {"intent": "refund request", "urgency": "MEDIUM"},
    "C-1003": {"intent": "refund request", "urgency": "LOW"},
    "C-1004": {"intent": "billing question", "urgency": "LOW"},
    "C-1005": {"intent": "service outage", "urgency": "HIGH"},
    "C-9999": {"intent": "invoice issue", "urgency": "LOW"},
}


def stub_parse(state):
    """Stands in for the LLM: finds the customer ID in the text."""
    for cid, fields in STUB_PARSES.items():
        if cid in state["raw_text"]:
            return {"customer_id": cid, **fields}
    return {"customer_id": None, "intent": None, "urgency": None}


@pytest.fixture
def temp_dbs(tmp_path, monkeypatch):
    """Fresh crm.db and tickets.db in a temp folder for every test."""
    crm = str(tmp_path / "crm.db")
    tickets = str(tmp_path / "tickets.db")
    monkeypatch.setattr(db, "DB_PATH", crm)
    monkeypatch.setattr(graph_module, "DB_PATH", crm)
    monkeypatch.setattr(store, "DB_PATH", tickets)
    db.setup_database()
    store.init_store()


@pytest.fixture
def stub_graph(temp_dbs):
    """The real graph with only the LLM node swapped out."""
    return graph_module.build_graph(parse_node=stub_parse)


@pytest.fixture
def client(stub_graph, monkeypatch):
    """FastAPI test client wired to the stub graph and temp databases."""
    from fastapi.testclient import TestClient

    import app as app_module

    monkeypatch.setattr(app_module, "graph", stub_graph)
    with TestClient(app_module.app) as c:
        yield c
