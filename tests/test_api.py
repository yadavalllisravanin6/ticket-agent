"""
Tests for the HTTP API and the approval flow. No LLM calls.
"""


def submit(client, customer="C-1002"):
    r = client.post("/ticket", json={"text": f"Hi, this is customer {customer}. Please help."})
    assert r.status_code == 201
    return r.json()


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_root_redirects_to_docs(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (302, 307)
    assert r.headers["location"] == "/docs"


def test_submit_saves_ticket_awaiting_approval(client):
    t = submit(client, "C-1002")
    assert t["ticket_id"].startswith("T-")
    assert t["status"] == "AWAITING_APPROVAL"
    assert t["agent_result"]["route"] == "escalate"
    assert t["reviewer_note"] is None


def test_submit_clean_customer_drafts(client):
    t = submit(client, "C-1001")
    assert t["agent_result"]["route"] == "draft_resolution"
    assert t["agent_result"]["rules_passed"] is True


def test_too_short_text_rejected(client):
    assert client.post("/ticket", json={"text": "hi"}).status_code == 422


def test_get_ticket(client):
    t = submit(client)
    r = client.get(f"/ticket/{t['ticket_id']}")
    assert r.status_code == 200
    assert r.json()["ticket_id"] == t["ticket_id"]


def test_get_missing_ticket_404(client):
    assert client.get("/ticket/T-NOPE").status_code == 404


def test_approve_with_note(client):
    t = submit(client, "C-1001")
    r = client.post(f"/ticket/{t['ticket_id']}/approve", json={"note": "Looks right."})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "APPROVED"
    assert body["reviewer_note"] == "Looks right."
    assert body["decided_at"] is not None


def test_reject_without_body(client):
    t = submit(client)
    r = client.post(f"/ticket/{t['ticket_id']}/reject")
    assert r.status_code == 200
    assert r.json()["status"] == "REJECTED"


def test_cannot_decide_twice(client):
    t = submit(client)
    client.post(f"/ticket/{t['ticket_id']}/approve")
    r = client.post(f"/ticket/{t['ticket_id']}/reject")
    assert r.status_code == 409
    # The first decision stands
    assert client.get(f"/ticket/{t['ticket_id']}").json()["status"] == "APPROVED"


def test_approve_missing_ticket_404(client):
    assert client.post("/ticket/T-NOPE/approve").status_code == 404


def test_list_and_filter_queue(client):
    a = submit(client, "C-1001")
    b = submit(client, "C-1002")
    client.post(f"/ticket/{a['ticket_id']}/approve")

    all_ids = {t["ticket_id"] for t in client.get("/tickets").json()}
    assert all_ids == {a["ticket_id"], b["ticket_id"]}

    pending = client.get("/tickets", params={"status": "AWAITING_APPROVAL"}).json()
    assert [t["ticket_id"] for t in pending] == [b["ticket_id"]]

    approved = client.get("/tickets", params={"status": "APPROVED"}).json()
    assert [t["ticket_id"] for t in approved] == [a["ticket_id"]]


def test_invalid_status_filter_rejected(client):
    assert client.get("/tickets", params={"status": "DONE"}).status_code == 422


def test_agent_failure_returns_502(client, monkeypatch):
    import app as app_module

    class Broken:
        def invoke(self, _):
            raise RuntimeError("Groq is down")

    monkeypatch.setattr(app_module, "graph", Broken())
    r = client.post("/ticket", json={"text": "C-1001 cannot log in"})
    assert r.status_code == 502
    # Nothing half-finished is saved
    assert client.get("/tickets").json() == []
