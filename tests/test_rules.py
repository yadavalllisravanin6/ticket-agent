"""
Tests for the deterministic parts of the agent: rule_check, the router, and
the full graph path for each mock customer. No LLM calls.
"""

import pytest

from graph import MAX_PAST_TICKETS_FOR_AUTO, route_after_rules, rule_check


def clean_state(**overrides):
    """A ticket that passes every rule. Each test breaks exactly one thing."""
    state = {
        "customer_id": "C-1001",
        "intent": "login issue",
        "urgency": "LOW",
        "account_status": "ACTIVE",
        "plan_tier": "ENTERPRISE",
        "past_tickets": 3,
    }
    state.update(overrides)
    return state


# ---------------------------------------------------------------- rule_check
def test_clean_ticket_passes():
    result = rule_check(clean_state())
    assert result == {"rules_passed": True, "rule_violations": []}


@pytest.mark.parametrize("status", ["DELINQUENT", "SUSPENDED"])
def test_blocked_account_status_fails(status):
    result = rule_check(clean_state(account_status=status))
    assert not result["rules_passed"]
    assert result["rule_violations"] == [f"Account is {status}"]


def test_unknown_customer_fails():
    result = rule_check(clean_state(account_status="UNKNOWN", customer_id="C-9999"))
    assert not result["rules_passed"]
    assert "not found in CRM" in result["rule_violations"][0]


def test_high_urgency_fails():
    result = rule_check(clean_state(urgency="HIGH"))
    assert result["rule_violations"] == ["HIGH urgency tickets go to a human"]


@pytest.mark.parametrize("urgency", [None, "", "URGENT", "low"])
def test_unrecognised_urgency_fails_safe(urgency):
    # Anything not exactly LOW/MEDIUM/HIGH escalates rather than auto-drafting
    result = rule_check(clean_state(urgency=urgency))
    assert not result["rules_passed"]
    assert "Urgency could not be determined" in result["rule_violations"][0]


def test_past_tickets_at_limit_passes():
    assert rule_check(clean_state(past_tickets=MAX_PAST_TICKETS_FOR_AUTO))["rules_passed"]


def test_past_tickets_over_limit_fails():
    result = rule_check(clean_state(past_tickets=MAX_PAST_TICKETS_FOR_AUTO + 1))
    assert not result["rules_passed"]


def test_refund_on_free_plan_fails():
    result = rule_check(clean_state(intent="Refund Request", plan_tier="FREE"))
    assert "FREE plan" in result["rule_violations"][0]


def test_non_refund_on_free_plan_passes():
    assert rule_check(clean_state(intent="login issue", plan_tier="FREE"))["rules_passed"]


def test_all_violations_are_collected():
    # C-1002's real profile breaks two rules; both must be reported
    result = rule_check(clean_state(account_status="DELINQUENT", past_tickets=12))
    assert len(result["rule_violations"]) == 2


# ---------------------------------------------------------------- router
def test_router_sends_pass_to_draft():
    assert route_after_rules({"rules_passed": True}) == "draft_resolution"


def test_router_sends_fail_to_escalate():
    assert route_after_rules({"rules_passed": False}) == "escalate"


# ---------------------------------------------------------------- full graph
@pytest.mark.parametrize(
    "customer, expected_route",
    [
        ("C-1001", "draft_resolution"),  # ACTIVE, ENTERPRISE, 3 tickets
        ("C-1002", "escalate"),          # DELINQUENT, 12 tickets
        ("C-1003", "escalate"),          # refund on FREE plan
        ("C-1004", "escalate"),          # SUSPENDED
        ("C-1005", "escalate"),          # HIGH urgency
        ("C-9999", "escalate"),          # not in CRM
    ],
)
def test_graph_routes_each_customer(stub_graph, customer, expected_route):
    result = stub_graph.invoke({"raw_text": f"Hi, this is customer {customer}."})
    assert result["route"] == expected_route
    assert result["status"] == "AWAITING_APPROVAL"


def test_graph_with_no_customer_id_escalates(stub_graph):
    result = stub_graph.invoke({"raw_text": "Something is broken, please help."})
    assert result["route"] == "escalate"
