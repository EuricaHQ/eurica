"""Tests for LLM advisory layer on has_critical_unresolved_participants.

Verifies:
- rule-based False + LLM says critical → True (escalation)
- rule-based True → LLM not called (short-circuit)
- LLM failure → fallback to rule-based result
- LLM returns invalid data → safe fallback
- advisor can be disabled (set to None)

Uses mock LLM functions — no real API calls.

Run: python3 test_critical_llm_advisory.py
"""

from machine.context import DecisionContext
from machine.transition import (
    _has_critical_unresolved_participants,
    _has_critical_unresolved_rule_based,
    set_critical_participant_advisor,
)


def _teardown():
    """Reset advisor to None after each test."""
    set_critical_participant_advisor(None)


# ---------------------------------------------------------------------------
# Context where rule-based returns False
# (majority locked: Italian has 3/5, threshold 3, eve missing)
# ---------------------------------------------------------------------------

def _majority_locked_ctx():
    return DecisionContext(
        question="Where to eat?",
        participants=["alice", "bob", "carol", "dave", "eve"],
        min_participants=3,
        decision_rule="majority",
        responses={
            "alice": ["Italian"], "bob": ["Italian"],
            "carol": ["Italian"], "dave": ["Thai"],
        },
        preferences=["Italian", "Italian", "Italian", "Thai"],
    )


# ---------------------------------------------------------------------------
# Context where rule-based returns True (consent + missing)
# ---------------------------------------------------------------------------

def _consent_missing_ctx():
    return DecisionContext(
        question="Where to eat?",
        participants=["alice", "bob", "carol"],
        min_participants=2,
        decision_rule="consent",
        responses={"alice": ["Italian"], "carol": ["Thai"]},
        preferences=["Italian", "Thai"],
    )


# ---------------------------------------------------------------------------
# Test: rule-based False + LLM escalates → True
# ---------------------------------------------------------------------------

def test_llm_escalates_when_rule_says_false():
    """LLM flags eve as critical despite majority being locked."""
    ctx = _majority_locked_ctx()

    # Precondition: rule-based says False
    assert not _has_critical_unresolved_rule_based(ctx)

    # Mock LLM that flags eve
    def mock_advisor(context, missing):
        assert "eve" in missing
        return {"critical_participants": ["eve"]}

    set_critical_participant_advisor(mock_advisor)
    try:
        assert _has_critical_unresolved_participants(ctx) is True
    finally:
        _teardown()


# ---------------------------------------------------------------------------
# Test: rule-based False + LLM says no one critical → False
# ---------------------------------------------------------------------------

def test_llm_agrees_not_critical():
    """LLM agrees no one is critical → result stays False."""
    ctx = _majority_locked_ctx()

    def mock_advisor(context, missing):
        return {"critical_participants": []}

    set_critical_participant_advisor(mock_advisor)
    try:
        assert _has_critical_unresolved_participants(ctx) is False
    finally:
        _teardown()


# ---------------------------------------------------------------------------
# Test: rule-based True → LLM not called
# ---------------------------------------------------------------------------

def test_llm_not_called_when_rule_says_true():
    """When rule-based returns True, LLM should never be invoked."""
    ctx = _consent_missing_ctx()

    # Precondition: rule-based says True
    assert _has_critical_unresolved_rule_based(ctx) is True

    call_count = 0

    def mock_advisor(context, missing):
        nonlocal call_count
        call_count += 1
        return {"critical_participants": []}

    set_critical_participant_advisor(mock_advisor)
    try:
        result = _has_critical_unresolved_participants(ctx)
        assert result is True
        assert call_count == 0, f"LLM was called {call_count} times, expected 0"
    finally:
        _teardown()


# ---------------------------------------------------------------------------
# Test: LLM raises exception → fallback to rule-based (False)
# ---------------------------------------------------------------------------

def test_llm_failure_falls_back():
    """LLM crashes → guard returns rule-based result (False)."""
    ctx = _majority_locked_ctx()

    def crashing_advisor(context, missing):
        raise RuntimeError("API timeout")

    set_critical_participant_advisor(crashing_advisor)
    try:
        # Should not crash; should return False (rule-based result)
        assert _has_critical_unresolved_participants(ctx) is False
    finally:
        _teardown()


# ---------------------------------------------------------------------------
# Test: LLM returns invalid data → safe fallback
# ---------------------------------------------------------------------------

def test_llm_returns_invalid_type():
    """LLM returns non-list → treated as empty → False."""
    ctx = _majority_locked_ctx()

    def bad_advisor(context, missing):
        return {"critical_participants": "eve"}  # string, not list

    set_critical_participant_advisor(bad_advisor)
    try:
        assert _has_critical_unresolved_participants(ctx) is False
    finally:
        _teardown()


def test_llm_returns_no_key():
    """LLM returns dict without expected key → empty → False."""
    ctx = _majority_locked_ctx()

    def bad_advisor(context, missing):
        return {"other_key": ["eve"]}

    set_critical_participant_advisor(bad_advisor)
    try:
        assert _has_critical_unresolved_participants(ctx) is False
    finally:
        _teardown()


def test_llm_returns_none():
    """LLM returns None → exception caught → False."""
    ctx = _majority_locked_ctx()

    def bad_advisor(context, missing):
        return None

    set_critical_participant_advisor(bad_advisor)
    try:
        assert _has_critical_unresolved_participants(ctx) is False
    finally:
        _teardown()


# ---------------------------------------------------------------------------
# Test: no advisor set → rule-based only
# ---------------------------------------------------------------------------

def test_no_advisor_set():
    """When no advisor is registered, guard uses rule-based only."""
    ctx = _majority_locked_ctx()
    set_critical_participant_advisor(None)

    # Rule-based says False, no advisor → False
    assert _has_critical_unresolved_participants(ctx) is False


# ---------------------------------------------------------------------------
# Test: LLM returns names not in missing → filtered out
# ---------------------------------------------------------------------------

def test_llm_returns_invalid_names():
    """LLM returns names not in missing list → filtered by OpenAILLM.
    At the guard level, any non-empty list escalates."""
    ctx = _majority_locked_ctx()

    # This advisor returns a name that IS in missing (eve)
    # plus a name that isn't (alice) — guard just checks len > 0
    def mock_advisor(context, missing):
        return {"critical_participants": ["alice", "eve"]}

    set_critical_participant_advisor(mock_advisor)
    try:
        # "alice" is not in missing but "eve" is — list is non-empty → True
        assert _has_critical_unresolved_participants(ctx) is True
    finally:
        _teardown()


if __name__ == "__main__":
    tests = [
        test_llm_escalates_when_rule_says_false,
        test_llm_agrees_not_critical,
        test_llm_not_called_when_rule_says_true,
        test_llm_failure_falls_back,
        test_llm_returns_invalid_type,
        test_llm_returns_no_key,
        test_llm_returns_none,
        test_no_advisor_set,
        test_llm_returns_invalid_names,
    ]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} LLM advisory tests passed.")
