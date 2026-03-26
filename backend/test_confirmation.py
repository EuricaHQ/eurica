"""Validation tests for the confirmation guard (spec v2.8.1).

Run: python test_confirmation.py
"""

from dataclasses import replace
from machine.states import State
from machine.events import Event
from machine.context import DecisionContext
from machine.transition import transition


def _base_context() -> DecisionContext:
    """Minimal context that satisfies participation (1 participant responded)."""
    return DecisionContext(
        decision_id="test",
        question="Where should we eat?",
        participants=["alice"],
        min_participants=1,
        responses={"alice": ["Italian"]},
        preferences=["Italian"],
    )


def test_consent_clean_auto_decide():
    """Consent + no fragility → DECIDING + DECISION_CONFIRMED → DECIDED."""
    ctx = _base_context()
    assert ctx.decision_rule == "consent"

    # Simulate system-event: AGGREGATION_COMPLETED in AGGREGATING → DECIDING
    state, actions, ctx = transition(State.AGGREGATING, Event.AGGREGATION_COMPLETED, ctx)
    assert state == State.DECIDING, f"expected deciding, got {state.value}"

    # System-event: DECISION_CONFIRMED (no confirmation required)
    state, actions, ctx = transition(State.DECIDING, Event.DECISION_CONFIRMED, ctx)
    assert state == State.DECIDED, f"expected decided, got {state.value}"


def test_majority_clean_auto_decide():
    """Majority rule + no fragility → same as consent: auto-decide."""
    ctx = replace(_base_context(), decision_rule="majority")

    state, actions, ctx = transition(State.AGGREGATING, Event.AGGREGATION_COMPLETED, ctx)
    assert state == State.DECIDING, f"expected deciding, got {state.value}"

    state, actions, ctx = transition(State.DECIDING, Event.DECISION_CONFIRMED, ctx)
    assert state == State.DECIDED, f"expected decided, got {state.value}"


def test_uncertainty_requires_confirmation():
    """Uncertainty present → confirmation_required → stays in DECIDING."""
    ctx = replace(_base_context(), uncertainties=["budget unclear"])

    state, actions, ctx = transition(State.AGGREGATING, Event.AGGREGATION_COMPLETED, ctx)
    assert state == State.DECIDING, f"expected deciding, got {state.value}"

    # No system-event should auto-confirm. Simulate: no DECISION_CONFIRMED fired.
    # Instead, send RESPONSE_RECEIVED — machine has no transition for this in DECIDING.
    state, actions, ctx = transition(State.DECIDING, Event.RESPONSE_RECEIVED, ctx)
    assert state == State.DECIDING, f"expected deciding (stuck), got {state.value}"
    assert actions == [], "expected no actions"


def test_initiator_approval_requires_confirmation():
    """Initiator approval required → confirmation_required → stays in DECIDING."""
    ctx = replace(_base_context(), requires_initiator_approval=True, initiator="alice")

    state, actions, ctx = transition(State.AGGREGATING, Event.AGGREGATION_COMPLETED, ctx)
    assert state == State.DECIDING, f"expected deciding, got {state.value}"

    # Same check: unrelated event should not advance
    state, actions, ctx = transition(State.DECIDING, Event.RESPONSE_RECEIVED, ctx)
    assert state == State.DECIDING, f"expected deciding (stuck), got {state.value}"
    assert actions == [], "expected no actions"


def test_explicit_approval_requires_confirmation():
    """Explicit approval required → confirmation_required → stays in DECIDING."""
    ctx = replace(_base_context(), requires_explicit_approval=True)

    state, actions, ctx = transition(State.AGGREGATING, Event.AGGREGATION_COMPLETED, ctx)
    assert state == State.DECIDING, f"expected deciding, got {state.value}"

    state, actions, ctx = transition(State.DECIDING, Event.RESPONSE_RECEIVED, ctx)
    assert state == State.DECIDING, f"expected deciding (stuck), got {state.value}"
    assert actions == [], "expected no actions"


if __name__ == "__main__":
    tests = [
        test_consent_clean_auto_decide,
        test_majority_clean_auto_decide,
        test_uncertainty_requires_confirmation,
        test_initiator_approval_requires_confirmation,
        test_explicit_approval_requires_confirmation,
    ]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} tests passed.")
