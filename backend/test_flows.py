"""Multi-step flow validation tests (spec v2.9 aligned).

Verify that event sequences produce correct state progressions
with no deadlocks.

Run: python3 test_flows.py
"""

from dataclasses import replace
from machine.states import State
from machine.events import Event
from machine.context import DecisionContext
from machine.transition import transition


def _base_context() -> DecisionContext:
    return DecisionContext(
        decision_id="flow",
        question="Where should we eat?",
        participants=["alice"],
        min_participants=1,
        responses={"alice": ["Italian"]},
        preferences=["Italian"],
    )


def _step(state, event, ctx, expected_state, label=""):
    next_state, actions, ctx = transition(state, event, ctx)
    tag = f" ({label})" if label else ""
    assert next_state == expected_state, (
        f"{tag} expected {expected_state.value}, got {next_state.value}"
    )
    return next_state, actions, ctx


def test_happy_path():
    """collecting → (AGGREGATION_COMPLETED) → aggregating → deciding → decided."""
    ctx = _base_context()

    # RESPONSE_RECEIVED stays in COLLECTING (spec v2.9)
    state, _, ctx = _step(
        State.COLLECTING, Event.RESPONSE_RECEIVED, ctx,
        State.COLLECTING, "collect stays",
    )
    # AGGREGATION_COMPLETED with participation satisfied → AGGREGATING
    state, _, ctx = _step(
        state, Event.AGGREGATION_COMPLETED, ctx,
        State.AGGREGATING, "collect→aggregate",
    )
    # AGGREGATION_COMPLETED in AGGREGATING → DECIDING (solution_found)
    state, _, ctx = _step(
        state, Event.AGGREGATION_COMPLETED, ctx,
        State.DECIDING, "aggregate→deciding",
    )
    # DECISION_CONFIRMED → DECIDED
    state, _, ctx = _step(
        state, Event.DECISION_CONFIRMED, ctx,
        State.DECIDED, "deciding→decided",
    )

    # Terminal: no further transitions
    state, actions, ctx = transition(state, Event.RESPONSE_RECEIVED, ctx)
    assert state == State.DECIDED, "terminal state must not change"
    assert actions == [], "terminal state must emit no actions"


def test_uncertainty_then_confirm():
    """collecting → aggregating → deciding → (wait) → confirmed → decided."""
    ctx = replace(_base_context(), uncertainties=["budget unclear"])

    state, _, ctx = _step(
        State.COLLECTING, Event.AGGREGATION_COMPLETED, ctx,
        State.AGGREGATING, "collect→aggregate",
    )
    state, _, ctx = _step(
        state, Event.AGGREGATION_COMPLETED, ctx,
        State.DECIDING, "aggregate→deciding",
    )

    # RESPONSE_RECEIVED self-transitions in DECIDING (spec v2.9)
    state, actions, ctx = transition(state, Event.RESPONSE_RECEIVED, ctx)
    assert state == State.DECIDING, "should stay in deciding with uncertainty"
    assert actions == [], "no actions on self-transition"

    # Explicit confirmation unblocks
    state, _, ctx = _step(
        state, Event.DECISION_CONFIRMED, ctx,
        State.DECIDED, "deciding→decided (confirmed)",
    )


def test_conflict_loop():
    """aggregating → resolving → (AGGREGATION_COMPLETED) → aggregating → deciding."""
    ctx = replace(_base_context(), conflicts=[{
        "participants": ["alice"],
        "source": "llm_signal",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "status": "open",
    }])

    # Aggregation sees conflicts → resolving
    state, _, ctx = _step(
        State.AGGREGATING, Event.AGGREGATION_COMPLETED, ctx,
        State.RESOLVING, "aggregate→resolving",
    )

    # RESPONSE_RECEIVED stays in RESOLVING (spec v2.9)
    ctx = replace(ctx, conflicts=[])
    state, _, ctx = _step(
        state, Event.RESPONSE_RECEIVED, ctx,
        State.RESOLVING, "resolving stays",
    )

    # AGGREGATION_COMPLETED is the return path → AGGREGATING
    state, _, ctx = _step(
        state, Event.AGGREGATION_COMPLETED, ctx,
        State.AGGREGATING, "resolving→aggregate",
    )

    # Second aggregation: no conflicts → deciding
    state, _, ctx = _step(
        state, Event.AGGREGATION_COMPLETED, ctx,
        State.DECIDING, "aggregate→deciding",
    )

    # Auto-finalize (no fragility)
    state, _, ctx = _step(
        state, Event.DECISION_CONFIRMED, ctx,
        State.DECIDED, "deciding→decided",
    )


def test_rejection_goes_to_aggregating():
    """deciding → REJECTED → aggregating (spec v2.9, not collecting)."""
    ctx = replace(_base_context(), uncertainties=["budget unclear"])

    state, _, ctx = _step(
        State.COLLECTING, Event.AGGREGATION_COMPLETED, ctx,
        State.AGGREGATING, "collect→aggregate",
    )
    state, _, ctx = _step(
        state, Event.AGGREGATION_COMPLETED, ctx,
        State.DECIDING, "aggregate→deciding",
    )

    # Rejection → back to AGGREGATING (spec v2.9)
    state, _, ctx = _step(
        state, Event.DECISION_REJECTED, ctx,
        State.AGGREGATING, "deciding→aggregating (rejected)",
    )

    # Re-aggregate: AGGREGATION_COMPLETED → deciding again
    ctx = replace(ctx, uncertainties=[])  # uncertainty resolved
    state, _, ctx = _step(
        state, Event.AGGREGATION_COMPLETED, ctx,
        State.DECIDING, "aggregate→deciding (retry)",
    )
    state, _, ctx = _step(
        state, Event.DECISION_CONFIRMED, ctx,
        State.DECIDED, "deciding→decided (retry)",
    )


def test_validation_returns_via_aggregating():
    """aggregating → validating → VALIDATION_COMPLETED → aggregating (spec v2.9).

    Validation always returns to AGGREGATING, never directly to DECIDING.
    """
    ctx = replace(_base_context(), constraints=["budget < 50"])

    # Aggregation sees constraints → validating
    state, _, ctx = _step(
        State.AGGREGATING, Event.AGGREGATION_COMPLETED, ctx,
        State.VALIDATING, "aggregate→validating",
    )

    # Validation completes → always back to AGGREGATING
    state, _, ctx = _step(
        state, Event.VALIDATION_COMPLETED, ctx,
        State.AGGREGATING, "validating→aggregating",
    )

    # Clear constraints so solution_found passes, re-aggregate → deciding
    ctx = replace(ctx, constraints=[])
    state, _, ctx = _step(
        state, Event.AGGREGATION_COMPLETED, ctx,
        State.DECIDING, "aggregate→deciding",
    )


def test_avoiding_returns_via_aggregating():
    """avoiding → AGGREGATION_COMPLETED → aggregating (spec v2.9)."""
    ctx = _base_context()

    # In AVOIDING, RESPONSE_RECEIVED stays
    state, _, ctx = _step(
        State.AVOIDING, Event.RESPONSE_RECEIVED, ctx,
        State.AVOIDING, "avoiding stays",
    )

    # AGGREGATION_COMPLETED is the return path → AGGREGATING
    state, _, ctx = _step(
        state, Event.AGGREGATION_COMPLETED, ctx,
        State.AGGREGATING, "avoiding→aggregating",
    )


def test_collecting_participation_not_satisfied():
    """AGGREGATION_COMPLETED in COLLECTING with insufficient participation → stay."""
    ctx = DecisionContext(
        decision_id="flow",
        question="Where should we eat?",
        participants=["alice"],
        min_participants=2,  # Need 2, only have 1
        responses={"alice": ["Italian"]},
    )

    state, _, ctx = _step(
        State.COLLECTING, Event.AGGREGATION_COMPLETED, ctx,
        State.COLLECTING, "collect stays (not enough participants)",
    )


def test_terminal_states_absorbing():
    """DECIDED and INFEASIBLE absorb all events."""
    ctx = _base_context()

    for terminal in [State.DECIDED, State.INFEASIBLE]:
        for event in Event:
            state, actions, _ = transition(terminal, event, ctx)
            assert state == terminal, (
                f"{terminal.value} + {event.value} should stay terminal"
            )
            assert actions == [], (
                f"{terminal.value} + {event.value} should emit no actions"
            )


if __name__ == "__main__":
    tests = [
        test_happy_path,
        test_uncertainty_then_confirm,
        test_conflict_loop,
        test_rejection_goes_to_aggregating,
        test_validation_returns_via_aggregating,
        test_avoiding_returns_via_aggregating,
        test_collecting_participation_not_satisfied,
        test_terminal_states_absorbing,
    ]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} flow tests passed.")
