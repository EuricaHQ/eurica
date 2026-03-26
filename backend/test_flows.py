"""Multi-step flow validation tests.

Verify that event sequences produce correct state progressions
with no deadlocks.

Run: python test_flows.py
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
    """collecting → aggregating → deciding → decided (auto-finalize)."""
    ctx = _base_context()

    state, _, ctx = _step(
        State.COLLECTING, Event.RESPONSE_RECEIVED, ctx,
        State.AGGREGATING, "collect→aggregate",
    )
    state, _, ctx = _step(
        state, Event.AGGREGATION_COMPLETED, ctx,
        State.DECIDING, "aggregate→deciding",
    )
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
        State.COLLECTING, Event.RESPONSE_RECEIVED, ctx,
        State.AGGREGATING, "collect→aggregate",
    )
    state, _, ctx = _step(
        state, Event.AGGREGATION_COMPLETED, ctx,
        State.DECIDING, "aggregate→deciding",
    )

    # Unrelated event must not advance
    state, actions, ctx = transition(state, Event.RESPONSE_RECEIVED, ctx)
    assert state == State.DECIDING, "should stay in deciding with uncertainty"
    assert actions == [], "no actions on unmatched event"

    # Explicit confirmation unblocks
    state, _, ctx = _step(
        state, Event.DECISION_CONFIRMED, ctx,
        State.DECIDED, "deciding→decided (confirmed)",
    )


def test_conflict_loop():
    """aggregating → resolving → aggregating → deciding."""
    ctx = replace(_base_context(), conflicts=[{"type": "preference_clash"}])

    # Aggregation sees conflicts → resolving
    state, _, ctx = _step(
        State.AGGREGATING, Event.AGGREGATION_COMPLETED, ctx,
        State.RESOLVING, "aggregate→resolving",
    )

    # Conflict resolved: clear conflicts, respond
    ctx = replace(ctx, conflicts=[])
    state, _, ctx = _step(
        state, Event.RESPONSE_RECEIVED, ctx,
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


def test_rejection_continues():
    """deciding (confirmation required) → REJECTED → collecting → normal flow."""
    ctx = replace(_base_context(), uncertainties=["budget unclear"])

    state, _, ctx = _step(
        State.COLLECTING, Event.RESPONSE_RECEIVED, ctx,
        State.AGGREGATING, "collect→aggregate",
    )
    state, _, ctx = _step(
        state, Event.AGGREGATION_COMPLETED, ctx,
        State.DECIDING, "aggregate→deciding",
    )

    # Rejection → back to collecting
    state, _, ctx = _step(
        state, Event.DECISION_REJECTED, ctx,
        State.COLLECTING, "deciding→collecting (rejected)",
    )

    # System continues: new response → aggregating again
    ctx = replace(ctx, uncertainties=[])  # uncertainty resolved
    state, _, ctx = _step(
        state, Event.RESPONSE_RECEIVED, ctx,
        State.AGGREGATING, "collect→aggregate (retry)",
    )
    state, _, ctx = _step(
        state, Event.AGGREGATION_COMPLETED, ctx,
        State.DECIDING, "aggregate→deciding (retry)",
    )
    state, _, ctx = _step(
        state, Event.DECISION_CONFIRMED, ctx,
        State.DECIDED, "deciding→decided (retry)",
    )


if __name__ == "__main__":
    tests = [
        test_happy_path,
        test_uncertainty_then_confirm,
        test_conflict_loop,
        test_rejection_continues,
    ]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} flow tests passed.")
