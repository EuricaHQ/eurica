"""Tests for spec v2.10.5 — Guard Priority Order & Conflict Activation.

Covers:
1. Conflict triggers RESOLVING from AGGREGATING (not VALIDATING)
2. Conflict blocks DECIDING (has_conflict overrides solution_found)
3. Guard priority enforced: conflict > validation > solution_found
4. Without conflict, validation still routes to VALIDATING
"""

import unittest
from dataclasses import replace

from machine.states import State
from machine.events import Event
from machine.context import DecisionContext
from machine.transition import transition


def _base_ctx(**overrides) -> DecisionContext:
    defaults = dict(
        question="When should we meet?",
        participants=["alice", "bob", "carol"],
        min_participants=2,
        responses={"alice": ["Wednesday"], "bob": ["Wednesday"]},
        preferences=["Wednesday", "Wednesday"],
    )
    defaults.update(overrides)
    return DecisionContext(**defaults)


def _open_conflict(actor: str = "carol") -> dict:
    return {
        "participants": [actor],
        "source": "llm_signal",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "status": "open",
    }


class TestGuardPriority(unittest.TestCase):

    def test_conflict_overrides_validation(self):
        """AGGREGATING + conflict + constraints → RESOLVING (not VALIDATING).

        Spec v2.10.5 §24: has_conflict has higher priority than needs_validation.
        """
        ctx = _base_ctx(
            conflicts=[_open_conflict()],
            constraints=["hard constraint"],
        )
        next_state, actions, _ = transition(
            State.AGGREGATING, Event.AGGREGATION_COMPLETED, ctx,
        )
        self.assertEqual(next_state, State.RESOLVING)

    def test_conflict_blocks_deciding(self):
        """AGGREGATING + conflict + all participation met → RESOLVING (not DECIDING).

        Even when solution_found conditions would be met, conflict takes priority.
        """
        ctx = _base_ctx(
            responses={"alice": ["Wed"], "bob": ["Wed"], "carol": ["Wed"]},
            preferences=["Wed", "Wed", "Wed"],
            conflicts=[_open_conflict()],
            min_participants=3,
        )
        next_state, actions, _ = transition(
            State.AGGREGATING, Event.AGGREGATION_COMPLETED, ctx,
        )
        self.assertEqual(next_state, State.RESOLVING)

    def test_validation_without_conflict_still_works(self):
        """AGGREGATING + constraints + no conflict → VALIDATING.

        When there's no conflict, needs_validation still works normally.
        """
        ctx = _base_ctx(
            constraints=["hard constraint"],
            conflicts=[],
        )
        next_state, actions, _ = transition(
            State.AGGREGATING, Event.AGGREGATION_COMPLETED, ctx,
        )
        self.assertEqual(next_state, State.VALIDATING)

    def test_solution_found_without_conflict_or_validation(self):
        """AGGREGATING + no conflict + no constraints + all responded → DECIDING."""
        ctx = _base_ctx(
            # All participants responded — no critical unresolved
            responses={"alice": ["Wed"], "bob": ["Wed"]},
            participants=["alice", "bob"],
            preferences=["Wed", "Wed"],
            conflicts=[],
            constraints=[],
        )
        next_state, actions, _ = transition(
            State.AGGREGATING, Event.AGGREGATION_COMPLETED, ctx,
        )
        self.assertEqual(next_state, State.DECIDING)

    def test_resolved_conflict_does_not_block(self):
        """AGGREGATING + resolved conflict only → does NOT trigger RESOLVING."""
        ctx = _base_ctx(
            # All participants responded
            responses={"alice": ["Wed"], "bob": ["Wed"]},
            participants=["alice", "bob"],
            preferences=["Wed", "Wed"],
            conflicts=[{
                "participants": ["bob"],
                "source": "llm_signal",
                "timestamp": "2026-01-01T00:00:00+00:00",
                "status": "resolved",
            }],
            constraints=[],
        )
        next_state, actions, _ = transition(
            State.AGGREGATING, Event.AGGREGATION_COMPLETED, ctx,
        )
        self.assertEqual(next_state, State.DECIDING)

    def test_conflict_plus_validation_plus_solution_all_present(self):
        """All guards would match — conflict wins due to priority."""
        ctx = _base_ctx(
            responses={"alice": ["Wed"], "bob": ["Wed"], "carol": ["Wed"]},
            preferences=["Wed", "Wed", "Wed"],
            conflicts=[_open_conflict()],
            constraints=["hard constraint"],
            min_participants=3,
        )
        next_state, actions, _ = transition(
            State.AGGREGATING, Event.AGGREGATION_COMPLETED, ctx,
        )
        self.assertEqual(next_state, State.RESOLVING)

    def test_resolving_action_is_resolve_conflict(self):
        """Transition to RESOLVING emits RESOLVE_CONFLICT action."""
        ctx = _base_ctx(conflicts=[_open_conflict()])
        _, actions, _ = transition(
            State.AGGREGATING, Event.AGGREGATION_COMPLETED, ctx,
        )
        action_types = [a.type.value for a in actions]
        self.assertIn("resolve_conflict", action_types)


if __name__ == "__main__":
    results = unittest.main(exit=False, verbosity=0)
    total = results.result.testsRun
    failures = len(results.result.failures) + len(results.result.errors)
    print(f"\n{total - failures}/{total} guard priority tests passed.")
