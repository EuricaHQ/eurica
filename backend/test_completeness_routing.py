"""Tests for completeness routing — spec v2.10.8 §37.4 + v2.10.9 governance.

1. Incomplete but compatible → NO DECIDING (stays in AGGREGATING)
2. Clarification triggered with correct interaction_type
3. Conflict still overrides completeness routing
4. Complete solution still leads to DECIDING
5. Empty preferences → no solution found (fallback)
6. proposed_dimensions has no effect on routing (v2.10.9)
7. expected_dimensions=None → dimension completeness satisfied (v2.10.9)

Run: python3 test_completeness_routing.py
"""

import unittest

from machine.states import State
from machine.events import Event
from machine.context import DecisionContext
from machine.actions import ActionType
from machine.transition import transition


def _base_context(**overrides) -> DecisionContext:
    defaults = dict(
        decision_id="comp",
        question="Where should we eat?",
        participants=["alice"],
        min_participants=1,
        responses={"alice": ["Wednesday evening"]},
    )
    defaults.update(overrides)
    return DecisionContext(**defaults)


class TestCompletenessRouting(unittest.TestCase):

    def test_incomplete_stays_in_aggregating(self):
        """solution_found but missing expected dimension → stays AGGREGATING."""
        ctx = _base_context(
            preferences=[{"value": "Wednesday", "dimension": "day"}],
            expected_dimensions=["day", "time"],
        )
        next_state, actions, _ = transition(
            State.AGGREGATING, Event.AGGREGATION_COMPLETED, ctx,
        )
        self.assertNotEqual(next_state, State.DECIDING)
        self.assertEqual(next_state, State.AGGREGATING)
        action_types = [a.type for a in actions]
        self.assertIn(ActionType.ASK_QUESTION, action_types)

    def test_incomplete_interaction_type_clarify_dimension(self):
        """Missing dimension → interaction_type = clarify_dimension."""
        ctx = _base_context(
            preferences=[{"value": "Wednesday", "dimension": "day"}],
            expected_dimensions=["day", "time"],
        )
        _, actions, _ = transition(
            State.AGGREGATING, Event.AGGREGATION_COMPLETED, ctx,
        )
        self.assertEqual(len(actions), 1)
        payload = actions[0].payload
        self.assertIsNotNone(payload)
        self.assertEqual(payload["interaction_type"], "clarify_dimension")
        self.assertEqual(payload["missing_dimensions"], ["time"])

    def test_complete_multi_dimension_reaches_deciding(self):
        """All expected dimensions covered → DECIDING."""
        ctx = _base_context(
            preferences=[
                {"value": "Wednesday", "dimension": "day"},
                {"value": "Evening", "dimension": "time"},
            ],
            expected_dimensions=["day", "time"],
        )
        next_state, actions, _ = transition(
            State.AGGREGATING, Event.AGGREGATION_COMPLETED, ctx,
        )
        self.assertEqual(next_state, State.DECIDING)
        action_types = [a.type for a in actions]
        self.assertIn(ActionType.PROPOSE_DECISION, action_types)

    def test_conflict_overrides_completeness(self):
        """Conflict priority is dominant — even if solution is incomplete."""
        ctx = _base_context(
            preferences=[{"value": "Wednesday", "dimension": "day"}],
            expected_dimensions=["day", "time"],
            conflicts=[{"status": "open", "dimension": "day"}],
        )
        next_state, _, _ = transition(
            State.AGGREGATING, Event.AGGREGATION_COMPLETED, ctx,
        )
        self.assertEqual(next_state, State.RESOLVING)

    def test_expected_dimensions_none_proceeds_to_deciding(self):
        """expected_dimensions=None → dimension completeness satisfied → DECIDING."""
        ctx = _base_context(
            preferences=[{"value": "Wednesday", "dimension": "day"}],
            expected_dimensions=None,
        )
        next_state, actions, _ = transition(
            State.AGGREGATING, Event.AGGREGATION_COMPLETED, ctx,
        )
        self.assertEqual(next_state, State.DECIDING)
        action_types = [a.type for a in actions]
        self.assertIn(ActionType.PROPOSE_DECISION, action_types)

    def test_expected_dimensions_empty_list_proceeds_to_deciding(self):
        """expected_dimensions=[] → same as None, no dimension evaluation."""
        ctx = _base_context(
            preferences=[{"value": "Wednesday", "dimension": "day"}],
            expected_dimensions=[],
        )
        next_state, actions, _ = transition(
            State.AGGREGATING, Event.AGGREGATION_COMPLETED, ctx,
        )
        self.assertEqual(next_state, State.DECIDING)
        action_types = [a.type for a in actions]
        self.assertIn(ActionType.PROPOSE_DECISION, action_types)

    def test_plain_string_preferences_proceeds_to_deciding(self):
        """Plain string preferences (no dimension info) → complete by default."""
        ctx = _base_context(
            preferences=["Italian"],
        )
        next_state, actions, _ = transition(
            State.AGGREGATING, Event.AGGREGATION_COMPLETED, ctx,
        )
        self.assertEqual(next_state, State.DECIDING)
        action_types = [a.type for a in actions]
        self.assertIn(ActionType.PROPOSE_DECISION, action_types)

    def test_empty_preferences_no_solution_found(self):
        """Empty preferences → solution_found is false → fallback to COLLECTING."""
        ctx = _base_context(preferences=[])
        next_state, _, _ = transition(
            State.AGGREGATING, Event.AGGREGATION_COMPLETED, ctx,
        )
        self.assertEqual(next_state, State.COLLECTING)

    def test_validation_overrides_completeness(self):
        """Constraints present → VALIDATING, regardless of completeness."""
        ctx = _base_context(
            preferences=[{"value": "Wednesday", "dimension": "day"}],
            expected_dimensions=["day", "time"],
            constraints=["Must be after 5pm"],
        )
        next_state, _, _ = transition(
            State.AGGREGATING, Event.AGGREGATION_COMPLETED, ctx,
        )
        self.assertEqual(next_state, State.VALIDATING)

    def test_all_expected_covered_proceeds_to_deciding(self):
        """All expected_dimensions covered → DECIDING."""
        ctx = _base_context(
            preferences=[{"value": "Wednesday", "dimension": "day"}],
            expected_dimensions=["day"],
        )
        next_state, _, _ = transition(
            State.AGGREGATING, Event.AGGREGATION_COMPLETED, ctx,
        )
        self.assertEqual(next_state, State.DECIDING)

    # --- v2.10.9: proposed_dimensions governance ---

    def test_proposed_dimensions_no_effect_on_deciding(self):
        """proposed_dimensions must NOT influence routing → DECIDING allowed."""
        ctx = _base_context(
            preferences=[{"value": "Wednesday", "dimension": "day"}],
            expected_dimensions=None,
            proposed_dimensions=["day", "time", "cuisine"],
        )
        next_state, actions, _ = transition(
            State.AGGREGATING, Event.AGGREGATION_COMPLETED, ctx,
        )
        self.assertEqual(next_state, State.DECIDING)
        action_types = [a.type for a in actions]
        self.assertIn(ActionType.PROPOSE_DECISION, action_types)

    def test_proposed_dimensions_no_effect_on_completeness(self):
        """proposed_dimensions set + expected_dimensions=None → complete."""
        ctx = _base_context(
            preferences=[{"value": "Wednesday", "dimension": "day"}],
            expected_dimensions=None,
            proposed_dimensions=["day", "time"],
        )
        next_state, _, _ = transition(
            State.AGGREGATING, Event.AGGREGATION_COMPLETED, ctx,
        )
        # proposed_dimensions must not trigger incompleteness
        self.assertEqual(next_state, State.DECIDING)

    def test_proposed_does_not_override_expected(self):
        """Only expected_dimensions drives incompleteness, not proposed."""
        ctx = _base_context(
            preferences=[
                {"value": "Wednesday", "dimension": "day"},
                {"value": "Evening", "dimension": "time"},
            ],
            expected_dimensions=["day", "time"],
            proposed_dimensions=["day", "time", "cuisine"],
        )
        next_state, _, _ = transition(
            State.AGGREGATING, Event.AGGREGATION_COMPLETED, ctx,
        )
        # expected satisfied (day+time) even though proposed has cuisine
        self.assertEqual(next_state, State.DECIDING)

    def test_clarify_dimension_only_with_expected(self):
        """clarify_dimension interaction_type requires expected_dimensions."""
        # With expected_dimensions → clarify_dimension
        ctx1 = _base_context(
            preferences=[{"value": "Wednesday", "dimension": "day"}],
            expected_dimensions=["day", "time"],
        )
        _, actions1, _ = transition(
            State.AGGREGATING, Event.AGGREGATION_COMPLETED, ctx1,
        )
        self.assertEqual(actions1[0].payload["interaction_type"], "clarify_dimension")

        # Without expected_dimensions → no incomplete route at all (goes to DECIDING)
        ctx2 = _base_context(
            preferences=[{"value": "Wednesday", "dimension": "day"}],
            expected_dimensions=None,
        )
        next_state, actions2, _ = transition(
            State.AGGREGATING, Event.AGGREGATION_COMPLETED, ctx2,
        )
        self.assertEqual(next_state, State.DECIDING)
        # No clarify payload on the PROPOSE_DECISION action
        self.assertIsNone(actions2[0].payload)

    def test_deterministic_across_runs(self):
        """Same input → same output across multiple runs."""
        ctx = _base_context(
            preferences=[{"value": "Wednesday", "dimension": "day"}],
            expected_dimensions=["day", "time"],
        )
        results = []
        for _ in range(5):
            state, actions, _ = transition(
                State.AGGREGATING, Event.AGGREGATION_COMPLETED, ctx,
            )
            results.append((state, [a.type for a in actions], actions[0].payload))
        for r in results[1:]:
            self.assertEqual(r, results[0])


if __name__ == "__main__":
    unittest.main()
