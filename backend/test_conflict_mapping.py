"""Tests for conflict mapping — spec v2.10.3 + v2.10.6 dimension awareness.

Covers:
1. Same-dimension conflict creates ctx.conflicts entry
2. Cross-dimension preferences do NOT create conflict
3. Missing dimension → no conflict assumed
4. Duplicate conflict is not appended (deduplication)
5. _has_conflict guard works from mapped context (open status)
6. Resolved conflicts are ignored by _has_conflict
7. _apply_actions stores structured preferences with participant
8. Backward compat: plain string preferences normalized
"""

import unittest

from machine.context import DecisionContext
from machine.transition import _has_conflict
from api.routes import _map_conflict_signal, _apply_actions, _normalize_preferences


def _pref(value: str, dimension: str | None = None, participant: str = "alice") -> dict:
    """Helper to create a structured preference."""
    entry = {"participant": participant, "value": value}
    if dimension:
        entry["dimension"] = dimension
    return entry


class TestConflictMapping(unittest.TestCase):

    # --- Dimension-aware conflict (v2.10.6) ---

    def test_same_dimension_conflict_creates_entry(self):
        """Same dimension + different value → conflict created."""
        ctx = DecisionContext(
            preferences=[_pref("Wednesday", "day")],
        )
        signals = {
            "conflict": True,
            "preferences": [{"value": "Thursday", "dimension": "day"}],
        }
        result = _map_conflict_signal(ctx, signals, actor="bob")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["status"], "open")
        self.assertEqual(result[0]["participants"], ["bob"])

    def test_cross_dimension_no_conflict(self):
        """Different dimensions → no conflict even if signal.conflict=True."""
        ctx = DecisionContext(
            preferences=[_pref("Wednesday", "day")],
        )
        signals = {
            "conflict": True,
            "preferences": [{"value": "Evening", "dimension": "time"}],
        }
        result = _map_conflict_signal(ctx, signals, actor="bob")
        self.assertEqual(len(result), 0)

    def test_missing_dimension_no_conflict(self):
        """Missing dimension on new pref → no conflict assumed."""
        ctx = DecisionContext(
            preferences=[_pref("Wednesday", "day")],
        )
        signals = {
            "conflict": True,
            "preferences": [{"value": "Thursday"}],  # no dimension
        }
        result = _map_conflict_signal(ctx, signals, actor="bob")
        self.assertEqual(len(result), 0)

    def test_missing_dimension_on_existing_no_conflict(self):
        """Missing dimension on existing pref → no conflict assumed."""
        ctx = DecisionContext(
            preferences=[_pref("Wednesday")],  # no dimension
        )
        signals = {
            "conflict": True,
            "preferences": [{"value": "Thursday", "dimension": "day"}],
        }
        result = _map_conflict_signal(ctx, signals, actor="bob")
        self.assertEqual(len(result), 0)

    def test_same_dimension_same_value_no_conflict(self):
        """Same dimension + same value → agreement, not conflict."""
        ctx = DecisionContext(
            preferences=[_pref("Wednesday", "day")],
        )
        signals = {
            "conflict": False,
            "preferences": [{"value": "Wednesday", "dimension": "day"}],
        }
        result = _map_conflict_signal(ctx, signals, actor="bob")
        self.assertEqual(len(result), 0)

    def test_constraint_vs_preference_same_dimension(self):
        """Hard constraint referencing existing pref value → conflict created."""
        ctx = DecisionContext(
            preferences=[_pref("Wednesday", "day")],
        )
        signals = {
            "conflict": True,
            "preferences": [],  # negation context — no preferences
            "constraints": ["I cannot do Wednesday at all"],
        }
        result = _map_conflict_signal(ctx, signals, actor="carol")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["status"], "open")
        self.assertEqual(result[0]["participants"], ["carol"])

    def test_constraint_unrelated_no_conflict(self):
        """Constraint that doesn't reference existing pref → no conflict."""
        ctx = DecisionContext(
            preferences=[_pref("Wednesday", "day")],
        )
        signals = {
            "conflict": True,
            "preferences": [],
            "constraints": ["I cannot bring food"],
        }
        result = _map_conflict_signal(ctx, signals, actor="carol")
        self.assertEqual(len(result), 0)

    def test_no_conflict_signal_no_entry(self):
        """signal.conflict=false → no change regardless of dimensions."""
        ctx = DecisionContext(
            preferences=[_pref("Wednesday", "day")],
        )
        signals = {
            "conflict": False,
            "preferences": [{"value": "Thursday", "dimension": "day"}],
        }
        result = _map_conflict_signal(ctx, signals, actor="bob")
        self.assertEqual(len(result), 0)

    # --- Deduplication ---

    def test_dedup_same_actor_same_source(self):
        """Same actor + same source + open → no duplicate."""
        existing = {
            "participants": ["bob"],
            "source": "llm_signal",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "status": "open",
        }
        ctx = DecisionContext(
            conflicts=[existing],
            preferences=[_pref("Wednesday", "day")],
        )
        signals = {
            "conflict": True,
            "preferences": [{"value": "Thursday", "dimension": "day"}],
        }
        result = _map_conflict_signal(ctx, signals, actor="bob")
        self.assertEqual(len(result), 1)  # no new entry

    def test_no_dedup_different_actor(self):
        """Different actor → new entry allowed."""
        existing = {
            "participants": ["alice"],
            "source": "llm_signal",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "status": "open",
        }
        ctx = DecisionContext(
            conflicts=[existing],
            preferences=[_pref("Wednesday", "day")],
        )
        signals = {
            "conflict": True,
            "preferences": [{"value": "Thursday", "dimension": "day"}],
        }
        result = _map_conflict_signal(ctx, signals, actor="bob")
        self.assertEqual(len(result), 2)

    def test_no_dedup_resolved_conflict(self):
        """Resolved conflict from same actor → new entry allowed."""
        existing = {
            "participants": ["bob"],
            "source": "llm_signal",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "status": "resolved",
        }
        ctx = DecisionContext(
            conflicts=[existing],
            preferences=[_pref("Wednesday", "day")],
        )
        signals = {
            "conflict": True,
            "preferences": [{"value": "Thursday", "dimension": "day"}],
        }
        result = _map_conflict_signal(ctx, signals, actor="bob")
        self.assertEqual(len(result), 2)

    # --- Guard behavior ---

    def test_has_conflict_guard_open(self):
        """_has_conflict returns True when open conflicts exist."""
        ctx = DecisionContext(conflicts=[{
            "participants": ["bob"],
            "source": "llm_signal",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "status": "open",
        }])
        self.assertTrue(_has_conflict(ctx))

    def test_has_conflict_guard_resolved_only(self):
        """_has_conflict returns False when all conflicts are resolved."""
        ctx = DecisionContext(conflicts=[{
            "participants": ["bob"],
            "source": "llm_signal",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "status": "resolved",
        }])
        self.assertFalse(_has_conflict(ctx))

    def test_has_conflict_guard_empty(self):
        """_has_conflict returns False on empty conflicts list."""
        ctx = DecisionContext()
        self.assertFalse(_has_conflict(ctx))

    # --- Preference normalization ---

    def test_normalize_structured_prefs(self):
        """Structured dict preferences normalized with participant."""
        result = _normalize_preferences(
            [{"value": "Wednesday", "dimension": "day"}], actor="alice",
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["participant"], "alice")
        self.assertEqual(result[0]["value"], "Wednesday")
        self.assertEqual(result[0]["dimension"], "day")

    def test_normalize_plain_string_prefs(self):
        """Plain string preferences normalized to dict with participant."""
        result = _normalize_preferences(["Wednesday"], actor="alice")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["participant"], "alice")
        self.assertEqual(result[0]["value"], "Wednesday")
        self.assertNotIn("dimension", result[0])

    def test_apply_actions_stores_structured_prefs(self):
        """_apply_actions stores structured preferences in context."""
        ctx = DecisionContext()
        signals = {
            "preferences": [{"value": "Wednesday", "dimension": "day"}],
            "constraints": [],
            "uncertainty": False,
            "objection": False,
            "conflict": False,
            "flexibility": "medium",
            "preference_strength": "none",
            "constraint_type": "none",
        }
        result = _apply_actions(ctx, [], signals, actor="alice")
        self.assertEqual(len(result.preferences), 1)
        self.assertEqual(result.preferences[0]["value"], "Wednesday")
        self.assertEqual(result.preferences[0]["dimension"], "day")
        self.assertEqual(result.preferences[0]["participant"], "alice")


if __name__ == "__main__":
    results = unittest.main(exit=False, verbosity=0)
    total = results.result.testsRun
    failures = len(results.result.failures) + len(results.result.errors)
    print(f"\n{total - failures}/{total} conflict mapping tests passed.")
