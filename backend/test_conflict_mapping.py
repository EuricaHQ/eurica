"""Tests for spec v2.10.3 — signal.conflict → ctx.conflicts mapping.

Covers:
1. Conflict signal creates structured ctx.conflicts entry
2. Duplicate conflict is not appended (deduplication)
3. _has_conflict guard works from mapped context (open status)
4. No conflict entry when signal.conflict is false
5. Resolved conflicts are ignored by _has_conflict
"""

import unittest

from machine.context import DecisionContext
from machine.transition import _has_conflict
from api.routes import _map_conflict_signal, _apply_actions


class TestConflictMapping(unittest.TestCase):

    def test_conflict_signal_creates_entry(self):
        """signal.conflict=true → one entry in ctx.conflicts."""
        ctx = DecisionContext()
        signals = {"conflict": True}
        result = _map_conflict_signal(ctx, signals, actor="bob")
        self.assertEqual(len(result), 1)
        entry = result[0]
        self.assertEqual(entry["participants"], ["bob"])
        self.assertEqual(entry["source"], "llm_signal")
        self.assertEqual(entry["status"], "open")
        self.assertIn("timestamp", entry)

    def test_no_conflict_signal_no_entry(self):
        """signal.conflict=false → no change to ctx.conflicts."""
        ctx = DecisionContext()
        signals = {"conflict": False}
        result = _map_conflict_signal(ctx, signals, actor="bob")
        self.assertEqual(len(result), 0)

    def test_dedup_same_actor_same_source(self):
        """Same actor + same source + open → no duplicate."""
        existing = {
            "participants": ["bob"],
            "source": "llm_signal",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "status": "open",
        }
        ctx = DecisionContext(conflicts=[existing])
        signals = {"conflict": True}
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
        ctx = DecisionContext(conflicts=[existing])
        signals = {"conflict": True}
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
        ctx = DecisionContext(conflicts=[existing])
        signals = {"conflict": True}
        result = _map_conflict_signal(ctx, signals, actor="bob")
        self.assertEqual(len(result), 2)

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

    def test_apply_actions_passes_actor(self):
        """_apply_actions with conflict signal creates entry via mapping."""
        ctx = DecisionContext()
        signals = {
            "preferences": [],
            "constraints": [],
            "uncertainty": False,
            "objection": False,
            "conflict": True,
            "flexibility": "medium",
            "preference_strength": "none",
            "constraint_type": "none",
        }
        result = _apply_actions(ctx, [], signals, actor="bob")
        self.assertEqual(len(result.conflicts), 1)
        self.assertEqual(result.conflicts[0]["participants"], ["bob"])
        self.assertEqual(result.conflicts[0]["status"], "open")

    def test_apply_actions_no_conflict(self):
        """_apply_actions without conflict signal leaves conflicts empty."""
        ctx = DecisionContext()
        signals = {
            "preferences": ["Wednesday"],
            "constraints": [],
            "uncertainty": False,
            "objection": False,
            "conflict": False,
            "flexibility": "medium",
            "preference_strength": "none",
            "constraint_type": "none",
        }
        result = _apply_actions(ctx, [], signals, actor="alice")
        self.assertEqual(len(result.conflicts), 0)


if __name__ == "__main__":
    results = unittest.main(exit=False, verbosity=0)
    total = results.result.testsRun
    failures = len(results.result.failures) + len(results.result.errors)
    print(f"\n{total - failures}/{total} conflict mapping tests passed.")
