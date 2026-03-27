"""Tests for has_critical_unresolved_participants guard (spec v2.9.2).

Verifies rule-based guard behavior across:
- consent rule (any missing → critical)
- majority / threshold rules (outcome-locked → not critical)
- participation constraint (min_participants)
- feasibility (uncertainty)
- initiator rule
- edge cases

Run: python3 test_critical_unresolved.py
"""

from dataclasses import replace
from machine.states import State
from machine.events import Event
from machine.context import DecisionContext
from machine.transition import (
    transition,
    _has_critical_unresolved_participants,
    _solution_found,
)


def _step(state, event, ctx, expected_state, label=""):
    next_state, actions, ctx = transition(state, event, ctx)
    tag = f" ({label})" if label else ""
    assert next_state == expected_state, (
        f"{tag} expected {expected_state.value}, got {next_state.value}"
    )
    return next_state, actions, ctx


# ---------------------------------------------------------------------------
# Step 1: no missing participants
# ---------------------------------------------------------------------------

def test_guard_all_responded():
    """No critical unresolved when everyone has responded."""
    ctx = DecisionContext(
        participants=["alice", "bob"],
        min_participants=2,
        responses={"alice": ["Italian"], "bob": ["Thai"]},
        preferences=["Italian", "Thai"],
    )
    assert not _has_critical_unresolved_participants(ctx)


def test_guard_no_participants_defined():
    """Edge: no participants → no one is unresolved."""
    ctx = DecisionContext(
        participants=[],
        min_participants=0,
        responses={},
        preferences=[],
    )
    assert not _has_critical_unresolved_participants(ctx)


# ---------------------------------------------------------------------------
# Step 2: participation constraint (min_participants)
# ---------------------------------------------------------------------------

def test_guard_min_participants_not_met():
    """Critical: need 3 respondents, only have 1."""
    ctx = DecisionContext(
        participants=["alice", "bob", "carol"],
        min_participants=3,
        responses={"alice": ["Italian"]},
        preferences=["Italian"],
    )
    assert _has_critical_unresolved_participants(ctx)


def test_guard_min_participants_met_but_consent_blocks():
    """min_participants met (2 >= 2), but consent rule still blocks."""
    ctx = DecisionContext(
        participants=["alice", "bob", "carol"],
        min_participants=2,
        decision_rule="consent",
        responses={"alice": ["Italian"], "bob": ["Thai"]},
        preferences=["Italian", "Thai"],
    )
    # carol missing + consent → critical (could object)
    assert _has_critical_unresolved_participants(ctx)


# ---------------------------------------------------------------------------
# Step 3: consent rule (any missing → critical)
# ---------------------------------------------------------------------------

def test_guard_consent_blocks_with_missing():
    """Consent rule: ANY missing participant is critical (could object)."""
    ctx = DecisionContext(
        participants=["alice", "bob", "carol"],
        min_participants=2,
        decision_rule="consent",
        responses={"alice": ["Italian"], "carol": ["Italian"]},
        preferences=["Italian", "Italian"],  # unanimous doesn't matter
    )
    # bob missing + consent → critical (could still object)
    assert _has_critical_unresolved_participants(ctx)


def test_guard_consent_ok_when_all_responded():
    """Consent rule: no critical when everyone has responded."""
    ctx = DecisionContext(
        participants=["alice", "bob"],
        min_participants=2,
        decision_rule="consent",
        responses={"alice": ["Italian"], "bob": ["Italian"]},
        preferences=["Italian", "Italian"],
    )
    assert not _has_critical_unresolved_participants(ctx)


# ---------------------------------------------------------------------------
# Step 4: majority / threshold rules
# ---------------------------------------------------------------------------

def test_guard_majority_outcome_locked():
    """Majority rule: leader has enough votes, missing can't flip it."""
    # 5 participants, 3 responded "Italian", 1 responded "Thai", 1 missing
    # majority threshold = 5 // 2 + 1 = 3 → leader has 3 → locked
    ctx = DecisionContext(
        participants=["alice", "bob", "carol", "dave", "eve"],
        min_participants=3,
        decision_rule="majority",
        responses={
            "alice": ["Italian"], "bob": ["Italian"],
            "carol": ["Italian"], "dave": ["Thai"],
        },
        preferences=["Italian", "Italian", "Italian", "Thai"],
    )
    # eve missing but Italian already has 3/5 (majority) → not critical
    assert not _has_critical_unresolved_participants(ctx)


def test_guard_majority_outcome_not_locked():
    """Majority rule: leader below threshold, missing votes matter."""
    # 5 participants, 2 "Italian", 1 "Thai", 2 missing
    # majority threshold = 3, leader has 2 → NOT locked
    ctx = DecisionContext(
        participants=["alice", "bob", "carol", "dave", "eve"],
        min_participants=2,
        decision_rule="majority",
        responses={
            "alice": ["Italian"], "bob": ["Italian"],
            "carol": ["Thai"],
        },
        preferences=["Italian", "Italian", "Thai"],
    )
    # dave + eve missing, Italian only has 2 < 3 → critical
    assert _has_critical_unresolved_participants(ctx)


def test_guard_threshold_locked():
    """Threshold rule: leader meets custom threshold."""
    ctx = DecisionContext(
        participants=["alice", "bob", "carol", "dave"],
        min_participants=2,
        decision_rule="threshold",
        decision_rule_threshold=2,
        responses={
            "alice": ["Italian"], "bob": ["Italian"],
            "carol": ["Thai"],
        },
        preferences=["Italian", "Italian", "Thai"],
    )
    # dave missing but Italian has 2 >= threshold(2) → not critical
    assert not _has_critical_unresolved_participants(ctx)


def test_guard_threshold_not_locked():
    """Threshold rule: leader below custom threshold."""
    ctx = DecisionContext(
        participants=["alice", "bob", "carol", "dave"],
        min_participants=2,
        decision_rule="threshold",
        decision_rule_threshold=3,
        responses={
            "alice": ["Italian"], "bob": ["Italian"],
            "carol": ["Thai"],
        },
        preferences=["Italian", "Italian", "Thai"],
    )
    # dave missing, Italian has 2 < threshold(3) → critical
    assert _has_critical_unresolved_participants(ctx)


def test_guard_majority_no_preferences():
    """Majority rule: no preferences yet → outcome open → critical."""
    ctx = DecisionContext(
        participants=["alice", "bob", "carol"],
        min_participants=1,
        decision_rule="majority",
        responses={"alice": ["hmm"]},
        preferences=[],
    )
    assert _has_critical_unresolved_participants(ctx)


# ---------------------------------------------------------------------------
# Step 4b: unanimity rule
# ---------------------------------------------------------------------------

def test_guard_unanimity_missing():
    """Unanimity rule: any missing participant is critical."""
    ctx = DecisionContext(
        participants=["alice", "bob", "carol"],
        min_participants=2,
        decision_rule="unanimity",
        responses={"alice": ["Italian"], "bob": ["Italian"]},
        preferences=["Italian", "Italian"],
    )
    # carol missing + unanimity → critical (must agree)
    assert _has_critical_unresolved_participants(ctx)


# ---------------------------------------------------------------------------
# Step 5: feasibility (uncertainty)
# ---------------------------------------------------------------------------

def test_guard_uncertainty_blocks():
    """Unresolved uncertainty makes missing participants critical."""
    ctx = DecisionContext(
        participants=["alice", "bob", "carol"],
        min_participants=2,
        decision_rule="initiator",
        initiator="alice",
        responses={"alice": ["Italian"], "bob": ["Italian"]},
        preferences=["Italian", "Italian"],
        uncertainties=["budget unclear"],
    )
    # initiator rule: alice responded, carol non-critical by rule...
    # BUT uncertainty exists → critical
    assert _has_critical_unresolved_participants(ctx)


# ---------------------------------------------------------------------------
# Step 5b: initiator rule
# ---------------------------------------------------------------------------

def test_guard_initiator_responded():
    """Initiator rule: initiator responded, others don't matter."""
    ctx = DecisionContext(
        participants=["alice", "bob", "carol"],
        min_participants=1,
        decision_rule="initiator",
        initiator="alice",
        responses={"alice": ["Italian"]},
        preferences=["Italian"],
    )
    # bob + carol missing, but initiator rule → only alice matters
    assert not _has_critical_unresolved_participants(ctx)


def test_guard_initiator_missing():
    """Initiator rule: initiator hasn't responded → critical."""
    ctx = DecisionContext(
        participants=["alice", "bob", "carol"],
        min_participants=1,
        decision_rule="initiator",
        initiator="bob",
        responses={"alice": ["Italian"]},
        preferences=["Italian"],
    )
    # bob (initiator) missing → critical
    assert _has_critical_unresolved_participants(ctx)


# ---------------------------------------------------------------------------
# Integration: solution_found includes the guard
# ---------------------------------------------------------------------------

def test_solution_found_blocked_by_critical():
    """solution_found returns False when critical unresolved (consent rule)."""
    ctx = DecisionContext(
        question="Where to eat?",
        participants=["alice", "bob", "carol"],
        min_participants=2,
        decision_rule="consent",
        responses={"alice": ["Italian"], "carol": ["Thai"]},
        preferences=["Italian", "Thai"],
    )
    # bob missing + consent → critical → solution NOT found
    assert not _solution_found(ctx)


def test_solution_found_allowed_majority_locked():
    """solution_found returns True when majority is locked in."""
    ctx = DecisionContext(
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
    # eve missing but Italian has majority (3/5) → solution found
    assert _solution_found(ctx)


# ---------------------------------------------------------------------------
# Transition-level: AGGREGATING routing
# ---------------------------------------------------------------------------

def test_aggregating_blocked_by_critical_participant():
    """AGGREGATION_COMPLETED in AGGREGATING → COLLECTING (not DECIDING)
    when critical unresolved participants exist (consent rule).
    """
    ctx = DecisionContext(
        question="Where to eat?",
        participants=["alice", "bob", "carol"],
        min_participants=2,
        decision_rule="consent",
        responses={"alice": ["Italian"], "carol": ["Thai"]},
        preferences=["Italian", "Thai"],
    )
    state, actions, ctx = _step(
        State.AGGREGATING, Event.AGGREGATION_COMPLETED, ctx,
        State.COLLECTING, "blocked: consent + bob missing",
    )


def test_aggregating_allowed_majority_locked():
    """AGGREGATION_COMPLETED → DECIDING when majority locked despite missing."""
    ctx = DecisionContext(
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
    state, actions, ctx = _step(
        State.AGGREGATING, Event.AGGREGATION_COMPLETED, ctx,
        State.DECIDING, "allowed: majority locked",
    )


def test_aggregating_all_responded():
    """AGGREGATION_COMPLETED → DECIDING when everyone has responded."""
    ctx = DecisionContext(
        question="Where to eat?",
        participants=["alice", "bob"],
        min_participants=2,
        decision_rule="consent",
        responses={"alice": ["Italian"], "bob": ["Thai"]},
        preferences=["Italian", "Thai"],
    )
    state, actions, ctx = _step(
        State.AGGREGATING, Event.AGGREGATION_COMPLETED, ctx,
        State.DECIDING, "all responded, divergent ok",
    )


if __name__ == "__main__":
    tests = [
        # Step 1: no missing
        test_guard_all_responded,
        test_guard_no_participants_defined,
        # Step 2: participation constraint
        test_guard_min_participants_not_met,
        test_guard_min_participants_met_but_consent_blocks,
        # Step 3: consent rule
        test_guard_consent_blocks_with_missing,
        test_guard_consent_ok_when_all_responded,
        # Step 4: majority / threshold
        test_guard_majority_outcome_locked,
        test_guard_majority_outcome_not_locked,
        test_guard_threshold_locked,
        test_guard_threshold_not_locked,
        test_guard_majority_no_preferences,
        # Step 4b: unanimity
        test_guard_unanimity_missing,
        # Step 5: feasibility
        test_guard_uncertainty_blocks,
        # Step 5b: initiator rule
        test_guard_initiator_responded,
        test_guard_initiator_missing,
        # Integration: solution_found
        test_solution_found_blocked_by_critical,
        test_solution_found_allowed_majority_locked,
        # Transition-level
        test_aggregating_blocked_by_critical_participant,
        test_aggregating_allowed_majority_locked,
        test_aggregating_all_responded,
    ]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} critical_unresolved tests passed.")
