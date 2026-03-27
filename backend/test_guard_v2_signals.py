"""Tests for guard v2: signal-aware critical_unresolved_participants.

Verifies that Signal Layer v2 (flexibility, preference_strength,
constraint_type) correctly refines the criticality assessment.

Run: python3 test_guard_v2_signals.py
"""

from machine.context import DecisionContext
from machine.transition import (
    _has_critical_unresolved_rule_based,
    _has_critical_unresolved_participants,
    _is_signal_environment_relaxed,
    _has_hard_constraints,
    _has_high_tension,
)


# ---------------------------------------------------------------------------
# 1. High flexibility + weak preference → consent allows early finalization
# ---------------------------------------------------------------------------

def test_consent_relaxed_allows_early_finalization():
    """Consent + all high flexibility + weak prefs → missing NOT critical."""
    ctx = DecisionContext(
        participants=["alice", "bob", "carol"],
        min_participants=2,
        decision_rule="consent",
        responses={"alice": ["Italian"], "carol": ["Italian"]},
        preferences=["Italian", "Italian"],
        flexibility_signals=["high", "high"],
        preference_strength_signals=["weak", "none"],
        constraint_type_signals=["none", "none"],
    )
    assert _is_signal_environment_relaxed(ctx)
    assert not _has_critical_unresolved_rule_based(ctx)


def test_consent_relaxed_with_none_prefs():
    """Consent + high flexibility + preference_strength=none → relaxed."""
    ctx = DecisionContext(
        participants=["alice", "bob"],
        min_participants=1,
        decision_rule="consent",
        responses={"alice": ["anything"]},
        preferences=["anything"],
        flexibility_signals=["high"],
        preference_strength_signals=["none"],
        constraint_type_signals=["none"],
    )
    assert _is_signal_environment_relaxed(ctx)
    assert not _has_critical_unresolved_rule_based(ctx)


def test_consent_not_relaxed_without_signals():
    """Consent + no v2 signals → NOT relaxed → conservative (critical)."""
    ctx = DecisionContext(
        participants=["alice", "bob", "carol"],
        min_participants=2,
        decision_rule="consent",
        responses={"alice": ["Italian"], "carol": ["Italian"]},
        preferences=["Italian", "Italian"],
        # No v2 signals at all
    )
    assert not _is_signal_environment_relaxed(ctx)
    assert _has_critical_unresolved_rule_based(ctx)


def test_consent_not_relaxed_with_medium_flexibility():
    """Consent + medium flexibility → NOT relaxed → still critical."""
    ctx = DecisionContext(
        participants=["alice", "bob", "carol"],
        min_participants=2,
        decision_rule="consent",
        responses={"alice": ["Italian"], "carol": ["Italian"]},
        preferences=["Italian", "Italian"],
        flexibility_signals=["high", "medium"],
        preference_strength_signals=["weak", "none"],
        constraint_type_signals=["none", "none"],
    )
    assert not _is_signal_environment_relaxed(ctx)
    assert _has_critical_unresolved_rule_based(ctx)


def test_consent_not_relaxed_with_strong_preference():
    """Consent + strong preference → NOT relaxed → still critical."""
    ctx = DecisionContext(
        participants=["alice", "bob", "carol"],
        min_participants=2,
        decision_rule="consent",
        responses={"alice": ["Italian"], "carol": ["Italian"]},
        preferences=["Italian", "Italian"],
        flexibility_signals=["high", "high"],
        preference_strength_signals=["strong", "none"],
        constraint_type_signals=["none", "none"],
    )
    assert not _is_signal_environment_relaxed(ctx)
    assert _has_critical_unresolved_rule_based(ctx)


# ---------------------------------------------------------------------------
# 2. Hard constraint → unresolved participant stays critical
# ---------------------------------------------------------------------------

def test_hard_constraint_blocks_regardless():
    """Hard constraint → critical even with initiator rule satisfied."""
    ctx = DecisionContext(
        participants=["alice", "bob", "carol"],
        min_participants=1,
        decision_rule="initiator",
        initiator="alice",
        responses={"alice": ["Italian"]},
        preferences=["Italian"],
        constraint_type_signals=["hard"],
    )
    assert _has_hard_constraints(ctx)
    assert _has_critical_unresolved_rule_based(ctx)


def test_hard_constraint_blocks_consent_relaxed():
    """Hard constraint overrides relaxed signal environment."""
    ctx = DecisionContext(
        participants=["alice", "bob", "carol"],
        min_participants=2,
        decision_rule="consent",
        responses={"alice": ["Italian"], "carol": ["Italian"]},
        preferences=["Italian", "Italian"],
        flexibility_signals=["high", "high"],
        preference_strength_signals=["weak", "none"],
        constraint_type_signals=["none", "hard"],  # one hard constraint
    )
    # Would be relaxed except for the hard constraint
    assert not _is_signal_environment_relaxed(ctx)
    assert _has_critical_unresolved_rule_based(ctx)


def test_hard_constraint_blocks_majority_locked():
    """Hard constraint → critical even when majority is vote-locked."""
    ctx = DecisionContext(
        participants=["alice", "bob", "carol", "dave", "eve"],
        min_participants=3,
        decision_rule="majority",
        responses={
            "alice": ["Italian"], "bob": ["Italian"],
            "carol": ["Italian"], "dave": ["Thai"],
        },
        preferences=["Italian", "Italian", "Italian", "Thai"],
        constraint_type_signals=["hard", "none", "none", "none"],
    )
    # Vote is locked (3/5 majority) but hard constraint present
    assert _has_critical_unresolved_rule_based(ctx)


def test_soft_constraint_does_not_block():
    """Soft constraint only → does NOT escalate to critical."""
    ctx = DecisionContext(
        participants=["alice", "bob", "carol"],
        min_participants=1,
        decision_rule="initiator",
        initiator="alice",
        responses={"alice": ["Italian"]},
        preferences=["Italian"],
        constraint_type_signals=["soft"],
    )
    assert not _has_hard_constraints(ctx)
    assert not _has_critical_unresolved_rule_based(ctx)


# ---------------------------------------------------------------------------
# 3. Strong preference + low flexibility → critical
# ---------------------------------------------------------------------------

def test_majority_locked_high_tension_stays_critical():
    """Majority locked BUT strong prefs + low flexibility → still critical."""
    ctx = DecisionContext(
        participants=["alice", "bob", "carol", "dave", "eve"],
        min_participants=3,
        decision_rule="majority",
        responses={
            "alice": ["Italian"], "bob": ["Italian"],
            "carol": ["Italian"], "dave": ["Thai"],
        },
        preferences=["Italian", "Italian", "Italian", "Thai"],
        flexibility_signals=["low", "low", "low", "low"],
        preference_strength_signals=["strong", "strong", "strong", "strong"],
        constraint_type_signals=["none", "none", "none", "none"],
    )
    assert _has_high_tension(ctx)
    # Vote locked at 3/5 but high tension → critical
    assert _has_critical_unresolved_rule_based(ctx)


def test_majority_locked_no_tension_not_critical():
    """Majority locked + no tension → not critical (existing behavior)."""
    ctx = DecisionContext(
        participants=["alice", "bob", "carol", "dave", "eve"],
        min_participants=3,
        decision_rule="majority",
        responses={
            "alice": ["Italian"], "bob": ["Italian"],
            "carol": ["Italian"], "dave": ["Thai"],
        },
        preferences=["Italian", "Italian", "Italian", "Thai"],
        flexibility_signals=["high", "medium", "high", "medium"],
        preference_strength_signals=["weak", "weak", "none", "weak"],
        constraint_type_signals=["none", "none", "none", "none"],
    )
    assert not _has_high_tension(ctx)
    assert not _has_critical_unresolved_rule_based(ctx)


# ---------------------------------------------------------------------------
# 4. Soft / weak signal mix → non-critical (early readiness)
# ---------------------------------------------------------------------------

def test_consent_soft_weak_allows_finalization():
    """Consent + all high flex + soft constraints + weak prefs → relaxed."""
    ctx = DecisionContext(
        participants=["alice", "bob", "carol"],
        min_participants=2,
        decision_rule="consent",
        responses={"alice": ["Italian"], "carol": ["Italian"]},
        preferences=["Italian", "Italian"],
        flexibility_signals=["high", "high"],
        preference_strength_signals=["weak", "weak"],
        constraint_type_signals=["soft", "soft"],  # soft, not hard
    )
    # Soft constraints don't block relaxed environment
    assert _is_signal_environment_relaxed(ctx)
    assert not _has_critical_unresolved_rule_based(ctx)


def test_uncertainty_overrides_relaxed():
    """Relaxed signals + uncertainty → still critical."""
    ctx = DecisionContext(
        participants=["alice", "bob", "carol"],
        min_participants=2,
        decision_rule="consent",
        responses={"alice": ["Italian"], "carol": ["Italian"]},
        preferences=["Italian", "Italian"],
        uncertainties=["budget unclear"],
        flexibility_signals=["high", "high"],
        preference_strength_signals=["weak", "none"],
        constraint_type_signals=["none", "none"],
    )
    assert not _is_signal_environment_relaxed(ctx)
    assert _has_critical_unresolved_rule_based(ctx)


# ---------------------------------------------------------------------------
# Edge: helpers in isolation
# ---------------------------------------------------------------------------

def test_helper_has_hard_constraints_empty():
    ctx = DecisionContext(constraint_type_signals=[])
    assert not _has_hard_constraints(ctx)


def test_helper_has_hard_constraints_mixed():
    ctx = DecisionContext(constraint_type_signals=["soft", "none", "hard"])
    assert _has_hard_constraints(ctx)


def test_helper_high_tension_needs_both():
    """High tension requires BOTH strong preference AND low flexibility."""
    ctx1 = DecisionContext(
        preference_strength_signals=["strong"],
        flexibility_signals=["high"],
    )
    assert not _has_high_tension(ctx1)

    ctx2 = DecisionContext(
        preference_strength_signals=["weak"],
        flexibility_signals=["low"],
    )
    assert not _has_high_tension(ctx2)

    ctx3 = DecisionContext(
        preference_strength_signals=["strong"],
        flexibility_signals=["low"],
    )
    assert _has_high_tension(ctx3)


if __name__ == "__main__":
    tests = [
        # 1. Relaxed consent
        test_consent_relaxed_allows_early_finalization,
        test_consent_relaxed_with_none_prefs,
        test_consent_not_relaxed_without_signals,
        test_consent_not_relaxed_with_medium_flexibility,
        test_consent_not_relaxed_with_strong_preference,
        # 2. Hard constraints
        test_hard_constraint_blocks_regardless,
        test_hard_constraint_blocks_consent_relaxed,
        test_hard_constraint_blocks_majority_locked,
        test_soft_constraint_does_not_block,
        # 3. High tension
        test_majority_locked_high_tension_stays_critical,
        test_majority_locked_no_tension_not_critical,
        # 4. Early readiness
        test_consent_soft_weak_allows_finalization,
        test_uncertainty_overrides_relaxed,
        # Helpers
        test_helper_has_hard_constraints_empty,
        test_helper_has_hard_constraints_mixed,
        test_helper_high_tension_needs_both,
    ]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} guard v2 signal tests passed.")
