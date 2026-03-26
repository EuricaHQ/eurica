from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DecisionContext:
    """Immutable-style context carried through every transition.

    Designed for multi-user but not fully implemented yet.
    All fields are plain data — no behaviour, no side effects.
    """

    decision_id: str = ""
    question: str = ""

    # Participants (multi-user ready, single-user for now)
    participants: list[str] = field(default_factory=list)
    min_participants: int = 1

    # Decision rule: how a decision is accepted
    # "consent" — no objections (default)
    # "majority" — >50% agree
    # "threshold" — at least decision_rule_threshold agree
    decision_rule: str = "consent"
    decision_rule_threshold: int | None = None

    # Collected responses per participant
    responses: dict[str, list[str]] = field(default_factory=dict)

    # Structured signal snapshots (populated by interpreter)
    preferences: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    uncertainties: list[str] = field(default_factory=list)
    objections: list[str] = field(default_factory=list)

    # Aggregation / resolution results (populated by actions later)
    aggregation_result: dict | None = None
    conflicts: list[dict] = field(default_factory=list)
    validation_errors: list[str] = field(default_factory=list)

    # Confirmation requirements (spec v2.8.1)
    requires_explicit_approval: bool = False
    requires_initiator_approval: bool = False
    initiator: str | None = None

    # Final outcome
    decision: str | None = None
    rejection_reason: str | None = None
