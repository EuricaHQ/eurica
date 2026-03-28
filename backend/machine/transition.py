from __future__ import annotations

from typing import Callable

from machine.states import State
from machine.events import Event
from machine.context import DecisionContext
from machine.actions import Action, ActionType


# Type aliases
Guard = Callable[[DecisionContext], bool]
TransitionResult = tuple[State, list[Action], DecisionContext]

# Terminal states — no transitions allowed out of these
_TERMINAL_STATES: frozenset[State] = frozenset({State.DECIDED, State.INFEASIBLE})


# ---------------------------------------------------------------------------
# Guards — pure predicates on context, no side effects
# ---------------------------------------------------------------------------

def _always(_ctx: DecisionContext) -> bool:
    return True


def _clarification_complete(ctx: DecisionContext) -> bool:
    """Stub guard: true when the question is sufficiently clear.

    Will be refined when LLM-driven clarification scoring is available.
    Currently: true if a question has been set (first response received).
    """
    return bool(ctx.question)


def _clarification_not_complete(ctx: DecisionContext) -> bool:
    return not _clarification_complete(ctx)


def _participation_satisfied(ctx: DecisionContext) -> bool:
    """Check if minimum participation requirement is met.

    Counts participants who have submitted at least one response.
    """
    return len(ctx.responses) >= ctx.min_participants


def _participation_not_satisfied(ctx: DecisionContext) -> bool:
    return not _participation_satisfied(ctx)


def _has_conflict(ctx: DecisionContext) -> bool:
    """Spec v2.10.3 §25.7.7: evaluate open conflicts only."""
    return any(c.get("status") == "open" for c in ctx.conflicts)


def _needs_validation(ctx: DecisionContext) -> bool:
    """True when constraints are present and need checking."""
    return len(ctx.constraints) > 0


def _is_avoidance(ctx: DecisionContext) -> bool:
    """Stub guard: true when avoidance behavior is detected.

    Currently checks if AVOIDANCE_DETECTED event context has been set.
    Will be refined with LLM-driven avoidance detection.
    """
    # Avoidance is signaled by the event arriving; guard confirms it.
    # For AGGREGATION_COMPLETED routing, this is a stub — avoidance
    # is primarily detected via the AVOIDANCE_DETECTED event path.
    return False


def _is_infeasible(ctx: DecisionContext) -> bool:
    """True when aggregation found no viable solution.

    Stub: will be refined when aggregation semantics evolve.
    """
    return False


def _has_hard_constraints(ctx: DecisionContext) -> bool:
    """True when any signal snapshot reported a hard constraint."""
    return "hard" in ctx.constraint_type_signals


def _has_high_tension(ctx: DecisionContext) -> bool:
    """True when responded participants show strong preferences + low flexibility.

    High tension means the outcome is sensitive to new input — missing
    participants are more likely to shift the result.
    """
    return (
        "strong" in ctx.preference_strength_signals
        and "low" in ctx.flexibility_signals
    )


def _is_signal_environment_relaxed(ctx: DecisionContext) -> bool:
    """True when v2 signals clearly indicate low-impact decision environment.

    Relaxed means:
    - at least one flexibility signal exists (we have data)
    - all flexibility signals are "high"
    - all preference_strength signals are "weak" or "none"
    - no hard constraints
    - no unresolved uncertainty

    Conservative: returns False when data is missing.
    """
    if not ctx.flexibility_signals:
        return False
    if _has_hard_constraints(ctx):
        return False
    if len(ctx.uncertainties) > 0:
        return False
    if any(f != "high" for f in ctx.flexibility_signals):
        return False
    if any(p not in ("weak", "none") for p in ctx.preference_strength_signals):
        return False
    return True


def _has_critical_unresolved_rule_based(ctx: DecisionContext) -> bool:
    """Rule-based evaluation of critical unresolved participants.

    Deterministic. No LLM. Pure predicate on context.

    A participant is critical unresolved if their missing input could
    still affect:
      1. decision rule outcome (consent / majority / threshold)
      2. participation constraint (min_participants)
      3. feasibility (hard constraints / unresolved uncertainty)

    Uses Signal Layer v2 (flexibility, preference_strength, constraint_type)
    to refine criticality when available. Falls back to conservative
    defaults when v2 signals are absent.

    Evaluated in order — first True wins.
    """
    responded = set(ctx.responses.keys())
    missing = [p for p in ctx.participants if p not in responded]

    # Step 1: no missing participants → not critical
    if not missing:
        return False

    n_responded = len(responded)
    total = len(ctx.participants)

    # Step 2: participation constraint
    if n_responded < ctx.min_participants:
        return True

    # Step 3: feasibility escalators (v2 + v1)
    # Hard constraints mean feasibility is still in flux — any missing
    # participant could introduce blocking constraints.
    if _has_hard_constraints(ctx):
        return True

    # Unresolved uncertainty affects feasibility regardless of rule.
    if len(ctx.uncertainties) > 0:
        return True

    # Step 4: decision rule sensitivity
    rule = ctx.decision_rule

    if rule == "consent":
        # Under consent, missing participants could still object.
        # BUT: if the signal environment is clearly relaxed (everyone
        # flexible, no strong preferences, no hard constraints),
        # the missing participant is unlikely to block.
        if _is_signal_environment_relaxed(ctx):
            return False
        return True

    if rule in ("majority", "threshold"):
        from collections import Counter
        counts = Counter(ctx.preferences)
        if not counts:
            return True
        leader_count = counts.most_common(1)[0][1]
        threshold = ctx.decision_rule_threshold or (total // 2 + 1)
        if leader_count < threshold:
            return True
        # Leader at or above threshold — outcome locked by votes.
        # BUT: high tension (strong prefs + low flexibility) means
        # the locked result may still face strong opposition.
        if _has_high_tension(ctx):
            return True
        return False

    if rule == "unanimity":
        return True

    # Step 5: initiator rule — only the initiator's input matters
    if rule == "initiator":
        if ctx.initiator and ctx.initiator not in responded:
            return True
        return False

    # Step 6: fallback — unknown rule, be conservative
    return True


def _has_critical_unresolved_participants(ctx: DecisionContext) -> bool:
    """Guard per spec v2.9.2 section 30.

    Two-layer evaluation:
    1. Rule-based (deterministic, always runs first)
    2. LLM advisory (optional, only if rule-based returns False)

    LLM is accessed via ctx.services["llm"] (dependency injection).
    LLM can ONLY escalate (mark as critical), never de-escalate.
    If no LLM is injected, or it fails → rule-based result stands.
    """
    responded = set(ctx.responses.keys())
    missing = [p for p in ctx.participants if p not in responded]

    if not missing:
        return False

    # Rule-based evaluation (deterministic baseline)
    rule_result = _has_critical_unresolved_rule_based(ctx)
    if rule_result:
        return True

    # Optional LLM advisory layer (via dependency injection)
    llm = ctx.services.get("llm")
    if llm is not None:
        try:
            compact_context = {
                "question": ctx.question,
                "participants": ctx.participants,
                "preferences": ctx.preferences,
                "constraints": ctx.constraints,
                "decision_rule": ctx.decision_rule,
                "flexibility_signals": ctx.flexibility_signals,
                "preference_strength_signals": ctx.preference_strength_signals,
                "constraint_type_signals": ctx.constraint_type_signals,
            }
            result = llm.evaluate_critical_participants(compact_context, missing)
            critical = result.get("critical_participants", [])
            if isinstance(critical, list) and len(critical) > 0:
                # Subset validation: only accept names actually in missing
                valid = [p for p in critical if p in missing]
                if valid:
                    return True
        except Exception:
            # LLM failure → fall back to rule-based result (False)
            pass

    return False


def _solution_found(ctx: DecisionContext) -> bool:
    """True when aggregation produced a viable solution.

    Per spec v2.9.2 section 28 (Decision Readiness):
      solution_found AND participation_satisfied
      AND NOT has_critical_unresolved_participants

    Requires at least one preference to constitute a solution.
    """
    return (
        len(ctx.preferences) > 0
        and _participation_satisfied(ctx)
        and not _has_conflict(ctx)
        and not _needs_validation(ctx)
        and not _is_infeasible(ctx)
        and not _has_critical_unresolved_participants(ctx)
    )


def _solution_complete(ctx: DecisionContext) -> bool:
    """Spec v2.10.8 §37.4: deterministic completeness evaluation.

    A solution is complete when:
    - no open conflicts
    - no critical unresolved participants
    - no materially missing dimensions

    This is NOT a guard — it is used for routing within solution_found.
    """
    if _has_conflict(ctx):
        return False
    if _has_critical_unresolved_participants(ctx):
        return False
    if _has_materially_missing_dimensions(ctx):
        return False
    return True


def _has_materially_missing_dimensions(ctx: DecisionContext) -> bool:
    """Spec v2.10.8 §37.4: check if expected dimensions are not yet covered.

    Uses ctx.expected_dimensions as the reference set. If no expected
    dimensions are defined, completeness cannot be evaluated and the
    solution is assumed complete (no implicit heuristics).
    """
    if not ctx.expected_dimensions:
        return False

    observed: set[str] = set()
    for p in ctx.preferences:
        if isinstance(p, dict) and p.get("dimension"):
            observed.add(p["dimension"])

    missing = set(ctx.expected_dimensions) - observed
    return len(missing) > 0


def _get_missing_dimensions(ctx: DecisionContext) -> list[str]:
    """Return list of expected dimensions not yet covered by preferences."""
    if not ctx.expected_dimensions:
        return []
    observed: set[str] = set()
    for p in ctx.preferences:
        if isinstance(p, dict) and p.get("dimension"):
            observed.add(p["dimension"])
    return sorted(set(ctx.expected_dimensions) - observed)


def _solution_found_and_complete(ctx: DecisionContext) -> bool:
    """Routing predicate: solution found AND complete per v2.10.8 §37.4."""
    return _solution_found(ctx) and _solution_complete(ctx)


def _solution_found_but_incomplete(ctx: DecisionContext) -> bool:
    """Routing predicate: solution found but NOT complete per v2.10.8 §37.4."""
    return _solution_found(ctx) and not _solution_complete(ctx)


def _confirmation_required(ctx: DecisionContext) -> bool:
    """Confirmation guard per spec v2.9 section 24.

    confirmation_required = true if:
        - decision rule requires it
        - decision is fragile (missing critical participants,
          narrow majority, weak aggregation, low confidence)
        - critical uncertainty exists
        - initiator approval required

    Stub: only evaluates structurally available indicators.
    Full fragility/confidence scoring comes later.
    """
    # Fragility: missing critical participants
    if not _participation_satisfied(ctx):
        return True

    # Critical uncertainty: unresolved uncertainties from participants
    if len(ctx.uncertainties) > 0:
        return True

    # Fragility: strong objections indicate fragile decision
    if len(ctx.objections) > 0:
        return True

    # Explicit approval requirement set on context
    if ctx.requires_explicit_approval:
        return True

    # Initiator must approve before finalizing
    if ctx.requires_initiator_approval:
        return True

    # No fragility or uncertainty detected
    return False


def _confirmation_not_required(ctx: DecisionContext) -> bool:
    return not _confirmation_required(ctx)


# ---------------------------------------------------------------------------
# Transition table — spec v2.9 section 28
#
# Each entry: (state, event, guard) → (next_state, actions)
#
# Guards are evaluated top-to-bottom. First match wins.
# Handlers do NOT contain control flow — only the table decides.
#
# Every (state, event) pair is explicitly handled.
# Terminal states (DECIDED, INFEASIBLE) are handled separately.
# ---------------------------------------------------------------------------

_TransitionEntry = tuple[State, Event, Guard, State, list[ActionType]]

_TRANSITION_TABLE: list[_TransitionEntry] = [

    # ===== CLARIFYING =====
    # RESPONSE_RECEIVED
    (State.CLARIFYING, Event.RESPONSE_RECEIVED, _clarification_complete,
     State.COLLECTING, [ActionType.ASK_QUESTION]),
    (State.CLARIFYING, Event.RESPONSE_RECEIVED, _clarification_not_complete,
     State.CLARIFYING, [ActionType.ASK_QUESTION]),
    # Self-transitions for irrelevant events
    (State.CLARIFYING, Event.AGGREGATION_COMPLETED, _always,
     State.CLARIFYING, []),
    (State.CLARIFYING, Event.VALIDATION_COMPLETED, _always,
     State.CLARIFYING, []),
    (State.CLARIFYING, Event.DECISION_CONFIRMED, _always,
     State.CLARIFYING, []),
    (State.CLARIFYING, Event.DECISION_REJECTED, _always,
     State.CLARIFYING, []),
    (State.CLARIFYING, Event.AVOIDANCE_DETECTED, _always,
     State.CLARIFYING, []),

    # ===== COLLECTING =====
    # RESPONSE_RECEIVED — always stay (aggregation is event-driven)
    (State.COLLECTING, Event.RESPONSE_RECEIVED, _always,
     State.COLLECTING, [ActionType.ASK_QUESTION]),
    # AGGREGATION_COMPLETED — the path to AGGREGATING
    (State.COLLECTING, Event.AGGREGATION_COMPLETED, _participation_satisfied,
     State.AGGREGATING, [ActionType.AGGREGATE]),
    (State.COLLECTING, Event.AGGREGATION_COMPLETED, _participation_not_satisfied,
     State.COLLECTING, []),
    # Self-transitions for irrelevant events
    (State.COLLECTING, Event.VALIDATION_COMPLETED, _always,
     State.COLLECTING, []),
    (State.COLLECTING, Event.DECISION_CONFIRMED, _always,
     State.COLLECTING, []),
    (State.COLLECTING, Event.DECISION_REJECTED, _always,
     State.COLLECTING, []),
    # AVOIDANCE_DETECTED — guarded
    (State.COLLECTING, Event.AVOIDANCE_DETECTED, _is_avoidance,
     State.AVOIDING, [ActionType.ADDRESS_AVOIDANCE]),
    (State.COLLECTING, Event.AVOIDANCE_DETECTED, _always,
     State.COLLECTING, []),

    # ===== AGGREGATING (structural hub) =====
    # RESPONSE_RECEIVED — self-transition
    (State.AGGREGATING, Event.RESPONSE_RECEIVED, _always,
     State.AGGREGATING, []),
    # AGGREGATION_COMPLETED — primary routing event
    # Guard priority order per spec v2.10.5 §24:
    #   1. is_infeasible (terminal — absolute priority)
    #   2. has_conflict (highest routing priority)
    #   3. needs_validation
    #   4. is_avoidance
    #   5. solution_found
    #   6. fallback → COLLECTING
    (State.AGGREGATING, Event.AGGREGATION_COMPLETED, _is_infeasible,
     State.INFEASIBLE, [ActionType.MARK_INFEASIBLE]),
    (State.AGGREGATING, Event.AGGREGATION_COMPLETED, _has_conflict,
     State.RESOLVING, [ActionType.RESOLVE_CONFLICT]),
    (State.AGGREGATING, Event.AGGREGATION_COMPLETED, _needs_validation,
     State.VALIDATING, [ActionType.VALIDATE_CONSTRAINT]),
    (State.AGGREGATING, Event.AGGREGATION_COMPLETED, _is_avoidance,
     State.AVOIDING, [ActionType.ADDRESS_AVOIDANCE]),
    # solution_found routing — spec v2.10.8 §37.4:
    #   complete → DECIDING
    #   incomplete → stay in AGGREGATING (clarification via action payload)
    (State.AGGREGATING, Event.AGGREGATION_COMPLETED, _solution_found_and_complete,
     State.DECIDING, [ActionType.PROPOSE_DECISION]),
    (State.AGGREGATING, Event.AGGREGATION_COMPLETED, _solution_found_but_incomplete,
     State.AGGREGATING, [ActionType.ASK_QUESTION]),
    # Fallback: no solution criteria met → back to collecting
    (State.AGGREGATING, Event.AGGREGATION_COMPLETED, _always,
     State.COLLECTING, [ActionType.ASK_QUESTION]),
    # VALIDATION_COMPLETED — self-transition
    (State.AGGREGATING, Event.VALIDATION_COMPLETED, _always,
     State.AGGREGATING, []),
    # DECISION_CONFIRMED — self-transition
    (State.AGGREGATING, Event.DECISION_CONFIRMED, _always,
     State.AGGREGATING, []),
    # DECISION_REJECTED — self-transition
    (State.AGGREGATING, Event.DECISION_REJECTED, _always,
     State.AGGREGATING, []),
    # AVOIDANCE_DETECTED — guarded
    (State.AGGREGATING, Event.AVOIDANCE_DETECTED, _is_avoidance,
     State.AVOIDING, [ActionType.ADDRESS_AVOIDANCE]),
    (State.AGGREGATING, Event.AVOIDANCE_DETECTED, _always,
     State.AGGREGATING, []),

    # ===== RESOLVING =====
    # RESPONSE_RECEIVED — always self-transition
    (State.RESOLVING, Event.RESPONSE_RECEIVED, _always,
     State.RESOLVING, [ActionType.RESOLVE_CONFLICT]),
    # AGGREGATION_COMPLETED — return path to AGGREGATING
    (State.RESOLVING, Event.AGGREGATION_COMPLETED, _always,
     State.AGGREGATING, [ActionType.AGGREGATE]),
    # Self-transitions for irrelevant events
    (State.RESOLVING, Event.VALIDATION_COMPLETED, _always,
     State.RESOLVING, []),
    (State.RESOLVING, Event.DECISION_CONFIRMED, _always,
     State.RESOLVING, []),
    (State.RESOLVING, Event.DECISION_REJECTED, _always,
     State.RESOLVING, []),
    # AVOIDANCE_DETECTED — guarded
    (State.RESOLVING, Event.AVOIDANCE_DETECTED, _is_avoidance,
     State.AVOIDING, [ActionType.ADDRESS_AVOIDANCE]),
    (State.RESOLVING, Event.AVOIDANCE_DETECTED, _always,
     State.RESOLVING, []),

    # ===== VALIDATING =====
    # RESPONSE_RECEIVED — self-transition
    (State.VALIDATING, Event.RESPONSE_RECEIVED, _always,
     State.VALIDATING, []),
    # AGGREGATION_COMPLETED — self-transition
    (State.VALIDATING, Event.AGGREGATION_COMPLETED, _always,
     State.VALIDATING, []),
    # VALIDATION_COMPLETED — always returns to AGGREGATING
    (State.VALIDATING, Event.VALIDATION_COMPLETED, _always,
     State.AGGREGATING, [ActionType.AGGREGATE]),
    # Self-transitions for irrelevant events
    (State.VALIDATING, Event.DECISION_CONFIRMED, _always,
     State.VALIDATING, []),
    (State.VALIDATING, Event.DECISION_REJECTED, _always,
     State.VALIDATING, []),
    # AVOIDANCE_DETECTED — guarded
    (State.VALIDATING, Event.AVOIDANCE_DETECTED, _is_avoidance,
     State.AVOIDING, [ActionType.ADDRESS_AVOIDANCE]),
    (State.VALIDATING, Event.AVOIDANCE_DETECTED, _always,
     State.VALIDATING, []),

    # ===== AVOIDING =====
    # RESPONSE_RECEIVED — self-transition
    (State.AVOIDING, Event.RESPONSE_RECEIVED, _always,
     State.AVOIDING, [ActionType.ADDRESS_AVOIDANCE]),
    # AGGREGATION_COMPLETED — return path to AGGREGATING
    (State.AVOIDING, Event.AGGREGATION_COMPLETED, _always,
     State.AGGREGATING, [ActionType.AGGREGATE]),
    # Self-transitions for irrelevant events
    (State.AVOIDING, Event.VALIDATION_COMPLETED, _always,
     State.AVOIDING, []),
    (State.AVOIDING, Event.DECISION_CONFIRMED, _always,
     State.AVOIDING, []),
    (State.AVOIDING, Event.DECISION_REJECTED, _always,
     State.AVOIDING, []),
    (State.AVOIDING, Event.AVOIDANCE_DETECTED, _always,
     State.AVOIDING, []),

    # ===== DECIDING =====
    # RESPONSE_RECEIVED — self-transition
    (State.DECIDING, Event.RESPONSE_RECEIVED, _always,
     State.DECIDING, []),
    # AGGREGATION_COMPLETED — self-transition
    (State.DECIDING, Event.AGGREGATION_COMPLETED, _always,
     State.DECIDING, []),
    # VALIDATION_COMPLETED — self-transition
    (State.DECIDING, Event.VALIDATION_COMPLETED, _always,
     State.DECIDING, []),
    # DECISION_CONFIRMED — always leads to DECIDED (spec v2.9.1)
    # confirmation_required controls HOW the event is triggered
    # (system-internal vs user-external), not IF the transition works.
    (State.DECIDING, Event.DECISION_CONFIRMED, _always,
     State.DECIDED, [ActionType.FINALIZE_DECISION]),
    # DECISION_REJECTED — back to AGGREGATING (spec v2.9)
    (State.DECIDING, Event.DECISION_REJECTED, _always,
     State.AGGREGATING, [ActionType.AGGREGATE]),
    # AVOIDANCE_DETECTED — self-transition
    (State.DECIDING, Event.AVOIDANCE_DETECTED, _always,
     State.DECIDING, []),
]


# ---------------------------------------------------------------------------
# Central transition function
# ---------------------------------------------------------------------------

def transition(
    state: State,
    event: Event,
    context: DecisionContext,
) -> TransitionResult:
    """Central transition function.

    (state, event, context) → (next_state, actions, updated_context)

    ALL control flow goes through here. No exceptions.
    This function is pure: no I/O, no LLM calls, no external systems.

    Evaluation order:
    1. Terminal state check (absorbing — all events → self)
    2. Walk table top-to-bottom
    3. First matching (state, event, guard) wins
    4. No match → raise (spec v2.9: missing transitions are forbidden)
    """
    # Terminal states are absorbing — no exit transitions
    if state in _TERMINAL_STATES:
        return (state, [], context)

    # Walk the table, first matching guard wins
    for t_state, t_event, guard, next_state, action_types in _TRANSITION_TABLE:
        if t_state == state and t_event == event and guard(context):
            actions = [Action(at) for at in action_types]
            # Spec v2.10.8 §37.4: attach clarification payload on incomplete routing
            if guard is _solution_found_but_incomplete:
                missing = _get_missing_dimensions(context)
                interaction_type = "clarify_dimension" if missing else "clarify"
                payload = {"interaction_type": interaction_type}
                if missing:
                    payload["missing_dimensions"] = missing
                actions = [Action(at, payload=payload) for at in action_types]
            return (next_state, actions, context)

    # Spec v2.9: no implicit no-op allowed. Every (state, event) must
    # be handled. If we reach here, the table is incomplete.
    raise ValueError(
        f"No transition defined for ({state.value}, {event.value}). "
        f"Spec v2.9 requires explicit handling of all (state, event) pairs."
    )
