from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

from fastapi import APIRouter

from models.schemas import MessageRequest, MessageResponse
from interpreter.signal_mapper import map_signals_to_event
from machine.states import State
from machine.events import Event
from machine.context import DecisionContext
from machine.actions import Action, ActionType
from machine.transition import transition
from llm.openai_llm import OpenAILLM
import store

router = APIRouter()

# LLM instance — swap implementation here (OpenAILLM, MockLLM, etc.)
_llm = OpenAILLM()

# Maximum system-event iterations to prevent infinite loops
_MAX_SYSTEM_ITERATIONS = 3


# ---------------------------------------------------------------------------
# Confirmation guard (mirror of machine/transition.py::_confirmation_required)
#
# Used ONLY by the system-event loop to decide whether to auto-emit
# DECISION_CONFIRMED. Must stay in sync with transition.py.
#
# Per spec v2.9.1: confirmation_required controls the SOURCE of the
# DECISION_CONFIRMED event (system vs user), NOT the transition behavior.
# ---------------------------------------------------------------------------

def _confirmation_required(ctx: DecisionContext) -> bool:
    if len(ctx.responses) < ctx.min_participants:
        return True
    if len(ctx.uncertainties) > 0:
        return True
    if len(ctx.objections) > 0:
        return True
    if ctx.requires_explicit_approval:
        return True
    if ctx.requires_initiator_approval:
        return True
    return False


# ---------------------------------------------------------------------------
# Context helpers
# ---------------------------------------------------------------------------

def _get_pref_value(p) -> str:
    """Extract value from a preference (string or dict with 'value' key)."""
    if isinstance(p, dict):
        return p.get("value", "")
    return str(p)


def _get_pref_dimension(p) -> str | None:
    """Extract dimension from a preference (None if string or missing)."""
    if isinstance(p, dict):
        return p.get("dimension")
    return None


def _has_same_dimension_conflict(
    new_prefs: list,
    existing_prefs: list,
    constraints: list | None = None,
) -> bool:
    """Spec v2.10.6 §18 Conflict Compatibility Rule.

    A conflict may only be created if preferences refer to the SAME
    dimension AND are incompatible. Different dimensions = complementary.
    Missing dimension = no conflict assumed.

    Also checks hard constraints: if a constraint text references an
    existing preference value, that's a same-dimension conflict.
    """
    # Check preference vs preference (different values, same dimension)
    for np in new_prefs:
        n_dim = _get_pref_dimension(np)
        n_val = _get_pref_value(np).lower()
        if not n_dim or not n_val:
            continue
        for ep in existing_prefs:
            e_dim = _get_pref_dimension(ep)
            e_val = _get_pref_value(ep).lower()
            if not e_dim:
                continue
            if n_dim == e_dim and n_val != e_val:
                return True

    # Check constraint vs existing preference (hard constraint negates
    # an existing preference value → same-dimension conflict)
    if constraints:
        for constraint_text in constraints:
            if not isinstance(constraint_text, str):
                continue
            c_lower = constraint_text.lower()
            for ep in existing_prefs:
                e_val = _get_pref_value(ep).lower()
                if e_val and e_val in c_lower:
                    return True

    return False


def _map_conflict_signal(
    context: DecisionContext,
    signals: dict,
    actor: str,
) -> list[dict]:
    """Spec v2.10.6 §18: dimension-aware conflict mapping.

    Creates a structured conflict entry ONLY when:
    1. signal.conflict == true
    2. New preferences conflict with existing in the SAME dimension
    3. Values are incompatible

    If dimension is missing on either side → no conflict created.
    Deduplication (§25.7.5): skip if open conflict from same actor exists.
    """
    if not signals.get("conflict"):
        return context.conflicts

    # Dimension-aware validation: only create conflict if same-dimension clash
    new_prefs = signals.get("preferences", [])
    new_constraints = signals.get("constraints", [])
    if not _has_same_dimension_conflict(new_prefs, context.preferences, new_constraints):
        return context.conflicts

    # Deduplication: same actor, same source, still open
    for existing in context.conflicts:
        if (
            existing.get("source") == "llm_signal"
            and existing.get("status") == "open"
            and actor in existing.get("participants", [])
        ):
            return context.conflicts

    entry = {
        "participants": [actor],
        "source": "llm_signal",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "open",
    }
    return context.conflicts + [entry]


def _normalize_preferences(raw_prefs: list, actor: str) -> list[dict]:
    """Normalize preferences to structured dicts with participant.

    Accepts both old format (strings) and new format (dicts with value/dimension).
    Spec v2.10.6 §25.8: preference = {value, dimension (optional)}.
    """
    result = []
    for p in raw_prefs:
        if isinstance(p, dict):
            entry = {
                "participant": actor,
                "value": p.get("value", ""),
            }
            if p.get("dimension"):
                entry["dimension"] = p["dimension"]
            result.append(entry)
        elif isinstance(p, str):
            result.append({"participant": actor, "value": p})
        # skip anything else
    return result


def _apply_actions(
    context: DecisionContext,
    actions: list[Action],
    signals: dict,
    actor: str = "",
) -> DecisionContext:
    """Merge signal data into context after transition.

    This is the ONLY place where LLM-extracted signals update the context.
    Actions are recorded but do not alter context here — they describe
    WHAT should happen, and the LLM response layer handles the HOW.

    Spec v2.10.6 §25.8: preferences stored as structured dicts with dimension.
    Spec v2.10.3 §25.7: signal.conflict is materialized into ctx.conflicts.
    """
    new_prefs = _normalize_preferences(signals.get("preferences", []), actor)
    return replace(
        context,
        preferences=context.preferences + new_prefs,
        constraints=context.constraints + signals.get("constraints", []),
        uncertainties=context.uncertainties + (
            ["uncertainty"] if signals.get("uncertainty") else []
        ),
        objections=context.objections + (
            ["objection"] if signals.get("objection") else []
        ),
        conflicts=_map_conflict_signal(context, signals, actor),
        flexibility_signals=context.flexibility_signals + (
            [signals["flexibility"]] if "flexibility" in signals else []
        ),
        preference_strength_signals=context.preference_strength_signals + (
            [signals["preference_strength"]] if "preference_strength" in signals else []
        ),
        constraint_type_signals=context.constraint_type_signals + (
            [signals["constraint_type"]] if "constraint_type" in signals else []
        ),
    )


def _context_to_dict(ctx: DecisionContext) -> dict:
    """Convert DecisionContext to a plain dict for LLM consumption.

    LLM receives read-only data — no internal types leak out.
    """
    return {
        "decision_id": ctx.decision_id,
        "question": ctx.question,
        "participants": ctx.participants,
        "min_participants": ctx.min_participants,
        "responses": ctx.responses,
        "preferences": ctx.preferences,
        "constraints": ctx.constraints,
        "uncertainties": ctx.uncertainties,
        "objections": ctx.objections,
        "conflicts": ctx.conflicts,
        "decision": ctx.decision,
    }


# ---------------------------------------------------------------------------
# System event derivation
# ---------------------------------------------------------------------------

def _derive_system_event(
    state: State,
    context: DecisionContext,
) -> Event | None:
    """Derive a system event from the current state.

    System events allow the machine to continue processing
    without waiting for external input. They are generated by
    the system, NOT by the LLM.

    Returns None if no system event should fire.
    """
    # COLLECTING: trigger aggregation check (guards decide the outcome)
    if state == State.COLLECTING:
        return Event.AGGREGATION_COMPLETED

    if state == State.AGGREGATING:
        return Event.AGGREGATION_COMPLETED

    if state == State.VALIDATING:
        return Event.VALIDATION_COMPLETED

    # Spec v2.9.1: confirmation_required controls SOURCE of event.
    # If false → system internally emits DECISION_CONFIRMED.
    # If true → wait for user-triggered DECISION_CONFIRMED.
    if state == State.DECIDING and not _confirmation_required(context):
        return Event.DECISION_CONFIRMED

    return None


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@router.post("/message", response_model=MessageResponse)
def post_message(req: MessageRequest):
    """Single endpoint for the decision loop.

    Flow (strict separation: LLM = interpretation + language,
    System = state + decisions):

    1. Load state + context
    2. Update context with new message
    3. LLM interprets message → signals (dict)
    4. Map signals → event (deterministic, no LLM)
    5. Transition (state machine decides)
    6. Apply actions + merge signals into context
    7. System event loop
    8. LLM generates response
    9. Persist + return
    """
    # 1. Load
    state, context = store.load(req.decision_id)

    # Inject services into context (dependency injection for guards)
    context = replace(context, services={"llm": _llm})

    # 2. Update context with the incoming message
    if not context.question:
        context = replace(context, question=req.message)

    # Append response from participant
    responses = dict(context.responses)
    participant_responses = list(responses.get(req.participant, []))
    participant_responses.append(req.message)
    responses[req.participant] = participant_responses
    context = replace(context, responses=responses)

    # Ensure participant is tracked
    if req.participant not in context.participants:
        participants = list(context.participants)
        participants.append(req.participant)
        context = replace(context, participants=participants)

    # 3. LLM interprets message → structured signals
    signals = _llm.interpret(req.message, _context_to_dict(context))

    # 4. Signals → Event (deterministic mapping, no LLM)
    event = map_signals_to_event(signals)

    # 5. Transition (state machine decides)
    all_actions: list[Action] = []
    next_state, actions, context = transition(state, event, context)
    all_actions.extend(actions)

    # 6. Apply actions + merge signals into context
    context = _apply_actions(context, actions, signals, actor=req.participant)

    # 7. System event loop — allow internal state progression
    for _ in range(_MAX_SYSTEM_ITERATIONS):
        system_event = _derive_system_event(next_state, context)
        if system_event is None:
            break

        prev_state = next_state
        next_state, actions, context = transition(next_state, system_event, context)
        all_actions.extend(actions)

        # If state didn't change, stop (prevent spin)
        if next_state == prev_state:
            break

    # 8. LLM generates response
    reply = _llm.generate(next_state.value, _context_to_dict(context))

    # 9. Persist
    store.save(req.decision_id, next_state, context)

    return MessageResponse(
        decision_id=req.decision_id,
        state=next_state.value,
        reply=reply,
        actions_executed=[a.type.value for a in all_actions],
    )
