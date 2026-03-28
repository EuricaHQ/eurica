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

def _map_conflict_signal(
    context: DecisionContext,
    signals: dict,
    actor: str,
) -> list[dict]:
    """Spec v2.10.3 §25.7.2: signal.conflict → ctx.conflicts mapping.

    Creates a structured conflict entry when signal.conflict is true.
    Deduplication (§25.7.5): skip if an open conflict from the same
    actor + source already exists.
    """
    if not signals.get("conflict"):
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

    Spec v2.10.3 §25.7: signal.conflict is materialized into ctx.conflicts
    so that guards operate exclusively on structured context.
    """
    return replace(
        context,
        preferences=context.preferences + signals.get("preferences", []),
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
