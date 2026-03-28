"""End-to-end conversation simulator for the Decision Coordinator.

Runs predefined multi-participant scenarios through the real system:
  LLM interpret → signal mapper → state machine → system events
  → targeting → interaction_type → LLM generate

Prints detailed per-step traces for observation and debugging.

Usage:
    cd backend
    python3 simulate_conversation.py            # all scenarios
    python3 simulate_conversation.py 1          # single scenario
    python3 simulate_conversation.py --mock     # use MockLLM (no API key needed)
"""

from __future__ import annotations

import sys
from dataclasses import replace

from machine.states import State
from machine.events import Event
from machine.context import DecisionContext
from machine.transition import transition, _confirmation_required
from interpreter.signal_mapper import map_signals_to_event
from machine.actions import Action
from api.routes import _map_conflict_signal
from llm.interface import LLM

# Maximum system-event iterations (mirrors routes.py)
_MAX_SYSTEM_ITERATIONS = 3


# ---------------------------------------------------------------------------
# Simulator-local smart mock (richer than MockLLM for readable traces)
# ---------------------------------------------------------------------------

class _SimulatorMockLLM(LLM):
    """Keyword-based mock with basic preference/conflict extraction.

    Good enough to make simulator traces meaningful without an API key.
    Not used in production or tests.
    """

    # Dimension lookup tables
    _DAYS = ("monday", "tuesday", "wednesday", "thursday", "friday",
             "saturday", "sunday")
    _TIMES = ("morning", "evening", "afternoon", "night")
    _FOODS = ("italian", "asian", "thai", "mexican", "sushi",
              "pizza", "burgers", "indian", "chinese", "japanese")

    def interpret(self, message: str, context: dict) -> dict:
        lower = message.lower()

        # Extract preferences with dimensions
        # Skip extraction when message is a hard constraint (negation context)
        is_negation = any(w in lower for w in ("cannot", "can't", "never",
                                                "not", "don't"))
        preferences: list[dict] = []
        if not is_negation:
            for day in self._DAYS:
                if day in lower:
                    preferences.append({"value": day.capitalize(), "dimension": "day"})
            for time in self._TIMES:
                if time in lower:
                    preferences.append({"value": time.capitalize(), "dimension": "time"})
            for food in self._FOODS:
                if food in lower:
                    preferences.append({"value": food.capitalize(), "dimension": "cuisine"})

        # Flexibility
        if any(w in lower for w in ("flexible", "anything", "don't mind",
                                     "whatever", "either", "any",
                                     "i guess", "is ok", "is fine i guess")):
            flexibility = "high"
        elif any(w in lower for w in ("definitely", "must", "only",
                                       "really prefer")):
            flexibility = "low"
        else:
            flexibility = "medium"

        # Preference strength
        if any(w in lower for w in ("definitely", "really", "must",
                                     "absolutely", "strongly")):
            preference_strength = "strong"
        elif any(w in lower for w in ("maybe", "could", "might",
                                       "perhaps")):
            preference_strength = "weak"
        elif preferences:
            preference_strength = "weak"
        else:
            preference_strength = "none"

        # Constraint type (computed before conflict to inform it)
        if any(w in lower for w in ("must", "cannot", "can't", "never")):
            constraint_type = "hard"
        elif any(w in lower for w in ("prefer not", "rather not")):
            constraint_type = "soft"
        else:
            constraint_type = "none"

        # Dimension-aware conflict detection (spec v2.10.6 §18)
        # Only flag conflict if same dimension + different value
        existing_prefs = context.get("preferences", [])
        conflict = False
        for np in preferences:
            n_dim = np.get("dimension")
            n_val = np.get("value", "").lower()
            if not n_dim:
                continue
            for ep in existing_prefs:
                if isinstance(ep, dict):
                    e_dim = ep.get("dimension")
                    e_val = ep.get("value", "").lower()
                else:
                    continue
                if e_dim == n_dim and n_val != e_val:
                    conflict = True
                    break
            if conflict:
                break

        # Hard constraint against same-dimension existing = conflict
        # When message is a negation, preferences are empty, so check
        # constraint text against existing preference values directly
        if not conflict and constraint_type == "hard" and existing_prefs:
            for ep in existing_prefs:
                if isinstance(ep, dict):
                    e_val = ep.get("value", "").lower()
                else:
                    continue
                if e_val and e_val in lower:
                    conflict = True
                    break

        # Constraints
        constraints: list[str] = []
        if constraint_type == "hard":
            constraints = [message.strip()]

        # Objection
        objection = any(w in lower for w in ("no way", "absolutely not",
                                              "refuse", "object"))

        return {
            "preferences": preferences,
            "constraints": constraints,
            "uncertainty": "?" in message or "not sure" in lower,
            "conflict": conflict,
            "objection": objection,
            "avoidance": any(w in lower for w in ("later", "not now",
                                                    "skip", "don't want to")),
            "flexibility": flexibility,
            "preference_strength": preference_strength,
            "constraint_type": constraint_type,
        }

    @staticmethod
    def _pref_values(prefs: list) -> list[str]:
        """Extract display values from structured or plain preferences."""
        result = []
        for p in prefs:
            if isinstance(p, dict):
                result.append(p.get("value", str(p)))
            else:
                result.append(str(p))
        return result

    def generate(self, state: str, context: dict) -> str:
        prefs = context.get("preferences", [])
        parts = context.get("participants", [])
        vals = self._pref_values(prefs)

        if state == "decided":
            if vals:
                from collections import Counter
                top = Counter(vals).most_common(1)[0][0]
                return f"Decision reached: {top}. Thanks everyone!"
            return "Decision has been finalized. Thanks!"

        if state == "deciding":
            if vals:
                from collections import Counter
                top = Counter(vals).most_common(1)[0][0]
                return f"It looks like '{top}' works. Can everyone confirm?"
            return "We have enough input. Shall we finalize?"

        if state == "collecting":
            responded = set(context.get("responses", {}).keys())
            missing = [p for p in parts if p not in responded]
            if missing:
                return f"Waiting for input from: {', '.join(missing)}"
            return "Thanks! Processing your responses..."

        if state == "resolving":
            return "There seems to be a disagreement. Can we find a compromise?"

        return f"[{state}] How would you like to proceed?"

    def evaluate_critical_participants(
        self, context: dict, missing: list[str],
    ) -> dict:
        return {"critical_participants": []}


# ---------------------------------------------------------------------------
# Targeting layer (spec v2.10.1 section 36)
# ---------------------------------------------------------------------------

def _compute_targeting(
    state: State,
    ctx: DecisionContext,
) -> dict:
    """Deterministic targeting: WHO should the system address next?

    Returns {recipients, priority, targeting_reason}.
    Follows spec v2.10.1 section 36.5 priority rules.
    """
    responded = set(ctx.responses.keys())
    missing = [p for p in ctx.participants if p not in responded]

    # Priority 1: critical unresolved participants
    # (imported lazily to avoid circular — already evaluated by guards)
    if missing and state in (State.COLLECTING, State.AGGREGATING):
        return {
            "recipients": missing,
            "priority": "high",
            "targeting_reason": "missing_preference",
        }

    # Priority 4: conflict participants (spec v2.10.5 §28)
    # Must target participants involved in conflicts, NOT full group
    if state == State.RESOLVING:
        conflict_participants: set[str] = set()
        for c in ctx.conflicts:
            if c.get("status") == "open":
                conflict_participants.update(c.get("participants", []))
        # Fallback: if no conflict participants found, use responded
        recipients = sorted(conflict_participants) if conflict_participants else list(responded)
        return {
            "recipients": recipients,
            "priority": "high",
            "targeting_reason": "conflict_participant",
        }

    # Deciding: target all for confirmation or inform
    if state == State.DECIDING:
        return {
            "recipients": list(ctx.participants),
            "priority": "medium",
            "targeting_reason": "confirmation" if _confirmation_required(ctx) else "inform",
        }

    # Terminal / default
    return {
        "recipients": list(ctx.participants),
        "priority": "low",
        "targeting_reason": "inform",
    }


# ---------------------------------------------------------------------------
# Interaction type selection (spec v2.10.1 section 36.10)
# ---------------------------------------------------------------------------

def _select_interaction_type(
    state: State,
    ctx: DecisionContext,
    targeting_reason: str,
) -> str:
    """Deterministic interaction_type = f(state, context, targeting_reason).

    Spec v2.10.1 section 36.10.1 baseline + 36.10.2 refinement.
    """
    # Targeting reason refinement (36.10.2) takes priority
    if targeting_reason == "missing_constraint":
        return "constraint"
    if targeting_reason == "missing_preference":
        return "preference"
    if targeting_reason == "conflict_participant":
        return "resolve_conflict"
    if targeting_reason == "uncertainty_only":
        return "clarify"

    # Baseline mapping (36.10.1)
    if state == State.CLARIFYING:
        return "clarify"

    if state == State.COLLECTING:
        if not ctx.preferences:
            return "preference"
        if not ctx.constraints:
            return "constraint"
        return "clarify"

    if state == State.RESOLVING:
        return "resolve_conflict"

    if state == State.DECIDING:
        if _confirmation_required(ctx):
            return "confirm"
        return "inform"

    if state == State.DECIDED:
        return "inform"

    if state == State.INFEASIBLE:
        return "inform"

    # Fallback (36.10.3)
    return "inform"


# ---------------------------------------------------------------------------
# Decision quality (spec v2.10.1 section 33 — stub)
# ---------------------------------------------------------------------------

def _evaluate_decision_quality(ctx: DecisionContext) -> str | None:
    """Simple heuristic for decision quality: high / medium / low.

    Only meaningful in DECIDING / DECIDED states.
    """
    if not ctx.preferences:
        return None

    responded = set(ctx.responses.keys())
    total = len(ctx.participants)
    ratio = len(responded) / total if total > 0 else 0

    has_conflicts = len(ctx.conflicts) > 0
    has_objections = len(ctx.objections) > 0
    has_uncertainty = len(ctx.uncertainties) > 0

    if has_conflicts or has_objections:
        return "low"
    if has_uncertainty or ratio < 0.75:
        return "medium"
    return "high"


# ---------------------------------------------------------------------------
# System event derivation (mirrors routes.py)
# ---------------------------------------------------------------------------

def _derive_system_event(
    state: State,
    ctx: DecisionContext,
) -> Event | None:
    if state == State.COLLECTING:
        return Event.AGGREGATION_COMPLETED
    if state == State.AGGREGATING:
        return Event.AGGREGATION_COMPLETED
    if state == State.VALIDATING:
        return Event.VALIDATION_COMPLETED
    if state == State.DECIDING and not _confirmation_required(ctx):
        return Event.DECISION_CONFIRMED
    return None


# ---------------------------------------------------------------------------
# Context helpers (mirrors routes.py::_apply_actions / _context_to_dict)
# ---------------------------------------------------------------------------

def _apply_actions(
    context: DecisionContext,
    actions: list[Action],
    signals: dict,
    actor: str = "",
) -> DecisionContext:
    """Merge signal data into context after transition.

    Uses the real _map_conflict_signal from routes.py (spec v2.10.3).
    No simulator-local workarounds.
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
# Printer
# ---------------------------------------------------------------------------

_SEP = "─" * 60


def _print_step(step: int, data: dict) -> None:
    print(f"\n{_SEP}")
    print(f"  STEP {step}")
    print(_SEP)
    print(f"  ACTOR:              {data['actor']}")
    print(f"  RECIPIENT(S):       {data['recipients']}")
    print(f"  MESSAGE:            {data['message']}")
    print(f"  SIGNALS:            {data['signals']}")
    print(f"  EVENT:              {data['event']}")
    print(f"  STATE BEFORE:       {data['state_before']}")
    print(f"  STATE AFTER:        {data['state_after']}")
    print(f"  SYSTEM EVENTS:      {data['system_events']}")
    print(f"  TARGETING:          {data['targeting']}")
    print(f"  INTERACTION TYPE:   {data['interaction_type']}")
    print(f"  DECISION QUALITY:   {data['decision_quality']}")
    print(f"  CONFIRM REQUIRED:   {data['confirmation_required']}")
    print(f"  CONFLICTS:          {data['conflicts']}")
    print(f"  OPEN CONFLICTS:     {data['open_conflict_count']}")
    print(f"  NEW CONFLICT:       {data['new_conflict']}")
    print(f"  ACTIONS:            {data['actions']}")
    print(f"  RESPONSE:           {data['response']}")


def _print_summary(scenario_name: str, ctx: DecisionContext, state: State,
                   n_steps: int, n_system: int) -> None:
    print(f"\n{'═' * 60}")
    print(f"  SUMMARY — {scenario_name}")
    print(f"{'═' * 60}")
    open_conflicts = [c for c in ctx.conflicts if c.get("status") == "open"]
    print(f"  Final state:            {state.value}")
    print(f"  Final decision:         {ctx.decision or '(none)'}")
    print(f"  Participants:           {ctx.participants}")
    print(f"  Responded:              {list(ctx.responses.keys())}")
    print(f"  User turns:             {n_steps}")
    print(f"  System events fired:    {n_system}")
    print(f"  Confirmation required:  {_confirmation_required(ctx)}")
    print(f"  Decision quality:       {_evaluate_decision_quality(ctx) or '(n/a)'}")
    print(f"  Open conflicts:         {len(open_conflicts)}")
    print()


# ---------------------------------------------------------------------------
# Core simulation loop
# ---------------------------------------------------------------------------

def _simulate_scenario(
    name: str,
    question: str,
    participants: list[str],
    min_participants: int,
    messages: list[tuple[str, str]],  # (participant, message)
    llm,
    decision_rule: str = "consent",
) -> None:
    print(f"\n{'█' * 60}")
    print(f"  SCENARIO: {name}")
    print(f"{'█' * 60}")
    print(f"  Question:         {question}")
    print(f"  Participants:     {participants}")
    print(f"  Min participants: {min_participants}")
    print(f"  Decision rule:    {decision_rule}")
    print(f"  Messages:         {len(messages)}")

    # Initialize state + context
    state = State.CLARIFYING
    ctx = DecisionContext(
        question=question,
        participants=list(participants),
        min_participants=min_participants,
        decision_rule=decision_rule,
        services={"llm": llm},
    )

    step = 0
    total_system_events = 0

    for participant, message in messages:
        step += 1
        state_before = state
        conflicts_before = list(ctx.conflicts)

        # Update context with message
        responses = dict(ctx.responses)
        participant_responses = list(responses.get(participant, []))
        participant_responses.append(message)
        responses[participant] = participant_responses
        ctx = replace(ctx, responses=responses)

        # Ensure participant is tracked
        if participant not in ctx.participants:
            parts = list(ctx.participants)
            parts.append(participant)
            ctx = replace(ctx, participants=parts)

        # LLM interprets → signals
        signals = llm.interpret(message, _context_to_dict(ctx))

        # Signals → event (deterministic)
        event = map_signals_to_event(signals)

        # Transition
        all_actions: list[Action] = []
        next_state, actions, ctx = transition(state, event, ctx)
        all_actions.extend(actions)

        # Apply actions + merge signals
        ctx = _apply_actions(ctx, actions, signals, actor=participant)

        # System event loop
        system_events = []
        for _ in range(_MAX_SYSTEM_ITERATIONS):
            sys_event = _derive_system_event(next_state, ctx)
            if sys_event is None:
                break

            prev = next_state
            next_state, actions, ctx = transition(next_state, sys_event, ctx)
            all_actions.extend(actions)
            system_events.append(sys_event.value)
            total_system_events += 1

            if next_state == prev:
                break

        state = next_state

        # Targeting
        targeting = _compute_targeting(state, ctx)

        # Interaction type
        interaction_type = _select_interaction_type(
            state, ctx, targeting["targeting_reason"],
        )

        # Decision quality
        quality = _evaluate_decision_quality(ctx)

        # Conflict visibility
        open_conflicts = [c for c in ctx.conflicts if c.get("status") == "open"]
        open_before = len([c for c in conflicts_before if c.get("status") == "open"])
        conflicts_before_count = open_before

        # LLM generates response
        response = llm.generate(state.value, _context_to_dict(ctx))

        _print_step(step, {
            "actor": f"user ({participant})",
            "recipients": targeting["recipients"],
            "message": message,
            "signals": signals,
            "event": event.value,
            "state_before": state_before.value,
            "state_after": state.value,
            "system_events": system_events or "(none)",
            "targeting": targeting,
            "interaction_type": interaction_type,
            "decision_quality": quality or "(n/a)",
            "confirmation_required": _confirmation_required(ctx),
            "conflicts": ctx.conflicts or "(none)",
            "open_conflict_count": len(open_conflicts),
            "new_conflict": len(open_conflicts) > conflicts_before_count,
            "actions": [a.type.value for a in all_actions] or "(none)",
            "response": response[:200] + "..." if len(response) > 200 else response,
        })

    _print_summary(name, ctx, state, step, total_system_events)


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

def _run_scenario_1(llm) -> None:
    """Hidden conflict: late hard constraint contradicts apparent consensus."""
    _simulate_scenario(
        name="Hidden Conflict",
        question="When should we meet?",
        participants=["alice", "bob", "carol"],
        min_participants=3,
        messages=[
            ("alice", "Wednesday works for me"),
            ("bob", "Wednesday is fine"),
            ("carol", "I cannot do Wednesday at all"),
        ],
        llm=llm,
    )


def _run_scenario_2(llm) -> None:
    """False consensus / fragile quality: weak preferences, no real conviction."""
    _simulate_scenario(
        name="False Consensus — Fragile Quality",
        question="Where should we eat?",
        participants=["alice", "bob"],
        min_participants=2,
        messages=[
            ("alice", "I guess Italian is ok"),
            ("bob", "Italian is fine I guess"),
        ],
        llm=llm,
    )


def _run_scenario_3(llm) -> None:
    """Missing participant but non-critical: carol doesn't respond."""
    _simulate_scenario(
        name="Missing Non-Critical Participant",
        question="When should we meet?",
        participants=["alice", "bob", "carol"],
        min_participants=2,
        messages=[
            ("alice", "I'm flexible"),
            ("bob", "I'm also flexible, evening is fine"),
        ],
        llm=llm,
    )


def _run_scenario_4(llm) -> None:
    """Multi-dimensional decision gap: day vs time on separate axes."""
    _simulate_scenario(
        name="Multi-Dimensional Decision Gap",
        question="When should we meet?",
        participants=["alice", "bob"],
        min_participants=2,
        messages=[
            ("alice", "Wednesday works"),
            ("bob", "Evening is best for me"),
        ],
        llm=llm,
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    args = sys.argv[1:]
    use_mock = "--mock" in args
    args = [a for a in args if a != "--mock"]

    if use_mock:
        llm = _SimulatorMockLLM()
        print("Using SimulatorMockLLM (no API calls)")
    else:
        try:
            from llm.openai_llm import OpenAILLM
            llm = OpenAILLM()
            # Force client init to fail fast if no API key
            llm._get_client()
            print("Using OpenAILLM")
        except Exception:
            print("OPENAI_API_KEY not set — falling back to SimulatorMockLLM")
            llm = _SimulatorMockLLM()

    scenarios = {
        "1": _run_scenario_1,
        "2": _run_scenario_2,
        "3": _run_scenario_3,
        "4": _run_scenario_4,
    }

    if args and args[0] in scenarios:
        scenarios[args[0]](llm)
    else:
        for fn in scenarios.values():
            fn(llm)


if __name__ == "__main__":
    main()
