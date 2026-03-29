# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Backend (Python/FastAPI)

cd backend

# Run the server
source .venv/bin/activate
uvicorn main:app --reload

# Run all tests
python -m pytest test_*.py

# Run a single test file
python test_flows.py
python test_guard_v2_signals.py

# Simulate multi-participant conversation without an API key (uses MockLLM)
python simulate_conversation.py

Tests requiring real LLM calls (test_e2e_smoke.py, test_openai_interpret.py, test_critical_llm_advisory.py) need OPENAI_API_KEY set in backend/.env.

### Frontend (Next.js)

cd frontend-web
npm run dev       # Dev server at http://localhost:3000
npm run build
npm run lint

Warning: This project uses Next.js 16 which has breaking changes. Before editing frontend code, read the relevant guide in frontend-web/node_modules/next/dist/docs/.

## Architecture

Eurica is a multi-party Decision Coordinator — a system where a group of participants collaboratively reach a decision.

### Core Principle: LLM is advisory, state machine is authoritative

The system enforces strict separation:
1. LLM interprets user messages → structured Signals
2. signal_mapper maps Signals → Event (deterministic)
3. State machine processes Event → State + Actions

The LLM never makes decisions.

### State Machine (backend/machine/)
- states.py — EFSM states:
  CLARIFYING → COLLECTING → AGGREGATING → RESOLVING → AVOIDING → VALIDATING → DECIDING → DECIDED / INFEASIBLE
- events.py — Events:
  RESPONSE_RECEIVED, AGGREGATION_COMPLETED, VALIDATION_COMPLETED, DECISION_CONFIRMED, DECISION_REJECTED, AVOIDANCE_DETECTED
- context.py — DecisionContext dataclass
- transition.py — FSM logic
- actions.py — ActionType enum

After a transition, the system may emit additional system events (max 3 iterations).

### LLM Layer (backend/llm/)
- interface.py
- openai_llm.py
- mock_llm.py

### Interpreter / Signal Layer (backend/interpreter/)
- signals.py
- signal_mapper.py

### API Layer (backend/api/routes.py)
- POST /chat
- Pipeline: interpret → map → transition → actions

### Frontend (frontend-web/)
- Next.js chat UI
- Calls backend at http://127.0.0.1:8000/chat

### Mobile (eurica-mobile/)
- Expo React Native project

## Spec Architecture

The specification is split into three layers:

### Core Spec (spec/core/)
Defines system behavior: states, transitions, guards, routing, targeting, interaction_type selection.

### Interaction Spec (spec/interaction/)
Defines communication behavior: phrasing rules, constraints, allowed/forbidden patterns.

### Meta Spec (spec/meta/)
Defines separation rules and the contract between Core and Interaction layers.

## Spec Authority

- ALWAYS follow latest Core Spec for: logic, transitions, guards, routing, targeting
- ALWAYS follow latest Interaction Spec for: message generation, phrasing constraints, allowed/forbidden patterns
- ALWAYS follow Meta Spec for: separation rules, architectural constraints
- Always use latest file in each spec directory
- Do NOT reference outdated spec versions
- Do NOT hardcode spec assumptions
- If code and spec differ → ALWAYS follow spec
- If spec is incomplete or unclear → ask for clarification, do NOT assume
## Spec Governance (CRITICAL)

- /spec is authoritative and controlled by Spec Master
- NEVER create or modify spec files autonomously
- Spec files may ONLY be created or updated when:
  → an explicit instruction is given to implement a provided spec
- In that case:
  → write the spec EXACTLY as provided
  → do NOT modify, extend, interpret, or restructure it
- NEVER increment spec versions unless explicitly instructed
- Any improvements, extensions, or ideas MUST be:
  → written to /spec/proposals/
  → NEVER directly applied to spec files

## Separation Enforcement (CRITICAL)

NEVER mix Core and Interaction concerns:
- NEVER encode phrasing rules in logic
- NEVER encode logic in prompts or message templates

The interface between both layers is interaction_type ONLY:
- Core Spec determines interaction_type
- Interaction Spec determines how interaction_type is phrased

### Communication Rules in Core Spec
If Core Spec contains communication-related constraints (phrasing rules, tone, message structure):
- Treat them as Interaction Spec rules
- Apply them ONLY in message generation (LLM layer)
- NEVER implement them in logic or transitions

### Implementation Rules
- Logic must ONLY depend on Core Spec
- Communication must ONLY depend on Interaction Spec
- DO NOT hardcode phrasing in targeting or transition logic
- DO NOT introduce behavioral assumptions outside specs
- DO NOT merge both layers into a single implementation

### LLM Behavior Constraint
- LLM output MUST comply with Interaction Spec
- LLM MUST NOT:
  - invent phrasing outside allowed patterns
  - introduce tone/style not defined in Interaction Spec
  - encode decision logic
- Enforce Interaction Spec constraints via validation after generation
- If constraints are violated → regenerate output

## STRICT DEVELOPMENT RULES (CRITICAL)

### Determinism
- The system must remain fully deterministic
- NO hidden heuristics
- NO implicit assumptions

### State Machine Safety
- NEVER change:
  - states
  - transitions
  - guard logic
  without explicit spec reference
- Guard priority MUST be preserved exactly

### Implementation Style
- Prefer minimal, surgical changes
- DO NOT refactor broadly unless explicitly requested
- Show diffs before large changes

### Spec Compliance (Implementation Detail)
- Incomplete solutions must NOT be finalized, even if compatible
- When completeness is uncertain → prefer clarification over decision
- Do NOT collapse multi-dimensional inputs into a single flat value

### Testing Discipline
- Always update tests when behavior changes
- Run full test suite before completion
- Add tests for edge cases

## Implementation Notes Governance

- /implementation/notes.md is the canonical place for:
  → implementation decisions
  → simplifications
  → temporary deviations from spec
- This file is:
  → non-authoritative
  → documentation only
  → separate from spec
- Claude MAY propose entries for this file
- Claude MUST NOT:
  - add entries autonomously
  - modify entries without instruction
- Updates to notes.md require explicit instruction
- Notes MUST reflect actual implemented behavior and include rationale and future intent
- Notes MUST NOT define system logic, override spec, or introduce new behavior
- Entries must NOT introduce new behavior

## Current Focus

- Dimension completeness (NOT yet enforced)
- Prevent premature decisions
