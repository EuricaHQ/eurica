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

Eurica is a multi-party Decision Coordinator — a system where a group of participants collaboratively reach a decision. The spec in spec/ is the authoritative design document.

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

## STRICT DEVELOPMENT RULES (CRITICAL)

### Spec Authority
- The specification (spec/*.md) is the single source of truth
- Always follow the latest specification in spec/
- If code and spec differ → ALWAYS follow spec
- NEVER introduce behavior not explicitly grounded in spec
- If spec is incomplete or unclear → ask for clarification, do NOT assume

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

### LLM Constraints
- LLM is interpretation only
- LLM must NOT:
  - decide transitions
  - override system logic
  - introduce new behavior

### Implementation Style
- Prefer minimal, surgical changes
- DO NOT refactor broadly unless explicitly requested
- Show diffs before large changes

### Testing Discipline
- Always update tests when behavior changes
- Run full test suite before completion
- Add tests for edge cases

## Current Focus

- Dimension completeness (NOT yet enforced)
- Prevent premature decisions