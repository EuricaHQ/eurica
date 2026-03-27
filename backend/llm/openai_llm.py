from __future__ import annotations

import json
import os

from openai import OpenAI

from llm.interface import LLM


# Strict signal schema — ONLY these keys are allowed out of interpret().
# No extra fields. Missing keys filled with defaults.
_SIGNAL_SCHEMA_LISTS = ("preferences", "constraints")
_SIGNAL_SCHEMA_BOOLS = ("uncertainty", "conflict", "objection", "avoidance")

_SAFE_DEFAULT: dict = {
    "preferences": [],
    "constraints": [],
    "uncertainty": False,
    "conflict": False,
    "objection": False,
    "avoidance": False,
}

_MODEL = "gpt-4o-mini"
_TEMPERATURE = 0.2
_TIMEOUT = 10  # seconds


class OpenAILLM(LLM):
    """OpenAI-backed LLM implementation.

    All provider-specific logic is contained in this file.
    Conforms to the LLM interface — swappable with MockLLM or any
    future provider (Mistral, etc.) without touching the rest of the system.
    """

    def __init__(self) -> None:
        self._client: OpenAI | None = None

    def _get_client(self) -> OpenAI:
        """Lazy client initialization — defers API key read to first use."""
        if self._client is None:
            self._client = OpenAI(
                api_key=os.environ["OPENAI_API_KEY"],
                timeout=_TIMEOUT,
            )
        return self._client

    def _call(self, prompt: str) -> str:
        """Send a single prompt to OpenAI and return the text content."""
        response = self._get_client().chat.completions.create(
            model=_MODEL,
            temperature=_TEMPERATURE,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""

    # -----------------------------------------------------------------
    # interpret()
    # -----------------------------------------------------------------

    def interpret(self, message: str, context: dict) -> dict:
        """Extract structured signals from a user message.

        Returns a dict with exact keys matching the LLM interface contract.
        On any failure (API error, bad JSON), returns safe defaults.
        """
        prompt = (
            "Extract structured signals from the following user message "
            "in the context of a group decision.\n"
            "\n"
            f"Decision question: {context.get('question', '(not set)')}\n"
            f"Participants so far: {context.get('participants', [])}\n"
            f"Existing preferences: {context.get('preferences', [])}\n"
            "\n"
            f"User message: \"{message}\"\n"
            "\n"
            "Return ONLY a JSON object with EXACTLY these keys:\n"
            "- preferences: list of strings (stated preferences)\n"
            "- constraints: list of strings (hard restrictions: must / cannot)\n"
            "- uncertainty: boolean (user expresses doubt or confusion)\n"
            "- conflict: boolean (user disagrees with existing preferences)\n"
            "- objection: boolean (user raises a strong objection)\n"
            "- avoidance: boolean (user avoids or deflects the decision)\n"
            "\n"
            "IMPORTANT: Constraints are hard restrictions (must / cannot). "
            "Uncertainty or hesitation is NOT a constraint.\n"
            "\n"
            "JSON ONLY. No explanation. No markdown. No wrapping. "
            "No extra keys."
        )

        try:
            raw = self._call(prompt)
            signals = json.loads(raw)
        except Exception:
            return dict(_SAFE_DEFAULT)

        # Strict schema enforcement: keep ONLY allowed keys, correct types
        result = {}
        for key in _SIGNAL_SCHEMA_LISTS:
            val = signals.get(key, [])
            result[key] = val if isinstance(val, list) else []
        for key in _SIGNAL_SCHEMA_BOOLS:
            val = signals.get(key, False)
            result[key] = val if isinstance(val, bool) else False

        return result

    # -----------------------------------------------------------------
    # generate()
    # -----------------------------------------------------------------

    def generate(self, state: str, context: dict) -> str:
        """Generate a natural language response for the current state."""
        compact = {
            "question": context.get("question"),
            "participants": context.get("participants"),
            "preferences": context.get("preferences"),
        }
        prompt = (
            "You are a Decision Coordinator. "
            "Be concise. Ask only what is necessary.\n"
            "\n"
            f"Current state: {state}\n"
            f"Context: {compact}\n"
            "\n"
            "Generate a short, helpful response appropriate for this state."
        )

        try:
            return self._call(prompt)
        except Exception:
            return f"[{state}] How would you like to proceed?"

    # -----------------------------------------------------------------
    # evaluate_critical_participants()
    # -----------------------------------------------------------------

    def evaluate_critical_participants(
        self, context: dict, missing: list[str],
    ) -> dict:
        """Ask LLM which missing participants could still influence the outcome.

        Advisory only. On any failure, returns empty list (safe default).
        """
        prompt = (
            "You are evaluating a group decision.\n"
            "\n"
            f"Decision question: {context.get('question', '(not set)')}\n"
            f"Decision rule: {context.get('decision_rule', 'consent')}\n"
            f"Current preferences: {context.get('preferences', [])}\n"
            f"Current constraints: {context.get('constraints', [])}\n"
            f"Missing participants (have not responded): {missing}\n"
            "\n"
            "Which of the missing participants could still influence "
            "the decision outcome? Consider whether their input could:\n"
            "- change the winning option\n"
            "- block consensus\n"
            "- introduce new constraints\n"
            "- affect feasibility\n"
            "\n"
            "Return ONLY a JSON object:\n"
            '{"critical_participants": ["name1", "name2"]}\n'
            "\n"
            "If none are critical, return:\n"
            '{"critical_participants": []}\n'
            "\n"
            "JSON ONLY. No explanation."
        )

        try:
            raw = self._call(prompt)
            result = json.loads(raw)
            participants = result.get("critical_participants", [])
            if not isinstance(participants, list):
                return {"critical_participants": []}
            # Sanitize: only keep names that are actually in missing
            valid = [p for p in participants if p in missing]
            return {"critical_participants": valid}
        except Exception:
            return {"critical_participants": []}
