from __future__ import annotations


class LLM:
    """Abstract LLM interface.

    Strict separation:
    - interpret(): user input → structured signals (dict)
    - generate(): state + context → natural language response

    LLM does NOT:
    - change state
    - trigger transitions
    - decide outcomes

    Rule: LLM suggests, system decides.
    """

    def interpret(self, message: str, context: dict) -> dict:
        """Extract structured signals from a user message.

        Args:
            message: Raw user input.
            context: Current decision context (read-only).

        Returns:
            Dict of structured signals. Strict schema — ONLY these keys:
            - preferences: list[str]
            - constraints: list[str]
            - uncertainty: bool
            - conflict: bool
            - objection: bool
            - avoidance: bool
        """
        raise NotImplementedError

    def generate(self, state: str, context: dict) -> str:
        """Generate a natural language response.

        Args:
            state: Current state name (after transition).
            context: Current decision context (read-only).

        Returns:
            Natural language response string.
        """
        raise NotImplementedError

    def evaluate_critical_participants(
        self, context: dict, missing: list[str],
    ) -> dict:
        """Advisory: identify which missing participants are critical.

        Called ONLY when rule-based logic says no one is critical,
        to give the LLM a chance to flag participants that structural
        rules might miss.

        LLM can ONLY escalate (mark as critical), never de-escalate.

        Args:
            context: Compact decision context (read-only).
            missing: List of participant names who haven't responded.

        Returns:
            Dict with key "critical_participants": list[str].
            Must be a subset of `missing`.
        """
        return {"critical_participants": []}
