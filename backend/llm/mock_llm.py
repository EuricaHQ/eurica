from __future__ import annotations

from llm.interface import LLM


class MockLLM(LLM):
    """Mock LLM for development and testing.

    interpret(): keyword-based signal extraction (no real LLM).
    generate(): template-based response (no real LLM).
    """

    def interpret(self, message: str, context: dict) -> dict:
        lower = message.lower()
        return {
            "preferences": [],
            "constraints": [],
            "uncertainty": False,
            "conflict": False,
            "objection": False,
            "avoidance": "later" in lower,
        }

    def generate(self, state: str, context: dict) -> str:
        return f"[{state}] What is your preference?"

    def evaluate_critical_participants(
        self, context: dict, missing: list[str],
    ) -> dict:
        return {"critical_participants": []}
