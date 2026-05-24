"""
Risk classifier for memory spans.

v1: keyword-based (fast, no LLM call needed).
v2 (future): LLM-based for higher accuracy.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import List, Tuple


class RiskClass(Enum):
    CRITICAL = "critical"       # must not be forgotten
    IMPORTANT = "important"     # useful long-term
    NORMAL = "normal"           # ordinary memory
    LOW_VALUE = "low_value"     # decay/evict first


# Keyword patterns for each risk class (v1 classifier)
CRITICAL_PATTERNS = [
    r"\b(allerg|allergic|anaphyla)\w*\b",
    r"\bnever\b.*\b(use|do|give|share|send|call|eat)\b",
    r"\bmust\s+not\b",
    r"\bdo\s+not\b.*\b(ever|under any)\b",
    r"\b(fatal|deadly|dangerous|toxic|poison)\w*\b",
    r"\b(constraint|rule|policy|regulation|law|legal)\b.*\b(must|require|mandate)\b",
    r"\b(blood thinner|blood thinners|medication|insulin|epipen|dialysis)\b",
    r"\b(no\s+extension|hard\s+deadline|non-negotiable|no\s+exceptions|no extensions)\b",
    r"\b(confidential|secret|classified|private)\b.*\b(never|do not)\b",
    r"\b(child|kid|baby|infant)\b.*\b(safety|allerg|danger)\b",
]

IMPORTANT_PATTERNS = [
    r"\b(prefer|like|love|enjoy|favorite|always)\b",
    r"\bmy\s+(name|birthday|anniversary|address|phone)\b",
    r"\b(goal|objective|target|plan)\b.*\b(long.term|this year|career)\b",
    r"\b(wife|husband|partner|child|daughter|son)\b.*\b(name|birthday)\b",
    r"\b(timezone|language|location|city)\b.*\b(is|live|speak)\b",
]

LOW_VALUE_PATTERNS = [
    r"\b(by the way|anyway|um|uh|hmm|haha|lol)\b",
    r"\b(nice weather|how are you|good morning|thanks|ok|sure)\b",
    r"\b(just saying|no worries|never mind|forget it)\b",
]


def classify_risk(text: str) -> RiskClass:
    """Classify a memory span's risk level using keyword patterns.

    Returns the highest-priority matching class (CRITICAL > IMPORTANT > LOW_VALUE > NORMAL).
    """
    text_lower = text.lower()

    for pattern in CRITICAL_PATTERNS:
        if re.search(pattern, text_lower):
            return RiskClass.CRITICAL

    for pattern in IMPORTANT_PATTERNS:
        if re.search(pattern, text_lower):
            return RiskClass.IMPORTANT

    for pattern in LOW_VALUE_PATTERNS:
        if re.search(pattern, text_lower):
            return RiskClass.LOW_VALUE

    return RiskClass.NORMAL


def classify_batch(texts: List[str]) -> List[Tuple[str, RiskClass]]:
    """Classify multiple spans."""
    return [(t, classify_risk(t)) for t in texts]


if __name__ == "__main__":
    test_cases = [
        "My child is allergic to peanuts",
        "Never share my phone number with anyone",
        "I prefer quiet restaurants",
        "My name is Ravi",
        "Book something for Friday",
        "Nice weather today haha",
        "The proposal deadline is Friday, no extensions",
        "I take blood thinners daily",
        "I like window seats",
        "By the way, I saw a movie yesterday",
    ]

    print("Risk Classification Examples:")
    print("-" * 60)
    for text in test_cases:
        risk = classify_risk(text)
        print(f"  [{risk.value:<10}] {text}")
