"""
Risk classifier for memory spans.

v1.5: keyword-based (fast, no LLM call needed).
      Expanded from v1 to cover medical conditions, legal deadlines,
      privacy constraints, and security secrets more broadly.
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


# Keyword patterns for each risk class (v1.5 classifier)
CRITICAL_PATTERNS = [
    # --- Allergies & food safety ---
    r"\b(allerg|allergic|anaphyla)\w*\b",
    r"\b(intoleran\w*|celiac|coeliac)\b",                       # lactose intolerant, celiac
    r"\bgluten\b.*\b(sick|disease|cannot|avoid)\b",             # celiac/gluten sensitivity

    # --- Medical / medication ---
    r"\b(blood thinner|blood thinners|warfarin|coumadin)\b",
    r"\b(medication|insulin|epipen|dialysis|suboxone)\b",
    r"\b(MAO inhibitor|MAOI|mao inhibitor)\w*\b",              # MAO inhibitors
    r"\b(pregnan\w*)\b.*\b(month|week|trimester|due)\b",       # pregnancy mentions with timing
    r"\b\d+\s*months?\s+pregnant\b",                            # "7 months pregnant"
    r"\b(life.threatening|anaphylactic|epipen)\b",

    # --- Explicit prohibitions ---
    r"\bnever\b.*\b(use|do|give|share|send|call|eat|drink|around|reveal|tell|discuss)\b",
    r"\bmust\s+not\b",
    r"\bmust\s+never\b",                                        # "must never be shared"
    r"\bdo\s+not\b.*\b(ever|under any)\b",
    r"\bcannot\s+(discuss|share|tell|reveal|disclose)\b",       # "I cannot discuss"
    r"\bnever\s+(reveal|disclose|discuss)\b",                   # "Never reveal my SSN"
    r"\bnever\s+tell\s+anyone\b",                               # "Never tell anyone"
    r"\bI\s+never\s+discuss\b",                                 # "I never discuss my compensation"

    # --- Danger / safety ---
    r"\b(fatal|deadly|dangerous|toxic|poison)\w*\b",
    r"\b(child|kid|baby|infant)\b.*\b(safety|allerg|danger)\b",

    # --- Legal / deadlines with consequences ---
    r"\b(constraint|rule|policy|regulation|law|legal)\b.*\b(must|require|mandate)\b",
    r"\b(no\s+extension|hard\s+deadline|non-negotiable|no\s+exceptions|no extensions)\b",
    r"\b(contract penalty|penalty clause|legal consequence)\b",
    r"\bdeadline\b.*\b(forfeit|permanent|penalty|ban|judgment)\b",  # deadline + consequence
    r"\b(forfeit|forfeits)\b.*\b(permanent|priority|right)\b",      # "forfeits our priority date"
    r"\bmissing\s+it\b.*\b(forfeit|penalty|ban|judgment)\b",        # "Missing it forfeits..."
    r"\bif\s+I\s+miss\b.*\b(default|judgment|penalty|ban)\b",      # "If I miss it, default judgment"
    r"\b(default\s+judgment|3.year.*ban|re.entry\s+ban)\b",         # specific legal consequences
    r"\b(overstay|overstaying)\b.*\b(ban|penalty|trigger)\b",       # visa overstay
    r"\blate\s+filing\b.*\b(penalty|incur)\b",                      # tax penalty
    r"\b(visa|h.1b)\b.*\b(expire|overstay)\b",                     # visa expiration

    # --- NDA / confidentiality ---
    r"\b(confidential|secret|classified|private)\b.*\b(never|do not)\b",
    r"\bNDA\b",                                                     # NDA mentions
    r"\bsigned\s+an?\s+NDA\b",                                     # "signed an NDA"
    r"\bnon.public\s+information\b",                                # insider trading / MNPI
    r"\bmaterial\s+non.public\b",                                   # "material non-public information"
    r"\b(insider|insider\s+trading)\b",

    # --- Security / secrets ---
    r"\b(password|api\s*key|private\s*key|ssh\s*key|secret\s*key)\b.*\b(never|must not|cannot|shared)\b",
    r"\bmust\s+never\s+be\s+shared\b",                             # "must never be shared"
    r"\bleaking\b.*\b(expose|customer|data|payment)\b",            # "Leaking it would expose"
    r"\b(production\s+database|prod\s+db)\b.*\b(password|credential)\b",

    # --- Privacy / protection ---
    r"\b(witness\s+protection|stalker|restraining\s+order)\b",     # safety situations
    r"\b(custody\s+dispute|custody)\b.*\b(never|do not|cannot)\b",
    r"\bidentity\s+theft\b",                                        # identity theft risk
    r"\bSSN\b.*\b(never|do not|cannot|reveal)\b",                  # SSN protection

    # --- Recovery / addiction ---
    r"\b(recovery|sober|sobriety|addiction|recovering)\b",
    r"\bmust never be around\b",

    # --- Competitors / embargo ---
    r"\b(competitor|competitors)\b.*\b(must not|cannot|never)\b",  # competitive secrets
    r"\b(embargo|launch\s+date)\b.*\b(before|until|cannot)\b",     # product embargo
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
