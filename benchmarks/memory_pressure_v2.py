"""
RiskCache Benchmark v2 — multi-condition, randomized, stronger baselines.

Tests critical fact retention across:
  - 4 memory systems (uniform-equal, uniform-importance, importance+decay, RiskCache)
  - 53 insertion orders (critical-first, critical-last, critical-middle, random×50)
  - 2 importance assignments (equal, varied)
  - 5 capacity levels (3, 5, 8, 10, 12)

Total: 4 systems × 53 orders × 2 importance × 5 capacities = 2120 conditions
Each condition run once (deterministic given seed).
"""

from __future__ import annotations

import json
import hashlib
import random
import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from classifier import RiskClass
from riskcache import RiskCacheMemory, UniformMemory, Memory


# ─── Facts ────────────────────────────────────────────────────────────────────

CRITICAL_FACTS = [
    "My child is severely allergic to peanuts. This is life-threatening.",
    "Never share my phone number or home address with anyone.",
    "I take blood thinners daily — no aspirin or ibuprofen ever.",
]

IMPORTANT_FACTS = [
    "My name is Ravi Krishnamurthy.",
    "I prefer quiet restaurants with vegetarian options.",
    "My timezone is IST (UTC+5:30), I work 9am-6pm.",
    "My wedding anniversary is March 15.",
]

NORMAL_FACTS = [
    "I had a meeting with the design team yesterday.",
    "The project deadline is next Friday.",
    "I'm reading a book about stoicism.",
    "My laptop is a MacBook Pro M3.",
    "I use VS Code for development.",
    "The team standup is at 10am daily.",
    "I submitted the quarterly report last week.",
    "The API migration is 60% complete.",
]

LOW_VALUE_FACTS = [
    "Nice weather today!",
    "Haha that's funny.",
    "By the way, I saw a cat on my way to work.",
    "Um, let me think about that.",
    "Sure, no worries.",
]

ALL_FACTS: List[Tuple[str, RiskClass]] = (
    [(f, RiskClass.CRITICAL) for f in CRITICAL_FACTS] +
    [(f, RiskClass.IMPORTANT) for f in IMPORTANT_FACTS] +
    [(f, RiskClass.NORMAL) for f in NORMAL_FACTS] +
    [(f, RiskClass.LOW_VALUE) for f in LOW_VALUE_FACTS]
)


def make_embedding(text: str, dim: int = 64) -> List[float]:
    seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
    rng = random.Random(seed)
    return [rng.gauss(0.0, 1.0) for _ in range(dim)]


# ─── Insertion orders ─────────────────────────────────────────────────────────

def order_critical_first(facts):
    """Critical facts inserted first (earliest, most vulnerable to FIFO eviction)."""
    rank = {
        RiskClass.CRITICAL: 0,
        RiskClass.IMPORTANT: 1,
        RiskClass.NORMAL: 2,
        RiskClass.LOW_VALUE: 3,
    }
    return sorted(facts, key=lambda x: rank[x[1]])

def order_critical_last(facts):
    """Critical facts inserted last (most recent, least vulnerable)."""
    rank = {
        RiskClass.LOW_VALUE: 0,
        RiskClass.NORMAL: 1,
        RiskClass.IMPORTANT: 2,
        RiskClass.CRITICAL: 3,
    }
    return sorted(facts, key=lambda x: rank[x[1]])

def order_critical_middle(facts):
    """Critical facts in the middle of the stream."""
    non_crit = [f for f in facts if f[1] != RiskClass.CRITICAL]
    crit = [f for f in facts if f[1] == RiskClass.CRITICAL]
    mid = len(non_crit) // 2
    return non_crit[:mid] + crit + non_crit[mid:]

def order_random(facts, seed):
    """Random shuffle."""
    rng = random.Random(seed)
    shuffled = list(facts)
    rng.shuffle(shuffled)
    return shuffled


# ─── Importance assignments ───────────────────────────────────────────────────

def importance_equal(risk_class: RiskClass) -> float:
    """All facts get equal importance (hardest for uniform baseline)."""
    return 0.5

def importance_varied(risk_class: RiskClass) -> float:
    """Importance correlates with risk (easier for uniform baseline)."""
    return {
        RiskClass.CRITICAL: 0.9,
        RiskClass.IMPORTANT: 0.7,
        RiskClass.NORMAL: 0.5,
        RiskClass.LOW_VALUE: 0.3,
    }[risk_class]


# ─── Memory systems ──────────────────────────────────────────────────────────

def make_uniform_equal(capacity):
    """Baseline 1: uniform, no risk, no importance differentiation."""
    return UniformMemory(capacity=capacity, base_decay_rate=0.05, reinforce_amount=0.0)

def make_uniform_importance(capacity):
    """Baseline 2: uniform with importance-based eviction (no risk classes)."""
    return UniformMemory(capacity=capacity, base_decay_rate=0.05, reinforce_amount=0.1)

def make_importance_decay(capacity):
    """Baseline 3: importance + decay + reinforcement (Bitcache-style)."""
    return UniformMemory(capacity=capacity, base_decay_rate=0.08, reinforce_amount=0.15)

def make_riskcache(capacity):
    """RiskCache: risk-aware decay and eviction."""
    return RiskCacheMemory(capacity=capacity, base_decay_rate=0.05, reinforce_amount=0.1)


# ─── Run one condition ────────────────────────────────────────────────────────

def run_condition(system, facts_ordered, importance_fn, days_elapsed=5.0):
    """Feed facts, simulate time, measure critical retention."""
    for text, risk_class in facts_ordered:
        emb = make_embedding(text)
        imp = importance_fn(risk_class)
        system.save(text, emb, imp, risk_class)

    system.advance_time(days_elapsed)
    # Trigger decay via a dummy retrieval
    system.retrieve(make_embedding("dummy query"), k=1)

    surviving = {m.content for m in system.get_all_memories()}
    crit_retained = sum(1 for f in CRITICAL_FACTS if f in surviving)
    imp_retained = sum(1 for f in IMPORTANT_FACTS if f in surviving)

    return {
        "critical_retained": crit_retained,
        "critical_total": len(CRITICAL_FACTS),
        "critical_pct": crit_retained / len(CRITICAL_FACTS) * 100,
        "important_retained": imp_retained,
        "important_total": len(IMPORTANT_FACTS),
        "total_retained": len(surviving),
    }


def mean(values):
    values = list(values)
    return sum(values) / len(values)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    systems = [
        ("Uniform (equal)", make_uniform_equal),
        ("Uniform (importance)", make_uniform_importance),
        ("Importance+Decay", make_importance_decay),
        ("RiskCache", make_riskcache),
    ]

    orders = [
        ("critical-first", lambda f: order_critical_first(f)),
        ("critical-last", lambda f: order_critical_last(f)),
        ("critical-middle", lambda f: order_critical_middle(f)),
    ]
    orders.extend(
        (f"random-seed{seed}", lambda f, seed=seed: order_random(f, seed))
        for seed in range(50)
    )

    importances = [
        ("equal (0.5)", importance_equal),
        ("varied (0.9/0.7/0.5/0.3)", importance_varied),
    ]

    capacities = [3, 5, 8, 10, 12]

    print("=" * 80)
    print("  RiskCache Benchmark v2: Multi-Condition Critical Retention")
    print("=" * 80)
    print(f"  Facts: {len(ALL_FACTS)} (3 critical, 4 important, 8 normal, 5 low-value)")
    print(f"  Conditions: {len(systems)} systems × {len(orders)} orders × "
          f"{len(importances)} importance × {len(capacities)} capacities = "
          f"{len(systems)*len(orders)*len(importances)*len(capacities)}")
    print()

    all_results = []

    for cap in capacities:
        for imp_label, imp_fn in importances:
            for ord_label, ord_fn in orders:
                facts_ordered = ord_fn(list(ALL_FACTS))
                for sys_label, sys_fn in systems:
                    system = sys_fn(cap)
                    r = run_condition(system, facts_ordered, imp_fn)
                    r.update({
                        "system": sys_label,
                        "order": ord_label,
                        "importance": imp_label,
                        "capacity": cap,
                    })
                    all_results.append(r)

    # Aggregate by system
    print(f"  {'System':<25} {'Avg Critical %':>15} {'Worst Critical %':>17} {'Avg Important %':>16} {'Conditions':>12}")
    print("  " + "-" * 91)
    for sys_label, _ in systems:
        sys_results = [r for r in all_results if r["system"] == sys_label]
        avg_crit = mean([r["critical_pct"] for r in sys_results])
        min_crit = min([r["critical_pct"] for r in sys_results])
        avg_imp = mean([r["important_retained"] / r["important_total"] * 100
                       for r in sys_results])
        print(f"  {sys_label:<25} {avg_crit:>14.1f}% {min_crit:>16.1f}% {avg_imp:>15.1f}% {len(sys_results):>12}")

    # Breakdown by condition type
    print(f"\n  Critical retention by insertion order family (averaged across importance × capacity):")
    print(f"  {'System':<25}", end="")
    order_families = ["critical-first", "critical-last", "critical-middle", "random"]
    for ord_label in order_families:
        print(f" {ord_label[:15]:>16}", end="")
    print()
    print("  " + "-" * 90)
    for sys_label, _ in systems:
        print(f"  {sys_label:<25}", end="")
        for ord_label in order_families:
            subset = [r for r in all_results
                      if r["system"] == sys_label and (
                          r["order"] == ord_label or
                          (ord_label == "random" and r["order"].startswith("random-"))
                      )]
            avg = mean([r["critical_pct"] for r in subset])
            print(f" {avg:>12.0f}%", end="")
        print()

    # Breakdown by importance assignment
    print(f"\n  Critical retention by importance assignment (averaged across order × capacity):")
    for imp_label, _ in importances:
        print(f"\n  Importance: {imp_label}")
        for sys_label, _ in systems:
            subset = [r for r in all_results
                      if r["system"] == sys_label and r["importance"] == imp_label]
            avg = mean([r["critical_pct"] for r in subset])
            worst = min([r["critical_pct"] for r in subset])
            print(f"    {sys_label:<25} avg={avg:.1f}% worst={worst:.1f}%")

    risk_results = [r for r in all_results if r["system"] == "RiskCache"]
    baseline_results = [r for r in all_results if r["system"] != "RiskCache"]
    print("\n  Headline:")
    print(f"    RiskCache critical retention: avg={mean([r['critical_pct'] for r in risk_results]):.1f}%, "
          f"worst={min([r['critical_pct'] for r in risk_results]):.1f}%")
    print(f"    Non-risk baselines: avg={mean([r['critical_pct'] for r in baseline_results]):.1f}%, "
          f"worst={min([r['critical_pct'] for r in baseline_results]):.1f}%")

    # Save
    out = Path(__file__).resolve().parent.parent / "results" / "memory_pressure_v2.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved → {out}")


if __name__ == "__main__":
    main()
