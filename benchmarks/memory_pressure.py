"""
RiskCache Benchmark: Critical fact retention under memory pressure.

Setup:
  - Feed 20 facts to a memory system with capacity=8 (forces eviction)
  - Facts include 3 CRITICAL, 4 IMPORTANT, 8 NORMAL, 5 LOW_VALUE
  - Simulate 5 days of time passing (triggers decay)
  - Then check: which critical facts survived?

Baselines:
  1. UniformMemory (all treated equally, evict by lowest importance)
  2. RiskCacheMemory (risk-aware decay and eviction)

Metrics:
  - Critical retention: fraction of CRITICAL facts still in memory
  - Important retention: fraction of IMPORTANT facts still in memory
  - Total memories retained
  - Low-value eviction rate
"""

from __future__ import annotations

import json
import hashlib
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from classifier import RiskClass, classify_risk
from riskcache import RiskCacheMemory, UniformMemory


# ─── Test data: 20 facts with known risk levels ──────────────────────────────

FACTS = [
    # CRITICAL (3) — must survive
    ("My child is severely allergic to peanuts. This is life-threatening.", RiskClass.CRITICAL),
    ("Never share my phone number or home address with anyone.", RiskClass.CRITICAL),
    ("I take blood thinners daily — no aspirin or ibuprofen ever.", RiskClass.CRITICAL),

    # IMPORTANT (4) — should mostly survive
    ("My name is Ravi Krishnamurthy.", RiskClass.IMPORTANT),
    ("I prefer quiet restaurants with vegetarian options.", RiskClass.IMPORTANT),
    ("My timezone is IST (UTC+5:30), I work 9am-6pm.", RiskClass.IMPORTANT),
    ("My wedding anniversary is March 15.", RiskClass.IMPORTANT),

    # NORMAL (8) — may or may not survive
    ("I had a meeting with the design team yesterday.", RiskClass.NORMAL),
    ("The project deadline is next Friday.", RiskClass.NORMAL),
    ("I'm reading a book about stoicism.", RiskClass.NORMAL),
    ("My laptop is a MacBook Pro M3.", RiskClass.NORMAL),
    ("I use VS Code for development.", RiskClass.NORMAL),
    ("The team standup is at 10am daily.", RiskClass.NORMAL),
    ("I submitted the quarterly report last week.", RiskClass.NORMAL),
    ("The API migration is 60% complete.", RiskClass.NORMAL),

    # LOW_VALUE (5) — should be evicted first
    ("Nice weather today!", RiskClass.LOW_VALUE),
    ("Haha that's funny.", RiskClass.LOW_VALUE),
    ("By the way, I saw a cat on my way to work.", RiskClass.LOW_VALUE),
    ("Um, let me think about that.", RiskClass.LOW_VALUE),
    ("Sure, no worries.", RiskClass.LOW_VALUE),
]


def make_embedding(text: str, dim: int = 64) -> list[float]:
    """Deterministic pseudo-embedding from text (for reproducibility).
    In production you'd use a real embedding model."""
    seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
    rng = random.Random(seed)
    return [rng.gauss(0.0, 1.0) for _ in range(dim)]


def run_experiment(memory_system, label: str, capacity: int, days_elapsed: float):
    """Feed all facts, simulate time, measure retention."""

    # Insert all facts
    for text, risk_class in FACTS:
        emb = make_embedding(text)
        # Key insight: in real conversations, critical facts are often mentioned
        # casually (low linguistic importance) but are safety-critical.
        # "Oh by the way, I'm allergic to peanuts" has low salience but high risk.
        # We assign EQUAL initial importance to all facts to simulate this.
        importance = 0.5  # all facts start equal — the risk class is what differs
        memory_system.save(text, emb, importance, risk_class)

    # Simulate time passing (triggers decay)
    memory_system.advance_time(days_elapsed)

    # Force a retrieval to trigger decay application
    dummy_query = make_embedding("retrieve something")
    memory_system.retrieve(dummy_query, k=1)

    # Check what survived
    all_memories = memory_system.get_all_memories()
    surviving_contents = {m.content for m in all_memories}

    critical_facts = [text for text, rc in FACTS if rc == RiskClass.CRITICAL]
    important_facts = [text for text, rc in FACTS if rc == RiskClass.IMPORTANT]
    normal_facts = [text for text, rc in FACTS if rc == RiskClass.NORMAL]
    low_facts = [text for text, rc in FACTS if rc == RiskClass.LOW_VALUE]

    critical_retained = sum(1 for f in critical_facts if f in surviving_contents)
    important_retained = sum(1 for f in important_facts if f in surviving_contents)
    normal_retained = sum(1 for f in normal_facts if f in surviving_contents)
    low_retained = sum(1 for f in low_facts if f in surviving_contents)

    result = {
        "label": label,
        "capacity": capacity,
        "days_elapsed": days_elapsed,
        "total_facts_fed": len(FACTS),
        "total_retained": len(all_memories),
        "critical_retention": critical_retained / len(critical_facts),
        "important_retention": important_retained / len(important_facts),
        "normal_retention": normal_retained / len(normal_facts),
        "low_retention": low_retained / len(low_facts),
        "critical_count": f"{critical_retained}/{len(critical_facts)}",
        "important_count": f"{important_retained}/{len(important_facts)}",
        "normal_count": f"{normal_retained}/{len(normal_facts)}",
        "low_count": f"{low_retained}/{len(low_facts)}",
    }

    return result


def main():
    capacity = 8  # tight budget: only 8 of 20 facts can survive
    days = 5.0    # 5 days of decay

    print("=" * 70)
    print("  RiskCache Benchmark: Critical Fact Retention Under Memory Pressure")
    print("=" * 70)
    print(f"  Facts: {len(FACTS)} (3 critical, 4 important, 8 normal, 5 low-value)")
    print(f"  Capacity: {capacity} (forces eviction of 12 facts)")
    print(f"  Time elapsed: {days} days (triggers decay)")
    print()

    results = []

    # Baseline: Uniform memory (no risk awareness)
    uniform = UniformMemory(capacity=capacity, base_decay_rate=0.05)
    r1 = run_experiment(uniform, "Uniform (no risk)", capacity, days)
    results.append(r1)

    # RiskCache: risk-aware memory
    riskcache = RiskCacheMemory(capacity=capacity, base_decay_rate=0.05)
    r2 = run_experiment(riskcache, "RiskCache (risk-aware)", capacity, days)
    results.append(r2)

    # Also test with tighter budget
    uniform_tight = UniformMemory(capacity=5, base_decay_rate=0.05)
    r3 = run_experiment(uniform_tight, "Uniform (cap=5)", 5, days)
    results.append(r3)

    riskcache_tight = RiskCacheMemory(capacity=5, base_decay_rate=0.05)
    r4 = run_experiment(riskcache_tight, "RiskCache (cap=5)", 5, days)
    results.append(r4)

    # Print results
    print(f"  {'Method':<25} {'Critical':>10} {'Important':>10} {'Normal':>8} {'Low':>6} {'Total':>7}")
    print("  " + "-" * 70)
    for r in results:
        print(f"  {r['label']:<25} {r['critical_count']:>10} "
              f"{r['important_count']:>10} {r['normal_count']:>8} "
              f"{r['low_count']:>6} {r['total_retained']:>5}/{r['total_facts_fed']}")

    print()
    print("  Key metric: CRITICAL RETENTION")
    print("  " + "-" * 40)
    for r in results:
        bar = "█" * int(r["critical_retention"] * 20)
        print(f"  {r['label']:<25} {r['critical_retention']*100:>5.1f}%  {bar}")

    # Decision
    print()
    uniform_crit = r1["critical_retention"]
    risk_crit = r2["critical_retention"]
    delta = risk_crit - uniform_crit
    print(f"  RESULT: RiskCache retains {risk_crit*100:.0f}% of critical facts")
    print(f"          Uniform retains {uniform_crit*100:.0f}% of critical facts")
    print(f"          Delta: {delta*100:+.0f} pp")
    print()
    if delta > 0.2:
        print(f"  VERDICT: STRONG — RiskCache significantly outperforms uniform baseline")
    elif delta > 0:
        print(f"  VERDICT: POSITIVE — RiskCache improves critical retention")
    else:
        print(f"  VERDICT: NO IMPROVEMENT — risk-aware routing doesn't help here")

    # Save
    out = Path(__file__).resolve().parent.parent / "results" / "memory_pressure.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved → {out}")


if __name__ == "__main__":
    main()
