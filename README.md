# RiskCache

**Risk-Aware Memory Retention for Long-Horizon AI Agents.**

RiskCache is a memory decision layer that sits above a vector memory engine
(Bitcache) and classifies each piece of information by risk level. Critical
memories (safety rules, allergies, hard constraints) are protected from decay
and eviction. Low-value memories are discarded first under memory pressure.

> **Status:** research prototype. Not production-ready.

---

## Problem

Current agent memory systems treat all memories equally. When memory is full,
they evict based on recency or importance score — but they don't know that
"user is allergic to peanuts" is fundamentally different from "user likes
Italian food." Under memory pressure, critical safety information can be
lost just as easily as small talk.

## Solution

RiskCache adds a **risk classifier** that labels each memory span:

| Risk class | Decay | Eviction | Example |
|---|---|---|---|
| `CRITICAL` | None | Never | "allergic to peanuts", "never share phone number" |
| `IMPORTANT` | Very slow | Only after all LowValue gone | "prefers morning meetings" |
| `NORMAL` | Default | Normal | "mentioned a project deadline" |
| `LOW_VALUE` | Fast | First to go | "said 'by the way'", repeated info |

## Architecture

```
User input / agent trace
        ↓
┌─────────────────────────┐
│   Risk Classifier       │  ← labels each span
│   (keyword or LLM)      │
└─────────────────────────┘
        ↓
┌─────────────────────────┐
│   Memory Router         │  ← decides storage form
│                         │
│   CRITICAL → prompt +   │
│              protected   │
│              memory      │
│   IMPORTANT → slow-decay │
│              memory      │
│   NORMAL → standard     │
│            memory        │
│   LOW_VALUE → fast-decay │
│              memory      │
└─────────────────────────┘
        ↓
┌─────────────────────────┐
│   Bitcache              │  ← storage engine
│   (vector + graph)      │
└─────────────────────────┘
```

## Research Question

> Can risk-aware memory routing improve retention of safety-critical facts
> under constrained memory budgets, while preserving retrieval speed?

## Expected Result

| Method | Critical retention | Task success | Tokens |
|---|---:|---:|---:|
| Full context (no memory) | 100% | high | very high |
| Bitcache (uniform) | ~40% | low | low |
| Bitcache (importance-only) | ~60% | medium | low |
| **RiskCache** | **~95%** | **high** | **low** |

## Project layout

```
riskcache/
├── src/
│   ├── classifier.py      Risk classifier (keyword + LLM)
│   ├── memory_router.py   Routes memories to storage with risk policies
│   └── riskcache.py       Main RiskCache class (wraps Bitcache)
├── benchmarks/
│   └── memory_pressure.py Benchmark: critical fact retention under pressure
├── data/
│   └── conversations.jsonl Test conversations with critical facts
├── results/
├── README.md
└── LICENSE
```

## Relationship to other projects

- **Bitcache** — the storage engine RiskCache uses (fast vector retrieval, decay, eviction)
- **CRISP** — routes which *tools* the agent sees. RiskCache routes which *memories* stay in context. Same pattern, different target.

## License

MIT
