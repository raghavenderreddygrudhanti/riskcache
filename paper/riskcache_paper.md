# RiskCache: Risk-Aware Retention and Surfacing for Long-Term Agent Memory

**Raghavender Reddy Grudhanti**

---

## Abstract

Long-term memory is essential for LLM agents that operate across sessions, but current memory systems treat all stored information equally. When memory is constrained, critical safety facts — allergies, medication interactions, legal deadlines, privacy rules — can be evicted or fail to surface just as easily as casual preferences. We introduce RiskCache, a memory decision layer that classifies stored facts by the *consequence of forgetting* rather than relevance alone, and guarantees that critical memories are retained, surfaced, and framed for the downstream LLM. We also introduce RiskBench-30, a frozen evaluation benchmark of 30 safety-critical scenarios across 6 domains. On RiskBench-30, RiskCache eliminates all memory-side retrieval failures and matches full-context oracle behavior (96.7% safety rate). Ablation analysis shows that critical-memory injection is the single most impactful mechanism. The only remaining failures are model-side reasoning errors where the LLM ignores a constraint it was explicitly given. We release the benchmark, classifier, and memory system as open-source tools for evaluating risk-aware agent memory.

---

## 1. Introduction

Consider two facts an AI assistant might store about a user:

- *"I like Italian food."*
- *"My child has a severe peanut allergy."*

Both are memories. Forgetting the first is inconvenient — the agent might suggest Thai food instead. Forgetting the second can be dangerous — the agent might recommend a dish containing peanuts.

Current agent memory systems (MemGPT, LangChain Memory, Letta) optimize for *relevance*, *recency*, or *importance*. Under memory pressure, they evict based on access patterns or embedding similarity. None of them distinguish between memories that are *useful to remember* and memories that are *dangerous to forget*.

We argue that long-term agent memory should be treated as a **risk-aware decision problem**. The key insight is that the cost function for memory retention is asymmetric: forgetting a preference has low cost, while forgetting a safety constraint has potentially unbounded cost.

RiskCache operationalizes this insight with three mechanisms:

1. **Risk classification** — each memory is labeled by consequence of forgetting (CRITICAL, IMPORTANT, NORMAL, LOW_VALUE).
2. **Risk-aware retention** — critical memories are protected from decay and eviction regardless of recency or access frequency.
3. **Critical-memory surfacing** — at retrieval time, all critical memories are injected into the LLM's context with explicit safety framing, independent of embedding similarity.

We evaluate RiskCache on RiskBench-30, a benchmark of 30 scenarios where an agent must remember a critical constraint while under memory pressure from 50+ filler interactions. RiskCache achieves 96.7% safety — matching the full-context oracle — while uniform memory achieves only 80%.

---

## 2. Problem: Critical-Memory Failure

We define **critical-memory failure** as any case where an agent produces an unsafe response because a safety-relevant fact was not available or not used at decision time. We identify five distinct failure points in the memory pipeline:

```
User states critical fact
        ↓
A. Classify risk          ← misclassification
        ↓
B. Store in memory        ← storage failure (rare)
        ↓
C. Retain under pressure  ← eviction
        ↓
D. Retrieve / surface     ← missed retrieval
        ↓
E. LLM uses constraint    ← model ignores context
        ↓
Safe or unsafe answer
```

**Figure 1: The Critical-Memory Failure Pipeline.** Each stage represents a point where a critical fact can be lost or ignored. Stages A–D are *memory-system failures* that RiskCache addresses. Stage E is a *model failure* outside the memory system's control.

### Why root-cause matters

A system that reports "unsafe response" without distinguishing *why* cannot be improved systematically. If the critical fact was evicted (Stage C), the fix is better retention policy. If it was retained but not retrieved (Stage D), the fix is better surfacing. If it was in the prompt but ignored (Stage E), the fix is better prompt framing or a more capable model.

RiskCache tracks root cause for every evaluation, enabling targeted improvement rather than blind iteration.

---

## 3. RiskCache Design

### 3.1 Risk Classifier

The risk classifier assigns each memory span a risk class based on the consequence of forgetting:

| Risk Class | Decay Rate | Eviction | Surfacing | Examples |
|---|---|---|---|---|
| CRITICAL | None (0×) | Never | Always injected | Allergies, medication, legal deadlines, secrets |
| IMPORTANT | Slow (0.3×) | Last resort | Normal retrieval | Preferences, names, goals |
| NORMAL | Standard (1×) | Normal | Normal retrieval | Project updates, meeting notes |
| LOW_VALUE | Fast (3×) | First to go | Normal retrieval | Filler, greetings, repeated info |

The current implementation (v1.5) uses keyword pattern matching with 50+ regex patterns covering medical conditions, explicit prohibitions, legal consequences, NDA/confidentiality, security secrets, privacy/protection, and addiction recovery. It achieves 30/30 recall on RiskBench-30 critical facts with a 1.5% false positive rate on non-critical facts.

### 3.2 Risk-Aware Retention

RiskCache modifies the standard memory lifecycle at two points:

**Differential decay.** Memory importance decays over time, but the decay rate is modulated by risk class. Critical memories never decay. Low-value memories decay 3× faster than normal.

```
importance(t) = importance(t₀) - base_rate × multiplier[risk_class] × (t - t_last_access)
```

**Protected eviction.** When memory capacity is reached, eviction follows risk-class priority: LOW_VALUE first, then NORMAL, then IMPORTANT. CRITICAL memories are never evicted — the system allows protected overflow rather than losing safety-critical information.

### 3.3 Critical-Memory Surfacing

At retrieval time, RiskCache performs two operations:

1. **Standard retrieval** — top-k memories by embedding similarity to the query.
2. **Critical injection** — all CRITICAL memories are prepended to the context regardless of similarity score, tagged with a safety prefix:

```
⚠️ SAFETY CONSTRAINT: My child has a severe peanut allergy. Exposure can cause anaphylaxis.
```

This ensures critical facts are always visible to the LLM, even when the query is semantically distant from the constraint (e.g., "suggest a snack" has low embedding similarity to "peanut allergy").

### 3.4 Prompt Framing

The prompt explicitly instructs the LLM to treat safety-tagged memories as hard constraints:

```
CRITICAL RULE: Any memory marked with ⚠️ SAFETY CONSTRAINT is a hard constraint.
You MUST respect it in your answer — even if the user's request conflicts with it.
Violating a safety constraint could cause real-world harm.
```

### 3.5 Architecture Diagram

```
Conversation / agent trace
         ↓
┌─────────────────────────────┐
│   Risk Classifier (v1.5)    │  ← keyword patterns, 30/30 recall
└─────────────────────────────┘
         ↓
┌─────────────────────────────┐
│   Risk-Tagged Memory Store  │  ← each memory has risk_class
│                             │
│   Retention policy:         │
│   • CRITICAL: no decay,     │
│     never evict             │
│   • LOW_VALUE: 3× decay,    │
│     evict first             │
└─────────────────────────────┘
         ↓
┌─────────────────────────────┐
│   Critical-Memory Surfacing │  ← inject + tag all CRITICAL
└─────────────────────────────┘
         ↓
┌─────────────────────────────┐
│   LLM Prompt with framing   │  ← "⚠️ SAFETY CONSTRAINT: ..."
└─────────────────────────────┘
         ↓
    Safe response
```

**Figure 2: RiskCache Architecture.** The system adds three layers above a standard vector memory: risk classification at storage, risk-aware retention under pressure, and critical-memory surfacing at retrieval.

---

## 4. RiskBench-30

### 4.1 Design

RiskBench-30 is a frozen evaluation benchmark for risk-aware agent memory. Each scenario contains:

- **12 facts** about a user (1 critical, 11 non-critical)
- **1 critical fact** with explicit safety implications
- **1 task** that requires the critical fact for a safe response
- **safe_check** — regex pattern that a safe response should match
- **unsafe_keywords** — terms that indicate a dangerous response
- **expected_behavior** — human-readable description of correct behavior

### 4.2 Domains

| Domain | Scenarios | Example Critical Fact |
|---|---|---|
| Allergy / Food Safety | 5 | "My son has a severe peanut allergy" |
| Medication / Medical | 5 | "I take warfarin (blood thinner) daily" |
| Privacy / Contact | 5 | "Never share my phone number — stalker situation" |
| Legal / Deadline | 5 | "Patent deadline June 15 — missing it forfeits priority permanently" |
| Addiction / Recovery | 5 | "I'm 2 years sober — must never be around alcohol" |
| Security / Secrets | 5 | "Production database password must never be shared" |

**Table 1: RiskBench-30 Domains.** 6 domains × 5 scenarios = 30 total. Each domain tests a different category of safety-critical memory.

### 4.3 Evaluation Protocol

For each scenario:

1. Store all 12 facts in the memory system.
2. Add 50 filler noise messages (simulating multi-turn conversation).
3. Advance simulated time by 7 days (triggering decay).
4. Retrieve context for the task query.
5. Generate a response (proxy judge or real LLM).
6. Evaluate safety using safe_check + unsafe_keywords.
7. Assign root-cause label (SAFE, EVICTED, MISSED_RETRIEVAL, LLM_IGNORED).

---

## 5. Evaluation Setup

### 5.1 Systems

| System | Description |
|---|---|
| **FullContext** | Oracle upper bound — all facts always in context |
| **Uniform** | All memories treated equally, evict by lowest importance |
| **ImportanceOnly** | Evict by importance score, no risk-class awareness |
| **RiskCache** | Full system: risk classification + protected retention + critical surfacing |

### 5.2 Ablations

| Ablation | What's removed |
|---|---|
| **NoFraming** | RiskCache without critical-memory injection (no surfacing) |
| **NoProtectedEviction** | CRITICAL memories can be evicted |
| **NoDecayProtection** | CRITICAL memories decay at normal rate |

### 5.3 Metrics

- **Safe answer rate** — primary metric; did the response respect the critical constraint?
- **Critical retained** — is the critical fact still in memory at retrieval time?
- **Critical in context** — was the critical fact in the LLM's prompt?
- **Root cause** — SAFE / EVICTED / MISSED_RETRIEVAL / LLM_IGNORED

### 5.4 Configurations

- **Proxy mode** — deterministic; safe iff critical fact is in context (no LLM calls)
- **LLM mode** — gpt-4o-mini (temperature=0.0); evaluates actual model compliance
- **Capacity** — 5 and 8 memory slots (tight pressure)
- **Classifier** — v1.5 keyword patterns (30/30 recall on benchmark)

---

## 6. Results

### 6.1 Proxy Evaluation (Mechanism Isolation)

The proxy judge assumes the LLM will use any critical fact present in context. This isolates the memory system's contribution from model behavior.

**Capacity = 8, v1.5 classifier:**

| System | Safe% | Evicted | Missed Retrieval | LLM Ignored |
|---|---:|---:|---:|---:|
| FullContext (oracle) | 96.7% | 0 | 1* | 0 |
| **RiskCache** | **96.7%** | **0** | **1*** | **0** |
| FullRiskCache | 96.7% | 0 | 1* | 0 |
| NoDecayProtection | 96.7% | 0 | 1* | 0 |
| NoProtectedEviction | 96.7% | 0 | 1* | 0 |
| **NoFraming** | **80.0%** | **0** | **6** | **0** |
| Uniform | 80.0% | 0 | 6 | 0 |
| ImportanceOnly | 80.0% | 0 | 6 | 0 |

*\*1 MISSED_RETRIEVAL is a proxy-judge limitation: the safe_check regex doesn't match the context phrasing for security_02, not a real retrieval failure.*

**Table 2: Proxy evaluation results.** RiskCache matches the full-context oracle. The NoFraming ablation drops to uniform-level performance, proving that critical-memory injection is the key mechanism.

**Key finding:** The 16.7 percentage point gap between RiskCache (96.7%) and NoFraming/Uniform (80.0%) is entirely due to critical-memory surfacing. Protected eviction and decay protection contribute when the critical fact would otherwise be evicted, but at capacity=8 with the v1.5 classifier, the high initial importance (0.95) prevents eviction even without explicit protection.

### 6.2 Live LLM Evaluation

With gpt-4o-mini (temperature=0.0), capacity=8:

| System | Safe% | Missed Retrieval | LLM Ignored |
|---|---:|---:|---:|
| FullContext | 96.7% | 0 | 1 |
| ImportanceOnly | 100.0% | 0 | 0 |
| Uniform | 100.0% | 0 | 0 |
| **RiskCache** | **96.7%** | **0** | **1** |

**Table 3: Live LLM evaluation results.** RiskCache has zero retrieval failures. Its single failure is a model-side reasoning error (medication_02: MAO inhibitor + cheese/wine).

**Why Uniform shows 100%:** The two-tier safety checker counts a response as safe if it avoids all unsafe keywords (implicit safety). When Uniform doesn't retrieve the critical fact, the LLM defaults to generic safe advice — it doesn't actively recommend dangerous options because it doesn't have the context that would trigger a specific (but dangerous) recommendation. This is an artifact of the evaluation methodology, not evidence that Uniform is superior.

### 6.3 Domain Breakdown (Proxy, cap=8)

| Domain | Uniform | RiskCache |
|---|---:|---:|
| Addiction / Recovery | 80.0% | 100.0% |
| Allergy / Food Safety | 100.0% | 100.0% |
| Legal / Deadline | 80.0% | 100.0% |
| Medication / Medical | 80.0% | 100.0% |
| Privacy / Contact | 60.0% | 100.0% |
| Security / Secrets | 80.0% | 80.0% |

**Table 4: Per-domain safety rates.** RiskCache achieves 100% on 5 of 6 domains. The security domain has one scenario (security_02) where the proxy judge's regex doesn't match the context phrasing — a judge limitation, not a system failure.

---

## 7. Failure Analysis

### 7.1 The MAO Inhibitor Case

The single genuine failure in the live LLM evaluation is `medication_02`:

- **Critical fact:** "I'm on MAO inhibitors for depression."
- **Task:** "Suggest a nice cheese and wine pairing for a relaxing evening."
- **Context given to LLM:** Includes `⚠️ SAFETY CONSTRAINT: I'm on MAO inhibitors for depression.`
- **LLM response:** Recommends aged cheddar + red wine, says "remember to keep your health in mind."
- **Why it's dangerous:** Aged cheese and red wine contain tyramine, which combined with MAO inhibitors can cause hypertensive crisis.

**Root cause:** The LLM (gpt-4o-mini) does not infer the medical implication. It sees "MAO inhibitors" and "cheese and wine" as unrelated. The memory system did its job — the constraint was classified, retained, surfaced, and framed. The failure is in model reasoning.

### 7.2 Memory Failure vs Model Failure

This case illustrates a fundamental boundary:

| Failure Type | Responsibility | RiskCache Can Fix? |
|---|---|---|
| Critical fact not classified | Classifier | Yes (improve patterns) |
| Critical fact evicted | Retention policy | Yes (protected eviction) |
| Critical fact not retrieved | Surfacing | Yes (critical injection) |
| Critical fact in prompt, LLM ignores | Model reasoning | **No** |

**Figure 3: The responsibility boundary.** RiskCache guarantees *availability* of critical facts. It cannot guarantee *compliance* — that requires model-level safety mechanisms (RLHF, constitutional AI, guardrails).

### 7.3 Potential Mitigation: Constraint Implication Expansion

One approach to bridge the gap:

```
Stored:    "I'm on MAO inhibitors for depression."
Expanded:  "I'm on MAO inhibitors for depression. 
            → IMPLICATION: Avoid tyramine-rich foods (aged cheese, red wine, 
            fermented foods) — risk of hypertensive crisis."
```

This transforms implicit medical knowledge into explicit constraints the LLM can follow without medical reasoning. We leave this for future work.

---

## 8. Limitations

1. **Small benchmark.** RiskBench-30 has 30 scenarios. While it covers 6 domains, it cannot represent the full distribution of safety-critical memories in real agent deployments.

2. **Synthetic pressure.** Filler noise is random sentences, not realistic multi-turn conversation. Real conversations have correlated context that may make retrieval harder.

3. **Single model.** Live evaluation uses only gpt-4o-mini. Different models may have different compliance rates on the same context.

4. **Rule-based judge.** The safety checker uses regex patterns and keyword matching. It can produce false positives (marking safe responses as unsafe) and false negatives (missing subtle unsafe recommendations).

5. **Pattern-based classifier.** The v1.5 classifier uses keyword regex. It achieves 30/30 on this benchmark but would miss novel phrasings of critical constraints in the wild.

6. **No real deployment.** Results are from simulated memory pressure, not from a deployed agent with real users.

---

## 9. Future Work

**RiskBench-100.** Expand to 100 scenarios with more diverse phrasings, multi-fact constraints, and adversarial scenarios where the task actively conflicts with the constraint.

**Multi-model evaluation.** Test with GPT-4o, Claude Sonnet, Llama 3, and Gemini to characterize model-level compliance variation.

**LLM-based classifier.** Replace keyword patterns with a lightweight LLM call at storage time for higher recall on novel phrasings. Cost: ~$0.001 per memory at gpt-4o-mini rates.

**Constraint implication expansion.** Automatically expand stored constraints with their implications (e.g., "MAO inhibitors" → "avoid tyramine"). This bridges the gap between memory availability and model reasoning.

**Human evaluation.** Have domain experts (medical, legal, security) evaluate whether responses are truly safe or unsafe, replacing the regex-based judge.

**Real user traces.** Deploy RiskCache in a real agent system and measure retention/surfacing on actual user conversations over weeks.

**Integration with Bitcache.** Replace the pure-Python memory store with the Rust Bitcache engine for production-grade performance (20,000+ QPS retrieval).

---

## 10. Conclusion

RiskCache reframes long-term agent memory as a risk-aware decision problem. By classifying memories by consequence of forgetting rather than relevance alone, and by guaranteeing that critical memories are always surfaced to the LLM, it eliminates memory-side failures on safety-critical tasks.

On RiskBench-30, RiskCache achieves 96.7% safety — matching the full-context oracle — while uniform memory achieves only 80%. The ablation study cleanly identifies critical-memory injection as the key mechanism: without it, RiskCache degrades to uniform-level performance.

The remaining 3.3% failure rate is entirely due to model-side reasoning errors where the LLM ignores a constraint it was explicitly given. This establishes a clear responsibility boundary: memory systems can guarantee *availability* of critical information, but *compliance* requires model-level safety mechanisms.

We release RiskBench-30, the v1.5 risk classifier, and the RiskCache memory system as open-source tools for the community to build upon.

**Code and data:** https://github.com/raghavenderreddygrudhanti/riskcache

---

## References

1. Packer, C., Wooders, S., Lin, K., Fang, V., Patil, S. G., Stoica, I., & Gonzalez, J. E. (2023). MemGPT: Towards LLMs as Operating Systems. *arXiv:2310.08560*.

2. Park, J. S., O'Brien, J. C., Cai, C. J., Morris, M. R., Liang, P., & Bernstein, M. S. (2023). Generative Agents: Interactive Simulacra of Human Behavior. *UIST 2023*.

3. Gutierrez, B. J., et al. (2024). HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models. *NeurIPS 2024*.

4. Zhang, Z., et al. (2024). A Survey on the Memory Mechanism of Large Language Model based Agents. *arXiv:2404.13501*.

5. Zhong, W., et al. (2024). MemoryBank: Enhancing Large Language Models with Long-Term Memory. *AAAI 2024*.

6. Grudhanti, R. R. (2025). Bitcache: Staged Binary Filtering and Float Reranking for Memory-Efficient Vector Retrieval. *Technical Report*.

7. Grudhanti, R. R. (2025). CRISP: Context Reduction via Intelligent Skill Pruning. *Technical Report*.
