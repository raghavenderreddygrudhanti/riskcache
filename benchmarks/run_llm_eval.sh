#!/bin/bash
# RiskBench-30 LLM evaluation script.
# Run this once with a valid OPENAI_API_KEY to produce keyword_classifier_llm.json
#
# Prerequisites:
#   export OPENAI_API_KEY="sk-..."
#   pip install openai  (if not already installed)
#
# This runs baselines only (no ablations) to keep LLM costs reasonable:
#   4 systems × 30 scenarios = 120 LLM calls ≈ $0.50 with gpt-4o-mini
#
# Expected runtime: ~3-5 minutes depending on API latency.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set"
    echo "  export OPENAI_API_KEY='sk-...'"
    exit 1
fi

echo "=== RiskBench-30 LLM Evaluation ==="
echo ""
echo "Running with keyword classifier + gpt-4o-mini (baselines only)..."
echo "This will make ~120 API calls."
echo ""

python benchmarks/riskbench_30.py \
    --provider openai \
    --model gpt-4o-mini \
    --capacity 5 \
    --no-ablations \
    --output keyword_classifier_llm

echo ""
echo "Done! Results saved to results/keyword_classifier_llm.json"
echo ""
echo "To also run with oracle classification + LLM:"
echo "  python benchmarks/riskbench_30.py --provider openai --model gpt-4o-mini --capacity 5 --oracle-classify --no-ablations --output oracle_classifier_llm"
