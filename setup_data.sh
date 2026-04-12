#!/usr/bin/env bash
# Download benchmark datasets for the evaluation framework.
# Run from the bias_framework/ directory.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/data"
mkdir -p "$DATA_DIR"

echo "═══════════════════════════════════════════════════"
echo "  Downloading benchmark datasets"
echo "═══════════════════════════════════════════════════"

# ── BBQ ────────────────────────────────────────────────
echo ""
echo "▶ BBQ (Parrish et al. 2022)"
mkdir -p "$DATA_DIR/bbq"
if [ ! -f "$DATA_DIR/bbq/Gender_identity.jsonl" ]; then
    echo "  Cloning BBQ repository..."
    git clone --depth 1 https://github.com/nyu-mll/BBQ.git /tmp/bbq_repo 2>/dev/null || true
    cp /tmp/bbq_repo/data/*.jsonl "$DATA_DIR/bbq/" 2>/dev/null || echo "  ⚠ Manual download needed: https://github.com/nyu-mll/BBQ"
    rm -rf /tmp/bbq_repo
    echo "  ✓ BBQ downloaded"
else
    echo "  ✓ BBQ already exists"
fi

# ── StereoSet ──────────────────────────────────────────
echo ""
echo "▶ StereoSet (Nadeem et al. 2021)"
mkdir -p "$DATA_DIR/stereoset"
if [ ! -f "$DATA_DIR/stereoset/dev.json" ]; then
    echo "  Downloading StereoSet..."
    curl -sL "https://raw.githubusercontent.com/moinnadeem/StereoSet/master/data/dev.json" \
        -o "$DATA_DIR/stereoset/dev.json" 2>/dev/null || echo "  ⚠ Manual download needed"
    echo "  ✓ StereoSet downloaded"
else
    echo "  ✓ StereoSet already exists"
fi

# ── RuBia ──────────────────────────────────────────────
echo ""
echo "▶ RuBia (Grigoreva et al. 2024)"
mkdir -p "$DATA_DIR/rubia"
if [ ! -f "$DATA_DIR/rubia/rubia_dataset.json" ]; then
    echo "  Cloning RuBia repository..."
    git clone --depth 1 https://github.com/vergrig/RuBia-Dataset.git /tmp/rubia_repo 2>/dev/null || true
    cp /tmp/rubia_repo/data/* "$DATA_DIR/rubia/" 2>/dev/null || \
    cp /tmp/rubia_repo/*.json "$DATA_DIR/rubia/" 2>/dev/null || \
    cp /tmp/rubia_repo/*.csv "$DATA_DIR/rubia/" 2>/dev/null || \
        echo "  ⚠ Manual download needed: https://github.com/vergrig/RuBia-Dataset"
    rm -rf /tmp/rubia_repo
    echo "  ✓ RuBia downloaded"
else
    echo "  ✓ RuBia already exists"
fi

# ── Ollama models ──────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════"
echo "  Ollama model checklist (pull manually):"
echo "═══════════════════════════════════════════════════"
echo "  ollama pull qwen2.5:7b-instruct"
echo "  ollama pull embeddinggemma"
echo "  ollama pull llama3.1:8b-instruct"
echo "  ollama pull mistral:7b-instruct-v0.3"
echo ""
echo "  For Russian (Vikhr requires manual GGUF import):"
echo "  1. Download from HuggingFace: Vikhrmodels/Vikhr-Nemo-12B-Instruct-R-21-09-24"
echo "  2. Convert to GGUF or find a pre-quantized version"
echo "  3. ollama create vikhr-nemo:12b-instruct -f Modelfile"
echo ""
echo "  Alternative for Russian without Vikhr:"
echo "  Qwen-2.5-7B-Instruct supports Russian natively."
echo ""
echo "═══════════════════════════════════════════════════"
echo "  Setup complete. Run: python run_evaluation.py"
echo "═══════════════════════════════════════════════════"
