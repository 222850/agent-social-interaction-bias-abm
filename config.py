"""
Configuration for Cross-Lingual LLM Bias Evaluation Framework.

Server: RTX 4060 Ti (16GB VRAM), 61GB RAM, 16 CPU cores, ~380GB disk.
All models served via Ollama in 4-bit quantization.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / "cache"
RESULTS_DIR = PROJECT_ROOT / "results"
for d in [DATA_DIR, CACHE_DIR, RESULTS_DIR]:
    d.mkdir(exist_ok=True)

# ── Ollama endpoint ────────────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"

# ── Models ─────────────────────────────────────────────────────────────
# Chosen for: recency, multilingual support, fit on 16GB VRAM (4-bit).
# Each model tag below corresponds to an Ollama-pulled model.

@dataclass
class ModelSpec:
    ollama_tag: str           # e.g. "qwen2.5:7b-instruct"
    display_name: str
    languages: list[str]
    vram_4bit_gb: float       # approximate 4-bit footprint
    hf_repo: Optional[str] = None  # for embedding extraction via HF

MODELS = {
    "qwen2.5-7b-instruct": ModelSpec(
        ollama_tag="qwen2.5:7b-instruct",
        display_name="Qwen-2.5-7B-Instruct",
        languages=["en", "ru", "zh"],
        vram_4bit_gb=4.5,
        hf_repo="Qwen/Qwen2.5-7B-Instruct",
    ),
    "llama3.1-8b": ModelSpec(
        ollama_tag="llama3.1:8b-instruct",
        display_name="Llama-3.1-8B-Instruct",
        languages=["en"],
        vram_4bit_gb=5.0,
        hf_repo="meta-llama/Llama-3.1-8B-Instruct",
    ),
    "mistral-7b": ModelSpec(
        ollama_tag="mistral:7b-instruct-v0.3",
        display_name="Mistral-7B-Instruct-v0.3",
        languages=["en", "fr", "de"],
        vram_4bit_gb=4.5,
        hf_repo="mistralai/Mistral-7B-Instruct-v0.3",
    ),
    "vikhr-nemo-12b": ModelSpec(
        ollama_tag="vikhr-nemo:12b-instruct",
        display_name="Vikhr-Nemo-12B-Instruct",
        languages=["ru", "en"],
        vram_4bit_gb=7.0,
        hf_repo="Vikhrmodels/Vikhr-Nemo-12B-Instruct-R-21-09-24",
    ),
}

# Models to run by default (adjust if VRAM tight)
DEFAULT_MODELS = ["qwen2.5-7b-instruct", "llama3.1-8b", "mistral-7b"]
RUSSIAN_MODELS = ["qwen2.5-7b-instruct", "vikhr-nemo-12b"]

# ── BBQ sensitive attributes (for spillover analysis) ──────────────────
BBQ_ATTRIBUTES = [
    "gender", "race_ethnicity", "age", "disability",
    "physical_appearance", "sexual_orientation",
    "religion", "nationality", "socioeconomic_status",
]

# ── Hiring simulation parameters ───────────────────────────────────────
HIRING_INDUSTRIES = ["healthcare", "finance", "technology"]
CANDIDATE_GENDERS = ["male", "female", "neutral"]

# ── Statistical thresholds ─────────────────────────────────────────────
ALPHA = 0.05
N_PERMUTATIONS = 10_000
BAYES_FACTOR_THRESHOLD = 3.0  # evidence threshold for pingouin

# ── Reproducibility ────────────────────────────────────────────────────
RANDOM_SEED = 42
LLM_TEMPERATURE = 0.0  # deterministic for reproducibility

# Dedicated embedding model for WEAT/SEAT and probing
EMBEDDING_MODEL = "embeddinggemma"
