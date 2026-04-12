# Bias Framework: LLM Bias Evaluation and Agent-Based Simulation

## Project overview

This repository contains an MVP prototype for evaluating social bias in large language models and studying how such bias can propagate through multi-agent decision systems.

The project combines benchmark-based bias evaluation with agent-based modeling and is connected to the broader thesis topic:

**“Моделирование процессов социального взаимодействия в агентных системах”**
(**Modeling Social Interaction Processes in Agent Systems**)

The core idea is that bias in LLM outputs should be studied not only as an isolated property of text generation, but also as part of a larger socio-technical system where model outputs influence institutional decisions, candidate trajectories, and aggregate group outcomes.

## Research motivation

Large language models may exhibit stereotypical or asymmetric behavior even when direct bias is not obvious in single prompts. This creates two related research questions:

1. How can we measure bias at the level of model outputs and representations?
2. How do such biases behave when the model is embedded into a larger decision-making process?

To address this, the project includes:

* benchmark-style evaluation for English and Russian
* encoded vs expressed bias analysis
* hiring-oriented counterfactual evaluation
* an agent-based hiring simulation with multiple fairness scenarios

## Project components

### 1. Benchmark-based evaluation

The repository includes several evaluation modules:

* **JobFair-style evaluation**
  Counterfactual resumes are scored under different gender conditions to check whether otherwise similar candidates receive different assessments.

* **WEAT / SEAT-style embedding analysis**
  Measures whether semantic associations in embedding space reflect stereotypical patterns.

* **Encoded vs Expressed bias**
  Distinguishes between:

  * bias directly visible in generated text
  * bias encoded in internal or embedding-level representations

* **RuBia / Russian evaluation support**
  Extends the evaluation pipeline to Russian-language prompts and materials.

### 2. Agent-based simulation

The simulation module models a hiring process where:

* candidates have attributes such as skill, experience, education, and salary expectations
* an LLM-based institution evaluates and ranks candidates
* fairness interventions can be applied at different stages

This allows us to move from “does the model say biased things?” to “what happens if the model becomes part of an institutional process?”

## Repository structure

```text
analysis/       Encoded vs expressed bias analysis
benchmarks/     Benchmark-style evaluation modules
simulation/     Agent-based hiring simulation
data/           Example inputs and placeholder datasets
models/         Notes on external models used in the project
docs/           Additional project and practice notes
```

Core files:

* `config.py` — model and path configuration
* `llm_client.py` — client for Ollama-based inference and embeddings
* `stats_utils.py` — statistical helper functions
* `run_evaluation.py` — main runner for evaluation stages
* `setup_data.sh` — benchmark data setup script

## Models used

Main generation model in MVP:

* `qwen2.5:7b-instruct`

Embedding model:

* `embeddinggemma`

The repository does not contain model weights. Models are expected to be available locally through Ollama.

## Installation

### 1. Create and activate virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start Ollama

```bash
nohup ollama serve > ollama.log 2>&1 &
```

### 4. Pull required models

```bash
ollama pull qwen2.5:7b-instruct
ollama pull embeddinggemma
```

### 5. Download / prepare benchmark data

```bash
bash setup_data.sh
```

## Example runs

### Encoded vs expressed bias

```bash
python run_evaluation.py --models qwen2.5-7b-instruct --stages encoded
```

### WEAT

```bash
python run_evaluation.py --models qwen2.5-7b-instruct --stages weat
```

### JobFair

```bash
python run_evaluation.py --models qwen2.5-7b-instruct --stages jobfair
```

### ABM simulation

```bash
python run_evaluation.py --models qwen2.5-7b-instruct --stages abm
```

## Main outputs

The project saves outputs into the `results/` directory, for example:

* `full_evaluation.json`
* `jobfair_en_qwen2.5-7b-instruct.csv`
* `encoded_expressed_qwen2.5-7b-instruct.csv`
* `abm_qwen2.5-7b-instruct.csv`

## What the metrics mean

### JobFair

* **mean_score_male / mean_score_female** — average candidate scores by gender
* **impact_ratio** — ratio of smaller to larger hire rate; values below 0.8 may indicate adverse impact
* **level_bias** — difference in average evaluation level
* **spread_bias** — difference in score variance or dispersion

### Encoded vs expressed bias

* **expressed_bias** — bias visible in generated text
* **probe_accuracy** — how well gender information can be recovered from embeddings
* **jailbreak_reactivation** — whether bias becomes stronger under adversarial prompting

### WEAT

* **effect size d** — strength of association between target and attribute groups
* **p-value** — statistical support for the association

### ABM

* **male_hire_rate / female_hire_rate** — hiring rates by gender
* **demographic_parity_diff** — absolute difference in hiring rates
* **impact_ratio** — fairness ratio across groups
* **score_gap** — difference in final scores
* **salary_gap** — difference in assigned salaries
* **amplification_factor** — whether a later stage makes disparities larger than earlier screening
* **gini_scores** — inequality in score distribution

## MVP status

This repository is currently an MVP:

* the main evaluation pipeline runs
* multiple bias-sensitive stages are implemented
* the ABM simulation produces interpretable outputs
* the project is structured for further extension

## Current limitations

* results depend on local Ollama setup
* some benchmark coverage is still limited compared to large-scale research pipelines
* probing uses embeddings rather than full hidden-state access
* ABM remains a stylized simulation rather than a full labor-market model
* Russian support is partial and mixed between translated prompts and dedicated resources

## Planned improvements

* improve reproducibility and dataset packaging
* expand Russian benchmark coverage
* add richer multi-round ABM dynamics
* separate screening, shortlisting, and final hiring analysis more clearly
* add visualization notebooks or dashboards

## Practice / development workflow

During practice, the project was improved through:

* opening issues
* fixing problems through pull requests
* bringing the repository to MVP state

Typical categories of fixes:

1. model configuration normalization
2. robust statistical utilities
3. stable ABM execution and non-empty scenario outputs

## Author

Project prepared as part of research for my thesis and production practice work in Financial University.