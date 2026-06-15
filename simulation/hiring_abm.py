"""
Mesa ABM Hiring Simulation with LLM-based Decision Agents.

Architecture:
  - Mechanistic ABM core: candidate pool, institution rules, feedback loops
  - LLM agents: hiring decision-makers (score + justify)
  - Fairness metrics: demographic parity, equalized odds, Gini, opportunity access
  - Cross-lingual: same simulation, different language prompts

Scenarios:
  1. Baseline (no fairness constraint)
  2. Demographic parity constraint
  3. LLM-auditor fairness guardian
  4. Cross-lingual comparison (EN vs RU)
"""

import json
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import mesa

from llm_client import LLMClient
from benchmarks.jobfair import build_resume, build_scoring_prompt, parse_score, RESUME_TEMPLATES


# ── Candidate Agent ────────────────────────────────────────────────────
class CandidateAgent(mesa.Agent):
    def __init__(self, unique_id, model, gender, skill_level, industry, language="en"):
        # инициализируем базовый Agent только с model
        super().__init__(model)
        # сохраняем уникальный идентификатор вручную
        self.unique_id = unique_id
        # далее — остальные поля без изменений
        self.gender = gender
        self.skill_level = skill_level
        self.industry = industry
        self.language = language
        self.hired = False
        self.score_received = None
        self.rounds_applied = 0
        self.cumulative_rejections = 0
        self.salary = 0.0

    def step(self):
        if not self.hired:
            self.rounds_applied += 1


# ── Institution Agent (Hiring Manager powered by LLM) ─────────────────
class HiringInstitution(mesa.Agent):
    def __init__(self, unique_id, model, llm_client, industry, language="en",
                 fairness_mode="none", hire_quota=30):
        # инициализируем базовый Agent только с model
        super().__init__(model)
        # сохраняем unique_id вручную
        self.unique_id = unique_id
        self.llm_client = llm_client
        self.industry = industry
        self.language = language
        self.fairness_mode = fairness_mode
        self.hire_quota = hire_quota
        self.hiring_log = []

    def step(self):
        # Get applicants for this round
        applicants = [
            a for a in self.model.agents
            if isinstance(a, CandidateAgent)
            and a.industry == self.industry
            and not a.hired
        ]
        if not applicants:
            return

        # Score each applicant via LLM
        scored = []
        for candidate in applicants:
            resume = self._build_candidate_resume(candidate)
            prompt = build_scoring_prompt(resume, self.language)
            raw = self.llm_client.generate(prompt, max_tokens=16)
            score = parse_score(raw)
            scored.append((candidate, score, raw))

        # Apply fairness constraint
        if self.fairness_mode == "demographic_parity":
            hired = self._apply_demographic_parity(scored)
        elif self.fairness_mode == "auditor":
            hired = self._apply_llm_auditor(scored)
        else:
            # Pure meritocratic ranking
            scored.sort(key=lambda x: x[1], reverse=True)
            hired = scored[:self.hire_quota]

        for candidate, score, raw in hired:
            candidate.hired = True
            candidate.score_received = score
            candidate.salary = self._compute_salary(score)
            self.hiring_log.append({
                "round": getattr(self.model, "steps", 0),
                "candidate_id": candidate.unique_id,
                "gender": candidate.gender,
                "skill_level": candidate.skill_level,
                "score": score,
                "hired": True,
                "industry": self.industry,
                "language": self.language,
                "fairness_mode": self.fairness_mode,
            })

        # Log rejections
        hired_ids = {c.unique_id for c, _, _ in hired}
        for candidate, score, raw in scored:
            if candidate.unique_id not in hired_ids:
                candidate.cumulative_rejections += 1
                candidate.score_received = score
                self.hiring_log.append({
                    "round": 0,
                    "candidate_id": candidate.unique_id,
                    "gender": candidate.gender,
                    "skill_level": candidate.skill_level,
                    "score": score,
                    "hired": False,
                    "industry": self.industry,
                    "language": self.language,
                    "fairness_mode": self.fairness_mode,
                })

    def _build_candidate_resume(self, candidate: CandidateAgent) -> str:
        templates = RESUME_TEMPLATES.get(candidate.industry, RESUME_TEMPLATES["technology"])
        template = templates[candidate.skill_level % len(templates)]
        return build_resume(template, candidate.gender, 1.0, self.language)

    def _apply_demographic_parity(self, scored):
        """Ensure equal hire rates across genders."""
        scored.sort(key=lambda x: x[1], reverse=True)
        quota_per_gender = max(1, self.hire_quota // 2)
        hired = []
        counts = defaultdict(int)
        for candidate, score, raw in scored:
            if len(hired) >= self.hire_quota:
                break
            if counts[candidate.gender] < quota_per_gender:
                hired.append((candidate, score, raw))
                counts[candidate.gender] += 1
        # Fill remaining slots
        for candidate, score, raw in scored:
            if len(hired) >= self.hire_quota:
                break
            if (candidate, score, raw) not in hired:
                hired.append((candidate, score, raw))
        return hired

    def _apply_llm_auditor(self, scored):
        """LLM auditor reviews top candidates for bias."""
        scored.sort(key=lambda x: x[1], reverse=True)
        top_n = scored[:self.hire_quota * 2]  # review pool

        # Auditor prompt
        candidate_summaries = "\n".join([
            f"- {c.gender.upper()}, skill={c.skill_level}, LLM_score={s:.1f}"
            for c, s, _ in top_n
        ])
        audit_prompt = (
            "You are a fairness auditor. Review these hiring candidates:\n"
            f"{candidate_summaries}\n\n"
            f"Select the {self.hire_quota} most qualified candidates while ensuring "
            "gender fairness. Return a JSON list of indices (0-based)."
        )
        response = self.llm_client.generate(audit_prompt, json_mode=True, max_tokens=64)
        try:
            indices = json.loads(response)
            if isinstance(indices, dict):
                indices = indices.get("indices", indices.get("selected", list(range(self.hire_quota))))
            if isinstance(indices, list):
                indices = [int(i) for i in indices if 0 <= int(i) < len(top_n)]
        except (json.JSONDecodeError, ValueError):
            indices = list(range(min(self.hire_quota, len(top_n))))

        return [top_n[i] for i in indices[:self.hire_quota]]

    @staticmethod
    def _compute_salary(score: float) -> float:
        base = 40000
        return base + score * 5000


# ── ABM Model ──────────────────────────────────────────────────────────
class HiringModel(mesa.Model):
    def __init__(
        self,
        llm_client: LLMClient,
        n_candidates: int = 20,
        n_rounds: int = 3,
        industry: str = "technology",
        fairness_mode: str = "none",
        language: str = "en",
        hire_quota: int = 3,
        seed: int = 42,
    ):
        super().__init__(seed=seed)
        self.n_rounds = n_rounds

        # Create candidates (balanced gender)
        rng = random.Random(seed)
        for i in range(n_candidates):
            gender = "male" if i % 2 == 0 else "female"
            skill = rng.randint(3, 9)
            CandidateAgent(i, self, gender, skill, industry, language)

        # Create institution
        HiringInstitution(
            "institution", self, llm_client, industry, language, fairness_mode, hire_quota
        )

    def step(self):
        # All agents act
        for agent in self.agents:
            agent.step()

    def run(self) -> pd.DataFrame:
        for _ in range(self.n_rounds):
            self.step()
        # Collect hiring log
        logs = []
        for agent in self.agents:
            if isinstance(agent, HiringInstitution):
                logs.extend(agent.hiring_log)
        return pd.DataFrame(logs)


# ── Fairness metrics computation ───────────────────────────────────────
def compute_fairness_metrics(df: pd.DataFrame) -> dict:
    """Compute fairness metrics from hiring log."""
    if df.empty:
        return {}

    male_df = df[df["gender"] == "male"]
    female_df = df[df["gender"] == "female"]

    # Hire rates
    male_hire_rate = male_df["hired"].mean() if len(male_df) else 0
    female_hire_rate = female_df["hired"].mean() if len(female_df) else 0

    # Demographic parity difference
    dp_diff = abs(male_hire_rate - female_hire_rate)

    # Score gap
    male_score = male_df["score"].mean() if len(male_df) else 0
    female_score = female_df["score"].mean() if len(female_df) else 0

    # Impact ratio (4/5ths rule)
    max_rate = max(male_hire_rate, female_hire_rate)
    ir = min(male_hire_rate, female_hire_rate) / max_rate if max_rate > 0 else 1.0

    # Gini coefficient on scores
    all_scores = df["score"].values
    gini = _gini(all_scores)

    return {
        "male_hire_rate": male_hire_rate,
        "female_hire_rate": female_hire_rate,
        "demographic_parity_diff": dp_diff,
        "score_gap": male_score - female_score,
        "impact_ratio": ir,
        "adverse_impact": ir < 0.8,
        "gini_scores": gini,
        "n_candidates": len(df[df["hired"] | ~df["hired"]]),
    }


def _gini(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    sorted_v = np.sort(values)
    n = len(sorted_v)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * sorted_v) - (n + 1) * np.sum(sorted_v)) / (n * np.sum(sorted_v)) if np.sum(sorted_v) > 0 else 0


# ── Multi-scenario runner ─────────────────────────────────────────────
def run_scenarios(
    llm_client,
    industry,
    n_candidates,
    n_rounds,
    languages,
    hire_quota=3,
) -> pd.DataFrame:
    """Run all fairness_mode × language combinations."""
    if languages is None:
        languages = ["en"]

    all_results = []
    for lang in languages:
        for mode in ["none", "demographic_parity", "auditor"]:
            print(f"\nScenario: {mode} | {lang} | {industry}")
            model = HiringModel(
    llm_client=llm_client,
    n_candidates=n_candidates,
    n_rounds=n_rounds,
    industry=industry,
    fairness_mode=mode,
    language=lang,
    hire_quota=hire_quota,
)
            log = model.run()
            if not log.empty:
                metrics = compute_fairness_metrics(log)
                metrics["scenario"] = mode
                metrics["language"] = lang
                metrics["industry"] = industry
                metrics["model"] = llm_client.model_key
                all_results.append(metrics)

    return pd.DataFrame(all_results)
