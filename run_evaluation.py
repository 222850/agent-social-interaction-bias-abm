#!/usr/bin/env python3
"""
Cross-Lingual LLM Bias Evaluation Framework — Main Runner.

Integrates:
  - VaNeu 4-stage framework (Manduru & Domeniconi 2025)
  - Bias spillover analysis (Paraschou et al. 2026)
  - Encoded vs Expressed bias (Bouchouchi et al. 2026)
  - JobFair hiring bias (Wang et al. 2024)
  - WEAT/SEAT embedding analysis
  - StereoSet iCAT
  - RuBia (Russian bias detection)
  - Mesa ABM hiring simulation (EN + RU)

Usage:
  python run_evaluation.py                    # run all
  python run_evaluation.py --models qwen2.5-7b-instruct --stages bbq jobfair
  python run_evaluation.py --models vikhr-nemo-12b --stages rubia weat abm --lang ru
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    MODELS, DEFAULT_MODELS, RUSSIAN_MODELS, RESULTS_DIR, BBQ_ATTRIBUTES,
)
from llm_client import LLMClient

# Benchmarks
from benchmarks.bbq_vaneu import VaNeuEvaluator, SpilloverEvaluator, load_bbq
from benchmarks.jobfair import JobFairEvaluator
from benchmarks.weat_seat import WEATEvaluator
from benchmarks.rubia_stereoset import (
    RuBiaEvaluator, StereoSetEvaluator, load_rubia, load_stereoset,
)

# Analysis
from analysis.encoded_expressed import EncodedExpressedEvaluator

# Simulation
from simulation.hiring_abm import run_scenarios


# ── Stage registry ─────────────────────────────────────────────────────
STAGES = [
    "bbq",          # VaNeu 4-stage + spillover readiness
    "stereoset",    # StereoSet iCAT
    "jobfair",      # JobFair hiring bias (EN)
    "jobfair_ru",   # JobFair hiring bias (RU)
    "weat",         # WEAT/SEAT embedding analysis
    "rubia",        # RuBia Russian bias
    "encoded",      # Encoded vs Expressed + jailbreak
    "abm",          # Mesa ABM hiring simulation
]


def run_stage(stage: str, client: LLMClient, lang: str = "en") -> dict:
    """Run a single evaluation stage. Returns results dict."""
    timestamp = datetime.now().isoformat()
    result = {"stage": stage, "model": client.model_key, "language": lang,
              "timestamp": timestamp}

    if stage == "bbq":
        print(f"\n{'='*60}")
        print(f"  BBQ + VaNeu 4-Stage Framework | {client.spec.display_name}")
        print(f"{'='*60}")
        try:
            evaluator = VaNeuEvaluator(client, max_items_per_attr=50)
            df = evaluator.run_full_evaluation()
            df.to_csv(RESULTS_DIR / f"bbq_vaneu_{client.model_key}.csv", index=False)
            result["vaneu_results"] = df.to_dict(orient="records")
            result["vacuous_neutral_attrs"] = df[df["is_vacuous"]]["attribute"].tolist()
            print(f"  ✓ {len(df)} attribute evaluations complete")
            for _, row in df.iterrows():
                flag = " ⚠ VACUOUS" if row["is_vacuous"] else ""
                print(f"    {row['attribute']:25s} bias={row['bias_ambig']:+.3f}  "
                      f"F1={row['f1_disambig']:.2f}  [{row['vaneu_category']}]{flag}")
        except FileNotFoundError as e:
            print(f"  ✗ {e}")
            result["error"] = str(e)

    elif stage == "stereoset":
        print(f"\n{'='*60}")
        print(f"  StereoSet iCAT | {client.spec.display_name}")
        print(f"{'='*60}")
        try:
            evaluator = StereoSetEvaluator(client, max_items=150)
            ss_result = evaluator.evaluate()
            result["stereoset"] = ss_result
            print(f"  ✓ LMS={ss_result['lms']:.1f}%  SS={ss_result['stereotype_score']:.1f}%  "
                  f"iCAT={ss_result['icat']:.2f}")
        except FileNotFoundError as e:
            print(f"  ✗ {e}")
            result["error"] = str(e)

    elif stage.startswith("jobfair"):
        is_ru = stage == "jobfair_ru"
        jf_lang = "ru" if is_ru else "en"
        print(f"\n{'='*60}")
        print(f"  JobFair Hiring Bias ({jf_lang.upper()}) | {client.spec.display_name}")
        print(f"{'='*60}")
        evaluator = JobFairEvaluator(client, language=jf_lang)
        df = evaluator.run()
        analysis = evaluator.analyze(df)
        df.to_csv(RESULTS_DIR / f"jobfair_{jf_lang}_{client.model_key}.csv", index=False)
        result["jobfair"] = {}
        for ind, metrics in analysis.items():
            result["jobfair"][ind] = {
                "level_bias_p": metrics["level_bias"]["p_value"],
                "level_bias_sig": metrics["level_bias"]["significant"],
                "spread_bias_p": metrics["spread_bias"]["p_value"],
                "impact_ratio": metrics["impact_ratio"],
                "taste_based": metrics["taste_based"],
                "mean_m": metrics["mean_score_male"],
                "mean_f": metrics["mean_score_female"],
                "bf10": metrics["bayes_factor"]["bf10"],
            }
            sig = "✗ SIG" if metrics["level_bias"]["significant"] else "  ok"
            print(f"  {ind:15s} M={metrics['mean_score_male']:.1f} F={metrics['mean_score_female']:.1f}  "
                  f"IR={metrics['impact_ratio']:.2f}  [{sig}]  "
                  f"taste={metrics['taste_based']}")

    elif stage == "weat":
        print(f"\n{'='*60}")
        print(f"  WEAT/SEAT Embedding Analysis | {client.spec.display_name}")
        print(f"{'='*60}")
        evaluator = WEATEvaluator(client)
        langs = [lang]
        if "ru" in client.spec.languages and lang == "en":
            langs.append("ru")
        weat_results = evaluator.run_all(langs)
        result["weat_seat"] = weat_results
        for r in weat_results:
            print(f"  {r['test']:25s} ({r['language']})  d={r.get('effect_size_d', 'N/A'):.3f}  "
                  f"p={r.get('p_value', 'N/A')}")

    elif stage == "rubia":
        print(f"\n{'='*60}")
        print(f"  RuBia Russian Bias | {client.spec.display_name}")
        print(f"{'='*60}")
        if "ru" not in client.spec.languages:
            print(f"  ✗ Model does not support Russian, skipping")
            result["error"] = "model does not support Russian"
            return result
        try:
            evaluator = RuBiaEvaluator(client, max_items=150)
            df = evaluator.evaluate()
            scores = evaluator.compute_scores(df)
            df.to_csv(RESULTS_DIR / f"rubia_{client.model_key}.csv", index=False)
            result["rubia"] = scores
            for domain, s in scores.items():
                print(f"  {domain:20s} bias={s['bias_score']:+.3f}  "
                      f"stereo_rate={s['stereotype_rate']:.1%}  n={s['n_items']}")
        except FileNotFoundError as e:
            print(f"  ✗ {e}")
            result["error"] = str(e)

    elif stage == "encoded":
        print(f"\n{'='*60}")
        print(f"  Encoded vs Expressed Bias | {client.spec.display_name}")
        print(f"{'='*60}")
        evaluator = EncodedExpressedEvaluator(client)
        langs = [lang]
        if "ru" in client.spec.languages and lang == "en":
            langs.append("ru")
        ee_results = evaluator.run_full(langs)
        result["encoded_expressed"] = {}
        for l, r in ee_results.items():
            result["encoded_expressed"][l] = {
                "expressed_bias": r["extrinsic"]["expressed_bias"],
                "probe_accuracy": r["intrinsic"]["probe_accuracy"],
                "encoded_detected": r["intrinsic"]["encoded_bias_detected"],
                "jailbreak_reactivation": r["jailbreak"]["reactivation_detected"],
                "alignment_gap": r["alignment_gap"]["gap_exists"],
            }
            gap_icon = "⚠ GAP" if r["alignment_gap"]["gap_exists"] else "  ok"
            jb_icon = "⚠ REACT" if r["jailbreak"]["reactivation_detected"] else "  ok"
            print(f"  {l.upper():4s} expressed={r['extrinsic']['expressed_bias']:+.3f}  "
                  f"probe={r['intrinsic']['probe_accuracy']:.2f}  "
                  f"[{gap_icon}] [{jb_icon}]")

    elif stage == "abm":
        print(f"\n{'='*60}")
        print(f"  Mesa ABM Hiring Simulation | {client.spec.display_name}")
        print(f"{'='*60}")
        langs = [lang]
        if "ru" in client.spec.languages and lang == "en":
            langs.append("ru")
        abm_df = run_scenarios(
            client, industry="technology",
            n_candidates=16, n_rounds=2, languages=langs,
        )
        if not abm_df.empty:
            abm_df.to_csv(RESULTS_DIR / f"abm_{client.model_key}.csv", index=False)
            result["abm"] = abm_df.to_dict(orient="records")
            for _, row in abm_df.iterrows():
                ai = "⚠ ADVERSE" if row.get("adverse_impact") else "  ok"
                print(f"  {row['scenario']:20s} ({row['language']})  "
                      f"DP={row['demographic_parity_diff']:.3f}  "
                      f"IR={row['impact_ratio']:.2f}  [{ai}]")

    return result


# ── Main orchestrator ──────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Cross-Lingual LLM Bias Evaluation")
    parser.add_argument("--models", nargs="+", default=None,
                        help=f"Models to evaluate. Available: {list(MODELS.keys())}")
    parser.add_argument("--stages", nargs="+", default=None,
                        help=f"Stages to run. Available: {STAGES}")
    parser.add_argument("--lang", default="en", choices=["en", "ru"],
                        help="Primary language")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    model_keys = args.models or DEFAULT_MODELS
    stages = args.stages or STAGES
    lang = args.lang

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  Cross-Lingual LLM Bias Evaluation Framework               ║")
    print("║  VaNeu + Spillover + Encoded/Expressed + JobFair + ABM      ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"\nModels:  {model_keys}")
    print(f"Stages:  {stages}")
    print(f"Language: {lang}")
    print(f"Results: {RESULTS_DIR}")

    all_results = []
    for model_key in model_keys:
        if model_key not in MODELS:
            print(f"\n⚠ Unknown model: {model_key}, skipping")
            continue
        client = LLMClient(model_key)
        print(f"\n\n{'#'*60}")
        print(f"  MODEL: {MODELS[model_key].display_name}")
        print(f"  Languages: {MODELS[model_key].languages}")
        print(f"  Cache: {client.cache_stats()}")
        print(f"{'#'*60}")

        for stage in stages:
            if stage == "rubia" and "ru" not in MODELS[model_key].languages:
                continue
            if stage == "jobfair_ru" and "ru" not in MODELS[model_key].languages:
                continue
            try:
                t0 = time.time()
                result = run_stage(stage, client, lang)
                result["elapsed_seconds"] = time.time() - t0
                all_results.append(result)
                print(f"  ⏱ {result['elapsed_seconds']:.0f}s")
            except Exception as e:
                print(f"  ✗ Stage {stage} failed: {e}")
                import traceback
                traceback.print_exc()
                all_results.append({
                    "stage": stage, "model": model_key,
                    "error": str(e),
                })

    # Save all results
    output_path = args.output or str(RESULTS_DIR / "full_evaluation.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n\n{'='*60}")
    print(f"  All results saved to: {output_path}")
    print(f"  CSV files in: {RESULTS_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
