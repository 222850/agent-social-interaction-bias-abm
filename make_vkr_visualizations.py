#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_vkr_visualizations.py

Генератор визуализаций для ВКР о кросс-лингвальной гендерной предвзятости
в LLM/SLM-агентах в задачах найма.

Как запускать из корня проекта/ноутбука:
    python make_vkr_visualizations.py --results-dir notebook_results --out-dir vkr_figures

Если CSV лежат в текущей папке:
    python make_vkr_visualizations.py --results-dir . --out-dir vkr_figures

Скрипт ожидает файлы вида:
    yandexgpt_api_expressed_bias_results.csv
    yandexgpt_api_encoded_results.csv
    yandexgpt_api_weat_results.csv
    yandexgpt_api_jobfair_results.csv
    yandexgpt_api_abm_n50_hq15_results.csv
    yandexgpt_api_abm_n100_hq30_results.csv
    1_yandexgpt_api_jobfair_raw_en_results.csv   # опционально
и аналогично для yandexgpt_5_pro, yandexgpt_5_1.

Выход: PNG 300 dpi + SVG для каждой визуализации.
PNG удобно вставлять в Word, SVG удобно хранить как векторную версию.
"""

from __future__ import annotations

import argparse
import ast
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D

# -----------------------------
# Общий стиль
# -----------------------------
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.family": "DejaVu Sans",
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "grid.linewidth": 0.7,
})

PALETTE = {
    "blue": "#345995",
    "cyan": "#03CEA4",
    "orange": "#FB4D3D",
    "yellow": "#F5B700",
    "purple": "#7B2CBF",
    "gray": "#6C757D",
    "light_gray": "#F3F5F7",
    "dark": "#1F2933",
    "green": "#2E7D32",
    "red": "#B00020",
    "cream": "#FFF8E7",
}

MODEL_ORDER = ["YandexGPT Lite", "YandexGPT 5 Pro", "YandexGPT 5.1"]
LANG_ORDER = ["EN", "RU"]
SCENARIO_ORDER = ["none", "baseline", "demographic_parity", "auditor", "soft_auditor", "anonymized"]

# -----------------------------
# Загрузка и нормализация CSV
# -----------------------------

def read_csv_safe(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8-sig")


def model_from_any(value: object) -> Optional[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    s = str(value).lower()
    if "5_1" in s or "5.1" in s or "5-1" in s:
        return "YandexGPT 5.1"
    if "5_pro" in s or "5-pro" in s or "5 pro" in s:
        return "YandexGPT 5 Pro"
    if "api" in s or "lite" in s:
        return "YandexGPT Lite"
    return str(value)


def model_from_filename(path: Path) -> str:
    return model_from_any(path.name) or path.stem


def normalize_language(x: object) -> str:
    s = str(x).lower()
    if s in {"en", "eng", "english"} or "_en" in s:
        return "EN"
    if s in {"ru", "rus", "russian"} or "_ru" in s:
        return "RU"
    return str(x).upper()


def add_common_columns(df: pd.DataFrame, source_path: Path, scenario: Optional[str] = None) -> pd.DataFrame:
    df = df.copy()
    df["source_file"] = source_path.name

    if "model" in df.columns:
        df["model_display"] = df["model"].map(model_from_any)
    elif "model_key" in df.columns:
        df["model_display"] = df["model_key"].map(model_from_any)
    elif "provider_model" in df.columns:
        df["model_display"] = df["provider_model"].map(model_from_any)
    else:
        df["model_display"] = model_from_filename(source_path)

    # Если модель не распознана из столбца, пробуем из имени файла
    df["model_display"] = df["model_display"].fillna(model_from_filename(source_path))

    if "language" not in df.columns:
        if "_en_" in source_path.name or source_path.name.endswith("_en_results.csv"):
            df["language"] = "EN"
        elif "_ru_" in source_path.name or source_path.name.endswith("_ru_results.csv"):
            df["language"] = "RU"
        else:
            df["language"] = "unknown"
    df["language"] = df["language"].map(normalize_language)

    if scenario is not None:
        df["abm_scale"] = scenario
    return df


def parse_dict_cell(x: object) -> Dict[str, object]:
    if isinstance(x, dict):
        return x
    if pd.isna(x):
        return {}
    s = str(x).strip()
    if not (s.startswith("{") and s.endswith("}")):
        return {}
    try:
        return ast.literal_eval(s)
    except Exception:
        return {}


def expand_dict_column(df: pd.DataFrame, col: str, prefix: str) -> pd.DataFrame:
    if col not in df.columns:
        return df
    parsed = df[col].map(parse_dict_cell)
    keys = sorted({k for d in parsed for k in d.keys()})
    for k in keys:
        df[f"{prefix}_{k}"] = parsed.map(lambda d: d.get(k, np.nan))
    return df


def load_many(results_dir: Path, patterns: Iterable[str], exclude_raw: bool = False, scenario: Optional[str] = None) -> pd.DataFrame:
    paths: List[Path] = []
    for pattern in patterns:
        paths.extend(sorted(results_dir.glob(pattern)))
    if exclude_raw:
        paths = [p for p in paths if "raw" not in p.name.lower()]
    frames = []
    for p in paths:
        try:
            df = read_csv_safe(p)
            frames.append(add_common_columns(df, p, scenario=scenario))
        except Exception as e:
            print(f"[WARN] Не удалось прочитать {p}: {e}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def first_existing(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    for n in names:
        if n in df.columns:
            return n
    return None


def load_all(results_dir: Path) -> Dict[str, pd.DataFrame]:
    expressed = load_many(results_dir, ["*expressed_bias_results.csv"], exclude_raw=True)
    encoded = load_many(results_dir, ["*encoded_results.csv"], exclude_raw=True)
    weat = load_many(results_dir, ["*weat_results.csv"], exclude_raw=True)
    jobfair = load_many(results_dir, ["*jobfair_results.csv"], exclude_raw=True)
    jobfair_raw = load_many(results_dir, ["1_*jobfair_raw_*_results.csv", "*jobfair_raw_*_results.csv"], exclude_raw=False)
    abm50 = load_many(results_dir, ["*abm_n50_hq15_results.csv"], exclude_raw=True, scenario="50/15")
    abm100 = load_many(results_dir, ["*abm_n100_hq30_results.csv"], exclude_raw=True, scenario="100/30")

    for df in [jobfair]:
        for col, pref in [("level_bias", "level_bias"), ("spread_bias", "spread_bias"),
                          ("level_test", "level_bias"), ("spread_test", "spread_bias")]:
            df = expand_dict_column(df, col, pref)
        if len(df):
            jobfair = df

    return {
        "expressed": expressed,
        "encoded": encoded,
        "weat": weat,
        "jobfair": jobfair,
        "jobfair_raw": jobfair_raw,
        "abm50": abm50,
        "abm100": abm100,
        "abm_all": pd.concat([d for d in [abm50, abm100] if len(d)], ignore_index=True, sort=False) if (len(abm50) or len(abm100)) else pd.DataFrame(),
    }

# -----------------------------
# Служебные функции для графиков
# -----------------------------

def ensure_out(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_both(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir = ensure_out(out_dir)
    png = out_dir / f"{stem}.png"
    svg = out_dir / f"{stem}.svg"
    fig.savefig(png, bbox_inches="tight", facecolor="white")
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[OK] {png}")


def placeholder(out_dir: Path, stem: str, title: str, reason: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.axis("off")
    ax.text(0.5, 0.62, title, ha="center", va="center", fontsize=16, weight="bold")
    ax.text(0.5, 0.42, reason, ha="center", va="center", fontsize=11, color=PALETTE["gray"], wrap=True)
    save_both(fig, out_dir, stem)


def ordered_unique(values: Iterable[object], preferred: List[str]) -> List[str]:
    vals = [v for v in pd.Series(list(values)).dropna().astype(str).unique().tolist()]
    return [v for v in preferred if v in vals] + [v for v in vals if v not in preferred]


def draw_box(ax, xy, w, h, text, fc, ec="#334155", fontsize=10, lw=1.2):
    box = FancyBboxPatch(
        xy, w, h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=lw, edgecolor=ec, facecolor=fc
    )
    ax.add_patch(box)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fontsize, wrap=True)
    return box


def arrow(ax, start, end, color="#334155"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, linewidth=1.3, color=color))

# -----------------------------
# Схемы методологии
# -----------------------------

def fig_methodology_pipeline(out_dir: Path):
    fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Гибридный пайплайн оценки гендерных смещений в LLM/SLM-агентах", pad=18, weight="bold")

    boxes = [
        (0.03, 0.55, 0.14, 0.22, "Входные данные\nEN/RU резюме\nконтрфактические пары", PALETTE["light_gray"]),
        (0.22, 0.55, 0.13, 0.22, "Модели\nYandexGPT Lite\n5 Pro / 5.1", "#E8F1FF"),
        (0.40, 0.55, 0.13, 0.22, "Одиночные\nпроверки\nexpressed / encoded", "#E9F7F3"),
        (0.58, 0.55, 0.13, 0.22, "Статический\nfairness-аудит\nWEAT/SEAT + JobFair", "#FFF3D6"),
        (0.76, 0.55, 0.18, 0.22, "Динамическая\nABM-симуляция\n50/15 и 100/30", "#F2E8FF"),
    ]
    centers = []
    for x, y, w, h, text, fc in boxes:
        draw_box(ax, (x, y), w, h, text, fc, fontsize=10)
        centers.append((x + w, y + h/2, x, y + h/2))
    for i in range(len(boxes) - 1):
        arrow(ax, (boxes[i][0] + boxes[i][2] + 0.015, boxes[i][1] + boxes[i][3]/2),
              (boxes[i+1][0] - 0.015, boxes[i+1][1] + boxes[i+1][3]/2))

    # Нижний слой метрик
    metric_boxes = [
        (0.19, 0.18, 0.18, 0.18, "Уровень представлений:\nprobe accuracy, WEAT d", "#F8FAFC"),
        (0.41, 0.18, 0.18, 0.18, "Уровень генерации:\nexpressed bias, gendered response", "#F8FAFC"),
        (0.63, 0.18, 0.18, 0.18, "Уровень решений:\nscore gap, impact ratio", "#F8FAFC"),
        (0.84, 0.18, 0.13, 0.18, "Системный уровень:\nDP diff, adverse impact", "#F8FAFC"),
    ]
    for x, y, w, h, text, fc in metric_boxes:
        draw_box(ax, (x, y), w, h, text, fc, fontsize=9, lw=0.9)
    ax.text(0.03, 0.27, "Единая логика:\nот скрытых ассоциаций\nк итоговому распределению найма", fontsize=10, va="center", color=PALETTE["dark"])
    save_both(fig, out_dir, "01_methodology_pipeline")


def fig_abm_architecture(out_dir: Path):
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Архитектура ABM/LLM-системы найма", pad=18, weight="bold")

    draw_box(ax, (0.05, 0.70), 0.22, 0.16, "Mesa-среда\nкандидаты, вакансии, раунды,\nстатус найма, сбор метрик", "#E8F1FF")
    draw_box(ax, (0.39, 0.70), 0.22, 0.16, "Orchestration layer\nпоследовательность вызовов,\nвалидация, парсинг", "#E9F7F3")
    draw_box(ax, (0.73, 0.70), 0.20, 0.16, "LLM-рекрутёр\nscore, hire/reject,\nкраткое обоснование", "#FFF3D6")

    draw_box(ax, (0.08, 0.36), 0.20, 0.14, "Fairness-сценарии\nnone / demographic parity\nSoft Auditor / anonymized", "#F2E8FF", fontsize=9)
    draw_box(ax, (0.40, 0.36), 0.20, 0.14, "Soft Auditor\nпроверка решения\nи fairness-сигнала", "#FFECEC", fontsize=9)
    draw_box(ax, (0.72, 0.36), 0.20, 0.14, "Метрики\nDP diff, impact ratio,\nscore gap, Gini", "#F8FAFC", fontsize=9)

    arrow(ax, (0.28, 0.78), (0.38, 0.78))
    arrow(ax, (0.62, 0.78), (0.72, 0.78))
    arrow(ax, (0.83, 0.69), (0.83, 0.52))
    arrow(ax, (0.72, 0.43), (0.61, 0.43))
    arrow(ax, (0.39, 0.43), (0.29, 0.43))
    arrow(ax, (0.18, 0.51), (0.18, 0.69))
    arrow(ax, (0.50, 0.51), (0.50, 0.69))
    arrow(ax, (0.61, 0.43), (0.71, 0.43))

    ax.text(0.5, 0.13,
            "Смысл схемы: Mesa хранит состояние симуляции, orchestration layer управляет LLM-вызовами,\n"
            "а итоговые fairness-метрики рассчитываются по результатам многошагового процесса найма.",
            ha="center", fontsize=10, color=PALETTE["gray"])
    save_both(fig, out_dir, "02_abm_llm_architecture")


def fig_bias_levels(out_dir: Path):
    fig, ax = plt.subplots(figsize=(11.5, 5.7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Уровни проявления гендерных смещений в исследовании", pad=18, weight="bold")
    levels = [
        ("Encoded / implicit", "ассоциации в представлениях\nembeddings, probing, WEAT/SEAT", "#E8F1FF"),
        ("Expressed", "гендерно маркированные\nответы и объяснения", "#E9F7F3"),
        ("Counterfactual", "различия в оценке\nпарных резюме JobFair", "#FFF3D6"),
        ("Systemic / ABM", "итоговые различия\nв найме и impact ratio", "#F2E8FF"),
    ]
    xs = [0.06, 0.30, 0.54, 0.78]
    for i, ((title, desc, fc), x) in enumerate(zip(levels, xs)):
        draw_box(ax, (x, 0.44), 0.18, 0.26, f"{title}\n\n{desc}", fc, fontsize=9.5)
        ax.text(x+0.09, 0.30, f"Уровень {i+1}", ha="center", va="center", fontsize=10, weight="bold", color=PALETTE["dark"])
        if i < 3:
            arrow(ax, (x+0.19, 0.57), (xs[i+1]-0.01, 0.57))
    ax.text(0.5, 0.13,
            "Одного уровня проверки недостаточно: внешне нейтральная генерация может сосуществовать\nсо скрытыми ассоциациями или системными различиями в динамической симуляции.",
            ha="center", fontsize=10, color=PALETTE["gray"])
    save_both(fig, out_dir, "03_bias_levels_map")

# -----------------------------
# Визуализации результатов
# -----------------------------

def fig_model_stage_matrix(data: Dict[str, pd.DataFrame], out_dir: Path):
    stages = ["expressed", "encoded", "weat", "jobfair", "abm50", "abm100"]
    rows = MODEL_ORDER
    matrix = np.zeros((len(rows), len(stages)))
    for j, st in enumerate(stages):
        df = data.get(st, pd.DataFrame())
        if len(df) and "model_display" in df.columns:
            available = set(df["model_display"].dropna())
            for i, m in enumerate(rows):
                matrix[i, j] = 1 if m in available else 0
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="Greens")
    ax.set_xticks(range(len(stages)), ["Expressed", "Encoded", "WEAT/SEAT", "JobFair", "ABM 50/15", "ABM 100/30"], rotation=25, ha="right")
    ax.set_yticks(range(len(rows)), rows)
    ax.set_title("Покрытие экспериментальных стадий по моделям", weight="bold", pad=14)
    for i in range(len(rows)):
        for j in range(len(stages)):
            ax.text(j, i, "есть" if matrix[i, j] else "нет", ha="center", va="center", fontsize=9,
                    color="white" if matrix[i, j] else PALETTE["gray"])
    ax.grid(False)
    save_both(fig, out_dir, "04_model_stage_coverage")


def fig_expressed_heatmap(data: Dict[str, pd.DataFrame], out_dir: Path):
    df = data["expressed"].copy()
    if df.empty:
        placeholder(out_dir, "05_expressed_bias_heatmap", "Expressed bias", "Файлы *expressed_bias_results.csv не найдены.")
        return
    metric = first_existing(df, ["expressed_bias", "bias_score", "mean_expressed_bias"])
    if metric is None:
        placeholder(out_dir, "05_expressed_bias_heatmap", "Expressed bias", "В CSV нет столбца expressed_bias / bias_score.")
        return
    df[metric] = to_num(df[metric])
    agg = df.groupby(["model_display", "language"], as_index=False)[metric].mean()
    models = ordered_unique(agg["model_display"], MODEL_ORDER)
    langs = ordered_unique(agg["language"], LANG_ORDER)
    pivot = agg.pivot(index="model_display", columns="language", values=metric).reindex(index=models, columns=langs)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    vals = pivot.values.astype(float)
    vmax = np.nanmax(np.abs(vals)) if np.isfinite(vals).any() else 1
    im = ax.imshow(vals, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(langs)), langs)
    ax.set_yticks(range(len(models)), models)
    ax.set_title("Выраженное гендерное смещение по моделям и языкам", weight="bold", pad=14)
    for i in range(len(models)):
        for j in range(len(langs)):
            v = vals[i, j]
            ax.text(j, i, "—" if np.isnan(v) else f"{v:.3f}", ha="center", va="center", fontsize=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    cbar.set_label(metric)
    ax.grid(False)
    save_both(fig, out_dir, "05_expressed_bias_heatmap")


def fig_gendered_response_bars(data: Dict[str, pd.DataFrame], out_dir: Path):
    df = data["expressed"].copy()
    if df.empty:
        placeholder(out_dir, "06_gendered_response_pct", "Доля гендерно маркированных ответов", "Файлы expressed не найдены.")
        return
    metric = first_existing(df, ["gendered_response_pct", "gendered_pct", "non_neutral_pct"])
    if metric is None:
        placeholder(out_dir, "06_gendered_response_pct", "Доля гендерно маркированных ответов", "В CSV нет столбца gendered_response_pct.")
        return
    df[metric] = to_num(df[metric])
    agg = df.groupby(["model_display", "language"], as_index=False)[metric].mean()
    models = ordered_unique(agg["model_display"], MODEL_ORDER)
    x = np.arange(len(models))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for idx, lang in enumerate(LANG_ORDER):
        vals = agg[agg["language"] == lang].set_index("model_display").reindex(models)[metric].values
        ax.bar(x + (idx - 0.5)*width, vals, width, label=lang)
    ax.set_xticks(x, models, rotation=15, ha="right")
    ax.set_ylabel(metric)
    ax.set_title("Гендерно маркированные ответы: EN vs RU", weight="bold", pad=14)
    ax.legend(title="Язык")
    save_both(fig, out_dir, "06_gendered_response_pct")


def fig_encoded_alignment(data: Dict[str, pd.DataFrame], out_dir: Path):
    df = data["encoded"].copy()
    if df.empty:
        placeholder(out_dir, "07_encoded_alignment", "Encoded bias и alignment gap", "Файлы *encoded_results.csv не найдены.")
        return
    acc_col = first_existing(df, ["probe_accuracy", "accuracy", "probe_acc"])
    gap_col = first_existing(df, ["alignment_gap", "gap"])
    jr_col = first_existing(df, ["jailbreak_reactivation", "reactivation", "jailbreak_reactivation_score"])
    if acc_col is None and gap_col is None and jr_col is None:
        placeholder(out_dir, "07_encoded_alignment", "Encoded bias и alignment gap", "В CSV нет probe_accuracy / alignment_gap / jailbreak_reactivation.")
        return
    for col in [acc_col, gap_col, jr_col]:
        if col:
            df[col] = to_num(df[col])
    agg_cols = [c for c in [acc_col, gap_col, jr_col] if c]
    agg = df.groupby(["model_display", "language"], as_index=False)[agg_cols].mean()

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    xcol = acc_col or gap_col or jr_col
    ycol = gap_col or jr_col or acc_col
    if xcol == ycol:
        ycol = None
    if ycol:
        for lang in ordered_unique(agg["language"], LANG_ORDER):
            sub = agg[agg["language"] == lang]
            ax.scatter(sub[xcol], sub[ycol], s=120, label=lang, alpha=0.85)
            for _, r in sub.iterrows():
                ax.annotate(r["model_display"].replace("YandexGPT ", ""), (r[xcol], r[ycol]), xytext=(6, 6), textcoords="offset points", fontsize=8)
        ax.set_xlabel(xcol)
        ax.set_ylabel(ycol)
    else:
        sub = agg.groupby("model_display", as_index=False)[xcol].mean()
        ax.bar(sub["model_display"], sub[xcol])
        ax.set_ylabel(xcol)
        ax.tick_params(axis="x", rotation=15)
    ax.set_title("Encoded bias: точность probing и разрыв выравнивания", weight="bold", pad=14)
    ax.legend(title="Язык") if ycol else None
    save_both(fig, out_dir, "07_encoded_alignment")


def fig_weat_lollipop(data: Dict[str, pd.DataFrame], out_dir: Path):
    df = data["weat"].copy()
    if df.empty:
        placeholder(out_dir, "08_weat_effect_sizes", "WEAT/SEAT effect sizes", "Файлы *weat_results.csv не найдены.")
        return
    effect_col = first_existing(df, ["effect_size_d", "effect_size", "cohens_d", "d"])
    p_col = first_existing(df, ["p_value", "p", "pval"])
    test_col = first_existing(df, ["test", "test_name", "weat_test"])
    if effect_col is None:
        placeholder(out_dir, "08_weat_effect_sizes", "WEAT/SEAT effect sizes", "В CSV нет столбца effect_size_d/effect_size.")
        return
    df[effect_col] = to_num(df[effect_col])
    if p_col:
        df[p_col] = to_num(df[p_col])
    df["label"] = df["model_display"] + " / " + df["language"].astype(str)
    if test_col:
        df["label"] += " / " + df[test_col].astype(str)
    plot_df = df.sort_values(effect_col).tail(30)
    fig, ax = plt.subplots(figsize=(10, max(5, 0.30*len(plot_df)+1.5)))
    y = np.arange(len(plot_df))
    ax.hlines(y, 0, plot_df[effect_col], color="#94A3B8", linewidth=2)
    sig = plot_df[p_col] < 0.05 if p_col else pd.Series(False, index=plot_df.index)
    colors = np.where(sig, PALETTE["orange"], PALETTE["blue"])
    ax.scatter(plot_df[effect_col], y, s=70, c=colors, zorder=3)
    ax.axvline(0, color="#111827", linewidth=1)
    ax.set_yticks(y, plot_df["label"])
    ax.set_xlabel(effect_col)
    ax.set_title("WEAT/SEAT: размеры эффекта по тестам", weight="bold", pad=14)
    if p_col:
        ax.legend(handles=[Line2D([0], [0], marker='o', color='w', label='p < 0.05', markerfacecolor=PALETTE["orange"], markersize=8),
                           Line2D([0], [0], marker='o', color='w', label='p ≥ 0.05', markerfacecolor=PALETTE["blue"], markersize=8)],
                  loc="lower right")
    save_both(fig, out_dir, "08_weat_effect_sizes")


def fig_jobfair_compression(data: Dict[str, pd.DataFrame], out_dir: Path):
    df = data["jobfair"].copy()
    if df.empty:
        placeholder(out_dir, "09_jobfair_score_compression", "JobFair score compression", "Файлы *jobfair_results.csv не найдены.")
        return
    score_cols = [c for c in ["mean_score_male", "mean_score_female", "mean_score_neutral"] if c in df.columns]
    if len(score_cols) < 2:
        placeholder(out_dir, "09_jobfair_score_compression", "JobFair score compression", "В CSV нет mean_score_male/female/neutral.")
        return
    for c in score_cols:
        df[c] = to_num(df[c])
    df["score_range"] = df[score_cols].max(axis=1) - df[score_cols].min(axis=1)
    group_cols = ["model_display", "language"]
    if "industry" in df.columns:
        group_cols.append("industry")
    agg = df.groupby(group_cols, as_index=False)["score_range"].mean()
    agg = agg.sort_values("score_range", ascending=False).head(25)
    fig, ax = plt.subplots(figsize=(10, max(4.8, 0.28*len(agg)+1.2)))
    labels = agg.apply(lambda r: f"{r['model_display']} / {r['language']}" + (f" / {r['industry']}" if 'industry' in agg.columns else ""), axis=1)
    ax.barh(np.arange(len(agg)), agg["score_range"])
    ax.set_yticks(np.arange(len(agg)), labels)
    ax.invert_yaxis()
    ax.set_xlabel("Размах средних score между male/female/neutral")
    ax.set_title("JobFair: насколько различаются средние оценки контрфактических резюме", weight="bold", pad=14)
    save_both(fig, out_dir, "09_jobfair_score_compression")


def fig_jobfair_raw_distributions(data: Dict[str, pd.DataFrame], out_dir: Path):
    df = data["jobfair_raw"].copy()
    if df.empty:
        placeholder(out_dir, "10_jobfair_raw_score_distribution", "Raw JobFair distributions", "Raw-файлы 1_*jobfair_raw_* не найдены. График опциональный.")
        return
    score_col = first_existing(df, ["score", "parsed_score", "rating", "candidate_score"])
    gender_col = first_existing(df, ["gender", "candidate_gender", "variant", "condition"])
    if score_col is None or gender_col is None:
        placeholder(out_dir, "10_jobfair_raw_score_distribution", "Raw JobFair distributions", "В raw CSV нет score и/или gender/variant.")
        return
    df[score_col] = to_num(df[score_col])
    df = df.dropna(subset=[score_col])
    if df.empty:
        placeholder(out_dir, "10_jobfair_raw_score_distribution", "Raw JobFair distributions", "Score не распарсился в числа.")
        return
    # Boxplot по гендерным вариантам и языкам
    groups = []
    labels = []
    for lang in ordered_unique(df["language"], LANG_ORDER):
        for g in ordered_unique(df[gender_col].astype(str), ["male", "female", "neutral"]):
            vals = df[(df["language"] == lang) & (df[gender_col].astype(str) == g)][score_col].dropna().values
            if len(vals):
                groups.append(vals)
                labels.append(f"{lang}\n{g}")
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.boxplot(groups, labels=labels, showfliers=False, patch_artist=True)
    ax.set_ylabel(score_col)
    ax.set_title("JobFair raw: распределения score по языкам и гендерным вариантам", weight="bold", pad=14)
    save_both(fig, out_dir, "10_jobfair_raw_score_distribution")


def abm_heatmap(data: Dict[str, pd.DataFrame], out_dir: Path, key: str, stem: str, title: str):
    df = data[key].copy()
    if df.empty:
        placeholder(out_dir, stem, title, f"Файлы для {key} не найдены.")
        return
    metric = first_existing(df, ["impact_ratio", "impact", "IR"])
    if metric is None:
        placeholder(out_dir, stem, title, "В CSV нет impact_ratio.")
        return
    df[metric] = to_num(df[metric])
    scen_col = first_existing(df, ["scenario", "intervention", "fairness_scenario", "condition"])
    if scen_col is None:
        df["scenario"] = "none"
        scen_col = "scenario"
    group_cols = ["model_display", "language", scen_col]
    agg = df.groupby(group_cols, as_index=False)[metric].mean()
    agg["row"] = agg["model_display"] + " / " + agg["language"]
    rows = ordered_unique(agg["row"], [m + " / " + l for m in MODEL_ORDER for l in LANG_ORDER])
    cols = ordered_unique(agg[scen_col], SCENARIO_ORDER)
    pivot = agg.pivot(index="row", columns=scen_col, values=metric).reindex(index=rows, columns=cols)
    vals = pivot.values.astype(float)
    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.42*len(rows)+1.2)))
    im = ax.imshow(vals, cmap="RdYlGn", vmin=0, vmax=max(1.2, np.nanmax(vals) if np.isfinite(vals).any() else 1.2))
    ax.set_xticks(range(len(cols)), cols, rotation=25, ha="right")
    ax.set_yticks(range(len(rows)), rows)
    ax.set_title(title, weight="bold", pad=14)
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = vals[i, j]
            txt = "—" if np.isnan(v) else f"{v:.2f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9)
    ax.axvline(-0.5, color="none")
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    cbar.set_label("impact_ratio")
    ax.grid(False)
    save_both(fig, out_dir, stem)


def fig_abm_scale_shift(data: Dict[str, pd.DataFrame], out_dir: Path):
    df = data["abm_all"].copy()
    if df.empty:
        placeholder(out_dir, "13_abm_scale_shift", "Масштабирование ABM", "ABM-файлы не найдены.")
        return
    metric = first_existing(df, ["impact_ratio", "impact", "IR"])
    if metric is None:
        placeholder(out_dir, "13_abm_scale_shift", "Масштабирование ABM", "В CSV нет impact_ratio.")
        return
    scen_col = first_existing(df, ["scenario", "intervention", "fairness_scenario", "condition"])
    if scen_col is None:
        df["scenario"] = "none"
        scen_col = "scenario"
    df[metric] = to_num(df[metric])
    # baseline/none для сопоставления масштаба
    base = df[df[scen_col].astype(str).str.lower().isin(["none", "baseline"])]
    if base.empty:
        base = df
    agg = base.groupby(["model_display", "language", "abm_scale"], as_index=False)[metric].mean()
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for (m, lang), sub in agg.groupby(["model_display", "language"]):
        sub = sub.set_index("abm_scale").reindex(["50/15", "100/30"]).reset_index()
        if sub[metric].notna().sum() >= 1:
            ax.plot(sub["abm_scale"], sub[metric], marker="o", linewidth=2, label=f"{m} / {lang}")
    ax.axhline(0.8, color=PALETTE["red"], linestyle="--", linewidth=1.2, label="adverse impact threshold = 0.8")
    ax.set_ylabel("impact_ratio")
    ax.set_xlabel("Масштаб ABM")
    ax.set_title("Изменение impact ratio при масштабировании ABM", weight="bold", pad=14)
    ax.legend(ncol=2, frameon=True)
    save_both(fig, out_dir, "13_abm_scale_shift")


def fig_interventions(data: Dict[str, pd.DataFrame], out_dir: Path):
    df = data["abm_all"].copy()
    if df.empty:
        placeholder(out_dir, "14_intervention_comparison", "Сравнение fairness-интервенций", "ABM-файлы не найдены.")
        return
    metric = first_existing(df, ["impact_ratio", "impact", "IR"])
    scen_col = first_existing(df, ["scenario", "intervention", "fairness_scenario", "condition"])
    if metric is None or scen_col is None:
        placeholder(out_dir, "14_intervention_comparison", "Сравнение fairness-интервенций", "В CSV нет impact_ratio и/или scenario/intervention.")
        return
    df[metric] = to_num(df[metric])
    agg = df.groupby([scen_col, "abm_scale"], as_index=False)[metric].mean()
    scenarios = ordered_unique(agg[scen_col], SCENARIO_ORDER)
    x = np.arange(len(scenarios))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for i, scale in enumerate(["50/15", "100/30"]):
        vals = agg[agg["abm_scale"] == scale].set_index(scen_col).reindex(scenarios)[metric].values
        ax.bar(x + (i - 0.5)*width, vals, width, label=scale)
    ax.axhline(0.8, color=PALETTE["red"], linestyle="--", linewidth=1.2, label="порог 0.8")
    ax.set_xticks(x, scenarios, rotation=20, ha="right")
    ax.set_ylabel("Средний impact_ratio")
    ax.set_title("Fairness-интервенции в ABM: сравнение impact ratio", weight="bold", pad=14)
    ax.legend()
    save_both(fig, out_dir, "14_intervention_comparison")


def fig_static_vs_dynamic(data: Dict[str, pd.DataFrame], out_dir: Path):
    job = data["jobfair"].copy()
    abm = data["abm_all"].copy()
    if job.empty or abm.empty:
        placeholder(out_dir, "15_static_vs_dynamic", "Статический JobFair vs динамическая ABM", "Нужны одновременно JobFair и ABM CSV.")
        return
    score_cols = [c for c in ["mean_score_male", "mean_score_female", "mean_score_neutral"] if c in job.columns]
    ir_col = first_existing(abm, ["impact_ratio", "impact", "IR"])
    scen_col = first_existing(abm, ["scenario", "intervention", "fairness_scenario", "condition"])
    if len(score_cols) < 2 or ir_col is None:
        placeholder(out_dir, "15_static_vs_dynamic", "Статический JobFair vs динамическая ABM", "Не хватает score columns или impact_ratio.")
        return
    for c in score_cols:
        job[c] = to_num(job[c])
    job["jobfair_range"] = job[score_cols].max(axis=1) - job[score_cols].min(axis=1)
    job_agg = job.groupby(["model_display", "language"], as_index=False)["jobfair_range"].mean()
    abm[ir_col] = to_num(abm[ir_col])
    if scen_col:
        abm_base = abm[abm[scen_col].astype(str).str.lower().isin(["none", "baseline"])]
        if abm_base.empty:
            abm_base = abm
    else:
        abm_base = abm
    abm_agg = abm_base.groupby(["model_display", "language"], as_index=False)[ir_col].mean()
    merged = job_agg.merge(abm_agg, on=["model_display", "language"], how="inner")
    if merged.empty:
        placeholder(out_dir, "15_static_vs_dynamic", "Статический JobFair vs динамическая ABM", "Нет пересечения по моделям/языкам.")
        return
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    for lang in ordered_unique(merged["language"], LANG_ORDER):
        sub = merged[merged["language"] == lang]
        ax.scatter(sub["jobfair_range"], sub[ir_col], s=130, label=lang, alpha=0.85)
        for _, r in sub.iterrows():
            ax.annotate(r["model_display"].replace("YandexGPT ", ""), (r["jobfair_range"], r[ir_col]), xytext=(7, 7), textcoords="offset points", fontsize=9)
    ax.axhline(0.8, color=PALETTE["red"], linestyle="--", linewidth=1.2)
    ax.set_xlabel("JobFair: размах средних score")
    ax.set_ylabel("ABM baseline: средний impact_ratio")
    ax.set_title("Статический скоринг и динамическая симуляция могут расходиться", weight="bold", pad=14)
    ax.legend(title="Язык")
    save_both(fig, out_dir, "15_static_vs_dynamic")


def fig_hypothesis_summary(data: Dict[str, pd.DataFrame], out_dir: Path):
    # Автоматическая мягкая сводка без сильных утверждений.
    labels = [
        "H1\nкросс-языковая\nасимметрия",
        "H2\nмежмодельная\nдифференциация",
        "H3\nencoded / expressed\nдиссоциация",
        "H4\nABM системные\nэффекты",
    ]
    statuses = []

    # H1: есть ли различия EN/RU в expressed или ABM
    h1 = 0.5
    exp = data["expressed"]
    if len(exp):
        mcol = first_existing(exp, ["expressed_bias", "bias_score", "mean_expressed_bias", "gendered_response_pct"])
        if mcol:
            tmp = exp.copy(); tmp[mcol] = to_num(tmp[mcol])
            wide = tmp.groupby("language")[mcol].mean()
            if set(["EN", "RU"]).issubset(wide.index) and abs(wide["EN"] - wide["RU"]) > 1e-9:
                h1 = 1.0
    statuses.append(h1)

    # H2: variance between models
    h2 = 0.5
    if len(exp):
        mcol = first_existing(exp, ["expressed_bias", "bias_score", "mean_expressed_bias", "gendered_response_pct"])
        if mcol:
            tmp = exp.copy(); tmp[mcol] = to_num(tmp[mcol])
            vals = tmp.groupby("model_display")[mcol].mean()
            if len(vals.dropna()) > 1 and vals.std() > 1e-9:
                h2 = 1.0
    statuses.append(h2)

    # H3: encoded detected + expressed not uniform
    enc = data["encoded"]
    h3 = 0.5 if len(enc) else 0.0
    if len(enc) and len(exp):
        h3 = 1.0
    statuses.append(h3)

    # H4: ABM present and impact ratio below threshold somewhere
    abm = data["abm_all"]
    h4 = 0.5 if len(abm) else 0.0
    ir = first_existing(abm, ["impact_ratio", "impact", "IR"]) if len(abm) else None
    if ir:
        vals = to_num(abm[ir])
        if (vals < 0.8).any():
            h4 = 1.0
    statuses.append(h4)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    colors = [PALETTE["green"] if s == 1 else PALETTE["yellow"] if s == 0.5 else PALETTE["gray"] for s in statuses]
    ax.bar(np.arange(len(labels)), statuses, color=colors)
    ax.set_ylim(0, 1.15)
    ax.set_xticks(np.arange(len(labels)), labels)
    ax.set_yticks([0, 0.5, 1.0], ["нет данных", "частично", "поддерживается"])
    ax.set_title("Сводная визуальная проверка гипотез по доступным CSV", weight="bold", pad=14)
    for i, s in enumerate(statuses):
        txt = "поддерживается" if s == 1 else "частично" if s == 0.5 else "нет данных"
        ax.text(i, s + 0.04, txt, ha="center", va="bottom", fontsize=9)
    save_both(fig, out_dir, "16_hypotheses_summary")


def fig_limitations_recommendations(out_dir: Path):
    fig, ax = plt.subplots(figsize=(12, 6.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Из результатов к требованиям аудита LLM/SLM в найме", pad=18, weight="bold")
    left = [
        "Ограниченный набор моделей",
        "API-модели могут обновляться",
        "Синтетические резюме ≠ реальный рынок труда",
        "Чувствительность к промптам и масштабу",
        "Static scoring не равен системному эффекту",
    ]
    right = [
        "Межмодельный аудит",
        "Фиксация версии API и конфигов",
        "Калибровка на HR-данных",
        "Robustness-checks: промпты, seed, реплики",
        "Обязательная динамическая симуляция",
    ]
    draw_box(ax, (0.06, 0.73), 0.34, 0.10, "Ограничения", "#FFECEC", fontsize=12)
    draw_box(ax, (0.60, 0.73), 0.34, 0.10, "Практические следствия", "#E9F7F3", fontsize=12)
    y0 = 0.60
    for i, (l, r) in enumerate(zip(left, right)):
        y = y0 - i*0.095
        draw_box(ax, (0.05, y), 0.36, 0.06, l, "#FFF7F7", fontsize=9, lw=0.8)
        draw_box(ax, (0.59, y), 0.36, 0.06, r, "#F3FFFA", fontsize=9, lw=0.8)
        arrow(ax, (0.42, y+0.03), (0.58, y+0.03), color=PALETTE["gray"])
    save_both(fig, out_dir, "17_limitations_to_audit_recommendations")


def write_manifest(out_dir: Path):
    rows = [
        ("01_methodology_pipeline.png", "Глава 2, после раздела 2.5 или в начале 2.8", "Схема всей гибридной методологии."),
        ("02_abm_llm_architecture.png", "Глава 2, раздел 2.5", "Архитектура Mesa + orchestration layer + LLM-рекрутёр + Soft Auditor."),
        ("03_bias_levels_map.png", "Глава 1, после 1.6 или в конце 1.7", "Теоретическая схема уровней смещения."),
        ("04_model_stage_coverage.png", "Глава 3, перед 3.1", "Покрытие стадий эксперимента по моделям."),
        ("05_expressed_bias_heatmap.png", "Глава 3, раздел 3.1", "Expressed bias по моделям и языкам."),
        ("06_gendered_response_pct.png", "Глава 3, раздел 3.1", "Доля гендерно маркированных ответов."),
        ("07_encoded_alignment.png", "Глава 3, раздел 3.1", "Encoded/probing и alignment gap."),
        ("08_weat_effect_sizes.png", "Глава 3, раздел 3.2", "WEAT/SEAT effect size по моделям/языкам."),
        ("09_jobfair_score_compression.png", "Глава 3, раздел 3.3", "Насколько JobFair-средние score различаются между вариантами резюме."),
        ("10_jobfair_raw_score_distribution.png", "Приложение или раздел 3.3", "Распределения raw JobFair score; вставлять только если raw score корректно распарсился."),
        ("11_abm_impact_ratio_50_15.png", "Глава 3, раздел 3.4", "ABM 50/15: impact ratio по моделям/языкам/сценариям."),
        ("12_abm_impact_ratio_100_30.png", "Глава 3, раздел 3.5", "ABM 100/30: impact ratio по моделям/языкам/сценариям."),
        ("13_abm_scale_shift.png", "Глава 3, конец 3.5 или глава 4, раздел 4.4", "Сравнение масштабов 50/15 и 100/30."),
        ("14_intervention_comparison.png", "Глава 3, раздел 3.6 или глава 4, раздел 4.6", "Сравнение fairness-интервенций."),
        ("15_static_vs_dynamic.png", "Глава 3, раздел 3.7 или глава 4, раздел 4.5", "Статический JobFair против динамической ABM."),
        ("16_hypotheses_summary.png", "Глава 3, раздел 3.8", "Компактная визуальная сводка гипотез."),
        ("17_limitations_to_audit_recommendations.png", "Глава 4, разделы 4.9-4.10 или заключение", "Ограничения и практические следствия."),
    ]
    df = pd.DataFrame(rows, columns=["figure", "where_to_insert", "meaning"])
    df.to_csv(out_dir / "FIGURE_INSERTION_PLAN.csv", index=False, encoding="utf-8-sig")
    md = out_dir / "FIGURE_INSERTION_PLAN.md"
    with md.open("w", encoding="utf-8") as f:
        f.write("# План вставки визуализаций в ВКР\n\n")
        for fig, where, meaning in rows:
            f.write(f"- **{fig}** — {where}. {meaning}\n")
    print(f"[OK] {md}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="notebook_results", help="Папка с CSV-файлами результатов")
    parser.add_argument("--out-dir", type=str, default="vkr_figures", help="Папка для сохранения графиков")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = ensure_out(Path(args.out_dir))
    if not results_dir.exists():
        raise FileNotFoundError(f"Папка с результатами не найдена: {results_dir.resolve()}")

    data = load_all(results_dir)
    print("\nНайденные строки по стадиям:")
    for k, df in data.items():
        print(f"  {k:12s}: {len(df)}")
    print()

    # Схемы методологии
    fig_methodology_pipeline(out_dir)
    fig_abm_architecture(out_dir)
    fig_bias_levels(out_dir)

    # Сводки и результаты
    fig_model_stage_matrix(data, out_dir)
    fig_expressed_heatmap(data, out_dir)
    fig_gendered_response_bars(data, out_dir)
    fig_encoded_alignment(data, out_dir)
    fig_weat_lollipop(data, out_dir)
    fig_jobfair_compression(data, out_dir)
    fig_jobfair_raw_distributions(data, out_dir)
    abm_heatmap(data, out_dir, "abm50", "11_abm_impact_ratio_50_15", "ABM 50 кандидатов / квота 15: impact ratio")
    abm_heatmap(data, out_dir, "abm100", "12_abm_impact_ratio_100_30", "ABM 100 кандидатов / квота 30: impact ratio")
    fig_abm_scale_shift(data, out_dir)
    fig_interventions(data, out_dir)
    fig_static_vs_dynamic(data, out_dir)
    fig_hypothesis_summary(data, out_dir)
    fig_limitations_recommendations(out_dir)
    write_manifest(out_dir)

    print("\nГотово. Вставляй PNG в Word, SVG сохраняй как векторные исходники.")


if __name__ == "__main__":
    main()
