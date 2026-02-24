"""
Basicness scoring pipeline.

Produces 5 settings:
  1. Optimized thresholds + correlation weights
  2. Optimized thresholds + literature weights
  3. Optimized thresholds + uniform weights
  4. Non-optimized thresholds + literature weights
  5. Non-optimized thresholds + uniform weights

Usage:
    uv run python run_pipeline.py                      # full pipeline
    uv run python run_pipeline.py --score-only          # skip agreement evaluation
    uv run python run_pipeline.py --agreements-only     # skip scoring (uses existing CSVs)
    uv run python run_pipeline.py --optimize-thresholds # find optimal thresholds per weight scheme
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
from sklearn.metrics import cohen_kappa_score

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CLASSIFICATION_DIR = os.path.join(PROJECT_ROOT, "src", "classification")
sys.path.insert(0, CLASSIFICATION_DIR)

from analyzer import OMWBasicnessAnalyzer  # noqa: E402
from utils import get_path_from_root  # noqa: E402

EXTRACTED_CSV = get_path_from_root(
    "results/omw/annotation_culture_gold_it_copy_omw_extracted.csv"
)
OPT_SET = get_path_from_root(
    "resources/annotation/culture_annotated_gold_it_majority_voting.xlsx"
)
CORRELATION_CSV = os.path.join(
    PROJECT_ROOT, "results/correlation/unfiltered/spearman_correlations.csv"
)
ANALYSIS_PREFIX = "annotation_culture_gold_it_copy_omw_scored"

HUMAN_ANNOTATION_DIR = os.path.join(PROJECT_ROOT, "data/human_annotations")
LANGUAGES = ["en", "it", "es", "nb"]

# ---------------------------------------------------------------------------
# Thresholds  (produced by --optimize-thresholds, DE + Nelder-Mead)
# ---------------------------------------------------------------------------
OPTIMISED_THRESHOLDS_CORRELATION = [
    0.4242346508257798,
    0.4678295133052386,
    0.5532618619010025,
]
OPTIMISED_THRESHOLDS_LITERATURE = [
    0.26897556686626234,
    0.4291733757725619,
    0.6742190517116484,
]
OPTIMISED_THRESHOLDS_UNIFORM = [
    0.10648787824674222,
    0.34000380747013714,
    0.4228698193707957,
]
NON_OPTIMISED_THRESHOLDS = [0.25, 0.5, 0.75]

# ---------------------------------------------------------------------------
# Weight presets
# ---------------------------------------------------------------------------
LITERATURE_WEIGHTS: dict[str, float] = {
    "word_frequency_weight": 1.0,
    "word_length_weight": -1.0,
    "n_hyponyms_weight": 0.0,
    "n_synonyms_weight": 0.0,
    "n_senses_weight": 0.0,
    "word_in_children_res_weight": 1.0,
    "word_in_second_lang_learn_res_weight": 1.0,
    "n_syn_senses_weight": 0.0,
}

UNIFORM_WEIGHTS: dict[str, float] = {
    "word_frequency_weight": 1.0,
    "word_length_weight": -1.0,
    "n_hyponyms_weight": -1.0,
    "n_synonyms_weight": -1.0,
    "n_senses_weight": -1.0,
    "word_in_children_res_weight": 1.0,
    "word_in_second_lang_learn_res_weight": 1.0,
    "n_syn_senses_weight": -1.0,
}

WEIGHT_SCHEMES = {
    "correlation": None,  # loaded at runtime from CORRELATION_CSV
    "literature": LITERATURE_WEIGHTS,
    "uniform": UNIFORM_WEIGHTS,
}

# All 5 settings: (label, weight_key, thresholds)
SETTINGS = [
    ("S1_opt_correlation", "correlation", OPTIMISED_THRESHOLDS_CORRELATION),
    ("S2_opt_literature", "literature", OPTIMISED_THRESHOLDS_LITERATURE),
    ("S3_opt_uniform", "uniform", OPTIMISED_THRESHOLDS_UNIFORM),
    ("S4_nonopt_literature", "literature", NON_OPTIMISED_THRESHOLDS),
    ("S5_nonopt_uniform", "uniform", NON_OPTIMISED_THRESHOLDS),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_human_annotations() -> dict[str, pd.DataFrame]:
    return {
        lang: pd.read_csv(
            os.path.join(
                HUMAN_ANNOTATION_DIR,
                f"culture_annotation_datasets_annotated_human_{lang}.csv",
            )
        )
        for lang in LANGUAGES
    }


def _precompute_pairs(
    scored_df: pd.DataFrame,
    human_dfs: dict[str, pd.DataFrame],
    languages: list[str] | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Pre-align scored data with human annotations, returning
    {lang: (scores_array, human_ranks_array)} with NaNs dropped.
    This is O(n) and only done *once*, outside the optimisation loop.
    """
    if languages is None:
        languages = LANGUAGES
    pairs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for lang in languages:
        filtered = scored_df[scored_df["Language"] == lang]
        merged = pd.merge(
            filtered, human_dfs[lang], on="ili", suffixes=("_feature", "_human")
        )
        scores = merged["basicness_score_experimental"].values.astype(np.float64)
        # Aggregate 3 annotators exactly as agreements.py does (median, then round)
        human_agg = (
            merged[["annotator_1", "annotator_2", "annotator_3"]]
            .median(axis=1)
            .round()
            .astype(int)
            .values
        )
        valid = ~np.isnan(scores)
        pairs[lang] = (scores[valid], human_agg[valid])
    return pairs


def _vectorised_kappa(
    params: np.ndarray,
    pairs: dict[str, tuple[np.ndarray, np.ndarray]],
) -> float:
    """
    Negative mean linear-weighted Cohen's kappa (minimise this).

    Threshold → rank mapping is fully vectorised via np.searchsorted,
    matching the ≤ semantics of analyzer.map_score_to_rank.
    """
    t = np.sort(params)  # enforce t1 ≤ t2 ≤ t3
    total = 0.0
    for scores, truth in pairs.values():
        predicted = np.searchsorted(t, scores, side="left") + 1
        total += cohen_kappa_score(truth, predicted, weights="linear")
    return -total / len(pairs)


def _print_agreements(name: str, scored_df: pd.DataFrame, human_dfs: dict) -> None:
    from src.analysis.agreements import calculate_feature_human_agreement

    print(f"\n  --- {name} ---")
    for lang, human_df in human_dfs.items():
        result = calculate_feature_human_agreement(
            scored_df, human_df, f"feature_human_{lang}", lang
        )
        o = result["overall"]
        print(
            f"    {lang}: κ={o['cohen_kappa']:.4f}  agree={o['percent_agreement']:.4f}"
            f"  bin={o.get('binary_agreement', float('nan')):.4f}"
            f"  MSE={o['mse']:.4f}"
        )


# ---------------------------------------------------------------------------
# Threshold optimisation
# ---------------------------------------------------------------------------
def optimize_thresholds(
    scored_df: pd.DataFrame,
    human_dfs: dict[str, pd.DataFrame],
    languages: list[str] | None = None,
) -> tuple[list[float], float, dict[str, float]]:
    """
    Find thresholds [t1, t2, t3] that maximise mean linear-weighted Cohen's κ
    across *languages* for the given scored DataFrame.

    Two-stage approach
    ──────────────────
    1. **Global**: ``scipy.optimize.differential_evolution`` with population-
       based stochastic search — no grid discretisation, explores the full
       continuous [score_min, score_max]³ space in parallel.
    2. **Local polish**: adaptive Nelder-Mead for fine-grained convergence.

    The objective is evaluated in O(n) per call (vectorised rank via
    ``np.searchsorted``) instead of the O(n·k) DataFrame re-ranking in the
    original grid search. Combined with DE's ~10 k evaluations (vs. ~100 k+
    cubic grid iterations) this gives **~100-1000× wall-clock speedup** while
    finding a *strictly better* optimum.
    """
    if languages is None:
        languages = LANGUAGES

    pairs = _precompute_pairs(scored_df, human_dfs, languages)
    n_total = sum(len(s) for s, _ in pairs.values())
    print(
        f"  Precomputed {n_total} score-truth pairs across {len(languages)} languages"
    )

    # Bounds from actual score range
    score_min = min(s.min() for s, _ in pairs.values())
    score_max = max(s.max() for s, _ in pairs.values())
    bounds = [(score_min, score_max)] * 3

    # Stage 1 — global search
    t0 = time.perf_counter()
    print(
        f"  Stage 1: differential evolution on [{score_min:.4f}, {score_max:.4f}]³ ..."
    )
    de_result = differential_evolution(
        _vectorised_kappa,
        bounds,
        args=(pairs,),
        seed=42,
        maxiter=1000,
        tol=1e-9,
        mutation=(0.5, 1.5),
        recombination=0.9,
        polish=False,
        updating="immediate",
    )
    de_kappa = -de_result.fun
    print(
        f"    DE converged={de_result.success}  κ̄={de_kappa:.6f}  ({time.perf_counter() - t0:.1f}s)"
    )

    # Stage 2 — local polish
    print("  Stage 2: Nelder-Mead polish ...")
    nm_result = minimize(
        _vectorised_kappa,
        np.sort(de_result.x),
        args=(pairs,),
        method="Nelder-Mead",
        options={"xatol": 1e-12, "fatol": 1e-12, "maxiter": 50_000, "adaptive": True},
    )
    best_thresholds = sorted(nm_result.x.tolist())
    best_kappa = -nm_result.fun
    elapsed = time.perf_counter() - t0
    print(f"    Polished κ̄={best_kappa:.6f}  ({elapsed:.1f}s total)")

    # Per-language breakdown
    t_arr = np.array(best_thresholds)
    lang_kappas = {}
    for lang in languages:
        scores, truth = pairs[lang]
        predicted = np.searchsorted(t_arr, scores, side="left") + 1
        lang_kappas[lang] = cohen_kappa_score(truth, predicted, weights="linear")

    return best_thresholds, best_kappa, lang_kappas


def run_optimization() -> None:
    """Score with each weight scheme, then find optimal thresholds for each."""
    print("Loading extracted synset data ...")
    df = pd.read_csv(EXTRACTED_CSV)
    print(f"  {len(df)} rows\n")

    analyzer = OMWBasicnessAnalyzer(df, opt_set_path=OPT_SET)
    human_dfs = _load_human_annotations()
    lang_weights = OMWBasicnessAnalyzer.load_correlation_weights(CORRELATION_CSV)

    results: dict[str, tuple[list[float], float, dict[str, float]]] = {}

    for scheme_name, fixed_weights in WEIGHT_SCHEMES.items():
        print(f"\n{'=' * 60}")
        print(f"  Optimising thresholds for: {scheme_name} weights")
        print(f"{'=' * 60}")

        # Configure weights
        if scheme_name == "correlation":
            analyzer.set_language_specific_weights(lang_weights)
        else:
            analyzer.set_weights(fixed_weights)
            analyzer.set_language_specific_weights(None)

        # Score with dummy thresholds (thresholds only affect rank, not score)
        scored_df = analyzer.analyze_lang_syn_group(
            thresholds=NON_OPTIMISED_THRESHOLDS,
        ).reset_index(drop=True)

        # Optimise
        best_t, best_k, lang_k = optimize_thresholds(scored_df, human_dfs)
        results[scheme_name] = (best_t, best_k, lang_k)

        print(f"\n  Thresholds : {best_t}")
        print(f"  Mean κ     : {best_k:.6f}")
        for lang, k in lang_k.items():
            print(f"    {lang}: κ={k:.6f}")

    # Print copy-pasteable summary
    print(f"\n{'=' * 60}")
    print("  COPY-PASTE SUMMARY")
    print(f"{'=' * 60}")
    for scheme_name, (thresholds, kappa, lang_k) in results.items():
        print(f"\n# {scheme_name} weights  (mean κ = {kappa:.6f})")
        print(f"OPTIMISED_THRESHOLDS_{scheme_name.upper()} = {thresholds}")


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def run_scoring() -> dict[str, pd.DataFrame]:
    print("Loading extracted synset data...")
    df = pd.read_csv(EXTRACTED_CSV)
    print(f"  {len(df)} rows\n")

    analyzer = OMWBasicnessAnalyzer(df, opt_set_path=OPT_SET)

    # Pre-load correlation weights (needed for Setting 1)
    lang_weights = OMWBasicnessAnalyzer.load_correlation_weights(CORRELATION_CSV)

    scored: dict[str, pd.DataFrame] = {}

    for label, weight_key, thresholds in SETTINGS:
        thresh_label = (
            "non-optimised" if thresholds is NON_OPTIMISED_THRESHOLDS else "optimised"
        )
        print(f"=== {label}: {weight_key} weights, {thresh_label} thresholds ===")

        if weight_key == "correlation":
            analyzer.set_language_specific_weights(lang_weights)
        else:
            weights = (
                LITERATURE_WEIGHTS if weight_key == "literature" else UNIFORM_WEIGHTS
            )
            analyzer.set_weights(weights)
            analyzer.set_language_specific_weights(None)

        scored[label] = analyzer.analyze_lang_syn_group(
            word=f"{ANALYSIS_PREFIX}_{label}",
            thresholds=thresholds,
        ).reset_index(drop=True)
        print(f"  → {len(scored[label])} rows scored\n")

    return scored


# ---------------------------------------------------------------------------
# Agreement evaluation
# ---------------------------------------------------------------------------
def run_agreements(scored: dict[str, pd.DataFrame] | None = None) -> None:
    human_dfs = _load_human_annotations()

    if scored is None:
        analysis_dir = get_path_from_root("analysis")
        scored = {}
        for label, _, _ in SETTINGS:
            path = os.path.join(analysis_dir, f"{ANALYSIS_PREFIX}_{label}_lang_syn.csv")
            scored[label] = pd.read_csv(path)

    print("\n=== Agreement evaluation ===")
    for label, df in scored.items():
        _print_agreements(label, df, human_dfs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Basicness scoring pipeline")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--score-only", action="store_true", help="Run scoring only")
    group.add_argument(
        "--agreements-only",
        action="store_true",
        help="Evaluate agreements from existing CSVs",
    )
    group.add_argument(
        "--optimize-thresholds",
        action="store_true",
        help="Find optimal thresholds per weight scheme (DE + Nelder-Mead)",
    )
    args = parser.parse_args()

    if args.optimize_thresholds:
        run_optimization()
    elif args.agreements_only:
        run_agreements()
    elif args.score_only:
        run_scoring()
    else:
        scored = run_scoring()
        run_agreements(scored)

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
