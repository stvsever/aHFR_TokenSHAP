# run_demo.py
from __future__ import annotations

import csv
import inspect
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, Optional

import torch

import ig_attribution as ig
import aHFR_TokenSHAP as hfr
import LLM_select as llm_sel


# -----------------------
# CONFIG
# -----------------------
PHENOTYPE = "DEPRESSION"
MODEL_NAME = "Qwen/Qwen2-7B-Instruct"
N_PROFILES = 100  # MUST be even if you want 50/50 CASE vs CONTROL ; set on 2 for testing purposes

IG_STEPS = 200

# HFR-TokenSHAP (Monte Carlo Shapley)
SHAPLEY_K = 5
SHAPLEY_RUNS = 3

# LLM-Select
LLM_SELECT_MODEL = "gpt-5-nano"  # NOTE: change to better model during 'actual' runs
LLM_SELECT_REPEATS = 3  # NOTE: increase during 'actual' runs
LLM_SELECT_TEMPERATURE = 0.7

PRINT_PROMPT_PER_SUBJECT = False
PRINT_LLM_GENERATION_PER_SUBJECT = False

# Show detailed Shapley permutation contributions? (very verbose + slow)
PRINT_SHAPLEY_VERBOSE_INTERNAL = True

# Show progress lines frequently
PROGRESS_EVERY_SEC = 2.0

EPS = 1e-9

# OUTPUT DIRS
# You can override with env var HFR_DEMO_RESULTS_DIR
DEFAULT_RESULTS_DIR = (
    "/Users/stijnvanseveren/PythonProjects/HFR_TokenSHAP/"
    "feature_importance_estimation/demonstration_results"
)
BASE_RESULTS_DIR = Path(os.environ.get("HFR_DEMO_RESULTS_DIR", DEFAULT_RESULTS_DIR)).expanduser()
TABLES_DIR = BASE_RESULTS_DIR / "tables"
VISUALS_DIR = BASE_RESULTS_DIR / "visuals"


# -----------------------
# HIERARCHICAL FEATURE SPACE
# -----------------------
# We create:
#   - 10 parent nodes (2 relevant + 8 distractor)
#   - 30 leaves (3 per parent)
#
# Relevant parents (each has 3 concrete sub-features):
RELEVANT_PARENTS = ["childhood_trauma", "nutrition"]

CHILDHOOD_TRAUMA_LEAVES = [
    "childhood_trauma_physical_abuse",
    "childhood_trauma_sexual_abuse",
    "childhood_trauma_emotional_neglect",
]

NUTRITION_LEAVES = [
    "nutrition_omega3_intake",
    "nutrition_vitamin_d_status",
    "nutrition_ultra_processed_food",
]

# Distractor parents (each has 3 concrete "sub-words" / components):
DISTRACTOR_PARENTS = [
    "colors",
    "chicken_soup",
    "computer",
    "floor",
    "window",
    "pencil",
    "bicycle",
    "cloud",
]

COLORS_LEAVES = [
    "colors_blue",
    "colors_red",
    "colors_green",
]
CHICKEN_SOUP_LEAVES = [
    "chicken_soup_bowl",
    "chicken_soup_recipe",
    "chicken_soup_smell",
]
COMPUTER_LEAVES = [
    "computer_keyboard",
    "computer_monitor",
    "computer_mouse",
]
FLOOR_LEAVES = [
    "floor_tile",
    "floor_carpet",
    "floor_wood",
]
WINDOW_LEAVES = [
    "window_glass",
    "window_frame",
    "window_curtains",
]
PENCIL_LEAVES = [
    "pencil_lead",
    "pencil_eraser",
    "pencil_shavings",
]
BICYCLE_LEAVES = [
    "bicycle_wheel",
    "bicycle_chain",
    "bicycle_handlebar",
]
CLOUD_LEAVES = [
    "cloud_cumulus",
    "cloud_rain",
    "cloud_shadow",
]

PARENT_TO_LEAVES: Dict[str, List[str]] = {
    "childhood_trauma": CHILDHOOD_TRAUMA_LEAVES,
    "nutrition": NUTRITION_LEAVES,
    "colors": COLORS_LEAVES,
    "chicken_soup": CHICKEN_SOUP_LEAVES,
    "computer": COMPUTER_LEAVES,
    "floor": FLOOR_LEAVES,
    "window": WINDOW_LEAVES,
    "pencil": PENCIL_LEAVES,
    "bicycle": BICYCLE_LEAVES,
    "cloud": CLOUD_LEAVES,
}

PARENT_NODES = RELEVANT_PARENTS + DISTRACTOR_PARENTS

# Leaves (atomic variables)
LEAF_IDS: List[str] = []
for p in PARENT_NODES:
    LEAF_IDS.extend(PARENT_TO_LEAVES[p])

# Leaf -> parent lookup
PARENT_OF_LEAF: Dict[str, str] = {lf: p for p, lfs in PARENT_TO_LEAVES.items() for lf in lfs}


def _surface_text(leaf_id: str) -> str:
    parent = PARENT_OF_LEAF[leaf_id]
    sub = leaf_id[len(parent) + 1 :] if leaf_id.startswith(parent + "_") else leaf_id
    parent_txt = parent.replace("_", " ")
    sub_txt = sub.replace("_", " ")
    if parent in RELEVANT_PARENTS:
        return f"{parent_txt}: {sub_txt}"
    return f"{parent_txt} — {sub_txt}"


SURFACE_TEXT = {lf: _surface_text(lf) for lf in LEAF_IDS}

# Ground-truth weights (leaf-level):
# - Relevant leaves ~ 1.0
# - Distractor leaves abs(weight) <= 0.2
TRUE_W: Dict[str, float] = {
    # childhood trauma (relevant)
    "childhood_trauma_physical_abuse": 1.0,
    "childhood_trauma_sexual_abuse": 1.0,
    "childhood_trauma_emotional_neglect": 1.0,
    # nutrition (relevant)
    "nutrition_omega3_intake": 1.0,
    "nutrition_vitamin_d_status": 1.0,
    "nutrition_ultra_processed_food": 1.0,
    # colors (distractor)
    "colors_blue": 0.00,
    "colors_red": 0.03,
    "colors_green": -0.02,
    # chicken_soup (distractor)
    "chicken_soup_bowl": 0.05,
    "chicken_soup_recipe": 0.04,
    "chicken_soup_smell": -0.01,
    # computer (distractor)
    "computer_keyboard": -0.08,
    "computer_monitor": 0.02,
    "computer_mouse": -0.04,
    # floor (distractor)
    "floor_tile": 0.02,
    "floor_carpet": 0.01,
    "floor_wood": -0.02,
    # window (distractor)
    "window_glass": 0.00,
    "window_frame": 0.06,
    "window_curtains": -0.01,
    # pencil (distractor)
    "pencil_lead": -0.12,
    "pencil_eraser": -0.05,
    "pencil_shavings": 0.02,
    # bicycle (distractor)
    "bicycle_wheel": 0.07,
    "bicycle_chain": -0.03,
    "bicycle_handlebar": 0.01,
    # cloud (distractor)
    "cloud_cumulus": -0.03,
    "cloud_rain": 0.02,
    "cloud_shadow": -0.01,
}
assert set(TRUE_W.keys()) == set(LEAF_IDS), "TRUE_W must contain exactly all leaves."


# -----------------------
# HFR HIERARCHY
# -----------------------
HIERARCHY_ROOT = "ALL_FEATURES"
HIERARCHY_CHILDREN: Dict[str, List[str]] = {HIERARCHY_ROOT: PARENT_NODES}
for parent in PARENT_NODES:
    HIERARCHY_CHILDREN[parent] = PARENT_TO_LEAVES[parent]

# Shapley at LEAF level (30 players), aggregate to parents (10).
# Passing hierarchy_children+cut_nodes triggers fixed-cut mode (leaf cut) in your HFR_TokenSHAP.
SHAPLEY_CUT_NODES = LEAF_IDS[:]  # tree cut at leaf level


# -----------------------
# UTIL
# -----------------------
def fmt_time(sec: float) -> str:
    if sec == float("inf"):
        return "ETA: ?"
    if sec < 60:
        return f"{sec:5.1f}s"
    m = int(sec // 60)
    s = sec - 60 * m
    if m < 60:
        return f"{m:02d}:{s:04.1f}"
    h = int(m // 60)
    m = m - 60 * h
    return f"{h:d}:{m:02d}:{s:04.1f}"


def normalize_abs_vector(vec: Dict[str, float]) -> Dict[str, float]:
    denom = sum(abs(v) for v in vec.values()) + EPS
    return {k: (abs(v) / denom) for k, v in vec.items()}


def safe_call(fn, **kwargs):
    """
    Call fn(**kwargs), but drop kwargs it doesn't accept.
    Keeps this script compatible if your modules differ slightly.
    """
    sig = inspect.signature(fn)
    ok = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return fn(**ok)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def mean_var_from_samples(x: List[float]) -> Tuple[float, float]:
    if not x:
        return 0.0, 0.0
    m = sum(x) / float(len(x))
    v = sum((xi - m) ** 2 for xi in x) / float(len(x))  # population variance
    return m, v


def _group_type(parent_id: str) -> str:
    return "relevant" if parent_id in RELEVANT_PARENTS else "distractor"


def parent_agg(scores_leaf: Dict[str, float]) -> Dict[str, float]:
    return {p: sum(scores_leaf[lf] for lf in PARENT_TO_LEAVES[p]) for p in PARENT_NODES}


def _eta(elapsed: float, done: int, total: int) -> float:
    if done <= 0:
        return float("inf")
    rate = elapsed / done
    return rate * max(total - done, 0)


def shapley_with_repeats_and_store_compat(
    *,
    score_fn,
    feature_ids: List[str],
    K: int,
    runs: int,
    seed: int,
    verbose: bool,
    progress_cb,
    # passthrough kwargs to HFR
    hierarchy_children=None,
    cut_nodes=None,
    root=None,
    leaf_ids=None,
    groups=None,
    adaptive_search=True,
    selection_temperature=0.0,
):
    """
    Compatibility layer:
      - If HFR_TokenSHAP exposes shapley_with_repeats_and_store, use it.
      - Else: implement equivalent behavior by repeated calls to monte_carlo_hfr_tokenshap
        and return (mean_phi, std_phi, all_phis).

    This preserves the rest of run_demo.py logic unchanged.
    """
    fn = getattr(hfr, "shapley_with_repeats_and_store", None)
    if callable(fn):
        out = safe_call(
            fn,
            score_fn=score_fn,
            feature_ids=feature_ids,
            hierarchy_children=hierarchy_children,
            root=root,
            leaf_ids=leaf_ids,
            cut_nodes=cut_nodes,
            groups=groups,
            K=K,
            runs=runs,
            seed=seed,
            verbose=verbose,
            progress_cb=progress_cb,
            adaptive_search=adaptive_search,
            selection_temperature=selection_temperature,
        )
        return out

    # Manual fallback
    all_phis: List[Dict[str, float]] = []
    t0 = time.perf_counter()

    for r in range(runs):

        def perm_progress(perm_done: int, K_eff: int, elapsed_perm: float, eta_perm: float):
            if progress_cb is None:
                return
            elapsed_total = time.perf_counter() - t0
            total_work = runs * K
            done_work = r * K + perm_done
            eta_total = _eta(elapsed_total, done_work, total_work)
            progress_cb(r + 1, runs, perm_done, K_eff, elapsed_total, eta_total)

        phi_r = safe_call(
            hfr.monte_carlo_hfr_tokenshap,
            score_fn=score_fn,
            feature_ids=feature_ids,
            hierarchy_children=hierarchy_children,
            root=root,
            leaf_ids=leaf_ids,
            cut_nodes=cut_nodes,
            groups=groups,
            K=K,
            seed=seed + 9973 * r,
            verbose=verbose,
            progress_cb=perm_progress,
            adaptive_search=adaptive_search,
            selection_temperature=selection_temperature,
            # NOTE: if your HFR exposes mixed_players knobs, safe_call will pass them only if provided.
        )
        all_phis.append(phi_r)

    keys = list(all_phis[0].keys())
    mean_phi: Dict[str, float] = {k: 0.0 for k in keys}
    std_phi: Dict[str, float] = {k: 0.0 for k in keys}

    for k in keys:
        vals = [d[k] for d in all_phis]
        m = sum(vals) / len(vals)
        mean_phi[k] = m
        if len(vals) == 1:
            std_phi[k] = 0.0
        else:
            var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)  # sample variance
            std_phi[k] = var**0.5

    return mean_phi, std_phi, all_phis


# -----------------------
# DATA + GROUND TRUTH
# -----------------------
def sample_one_profile(rng: random.Random) -> Dict[str, int]:
    """Leaf values in {-1,+1}."""
    return {fid: rng.choice([-1, +1]) for fid in LEAF_IDS}


def ground_truth_logit(profile: Dict[str, int]) -> float:
    return sum(TRUE_W[f] * float(profile[f]) for f in LEAF_IDS)


def ground_truth_label(profile: Dict[str, int]) -> str:
    return f"{PHENOTYPE} CASE" if ground_truth_logit(profile) > 0 else f"{PHENOTYPE} CONTROL"


def make_balanced_profiles(n: int = 20, seed: int = 1, max_draws: int = 500_000) -> List[Dict[str, int]]:
    """Return EXACTLY equal CASE vs CONTROL (50/50)."""
    if n % 2 != 0:
        raise ValueError("N_PROFILES must be even to enforce exact 50/50 CASE vs CONTROL.")

    target = n // 2
    rng = random.Random(seed)

    cases: List[Dict[str, int]] = []
    ctrls: List[Dict[str, int]] = []

    draws = 0
    while (len(cases) < target) or (len(ctrls) < target):
        draws += 1
        if draws > max_draws:
            raise RuntimeError(
                f"Failed to sample balanced dataset within max_draws={max_draws}. "
                f"Current counts: CASE={len(cases)}, CONTROL={len(ctrls)}."
            )

        p = sample_one_profile(rng)
        lab = ground_truth_label(p)
        if lab.endswith("CASE"):
            if len(cases) < target:
                cases.append(p)
        else:
            if len(ctrls) < target:
                ctrls.append(p)

    profiles = cases + ctrls
    rng.shuffle(profiles)
    print(f"[DATA] Balanced dataset built: CASE={len(cases)} CONTROL={len(ctrls)} (draws={draws})")
    return profiles


# -----------------------
# PROMPT BUILDING
# -----------------------
def build_prompt_and_spans(profile: Dict[str, int], active: Set[str]) -> Tuple[str, Dict[str, Tuple[int, int]]]:
    """
    Template-preserving ablation:
      - inactive leaves keep line structure but set value=+0.00 and vote=UNKNOWN
    """
    header = (
        f"You are a classifier. Output exactly one label:\n"
        f"  '{PHENOTYPE} CASE'  or  '{PHENOTYPE} CONTROL'\n\n"
        f"Task:\n"
        f"  Compute a risk score as: score = sum(weight * value) across the feature lines below.\n"
        f"  Then output '{PHENOTYPE} CASE' if score > 0 else '{PHENOTYPE} CONTROL'.\n\n"
        f"Important:\n"
        f"  Only TWO parent groups are clinically relevant:\n"
        f"    1) childhood trauma (3 sub-features)\n"
        f"    2) nutrition (3 sub-features)\n"
        f"  These have weight ≈ 1.0 per sub-feature.\n"
        f"  The other EIGHT parent groups are distractors with tiny weights (|weight| ≤ 0.2), so they should barely matter.\n\n"
        f"Interpretation of values:\n"
        f"  value=+1.00 suggests risk, value=-1.00 suggests protection.\n"
        f"  value=+0.00 and vote=UNKNOWN means missing/ablated.\n\n"
        f"Features (leaf nodes):\n"
    )

    lines: List[str] = []
    spans: Dict[str, Tuple[int, int]] = {}
    cursor = len(header)

    for leaf_id in LEAF_IDS:
        parent = PARENT_OF_LEAF[leaf_id]

        if leaf_id in active:
            v = profile[leaf_id]
            val_str = f"{float(v):+.2f}"
            vote = "CASE" if v > 0 else "CONTROL"
        else:
            val_str = f"{0.0:+.2f}"
            vote = "UNKNOWN"

        w = TRUE_W[leaf_id]
        gtype = _group_type(parent)
        text = SURFACE_TEXT.get(leaf_id, leaf_id.replace("_", " "))

        line = (
            f"@@ LEAF_ID={leaf_id} | parent={parent} | group_type={gtype} | "
            f"text='{text}' | weight={w:+.2f} | value={val_str} | vote={vote} @@\n"
        )

        start = cursor
        end = cursor + len(line)
        spans[leaf_id] = (start, end)
        lines.append(line)
        cursor = end

    # Important: we end with "Label: DEPRESSION" so the next token(s) are " CASE"/" CONTROL"
    footer = f"\nAnswer with the label only.\nLabel: {PHENOTYPE}"
    prompt = header + "".join(lines) + footer
    return prompt, spans


def build_prompt_only(profile: Dict[str, int], active: Set[str]) -> str:
    prompt, _ = build_prompt_and_spans(profile, active)
    return prompt


# -----------------------
# MAIN
# -----------------------
def main():
    global BASE_RESULTS_DIR, TABLES_DIR, VISUALS_DIR

    # Prefer deterministic-ish behavior in demo
    torch.manual_seed(0)
    random.seed(0)

    # Ensure dirs (fallback to local if the absolute path fails)
    try:
        ensure_dir(TABLES_DIR)
        ensure_dir(VISUALS_DIR)
    except Exception as e:
        print(f"[WARN] Could not create results dirs under {BASE_RESULTS_DIR}: {e}")
        local_base = Path("./demonstration_results").resolve()
        print(f"[WARN] Falling back to local: {local_base}")

        BASE_RESULTS_DIR = local_base
        TABLES_DIR = BASE_RESULTS_DIR / "tables"
        VISUALS_DIR = BASE_RESULTS_DIR / "visuals"

        ensure_dir(TABLES_DIR)
        ensure_dir(VISUALS_DIR)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("ig_attribution loaded from:", getattr(ig, "__file__", "<unknown>"))
    print("HFR_TokenSHAP loaded from:", getattr(hfr, "__file__", "<unknown>"))
    print("LLM_select loaded from:", getattr(llm_sel, "__file__", "<unknown>"))

    # ---------------------------------------------------------------------
    # COMPATIBILITY PATCH (important):
    # Some older HFR_TokenSHAP versions exposed build_groups_from_tree_cut().
    # This demo only needs leaf-level players for sanity checks/printing.
    #
    # The actual Shapley call below uses the fixed-cut interface
    # (hierarchy_children + cut_nodes), which your provided HFR_TokenSHAP supports.
    # ---------------------------------------------------------------------
    groups: Optional[Dict[str, List[str]]] = None
    build_groups_fn = getattr(hfr, "build_groups_from_tree_cut", None)
    if callable(build_groups_fn):
        try:
            groups = safe_call(
                build_groups_fn,
                hierarchy_children=HIERARCHY_CHILDREN,
                leaf_ids=LEAF_IDS,
                cut_nodes=SHAPLEY_CUT_NODES,
                root=HIERARCHY_ROOT,
                require_nonempty_groups=True,
            )
        except Exception as e:
            print(f"[WARN] hfr.build_groups_from_tree_cut failed ({e}); using trivial leaf groups.")
            groups = None

    if not isinstance(groups, dict) or not groups:
        groups = {lf: [lf] for lf in LEAF_IDS}

    player_leaf_ids = list(groups.keys())
    if set(player_leaf_ids) != set(LEAF_IDS):
        raise RuntimeError(
            "Leaf-level player groups mismatch. Expected players == LEAF_IDS. "
            f"Got {len(player_leaf_ids)} players."
        )

    print("\n[HFR] Player set: LEAF cut (30 players). Parents will be aggregated post-hoc (10).")

    if N_PROFILES % 2 != 0:
        raise ValueError("Set N_PROFILES to an even number to enforce exact 50/50 CASE/CONTROL.")

    # Load model
    model, tokenizer, device = safe_call(ig.load_model, model_name=MODEL_NAME, prefer_mps=True)
    labels = safe_call(ig.prepare_label_tokens, tokenizer=tokenizer, case_str=" CASE", control_str=" CONTROL")

    print(f"\nModel: {MODEL_NAME} | device: {device}")
    print(f"CASE token ids: {labels.case_ids} | CONTROL token ids: {labels.control_ids}")
    if not getattr(labels, "single_token_labels", True):
        print(
            "NOTE: tokenizer makes labels multi-token.\n"
            "If your ig_attribution implementation assumes single-token labels,\n"
            "update ig_attribution accordingly.\n"
        )

    profiles = make_balanced_profiles(n=N_PROFILES, seed=1)

    # -----------------------
    # LLM-SELECT (global prior)
    # -----------------------
    print("\n" + "=" * 100)
    print("[LLM-SELECT] Running feature-name-only importance prior...")
    llm_leaf_runs: List[Dict[str, Any]] = []
    llm_parent_runs: List[Dict[str, Any]] = []
    try:
        out = safe_call(
            llm_sel.get_llm_select_scores,
            phenotype=PHENOTYPE,
            leaf_ids=LEAF_IDS,
            parent_ids=PARENT_NODES,
            repeats=LLM_SELECT_REPEATS,
            model=LLM_SELECT_MODEL,
            # temperature=LLM_SELECT_TEMPERATURE,
        )
        # Expected: (llm_leaf_runs, llm_parent_runs)
        if isinstance(out, tuple) and len(out) == 2:
            llm_leaf_runs, llm_parent_runs = out
        else:
            raise RuntimeError("LLM_select.get_llm_select_scores returned unexpected format.")
    except Exception as e:
        print(f"[WARN] LLM-Select failed (continuing without it): {e}")
        llm_leaf_runs, llm_parent_runs = [], []

    # -----------------------
    # ROW TABLES (long)
    # -----------------------
    subjects_rows: List[Dict[str, Any]] = []
    leaves_rows: List[Dict[str, Any]] = []

    # results1/2/3
    results1_hfr_rows: List[Dict[str, Any]] = []
    results1_hfr_runs_rows: List[Dict[str, Any]] = []
    results2_ig_rows: List[Dict[str, Any]] = []
    results3_llm_rows: List[Dict[str, Any]] = []

    # Summary: collect normalized abs per feature
    norm_samples: Dict[Tuple[str, str, str], List[float]] = {}

    def _push_sample(method: str, level: str, fid: str, val: float) -> None:
        norm_samples.setdefault((method, level, fid), []).append(float(val))

    subj_times: List[float] = []
    t_start_all = time.perf_counter()

    for i, prof in enumerate(profiles, start=1):
        t_subj0 = time.perf_counter()

        subject_id = i
        active_all = set(LEAF_IDS)
        prompt, spans = build_prompt_and_spans(prof, active=active_all)
        baseline_prompt = build_prompt_only(prof, active=set())  # all ablated (template-preserving)

        gt_log = ground_truth_logit(prof)
        gt_lab = ground_truth_label(prof)

        print("\n" + "=" * 100)
        print(f"SUBJECT {i:02d}/{N_PROFILES}")

        if PRINT_PROMPT_PER_SUBJECT:
            print("\n--- PROMPT ---")
            print(prompt)

        # score
        t0 = time.perf_counter()
        s, lp_case, lp_control = safe_call(
            ig.score_case_control,
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            labels=labels,
            device=device,
        )
        t_score = time.perf_counter() - t0
        pred_lab = f"{PHENOTYPE} CASE" if s > 0 else f"{PHENOTYPE} CONTROL"

        # generation (optional)
        if PRINT_LLM_GENERATION_PER_SUBJECT:
            t0 = time.perf_counter()
            gen = safe_call(
                ig.generate_next_token,
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                device=device,
                max_new_tokens=1,
            )
            t_gen = time.perf_counter() - t0
            print("\n--- LLM OUTPUT (next token, greedy) ---")
            print(repr(gen))
        else:
            t_gen = 0.0

        print("\n--- SCORES ---")
        print(f"GroundTruth: label={gt_lab}  logit={gt_log:+.3f}")
        print(f"ModelScore : pred={pred_lab}  s={s:+.4f}  (lp_case={lp_case:+.4f}, lp_ctrl={lp_control:+.4f})")
        print(f"Timing: score={t_score:.2f}s | gen={t_gen:.2f}s")

        # store per-leaf ground truth & values
        for leaf_id in LEAF_IDS:
            parent = PARENT_OF_LEAF[leaf_id]
            leaves_rows.append(
                dict(
                    run_id=run_id,
                    subject_id=subject_id,
                    leaf_id=leaf_id,
                    parent_id=parent,
                    group_type=_group_type(parent),
                    value=int(prof[leaf_id]),
                    true_weight=float(TRUE_W[leaf_id]),
                    true_contribution=float(TRUE_W[leaf_id]) * float(prof[leaf_id]),
                )
            )

        # -----------------------
        # IG (leaf-level), then aggregate to parents
        # -----------------------
        print("\n--- IG RUN ---")
        last_print = [0.0]
        ig_t0 = time.perf_counter()

        def ig_progress(step_idx: int, steps: int, elapsed: float, eta: float):
            now = time.perf_counter()
            if (now - last_print[0]) >= PROGRESS_EVERY_SEC or step_idx == steps:
                last_print[0] = now
                print(f"  IG step {step_idx:02d}/{steps} | elapsed {fmt_time(elapsed)} | eta {fmt_time(eta)}")

        ig_leaf_scores: Dict[str, float] = safe_call(
            ig.integrated_gradients_feature_importance,
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            baseline_prompt=baseline_prompt,
            feature_spans=spans,
            labels=labels,
            device=device,
            steps=IG_STEPS,
            baseline_mode="mask",
            span_mode="value",
            progress_cb=ig_progress,
        )
        if ig_leaf_scores is None:
            raise RuntimeError("IG did not return scores. Check ig_attribution.integrated_gradients_feature_importance.")

        t_ig = time.perf_counter() - ig_t0
        print(f"  IG done in {fmt_time(t_ig)}")

        ig_leaf_norm = normalize_abs_vector(ig_leaf_scores)
        ig_parent_scores = parent_agg(ig_leaf_scores)
        ig_parent_norm = normalize_abs_vector(ig_parent_scores)

        # Store IG: leaves
        for leaf_id in LEAF_IDS:
            parent = PARENT_OF_LEAF[leaf_id]
            results2_ig_rows.append(
                dict(
                    run_id=run_id,
                    subject_id=subject_id,
                    feature_level="leaf",
                    feature_id=leaf_id,
                    parent_id=parent,
                    group_type=_group_type(parent),
                    importance=float(ig_leaf_scores[leaf_id]),
                    abs_importance=abs(float(ig_leaf_scores[leaf_id])),
                    norm_abs=float(ig_leaf_norm[leaf_id]),
                )
            )
            _push_sample("IG", "leaf", leaf_id, ig_leaf_norm[leaf_id])

        # Store IG: parents
        for pid in PARENT_NODES:
            results2_ig_rows.append(
                dict(
                    run_id=run_id,
                    subject_id=subject_id,
                    feature_level="parent",
                    feature_id=pid,
                    parent_id="",
                    group_type=_group_type(pid),
                    importance=float(ig_parent_scores[pid]),
                    abs_importance=abs(float(ig_parent_scores[pid])),
                    norm_abs=float(ig_parent_norm[pid]),
                )
            )
            _push_sample("IG", "parent", pid, ig_parent_norm[pid])

        # -----------------------
        # HFR-TokenSHAP (Shapley MC, leaf players), then aggregate to parents
        # IMPORTANT: adaptive_search is explicitly OFF here.
        #
        # Compatibility note:
        #   - Your provided HFR_TokenSHAP supports fixed-cut mode via (hierarchy_children + cut_nodes).
        #   - Newer revisions may not ship shapley_with_repeats_and_store, so we wrap it.
        # -----------------------
        #print("\n--- HFR-TokenSHAP RUN (Shapley MC, leaf players; adaptive_search=False) ---")
        last_print = [0.0]
        shap_t0 = time.perf_counter()

        def score_fn(active_leaves: Set[str]) -> float:
            ptxt = build_prompt_only(prof, active_leaves)
            s_local, _, _ = safe_call(
                ig.score_case_control,
                model=model,
                tokenizer=tokenizer,
                prompt=ptxt,
                labels=labels,
                device=device,
            )
            return float(s_local)

        def shap_progress(run_idx: int, runs: int, perm_done: int, K: int, elapsed_total: float, eta_total: float):
            now = time.perf_counter()
            if (now - last_print[0]) >= PROGRESS_EVERY_SEC or (run_idx == runs and perm_done == K):
                last_print[0] = now
                total_perm = runs * K
                done_perm = (run_idx - 1) * K + perm_done
                print(
                    f"  Shapley progress: run {run_idx}/{runs}, perm {perm_done}/{K} "
                    f"({done_perm}/{total_perm}) | elapsed {fmt_time(elapsed_total)} | eta {fmt_time(eta_total)}"
                )

        shap_mean, shap_std, shap_per_run = shapley_with_repeats_and_store_compat(
            score_fn=score_fn,
            feature_ids=LEAF_IDS,
            groups=None,
            K=SHAPLEY_K,
            runs=SHAPLEY_RUNS,
            seed=1000 + i,
            verbose=PRINT_SHAPLEY_VERBOSE_INTERNAL,
            progress_cb=shap_progress,
            hierarchy_children=HIERARCHY_CHILDREN,
            cut_nodes=SHAPLEY_CUT_NODES,
            root=HIERARCHY_ROOT,
            leaf_ids=LEAF_IDS,
            # explicitly disable adaptive mode
            adaptive_search=True,
            selection_temperature=0.0,
        )
        if shap_mean is None or shap_std is None or shap_per_run is None:
            raise RuntimeError(
                "HFR-TokenSHAP did not return expected outputs. "
                "Check HFR_TokenSHAP.* API."
            )

        t_shap = time.perf_counter() - shap_t0
        print(f"  Shapley done in {fmt_time(t_shap)}")

        # Per-run storage (leaf + parent)
        for r_idx, phi_r in enumerate(shap_per_run, start=1):
            # leaf rows
            for leaf_id in LEAF_IDS:
                parent = PARENT_OF_LEAF[leaf_id]
                results1_hfr_runs_rows.append(
                    dict(
                        run_id=run_id,
                        subject_id=subject_id,
                        run_idx=r_idx,
                        feature_level="leaf",
                        feature_id=leaf_id,
                        parent_id=parent,
                        group_type=_group_type(parent),
                        importance=float(phi_r[leaf_id]),
                    )
                )
            # parent rows (sum leaves)
            for pid in PARENT_NODES:
                pval = sum(float(phi_r[lf]) for lf in PARENT_TO_LEAVES[pid])
                results1_hfr_runs_rows.append(
                    dict(
                        run_id=run_id,
                        subject_id=subject_id,
                        run_idx=r_idx,
                        feature_level="parent",
                        feature_id=pid,
                        parent_id="",
                        group_type=_group_type(pid),
                        importance=float(pval),
                    )
                )

        # Leaf normalized
        shap_leaf_norm = normalize_abs_vector(shap_mean)

        # Parent mean/std computed from per-run aggregates
        shap_parent_runs: List[Dict[str, float]] = []
        for phi_r in shap_per_run:
            shap_parent_runs.append(
                {pid: sum(float(phi_r[lf]) for lf in PARENT_TO_LEAVES[pid]) for pid in PARENT_NODES}
            )

        shap_parent_mean = {
            pid: sum(r[pid] for r in shap_parent_runs) / float(len(shap_parent_runs))
            for pid in PARENT_NODES
        }
        shap_parent_std = {}
        for pid in PARENT_NODES:
            m = shap_parent_mean[pid]
            shap_parent_std[pid] = (
                sum((r[pid] - m) ** 2 for r in shap_parent_runs) / float(len(shap_parent_runs))
            ) ** 0.5

        shap_parent_norm = normalize_abs_vector(shap_parent_mean)

        # Store HFR: leaves
        for leaf_id in LEAF_IDS:
            parent = PARENT_OF_LEAF[leaf_id]
            results1_hfr_rows.append(
                dict(
                    run_id=run_id,
                    subject_id=subject_id,
                    feature_level="leaf",
                    feature_id=leaf_id,
                    parent_id=parent,
                    group_type=_group_type(parent),
                    importance=float(shap_mean[leaf_id]),
                    abs_importance=abs(float(shap_mean[leaf_id])),
                    norm_abs=float(shap_leaf_norm[leaf_id]),
                    extra_std=float(shap_std[leaf_id]),
                )
            )
            _push_sample("HFR_TokenSHAP", "leaf", leaf_id, shap_leaf_norm[leaf_id])

        # Store HFR: parents
        for pid in PARENT_NODES:
            results1_hfr_rows.append(
                dict(
                    run_id=run_id,
                    subject_id=subject_id,
                    feature_level="parent",
                    feature_id=pid,
                    parent_id="",
                    group_type=_group_type(pid),
                    importance=float(shap_parent_mean[pid]),
                    abs_importance=abs(float(shap_parent_mean[pid])),
                    norm_abs=float(shap_parent_norm[pid]),
                    extra_std=float(shap_parent_std[pid]),
                )
            )
            _push_sample("HFR_TokenSHAP", "parent", pid, shap_parent_norm[pid])

        # Subject timing
        t_subj = time.perf_counter() - t_subj0
        subj_times.append(t_subj)
        avg = sum(subj_times) / len(subj_times)
        remain = avg * (N_PROFILES - i)
        elapsed_all = time.perf_counter() - t_start_all

        print("\n--- SUBJECT TIMING ---")
        print(
            f"Subject total: {fmt_time(t_subj)} | total elapsed: {fmt_time(elapsed_all)} | "
            f"ETA remaining: {fmt_time(remain)}"
        )

        subjects_rows.append(
            dict(
                run_id=run_id,
                subject_id=subject_id,
                gt_label=gt_lab,
                gt_logit=float(gt_log),
                pred_label=pred_lab,
                model_s=float(s),
                lp_case=float(lp_case),
                lp_control=float(lp_control),
                timing_score_s=float(t_score),
                timing_gen_s=float(t_gen),
                timing_ig_s=float(t_ig),
                timing_shapley_s=float(t_shap),
                ig_steps=int(IG_STEPS),
                shapley_k=int(SHAPLEY_K),
                shapley_runs=int(SHAPLEY_RUNS),
                model_name=str(MODEL_NAME),
                device=str(device),
            )
        )

    # -----------------------
    # RESULTS3 (LLM-SELECT) rows + samples for final summary
    # -----------------------
    if llm_leaf_runs and llm_parent_runs:
        for r_idx, run_scores in enumerate(llm_leaf_runs, start=1):
            # expected keys: run_scores["raw"], run_scores["norm"]
            for leaf_id in LEAF_IDS:
                parent = PARENT_OF_LEAF[leaf_id]
                results3_llm_rows.append(
                    dict(
                        run_id=run_id,
                        repeat_idx=r_idx,
                        feature_level="leaf",
                        feature_id=leaf_id,
                        parent_id=parent,
                        group_type=_group_type(parent),
                        score_raw=int(run_scores["raw"][leaf_id]),
                        norm_abs=float(run_scores["norm"][leaf_id]),
                        model=LLM_SELECT_MODEL,
                        temperature=float(LLM_SELECT_TEMPERATURE),
                    )
                )
                _push_sample("LLM_Select", "leaf", leaf_id, run_scores["norm"][leaf_id])

        for r_idx, run_scores in enumerate(llm_parent_runs, start=1):
            for pid in PARENT_NODES:
                results3_llm_rows.append(
                    dict(
                        run_id=run_id,
                        repeat_idx=r_idx,
                        feature_level="parent",
                        feature_id=pid,
                        parent_id="",
                        group_type=_group_type(pid),
                        score_raw=int(run_scores["raw"][pid]),
                        norm_abs=float(run_scores["norm"][pid]),
                        model=LLM_SELECT_MODEL,
                        temperature=float(LLM_SELECT_TEMPERATURE),
                    )
                )
                _push_sample("LLM_Select", "parent", pid, run_scores["norm"][pid])

    # -----------------------
    # FINAL SUMMARY TABLE (means + variance across methods)
    # -----------------------
    final_rows: List[Dict[str, Any]] = []

    def _get_mv(method: str, level: str, fid: str) -> Tuple[float, float, int]:
        xs = norm_samples.get((method, level, fid), [])
        m, v = mean_var_from_samples(xs)
        return m, v, len(xs)

    for leaf_id in LEAF_IDS:
        parent = PARENT_OF_LEAF[leaf_id]
        ig_m, ig_v, ig_n = _get_mv("IG", "leaf", leaf_id)
        hfr_m, hfr_v, hfr_n = _get_mv("HFR_TokenSHAP", "leaf", leaf_id)
        llm_m, llm_v, llm_n = _get_mv("LLM_Select", "leaf", leaf_id)
        final_rows.append(
            dict(
                feature_level="leaf",
                feature_id=leaf_id,
                parent_id=parent,
                group_type=_group_type(parent),
                IG_mean=ig_m,
                IG_var=ig_v,
                IG_n=ig_n,
                HFR_mean=hfr_m,
                HFR_var=hfr_v,
                HFR_n=hfr_n,
                LLMSelect_mean=llm_m,
                LLMSelect_var=llm_v,
                LLMSelect_n=llm_n,
            )
        )

    for pid in PARENT_NODES:
        ig_m, ig_v, ig_n = _get_mv("IG", "parent", pid)
        hfr_m, hfr_v, hfr_n = _get_mv("HFR_TokenSHAP", "parent", pid)
        llm_m, llm_v, llm_n = _get_mv("LLM_Select", "parent", pid)
        final_rows.append(
            dict(
                feature_level="parent",
                feature_id=pid,
                parent_id="",
                group_type=_group_type(pid),
                IG_mean=ig_m,
                IG_var=ig_v,
                IG_n=ig_n,
                HFR_mean=hfr_m,
                HFR_var=hfr_v,
                HFR_n=hfr_n,
                LLMSelect_mean=llm_m,
                LLMSelect_var=llm_v,
                LLMSelect_n=llm_n,
            )
        )

    # -----------------------
    # WRITE TABLES
    # -----------------------
    subjects_path = TABLES_DIR / f"subjects_{run_id}.csv"
    leaves_path = TABLES_DIR / f"leaves_{run_id}.csv"

    results1_path = TABLES_DIR / f"results1_hfr_tokenshapley_{run_id}.csv"
    results1_runs_path = TABLES_DIR / f"results1_hfr_tokenshapley_runs_{run_id}.csv"
    results2_path = TABLES_DIR / f"results2_ig_{run_id}.csv"
    results3_path = TABLES_DIR / f"results3_llm_select_{run_id}.csv"
    final_path = TABLES_DIR / f"final_feature_summary_{run_id}.csv"
    meta_path = TABLES_DIR / f"run_meta_{run_id}.json"

    write_csv(subjects_path, subjects_rows, fieldnames=list(subjects_rows[0].keys()) if subjects_rows else ["run_id"])
    write_csv(leaves_path, leaves_rows, fieldnames=list(leaves_rows[0].keys()) if leaves_rows else ["run_id"])

    write_csv(
        results1_path,
        results1_hfr_rows,
        fieldnames=list(results1_hfr_rows[0].keys()) if results1_hfr_rows else ["run_id"],
    )
    write_csv(
        results1_runs_path,
        results1_hfr_runs_rows,
        fieldnames=list(results1_hfr_runs_rows[0].keys()) if results1_hfr_runs_rows else ["run_id"],
    )
    write_csv(
        results2_path,
        results2_ig_rows,
        fieldnames=list(results2_ig_rows[0].keys()) if results2_ig_rows else ["run_id"],
    )
    write_csv(
        results3_path,
        results3_llm_rows,
        fieldnames=list(results3_llm_rows[0].keys()) if results3_llm_rows else ["run_id"],
    )
    write_csv(final_path, final_rows, fieldnames=list(final_rows[0].keys()) if final_rows else ["feature_id"])

    meta = dict(
        run_id=run_id,
        created_at=datetime.now().isoformat(timespec="seconds"),
        phenotype=PHENOTYPE,
        hf_model_name=MODEL_NAME,
        n_profiles=N_PROFILES,
        ig_steps=IG_STEPS,
        shapley_k=SHAPLEY_K,
        shapley_runs=SHAPLEY_RUNS,
        hfr_adaptive_search=True,
        hfr_selection_temperature=0.0,
        llm_select_model=LLM_SELECT_MODEL,
        llm_select_repeats=LLM_SELECT_REPEATS,
        llm_select_temperature=LLM_SELECT_TEMPERATURE,
        parent_nodes=PARENT_NODES,
        relevant_parents=RELEVANT_PARENTS,
        distractor_parents=DISTRACTOR_PARENTS,
        leaf_ids=LEAF_IDS,
        parent_to_leaves=PARENT_TO_LEAVES,
        hierarchy_root=HIERARCHY_ROOT,
        hierarchy_children=HIERARCHY_CHILDREN,
        paths=dict(
            base_results_dir=str(BASE_RESULTS_DIR),
            tables_dir=str(TABLES_DIR),
            visuals_dir=str(VISUALS_DIR),
            subjects_csv=str(subjects_path),
            leaves_csv=str(leaves_path),
            results1_hfr=str(results1_path),
            results1_hfr_runs=str(results1_runs_path),
            results2_ig=str(results2_path),
            results3_llm_select=str(results3_path),
            final_summary=str(final_path),
        ),
    )
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\n" + "#" * 100)
    print("DONE — TABLES WRITTEN")
    print("  ", subjects_path)
    print("  ", leaves_path)
    print("  ", results1_path)
    print("  ", results1_runs_path)
    print("  ", results2_path)
    print("  ", results3_path)
    print("  ", final_path)
    print("  ", meta_path)


if __name__ == "__main__":
    main()
