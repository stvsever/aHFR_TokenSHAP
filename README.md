# HFR-TokenSHAP: A TokenSHAP-extended Algorithm for Hierarchical Feature Importance Estimation


This repository provides a **reproducible demonstration** of *feature-level* attribution for LLM-based binary phenotype classification prompts, together with the accompanying manuscript PDF.

**Manuscript (PDF):** `paper_HFR_TokenSHAP.pdf`  
**Repository:** `stvsever/PAPER_HFR_TokenSHAP`

---

## Abstract (Repository-level)

Token-level attribution methods (e.g., gradient-based saliency) can assign high importance to prompt scaffolding (e.g., headers, separators, boilerplate) even when the scientific question is **which structured features drive the decision**. **HFR-TokenSHAP** restricts Shapley “players” to semantically meaningful **feature nodes** (optionally organized into hierarchical groups) and evaluates contributions via a **decision-aligned value function** (e.g., label log-odds). The result is an explanation over *feature units* rather than over all prompt tokens. This hierarchically restricted feature (HFR) extension reduces the computational burden of Monte Carlo Shapley estimation by restricting the sampling space to feature-level hierarchy nodes, rather than individual prompt tokens that are not the explanatory object of interest.

---

## Figure: Example Prompt Overlay

The following figure 1 shows a qualitative example prompt with an overlay of feature-importance scores (Integrated Gradients vs. HFR-TokenSHAP-style restricted Shapley), illustrating increased emphasis on clinically relevant features such as `sleep_quality` and `childhood_trauma_exposure` relative to distractor “word-features”.

![Example prompt overlay](https://github.com/stvsever/PAPER_HFR_TokenSHAP/blob/main/demonstration_results/visuals/example_prompt_overlay.png)

---

## Repository Structure

- `utils/demo_run.py`  
  Main entry point. Generates pseudo-profiles, constructs prompts, computes:
  1) Integrated Gradients (IG) feature importances,  
  2) Monte Carlo Shapley feature importances under a *feature-restricted* player set (HFR-TokenSHAP-style),  
  and writes tables/plots to `TABLES_DIR`.

- `utils/ig_attribution.py`  
  Model loading utilities, decision scoring (e.g., log-odds), and IG helper functions.

- `utils/group_shapley.py`  
  Monte Carlo Shapley estimation via permutation sampling over feature “players” (with optional grouping / hierarchical restriction).

- `demonstration_results/`  
  Generated tables and visuals (e.g., prompt overlays, aggregated plots).

---
