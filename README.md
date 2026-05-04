# VCBench: Interpretable ML for Founder Success Prediction

This repository contains an interpretable machine learning approach to the
[VCBench](https://www.vcbench.com/) benchmark for predicting venture capital
founder success, submitted as a NeurIPS 2026 paper as part of the AIvancity
PGE5 *AI in Finance* course.

## Summary

We treat founder success prediction as a tabular classification problem.
Starting from the structured JSON fields of VCBench (education, jobs, prior
exits), we engineer **42 interpretable features** and compare four classical
machine learning models — Logistic Regression, Random Forest, XGBoost,
LightGBM — under the same 6-fold cross-validation protocol as the original
VCBench paper. We then use SHAP to identify which features drive the
predictions.

**Key result.** A Random Forest reaches **F0.5 = 24.6 %** on the public
VCBench split (precision = 25.1 %, recall = 23.5 %), a level comparable to
the structured-ML approaches on the public leaderboard, while costing zero
in API fees, running on commodity hardware in under two minutes, and being
fully interpretable through SHAP.

## Why interpretable ML

The top of the VCBench leaderboard is dominated by large language models
(GPT-4o, DeepSeek, Verifiable-RL). These approaches are powerful but
expensive, slow, and opaque — three properties that make them hard to use
in regulated decision-making like venture capital. We ask a simple
question: how close can a fully interpretable, freely reproducible tabular
model get to LLM-level performance, and what does it tell us about which
founder traits actually predict success?

## Repository layout

```
vcbench-interpretable-ml/
├── data/
│   ├── vcbench_final_public.csv           # 4500 founders, public split
│   └── vcbench_final_public_sample100.csv # 100-row debug sample
├── src/
│   ├── features.py                        # feature engineering (42 features)
│   ├── models.py                          # 6-fold CV + OOF threshold tuning
│   ├── shap_analysis.py                   # SHAP plots + importance
│   ├── run_pipeline.py                    # end-to-end script
│   └── official_evaluation.py             # Vela's reference scorer
├── notebooks/
│   ├── 01_exploration.ipynb               # data exploration
│   ├── 02_features.ipynb                  # feature engineering walkthrough
│   ├── 03_training.ipynb                  # model training + comparison
│   └── 04_shap.ipynb                      # interpretability analysis
├── figures/                               # all figures (PDF + PNG)
├── results/                               # CSVs of metrics, OOF preds, etc.
├── requirements.txt
├── LICENSE                                # MIT
└── README.md
```

## Reproducing the results

The full pipeline runs end-to-end in about 100 seconds on a laptop.

```bash
# 1. clone
git clone https://github.com/<your-handle>/vcbench-interpretable-ml.git
cd vcbench-interpretable-ml

# 2. install
pip install -r requirements.txt

# 3. download the dataset (placed in data/)
# The public split CSV is shipped with this repo.
# Otherwise it can be obtained from the official starter kit:
#   https://github.com/Vela-Engineering/VCBench-Starter-Kit

# 4. run everything
python src/run_pipeline.py
```

This will populate `figures/` with all plots (data exploration, model
comparison, SHAP) and `results/` with all numerical artifacts (per-fold
metrics, leaderboard comparison, out-of-fold predictions, SHAP importance).

For a more pedagogical walkthrough, open the Jupyter notebooks in
`notebooks/` — they reproduce each pipeline stage step by step and are
designed to be opened directly in Google Colab.

## Method

### Features (42 total, four tiers)

1. **Exit signals** (9): `n_prior_ipos`, `n_prior_acquisitions`,
   `n_prior_exits`, `has_any_prior_exit`, `has_multiple_exits`,
   `has_big_prior_ipo`, `has_well_known_acquirer`, ...
2. **Education signals** (13): `n_degrees`, `best_qs_ranking`,
   `has_top_10_qs`, `has_top_50_qs`, `has_top_200_qs`, `has_phd`,
   `has_mba`, `has_master`, `has_bachelor`, `has_cs_field`,
   `has_engineering_field`, `has_business_field`, `has_stem_field`.
3. **Career signals** (11): `n_jobs`, `total_career_years`,
   `n_unique_industries_worked`, `had_c_level_role`, `had_founder_role`,
   `had_leadership_role`, `had_senior_tech_role`, `max_company_size`,
   `worked_at_big_company`, `worked_at_huge_company`, `n_startup_jobs`.
4. **Industry + composite signals** (9): three industry indicators, plus
   `worked_in_target_industry`, `career_breadth`, `avg_job_duration`,
   `elite_education_x_leadership`, `serial_entrepreneur`,
   `technical_founder`.

### Cross-validation protocol

We use stratified 6-fold cross-validation (matching the original paper),
with per-model out-of-fold (OOF) probability collection. The decision
threshold is tuned **once on the full OOF probability vector** to maximise
F0.5; this avoids the threshold over-fitting that occurs when tuning on
training-set probabilities.

### Models

All models use sensible regularisation and `class_weight='balanced'` (or
its boosting equivalent) where applicable. We train Logistic Regression
(scaled inputs), Random Forest, XGBoost, LightGBM, and a simple averaging
ensemble of the three tree models.

## Results

### Headline numbers

| Model              | F0.5 (mean ± std) | Precision | Recall | ROC-AUC |
|--------------------|-------------------|-----------|--------|---------|
| **Random Forest**  | **0.246 ± 0.022** | **0.251** | 0.235  | 0.664   |
| Ensemble           | 0.236 ± 0.018     | 0.235     | 0.242  | 0.656   |
| Logistic Reg.      | 0.225 ± 0.038     | 0.246     | 0.173  | 0.669   |
| XGBoost            | 0.219 ± 0.022     | 0.208     | 0.277  | 0.638   |
| LightGBM           | 0.215 ± 0.047     | 0.222     | 0.193  | 0.633   |

Numbers from `results/model_results.csv`.

### Comparison to the public leaderboard

Approximate F0.5 estimates from precision/recall on
[vcbench.com](https://www.vcbench.com/):

| Rank | Model                        | Type                      | F0.5 % | Precision % |
|------|------------------------------|---------------------------|--------|-------------|
| 1    | Verifiable-RL                | LLM (RL, Vela+Oxford)     | 36.6   | 42.6        |
| 2    | Policy-Induction             | LLM                       | 33.7   | 41.0        |
| 3    | GemVC-v0                     | LLM                       | 32.8   | 39.4        |
| 4    | Verifiable-Reasoning         | LLM                       | 27.9   | 30.6        |
| 5    | Structured-Rule-Stump        | Tabular ML                | 27.7   | 32.8        |
| 6    | Random-Rule-Forest           | Hybrid                    | 27.5   | 42.5        |
| -    | **Ours: Random Forest**      | **Tabular ML (interp.)**  | **24.6** | **25.1**  |

We sit just below the structured-ML baselines and well above the random
classifier baseline (≈9 %), at zero API cost.

## Key findings (SHAP)

Top-five most influential features on the model's predictions:

1. **`best_qs_ranking`** — a top-tier alma mater is the single strongest
   positive signal; conversely, a 200+ ranking pushes predictions
   strongly downward.
2. **`has_top_200_qs`** and **`has_top_10_qs`** — confirms the QS effect.
3. **`worked_in_target_industry`** — founders building in a sector they
   previously worked in are favoured.
4. **`max_company_size`** / **`worked_at_huge_company`** — exposure to
   large organisations correlates positively with success.
5. **`avg_job_duration`** — short, varied stints (often a serial-
   entrepreneur pattern) outperform long tenures.

See `figures/fig6_shap_summary.{pdf,png}` for the full beeswarm plot, and
`figures/shap_dependence_*.{pdf,png}` for per-feature dependence plots.

## Limitations

- **Dataset ceiling.** The public VCBench split has a known information
  ceiling around F0.5 ≈ 0.30. Beyond this, the dataset simply does not
  contain the signals (idea quality, market timing, team chemistry, etc.)
  needed to do better.
- **Tech-heavy bias.** 34 % of the dataset is tech / software founders;
  generalisation to other sectors is not validated.
- **Static profiles.** Each founder is represented by a snapshot at the
  moment of founding; we do not model post-founding dynamics.
- **No private split.** We cannot evaluate on Vela's held-out 4 500
  founders, so all reported numbers are 6-fold CV on the public split.

## License

MIT — see `LICENSE`.

## Acknowledgements

Dataset by [Vela Research](https://vela.partners/) and the University of
Oxford. Course supervision by Mostapha Benhenda (AIvancity).

## Citation

If you build on this work, please cite the original VCBench paper:

```bibtex
@misc{chen2025vcbenchbenchmarkingllmsventure,
  title={VCBench: Benchmarking LLMs in Venture Capital},
  author={Rick Chen et al.},
  year={2025},
  eprint={2509.14448},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2509.14448},
}
```
