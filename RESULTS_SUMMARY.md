# Results summary

Run on the public VCBench split (4500 founders, 9% success rate).

## Headline numbers (6-fold stratified CV)

| Model              | F0.5 (mean ± std) | Precision | Recall | ROC-AUC | Threshold |
|--------------------|-------------------|-----------|--------|---------|-----------|
| **Random Forest**  | **0.246 ± 0.022** | **0.251** | 0.235  | 0.664   | 0.493     |
| Ensemble (avg)     | 0.236 ± 0.018     | 0.235     | 0.242  | 0.656   | 0.296     |
| Logistic Reg.      | 0.225 ± 0.038     | 0.246     | 0.173  | 0.669   | 0.749     |
| XGBoost            | 0.219 ± 0.022     | 0.208     | 0.277  | 0.638   | 0.177     |
| LightGBM           | 0.215 ± 0.047     | 0.222     | 0.193  | 0.633   | 0.231     |

## Top features (mean |SHAP|, computed on Random Forest)

| Rank | Feature                   | Interpretation |
|------|---------------------------|----------------|
| 1    | `best_qs_ranking`         | Lower (more prestigious) ranking → higher predicted success |
| 2    | `has_top_200_qs`          | Having any degree from a top-200 university boosts predictions |
| 3    | `max_company_size`        | Exposure to large organisations correlates positively |
| 4    | `worked_in_target_industry` | Industry-aligned background is favoured |
| 5    | `avg_job_duration`        | Shorter, varied stints (serial-entrepreneur pattern) outperform long tenures |
| 6    | `total_career_years`      | Mid-career founders are favoured (saturating effect) |
| 7    | `n_degrees`               | More education modestly negative (over-credentialed) |
| 8    | `worked_at_huge_company`  | 5000+-employee experience helps |
| 9    | `has_top_10_qs`           | Elite university gives a clear bump |
| 10   | `career_breadth`          | Industry diversity per job, mild negative |

## Comparison to public leaderboard

| Rank | Model                 | Type                     | F0.5 % | Precision % | Recall % |
|------|-----------------------|--------------------------|--------|-------------|----------|
| 1    | Verifiable-RL         | LLM (RL)                 | 36.6   | 42.6        | 23.6     |
| 2    | Policy-Induction      | LLM                      | 33.7   | 41.0        | 20.2     |
| 3    | GemVC-v0              | LLM                      | 32.8   | 39.4        | 20.3     |
| 4    | Verifiable-Reasoning  | LLM                      | 27.9   | 30.6        | 21.0     |
| 5    | Structured-Rule-Stump | Tabular ML               | 27.7   | 32.8        | 18.0     |
| 6    | Random-Rule-Forest    | Hybrid                   | 27.5   | 42.5        | 12.1     |
| —    | **Ours: Random Forest** | **Tabular ML (interp.)** | **24.6** | **25.1** | 23.5 |

We sit just below the structured-ML baselines (27.5 % – 27.7 %) and well
above the random / market baselines (~9 %), at zero API cost, on commodity
hardware, and with full SHAP-based interpretability.
