"""
Main pipeline: runs the full end-to-end experiment for the VCBench
founder success prediction project.

Steps:
  1. Load and explore the dataset.
  2. Build features.
  3. Run 6-fold CV for all models with OOF threshold tuning.
  4. Compare results to the public VCBench leaderboard.
  5. Train the best model on full data and run SHAP analysis.
  6. Generate all figures (data exploration + SHAP).
  7. Save metrics and OOF predictions.

Usage:
    python src/run_pipeline.py
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from features import build_features, feature_columns, safe_parse, parse_qs_ranking
from models import run_all_models, evaluate_with_oof_threshold, get_models
from shap_analysis import (
    fit_best_model, compute_shap_values,
    plot_shap_summary, plot_shap_bar, plot_shap_dependence,
)


# ------------ paths ------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "vcbench_final_public.csv")
FIG_DIR = os.path.join(ROOT, "figures")
RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ------------ plot style ------------
sns.set_style("whitegrid")
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def save_fig(fig, name):
    fig.savefig(os.path.join(FIG_DIR, f"{name}.pdf"))
    fig.savefig(os.path.join(FIG_DIR, f"{name}.png"), dpi=200)


# ============================================================
# 1. EXPLORATORY FIGURES
# ============================================================
def make_exploratory_figures(df_raw, df_feat):
    print("\n[1] Generating exploratory figures...")

    # --- Fig 1: class distribution ---
    fig, ax = plt.subplots(figsize=(7, 4))
    counts = df_raw["success"].value_counts().sort_index()
    bars = ax.bar(["Failure (0)", "Success (1)"], counts.values,
                  color=["#E63946", "#2A9D8F"], edgecolor="black", linewidth=0.8)
    for bar, count in zip(bars, counts.values):
        pct = count / len(df_raw) * 100
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                f"{count}\n({pct:.1f}%)", ha="center", fontsize=11,
                fontweight="bold")
    ax.set_ylabel("Number of founders")
    ax.set_title("VCBench public split: class distribution (4,500 founders)")
    ax.set_ylim(0, max(counts.values) * 1.15)
    plt.tight_layout()
    save_fig(fig, "fig1_class_distribution")
    plt.close(fig)

    # --- Fig 2: top 10 industries with success rate ---
    top10 = df_raw["industry"].value_counts().head(10).index
    ind_stats = df_raw[df_raw["industry"].isin(top10)].groupby("industry").agg(
        n=("success", "count"),
        success_rate=("success", lambda x: x.mean() * 100),
    ).sort_values("n", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    y_pos = np.arange(len(ind_stats))
    ax.barh(y_pos, ind_stats["n"].values,
            color="#264653", edgecolor="black", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([s[:42] for s in ind_stats.index])
    for i, (n, sr) in enumerate(zip(ind_stats["n"].values,
                                    ind_stats["success_rate"].values)):
        ax.text(n + 5, i, f"  {sr:.1f}% success",
                va="center", fontsize=10, color="#E76F51", fontweight="bold")
    ax.set_xlabel("Number of founders")
    ax.set_title("Top 10 industries: founder count and success rate "
                 "(baseline = 9%)")
    plt.tight_layout()
    save_fig(fig, "fig2_top_industries")
    plt.close(fig)

    # --- Fig 3: QS ranking tier vs success rate ---
    def qs_tier(qs):
        if qs is None or qs >= 999:
            return "No ranking"
        if qs <= 10: return "Top 10"
        if qs <= 50: return "Top 11–50"
        if qs <= 100: return "Top 51–100"
        if qs <= 200: return "Top 101–200"
        return "200+"

    df_raw = df_raw.copy()
    edus = df_raw["educations_json"].apply(safe_parse)
    def best_qs(es):
        ranks = [parse_qs_ranking(e.get("qs_ranking", "")) for e in es]
        ranks = [r for r in ranks if r is not None]
        return min(ranks) if ranks else 999.0
    df_raw["best_qs"] = edus.apply(best_qs)
    df_raw["qs_tier"] = df_raw["best_qs"].apply(qs_tier)

    qs_order = ["Top 10", "Top 11–50", "Top 51–100", "Top 101–200", "200+",
                "No ranking"]
    qs_data = df_raw.groupby("qs_tier").agg(
        n=("success", "count"),
        sr=("success", lambda x: x.mean() * 100),
    )
    qs_data = qs_data.reindex(qs_order)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#2A9D8F", "#52B788", "#95D5B2", "#F4A261", "#E76F51", "#9D9D9D"]
    bars = ax.bar(qs_order, qs_data["sr"].values, color=colors,
                  edgecolor="black", linewidth=0.8)
    ax.axhline(9.0, color="black", linestyle="--", linewidth=1,
               label="Baseline (9%)")
    for bar, n, sr in zip(bars, qs_data["n"].values, qs_data["sr"].values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                f"{sr:.1f}%\n(n={int(n)})", ha="center", fontsize=9)
    ax.set_ylabel("Success rate (%)")
    ax.set_xlabel("Best QS world ranking among founder's degrees")
    ax.set_title("Education prestige strongly predicts founder success")
    ax.legend()
    ax.set_ylim(0, max(qs_data["sr"].values) * 1.25)
    plt.xticks(rotation=15)
    plt.tight_layout()
    save_fig(fig, "fig3_qs_ranking")
    plt.close(fig)

    # --- Fig 4: prior exits vs success rate ---
    n_exits = df_feat["n_prior_exits"].astype(int)
    df_tmp = pd.DataFrame({"n_exits": n_exits, "success": df_feat["success"]})
    df_tmp["bucket"] = df_tmp["n_exits"].apply(
        lambda x: "4+" if x >= 4 else str(int(x))
    )
    bucket_order = ["0", "1", "2", "3", "4+"]
    exit_data = df_tmp.groupby("bucket").agg(
        n=("success", "count"),
        sr=("success", lambda x: x.mean() * 100),
    ).reindex(bucket_order).dropna()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["#9D9D9D", "#52B788", "#2A9D8F", "#1D7874", "#0E4D45"]
    bars = ax.bar(exit_data.index, exit_data["sr"].values,
                  color=colors[:len(exit_data)],
                  edgecolor="black", linewidth=0.8)
    ax.axhline(9.0, color="black", linestyle="--", linewidth=1,
               label="Baseline (9%)")
    for bar, n, sr in zip(bars, exit_data["n"].values, exit_data["sr"].values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{sr:.1f}%\n(n={int(n)})", ha="center", fontsize=9)
    ax.set_ylabel("Success rate (%)")
    ax.set_xlabel("Number of prior exits (IPO + acquisition)")
    ax.set_title("Track record: prior exits dramatically predict future success")
    ax.legend()
    ax.set_ylim(0, max(exit_data["sr"].values) * 1.25)
    plt.tight_layout()
    save_fig(fig, "fig4_prior_exits")
    plt.close(fig)

    print("    → 4 exploratory figures saved.")


# ============================================================
# 2. MODEL COMPARISON FIGURE
# ============================================================
def make_model_comparison_figure(summary):
    print("\n[2] Generating model comparison figure...")
    df_plot = summary.copy().sort_values("f0.5_mean")
    fig, ax = plt.subplots(figsize=(9, 5))
    ypos = np.arange(len(df_plot))
    bars = ax.barh(
        ypos, df_plot["f0.5_mean"].values,
        xerr=df_plot["f0.5_std"].values,
        color=["#264653" if i < len(df_plot) - 1 else "#E76F51"
               for i in range(len(df_plot))],
        edgecolor="black", linewidth=0.7,
        error_kw={"ecolor": "black", "capsize": 3, "linewidth": 1},
    )
    ax.set_yticks(ypos)
    ax.set_yticklabels(df_plot["model"].values)
    ax.axvline(0.251, color="red", linestyle="--", linewidth=1,
               label="Course target F0.5 ≥ 0.251")
    for bar, val in zip(bars, df_plot["f0.5_mean"].values):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=10)
    ax.set_xlabel("F0.5 (mean ± std across 6 folds)")
    ax.set_title("Model comparison on VCBench public split (6-fold CV)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    save_fig(fig, "fig5_model_comparison")
    plt.close(fig)
    print("    → Model comparison saved.")


# ============================================================
# 3. SHAP ANALYSIS
# ============================================================
def run_shap(X, y):
    print("\n[3] Running SHAP analysis on best model (Random Forest)...")
    model = fit_best_model(X, y, seed=42)
    shap_values, X_sample, idx, feature_names = compute_shap_values(
        model, X, max_samples=2000, seed=42,
    )

    plot_shap_summary(
        shap_values, X_sample, feature_names,
        os.path.join(FIG_DIR, "fig6_shap_summary.pdf"),
    )

    top_features = plot_shap_bar(
        shap_values, X_sample, feature_names,
        os.path.join(FIG_DIR, "fig7_shap_bar.pdf"),
    )

    top_names = [t[0] for t in top_features][::-1]  # most important first
    plot_shap_dependence(
        shap_values, X_sample, feature_names, top_names,
        save_dir=FIG_DIR, n_top=4,
    )

    importance = pd.DataFrame(
        [{"feature": n, "mean_abs_shap": v} for n, v in top_features[::-1]]
    )
    importance.to_csv(
        os.path.join(RESULTS_DIR, "shap_feature_importance.csv"), index=False,
    )
    print(f"    → SHAP figures + importance CSV saved.")
    print(f"    → Top 5 features: {top_names[:5]}")
    return importance


# ============================================================
# 4. LEADERBOARD COMPARISON TABLE
# ============================================================
LEADERBOARD = pd.DataFrame([
    {"model": "Verifiable-RL (SOTA)", "team": "Vela + Oxford",
     "precision": 42.6, "recall": 23.6, "f0.5_est": 36.6,
     "type": "LLM (RL)"},
    {"model": "Policy-Induction", "team": "Vela + Oxford",
     "precision": 41.0, "recall": 20.2, "f0.5_est": 33.7,
     "type": "LLM"},
    {"model": "GemVC-v0", "team": "Independent",
     "precision": 39.4, "recall": 20.3, "f0.5_est": 32.8,
     "type": "LLM"},
    {"model": "Structured-Rule-Stump", "team": "Independent",
     "precision": 32.8, "recall": 18.0, "f0.5_est": 27.7,
     "type": "Tabular ML"},
    {"model": "Random-Rule-Forest", "team": "Vela + Oxford",
     "precision": 42.5, "recall": 12.1, "f0.5_est": 27.5,
     "type": "Hybrid"},
    {"model": "Verifiable-Reasoning", "team": "Vela + Oxford",
     "precision": 30.6, "recall": 21.0, "f0.5_est": 27.9,
     "type": "LLM"},
])


def save_leaderboard_comparison(summary):
    """Build a comparison table: our results vs the public leaderboard."""
    print("\n[4] Building leaderboard comparison...")
    our_best = summary.iloc[0].copy()
    rows = []
    for _, row in LEADERBOARD.iterrows():
        rows.append({
            "model": row["model"],
            "type": row["type"],
            "precision_pct": row["precision"],
            "recall_pct": row["recall"],
            "f0.5_pct": row["f0.5_est"],
            "is_ours": False,
        })
    rows.append({
        "model": f"Ours: {our_best['model']}",
        "type": "Tabular ML (Interpretable)",
        "precision_pct": our_best["precision_mean"] * 100,
        "recall_pct": our_best["recall_mean"] * 100,
        "f0.5_pct": our_best["f0.5_mean"] * 100,
        "is_ours": True,
    })
    df_comp = pd.DataFrame(rows).sort_values("f0.5_pct", ascending=False)
    df_comp.to_csv(os.path.join(RESULTS_DIR, "leaderboard_comparison.csv"),
                   index=False)
    print(df_comp.to_string(index=False, float_format="%.1f"))
    return df_comp


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("VCBench Founder Success Prediction — Full Pipeline")
    print("=" * 70)
    t0 = time.time()

    # Load
    print(f"\n[0] Loading {DATA_PATH}...")
    df_raw = pd.read_csv(DATA_PATH)
    print(f"    → {len(df_raw)} rows, success rate "
          f"{df_raw['success'].mean()*100:.2f}%")

    # Features
    print(f"\n[FE] Building features...")
    df_feat = build_features(df_raw)
    feats = feature_columns(df_feat)
    X = df_feat[feats]
    y = df_feat["success"]
    print(f"    → {len(feats)} features")
    df_feat.to_csv(os.path.join(RESULTS_DIR, "features.csv"), index=False)

    # Exploratory figures
    make_exploratory_figures(df_raw, df_feat)

    # CV evaluation
    print(f"\n[CV] Running 6-fold CV with OOF threshold tuning...")
    summary, folds, oof = run_all_models(X, y, n_folds=6, seed=42)
    print()
    print("=" * 70)
    print("RESULTS — sorted by F0.5")
    print("=" * 70)
    print(summary[["model", "threshold", "f0.5_mean", "f0.5_std",
                   "precision_mean", "recall_mean",
                   "roc_auc_mean"]].to_string(index=False,
                                              float_format="%.4f"))
    summary.to_csv(os.path.join(RESULTS_DIR, "model_results.csv"), index=False)

    # Save per-fold details
    for name, df_folds in folds.items():
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
        df_folds.to_csv(
            os.path.join(RESULTS_DIR, f"folds_{safe_name}.csv"), index=False,
        )

    # Save OOF predictions for the best model
    best_name = summary.iloc[0]["model"]
    if best_name in oof:
        oof_df = pd.DataFrame({
            "founder_uuid": df_raw["founder_uuid"],
            "true_success": y.values,
            "predicted_proba": oof[best_name]["proba"],
            "predicted_label": oof[best_name]["pred"],
        })
        oof_df.to_csv(os.path.join(RESULTS_DIR, "oof_predictions_best.csv"),
                      index=False)
        print(f"\n    → OOF predictions for best model ({best_name}) saved.")

    # Model comparison plot
    make_model_comparison_figure(summary)

    # Leaderboard comparison
    save_leaderboard_comparison(summary)

    # SHAP
    run_shap(X, y)

    # Final summary
    print(f"\n{'=' * 70}")
    print(f"PIPELINE COMPLETE in {time.time() - t0:.1f}s")
    print(f"{'=' * 70}")
    print(f"Best model        : {best_name}")
    print(f"F0.5 (mean ± std) : {summary.iloc[0]['f0.5_mean']:.4f} "
          f"± {summary.iloc[0]['f0.5_std']:.4f}")
    print(f"Precision (mean)  : {summary.iloc[0]['precision_mean']*100:.2f}%")
    print(f"Recall (mean)     : {summary.iloc[0]['recall_mean']*100:.2f}%")
    print(f"ROC-AUC (mean)    : {summary.iloc[0]['roc_auc_mean']:.4f}")
    print(f"\nFigures in : {FIG_DIR}")
    print(f"Results in : {RESULTS_DIR}")


if __name__ == "__main__":
    main()
