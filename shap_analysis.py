"""
SHAP analysis for interpretability.

Trains the best model on the full dataset and computes SHAP values to
understand which features drive predictions globally and individually.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")


def fit_best_model(X, y, seed=42):
    """Fit a Random Forest (our best model) on the full data for SHAP."""
    model = RandomForestClassifier(
        n_estimators=600,
        max_depth=10,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
    )
    model.fit(X, y)
    return model


def compute_shap_values(model, X, max_samples=2000, seed=42):
    """Compute SHAP values using TreeExplainer.
    For large datasets we sample to keep it tractable."""
    X_arr = X.values if isinstance(X, pd.DataFrame) else X
    feature_names = list(X.columns) if isinstance(X, pd.DataFrame) else None

    if len(X_arr) > max_samples:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(X_arr), max_samples, replace=False)
        X_sample = X_arr[idx]
    else:
        idx = np.arange(len(X_arr))
        X_sample = X_arr

    explainer = shap.TreeExplainer(model)
    raw_shap = explainer.shap_values(X_sample)

    # For binary classification, sklearn RF returns a list [class0, class1]
    # We want the SHAP values for class 1 (success).
    if isinstance(raw_shap, list) and len(raw_shap) == 2:
        shap_values = raw_shap[1]
    elif raw_shap.ndim == 3:
        # Newer SHAP returns (n_samples, n_features, n_classes)
        shap_values = raw_shap[:, :, 1]
    else:
        shap_values = raw_shap

    return shap_values, X_sample, idx, feature_names


def plot_shap_summary(shap_values, X_sample, feature_names, save_path,
                      max_display=15):
    """Beeswarm-style summary plot of SHAP values."""
    plt.figure()
    shap.summary_plot(
        shap_values, X_sample, feature_names=feature_names,
        max_display=max_display, show=False, plot_size=(10, 7),
    )
    plt.title("SHAP feature importance — what drives founder success predictions",
              fontsize=12, pad=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.savefig(save_path.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close()


def plot_shap_bar(shap_values, X_sample, feature_names, save_path,
                  max_display=15):
    """Mean absolute SHAP bar plot — global importance."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:max_display]
    names = [feature_names[i] for i in order][::-1]
    vals = mean_abs[order][::-1]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(names, vals, color="#264653", edgecolor="black", linewidth=0.5)
    for bar, v in zip(bars, vals):
        ax.text(v + max(vals) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{v:.4f}", va="center", fontsize=9)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Top {max_display} features by global importance")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.savefig(save_path.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close()
    return list(zip(names[::-1], vals[::-1]))


def plot_shap_dependence(shap_values, X_sample, feature_names, top_features,
                         save_dir, n_top=4):
    """Dependence plots for the top N features."""
    if isinstance(X_sample, np.ndarray):
        X_sample_df = pd.DataFrame(X_sample, columns=feature_names)
    else:
        X_sample_df = X_sample

    saved = []
    for feat in top_features[:n_top]:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        feat_idx = feature_names.index(feat)
        ax.scatter(
            X_sample_df[feat], shap_values[:, feat_idx],
            alpha=0.4, s=20, color="#264653", edgecolor="white", linewidth=0.3,
        )
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.set_xlabel(feat)
        ax.set_ylabel("SHAP value (impact on prediction)")
        ax.set_title(f"How '{feat}' affects predictions")
        plt.tight_layout()
        path = os.path.join(save_dir, f"shap_dependence_{feat}.pdf")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.savefig(path.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
        plt.close()
        saved.append(path)
    return saved
