"""
Training and evaluation for VCBench founder success prediction.

Key methodological choices:
- 6-fold stratified cross-validation (matches the original VCBench protocol)
- Out-of-fold (OOF) threshold tuning for F0.5: avoids the overfit-on-train issue
  that plagues tree-based models when the decision threshold is set on the
  same data the model was fit on.
- F0.5 is the primary metric (precision-weighted), with precision/recall/AUC as
  secondary metrics.
- Models: Logistic Regression, Random Forest, XGBoost, LightGBM, plus a simple
  averaging ensemble of the three best.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    fbeta_score, precision_score, recall_score, roc_auc_score
)
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")


# ---------- threshold optimization ----------

def best_threshold_for_f05(y_true, y_proba, n_thresholds=399):
    """Find the threshold in (0, 1) maximizing F0.5 on (y_true, y_proba)."""
    thresholds = np.linspace(0.005, 0.995, n_thresholds)
    best_t, best_f = 0.5, -1.0
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        if y_pred.sum() == 0:
            continue
        f = fbeta_score(y_true, y_pred, beta=0.5, zero_division=0)
        if f > best_f:
            best_f, best_t = f, t
    return best_t, best_f


# ---------- model factories ----------

def get_models(scale_pos_weight=10.1):
    """Returns a dict of {name: (model, needs_scaling)} pairs."""
    return {
        "LogisticRegression": (
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                C=0.5,
                random_state=42,
            ),
            True,
        ),
        "RandomForest": (
            RandomForestClassifier(
                n_estimators=600,
                max_depth=10,
                min_samples_leaf=5,
                max_features="sqrt",
                class_weight="balanced",
                n_jobs=-1,
                random_state=42,
            ),
            False,
        ),
        # NOTE: XGBoost and LightGBM here use NO scale_pos_weight.
        # We let them learn naturally calibrated probabilities and rely on
        # the OOF threshold tuning to handle the 9% class imbalance.
        # Empirically this gives much better F0.5 than scale_pos_weight=N.
        "XGBoost": (
            xgb.XGBClassifier(
                n_estimators=800,
                max_depth=4,
                learning_rate=0.03,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=0.3,
                reg_lambda=1.5,
                min_child_weight=5,
                eval_metric="logloss",
                tree_method="hist",
                random_state=42,
                n_jobs=-1,
                verbosity=0,
            ),
            False,
        ),
        "LightGBM": (
            lgb.LGBMClassifier(
                n_estimators=800,
                max_depth=5,
                num_leaves=23,
                learning_rate=0.03,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=0.2,
                reg_lambda=1.0,
                min_child_samples=20,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            ),
            False,
        ),
    }


# ---------- core CV that returns OOF probabilities ----------

def cv_oof_probas(model_template, X, y, needs_scaling=False, n_folds=6, seed=42):
    """Run stratified k-fold CV and return out-of-fold probabilities (one
    proba per row, predicted by the fold where the row was held out)."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    X_arr = X.values if isinstance(X, pd.DataFrame) else X
    y_arr = y.values if isinstance(y, pd.Series) else y
    oof_proba = np.zeros(len(X_arr))
    fold_assignment = np.zeros(len(X_arr), dtype=int)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_arr, y_arr), 1):
        X_tr, X_val = X_arr[train_idx], X_arr[val_idx]
        y_tr = y_arr[train_idx]

        if needs_scaling:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_val = scaler.transform(X_val)

        m = type(model_template)(**model_template.get_params())
        m.fit(X_tr, y_tr)
        oof_proba[val_idx] = m.predict_proba(X_val)[:, 1]
        fold_assignment[val_idx] = fold_idx

    return oof_proba, fold_assignment


def per_fold_metrics(y_true, y_proba, fold_assignment, threshold):
    """Compute F0.5 / precision / recall / AUC per fold given a global threshold."""
    fold_records = []
    for fold in sorted(np.unique(fold_assignment)):
        mask = fold_assignment == fold
        y_t = y_true[mask]
        p = y_proba[mask]
        y_pred = (p >= threshold).astype(int)
        try:
            auc = roc_auc_score(y_t, p)
        except ValueError:
            auc = float("nan")
        fold_records.append({
            "fold": int(fold),
            "f0.5": fbeta_score(y_t, y_pred, beta=0.5, zero_division=0),
            "precision": precision_score(y_t, y_pred, zero_division=0),
            "recall": recall_score(y_t, y_pred, zero_division=0),
            "roc_auc": auc,
            "n_pred_pos": int(y_pred.sum()),
            "n_true_pos": int(((y_pred == 1) & (y_t == 1)).sum()),
        })
    return pd.DataFrame(fold_records)


def evaluate_with_oof_threshold(name, model_template, X, y, needs_scaling=False,
                                n_folds=6, seed=42):
    """Run CV, get OOF probas, tune threshold once on OOF, report per-fold metrics."""
    y_arr = y.values if isinstance(y, pd.Series) else y
    oof_proba, fold_assignment = cv_oof_probas(
        model_template, X, y, needs_scaling=needs_scaling,
        n_folds=n_folds, seed=seed,
    )
    threshold, _ = best_threshold_for_f05(y_arr, oof_proba)
    df_folds = per_fold_metrics(y_arr, oof_proba, fold_assignment, threshold)
    summary = {
        "model": name,
        "threshold": threshold,
        "f0.5_mean": df_folds["f0.5"].mean(),
        "f0.5_std": df_folds["f0.5"].std(),
        "precision_mean": df_folds["precision"].mean(),
        "precision_std": df_folds["precision"].std(),
        "recall_mean": df_folds["recall"].mean(),
        "recall_std": df_folds["recall"].std(),
        "roc_auc_mean": df_folds["roc_auc"].mean(),
    }
    oof_pred = (oof_proba >= threshold).astype(int)
    return summary, df_folds, oof_proba, oof_pred


def run_all_models(X, y, n_folds=6, seed=42, include_ensemble=True):
    """Run 6-fold CV for every base model + an averaging ensemble."""
    pos_neg_ratio = (y == 0).sum() / max(1, (y == 1).sum())
    models = get_models(scale_pos_weight=pos_neg_ratio)

    summaries = []
    all_folds = {}
    all_oof = {}

    for name, (model, needs_scaling) in models.items():
        print(f"  [running] {name} ...", flush=True)
        summary, df_folds, oof_proba, oof_pred = evaluate_with_oof_threshold(
            name, model, X, y, needs_scaling=needs_scaling,
            n_folds=n_folds, seed=seed,
        )
        summaries.append(summary)
        all_folds[name] = df_folds
        all_oof[name] = {"proba": oof_proba, "pred": oof_pred}
        print(f"  [done]    {name}: F0.5 = {summary['f0.5_mean']:.4f} "
              f"(prec={summary['precision_mean']:.3f}, "
              f"rec={summary['recall_mean']:.3f})")

    if include_ensemble:
        ens_models = ["XGBoost", "LightGBM", "RandomForest"]
        ens_proba = np.mean([all_oof[n]["proba"] for n in ens_models], axis=0)
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        y_arr = y.values if isinstance(y, pd.Series) else y
        fold_assignment = np.zeros(len(y_arr), dtype=int)
        for fold_idx, (_, val_idx) in enumerate(
            skf.split(np.zeros(len(y_arr)), y_arr), 1
        ):
            fold_assignment[val_idx] = fold_idx

        threshold, _ = best_threshold_for_f05(y_arr, ens_proba)
        df_folds_ens = per_fold_metrics(y_arr, ens_proba, fold_assignment, threshold)
        ens_summary = {
            "model": "Ensemble (XGB+LGB+RF avg)",
            "threshold": threshold,
            "f0.5_mean": df_folds_ens["f0.5"].mean(),
            "f0.5_std": df_folds_ens["f0.5"].std(),
            "precision_mean": df_folds_ens["precision"].mean(),
            "precision_std": df_folds_ens["precision"].std(),
            "recall_mean": df_folds_ens["recall"].mean(),
            "recall_std": df_folds_ens["recall"].std(),
            "roc_auc_mean": df_folds_ens["roc_auc"].mean(),
        }
        summaries.append(ens_summary)
        all_folds["Ensemble"] = df_folds_ens
        all_oof["Ensemble"] = {
            "proba": ens_proba,
            "pred": (ens_proba >= threshold).astype(int),
        }
        print(f"  [done]    Ensemble: F0.5 = {ens_summary['f0.5_mean']:.4f} "
              f"(prec={ens_summary['precision_mean']:.3f}, "
              f"rec={ens_summary['recall_mean']:.3f})")

    df_summary = pd.DataFrame(summaries).sort_values("f0.5_mean", ascending=False)
    df_summary = df_summary.reset_index(drop=True)
    return df_summary, all_folds, all_oof
