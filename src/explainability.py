"""
explainability.py
=================
Model explainability for IMDb sentiment pipelines.

Three methods:
  1. Coefficient plots  — top positive/negative words (LogReg, LinearSVC)
  2. SHAP               — feature importance for TF-IDF + tree models
  3. LIME               — per-sample explanation for any pipeline

Usage:
    from src.explainability import plot_coefficients, explain_with_shap, explain_with_lime
"""

import logging
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os

matplotlib.use("Agg")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Coefficient plots (LogReg / LinearSVC)
# ─────────────────────────────────────────────────────────────────────────────

def plot_coefficients(
    pipeline,
    name:     str,
    top_n:    int = 20,
    save_dir: str = "outputs",
) -> str:
    """
    Plot top-N positive and negative word weights from a linear model.

    Works with: LogisticRegression, LinearSVC (via CalibratedClassifierCV).
    The vectorizer step must be named 'tfidf' in the pipeline.

    Args:
        pipeline : fitted sklearn Pipeline with 'tfidf' + linear classifier
        name     : model name for plot title and filename
        top_n    : number of words to show per class
        save_dir : output directory

    Returns:
        Path to saved plot.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Extract vectorizer
    if "tfidf" not in pipeline.named_steps:
        raise ValueError("plot_coefficients requires a 'tfidf' step in the pipeline.")
    vectorizer   = pipeline.named_steps["tfidf"]
    feature_names = np.array(vectorizer.get_feature_names_out())

    # Extract coefficients
    clf = pipeline.named_steps["clf"]
    coef = _extract_coef(clf)
    if coef is None:
        raise ValueError(f"Cannot extract coefficients from {type(clf).__name__}")

    # Top positive (→ positive sentiment) and negative (→ negative sentiment)
    top_pos_idx = np.argsort(coef)[-top_n:][::-1]
    top_neg_idx = np.argsort(coef)[:top_n]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, idx, color, title in zip(
        axes,
        [top_pos_idx, top_neg_idx],
        ["#4C8BF5", "#E8534C"],
        [f"Top {top_n} → Positive Sentiment", f"Top {top_n} → Negative Sentiment"],
    ):
        words  = feature_names[idx]
        values = coef[idx]
        ax.barh(words[::-1], values[::-1], color=color, edgecolor="white")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Coefficient Weight")
        ax.axvline(0, color="black", linewidth=0.8)

    plt.suptitle(f"Feature Coefficients — {name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = f"{save_dir}/coef_{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Coefficient plot saved → %s", path)
    return path


def _extract_coef(clf) -> np.ndarray:
    """Extract 1-D coefficient array from various classifier types."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.svm import LinearSVC
    from sklearn.linear_model import LogisticRegression

    if isinstance(clf, LogisticRegression):
        return clf.coef_[0]
    elif isinstance(clf, LinearSVC):
        return clf.coef_[0]
    elif isinstance(clf, CalibratedClassifierCV):
        base = clf.estimator
        if hasattr(base, "coef_"):
            return base.coef_[0]
        # Average calibrated classifiers
        coefs = [c.coef_[0] for c in clf.calibrated_classifiers_
                 if hasattr(c, "coef_")]
        if coefs:
            return np.mean(coefs, axis=0)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 2. SHAP — TF-IDF + tree models (XGBoost)
# ─────────────────────────────────────────────────────────────────────────────

def explain_with_shap(
    pipeline,
    name:          str,
    X_sample:      np.ndarray,
    n_samples:     int  = 200,
    save_dir:      str  = "outputs",
) -> str:
    """
    Generate SHAP summary plot for a TF-IDF + XGBoost pipeline.

    Best suited for tree-based models. For linear models, use
    plot_coefficients() instead.

    Args:
        pipeline  : fitted Pipeline with 'tfidf' step
        name      : model name for title/filename
        X_sample  : text array to explain (subset for speed)
        n_samples : max samples to use (SHAP is slow on large sets)
        save_dir  : output directory

    Returns:
        Path to saved SHAP summary plot.
    """
    try:
        import shap
    except ImportError:
        raise ImportError("Install SHAP: pip install shap")

    os.makedirs(save_dir, exist_ok=True)

    if "tfidf" not in pipeline.named_steps:
        raise ValueError("explain_with_shap requires a 'tfidf' step in the pipeline.")

    # Transform text → TF-IDF features
    tfidf       = pipeline.named_steps["tfidf"]
    clf         = pipeline.named_steps["clf"]
    feature_names = tfidf.get_feature_names_out()

    idx      = np.random.choice(len(X_sample), min(n_samples, len(X_sample)), replace=False)
    X_subset = tfidf.transform(X_sample[idx])

    logger.info("Computing SHAP values for %s (n=%d)...", name, len(idx))

    try:
        # TreeExplainer for XGBoost / tree models
        explainer   = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_subset)
    except Exception:
        # LinearExplainer fallback for linear models
        logger.info("TreeExplainer failed — using LinearExplainer")
        explainer   = shap.LinearExplainer(clf, X_subset)
        shap_values = explainer.shap_values(X_subset)

    # Summary plot
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(
        shap_values, X_subset,
        feature_names=feature_names,
        max_display=20,
        show=False,
        plot_type="bar",
    )
    plt.title(f"SHAP Feature Importance — {name}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = f"{save_dir}/shap_{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("SHAP plot saved → %s", path)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 3. LIME — per-sample explanation (any pipeline)
# ─────────────────────────────────────────────────────────────────────────────

def explain_with_lime(
    pipeline,
    name:       str,
    texts:      np.ndarray,
    labels:     np.ndarray,
    n_samples:  int = 5,
    save_dir:   str = "outputs",
) -> list:
    """
    Generate LIME explanations for n_samples texts.

    Explains which words pushed the prediction toward positive/negative.
    Works with ANY pipeline that has predict_proba — TF-IDF, W2V, or SBERT.

    Args:
        pipeline  : fitted sklearn Pipeline
        texts     : text array
        labels    : true labels (used to pick interesting samples)
        n_samples : number of samples to explain
        save_dir  : directory to save HTML explanation files

    Returns:
        List of dicts with idx, true_label, pred_label, explanation_path.
    """
    try:
        from lime.lime_text import LimeTextExplainer
    except ImportError:
        raise ImportError("Install LIME: pip install lime")

    os.makedirs(save_dir, exist_ok=True)

    if not hasattr(pipeline, "predict_proba"):
        raise ValueError(
            "LIME requires predict_proba. Wrap LinearSVC in CalibratedClassifierCV."
        )

    explainer  = LimeTextExplainer(class_names=["negative", "positive"])
    label_map  = {0: "negative", 1: "positive"}
    y_pred     = pipeline.predict(texts)

    # Pick a mix: correct + incorrect predictions
    correct_idx   = np.where(y_pred == labels)[0]
    incorrect_idx = np.where(y_pred != labels)[0]
    n_correct     = min(n_samples // 2, len(correct_idx))
    n_incorrect   = min(n_samples - n_correct, len(incorrect_idx))

    chosen = np.concatenate([
        np.random.choice(correct_idx,   n_correct,   replace=False) if n_correct   > 0 else [],
        np.random.choice(incorrect_idx, n_incorrect, replace=False) if n_incorrect > 0 else [],
    ]).astype(int)

    results = []
    for i, idx in enumerate(chosen):
        text       = str(texts[idx])
        true_label = label_map[int(labels[idx])]
        pred_label = label_map[int(y_pred[idx])]
        correct    = true_label == pred_label

        logger.info(
            "LIME [%d/%d] idx=%d | true=%s | pred=%s | correct=%s",
            i + 1, len(chosen), idx, true_label, pred_label, correct
        )

        exp = explainer.explain_instance(
            text,
            pipeline.predict_proba,
            num_features=10,
            num_samples=500,
        )

        # Save HTML
        html_path = f"{save_dir}/lime_{name}_sample{idx}.html"
        exp.save_to_file(html_path)

        # Print top words to console
        print(f"\nLIME — {name} | idx={idx} | true={true_label} | pred={pred_label}")
        for word, weight in exp.as_list():
            direction = "→ POS" if weight > 0 else "→ NEG"
            print(f"  {word:<20} {weight:+.4f}  {direction}")

        results.append({
            "idx":              idx,
            "true_label":       true_label,
            "pred_label":       pred_label,
            "correct":          correct,
            "explanation_path": html_path,
        })

    logger.info("LIME explanations saved to %s/", save_dir)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Bulk explainability: run all three methods for best model
# ─────────────────────────────────────────────────────────────────────────────

def explain_best_model(
    pipeline,
    name:      str,
    X_test:    np.ndarray,
    y_test:    np.ndarray,
    save_dir:  str = "outputs",
):
    """
    Run all applicable explainability methods for the best model.

    - Coefficient plot if linear model with TF-IDF
    - SHAP if XGBoost with TF-IDF
    - LIME for any model (5 samples)

    Args:
        pipeline : fitted best pipeline
        name     : e.g. 'tfidf_linearsvc_tuned'
        X_test   : test texts
        y_test   : test labels
        save_dir : output directory
    """
    paths = {}

    # Coefficient plot (linear + tfidf only)
    if "tfidf" in pipeline.named_steps:
        clf = pipeline.named_steps["clf"]
        if _extract_coef(clf) is not None:
            try:
                paths["coef"] = plot_coefficients(pipeline, name, save_dir=save_dir)
            except Exception as e:
                logger.warning("Coefficient plot failed: %s", e)

        # SHAP (XGBoost + tfidf)
        from xgboost import XGBClassifier
        if isinstance(clf, XGBClassifier):
            try:
                paths["shap"] = explain_with_shap(pipeline, name, X_test, save_dir=save_dir)
            except Exception as e:
                logger.warning("SHAP failed: %s", e)

    # LIME (any pipeline with predict_proba)
    try:
        paths["lime"] = explain_with_lime(pipeline, name, X_test, y_test,
                                           n_samples=5, save_dir=save_dir)
    except Exception as e:
        logger.warning("LIME failed: %s", e)

    return paths


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    texts  = np.array(["great film loved brilliant acting"] * 80 +
                       ["terrible movie awful boring waste"] * 80)
    labels = np.array([1] * 80 + [0] * 80)

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf",   LogisticRegression(max_iter=200)),
    ])
    pipe.fit(texts, labels)

    plot_coefficients(pipe, "test_logreg", top_n=10, save_dir="/tmp/xpl_test")
    explain_with_lime(pipe, "test_logreg", texts[:20], labels[:20],
                      n_samples=2, save_dir="/tmp/xpl_test")

    print("\n✓ explainability.py self-test passed")
