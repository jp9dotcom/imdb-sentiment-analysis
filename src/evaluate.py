"""
evaluate.py
===========
Full evaluation suite for trained IMDb sentiment pipelines.

Covers:
  - Accuracy, Precision, Recall, Macro F1, ROC-AUC
  - Confusion matrix (plot + raw)
  - Classification report
  - Inference speed benchmark
  - Error analysis (worst-N misclassified samples)
  - Cross-model comparison table

Usage:
    from src.evaluate import evaluate_pipeline, compare_models, error_analysis
"""

import time
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report,
    RocCurveDisplay,
)

logger = logging.getLogger(__name__)
matplotlib.use("Agg")


# ─────────────────────────────────────────────────────────────────────────────
# Core evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_pipeline(
    name:      str,
    pipeline,
    X:         np.ndarray,
    y:         np.ndarray,
    split:     str = "test",
    n_speed:   int = 200,
    save_dir:  str = "outputs",
) -> dict:
    """
    Full evaluation of one pipeline on a given split.

    Args:
        name      : model name e.g. 'tfidf_linearsvc'
        pipeline  : fitted sklearn Pipeline
        X         : text array
        y         : true labels
        split     : label for plots e.g. 'val' or 'test'
        n_speed   : number of samples for inference speed benchmark
        save_dir  : directory to save plots

    Returns:
        dict with all metrics + paths to saved plots
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    y_pred = pipeline.predict(X)

    # Probabilities for ROC-AUC
    if hasattr(pipeline, "predict_proba"):
        y_prob = pipeline.predict_proba(X)[:, 1]
        roc_auc = round(roc_auc_score(y, y_prob), 4)
    else:
        y_prob  = None
        roc_auc = None
        logger.warning("%s: predict_proba not available — ROC-AUC skipped", name)

    metrics = {
        "accuracy":  round(accuracy_score(y, y_pred), 4),
        "macro_f1":  round(f1_score(y, y_pred, average="macro"), 4),
        "precision": round(precision_score(y, y_pred, average="macro"), 4),
        "recall":    round(recall_score(y, y_pred, average="macro"), 4),
        "roc_auc":   roc_auc,
    }

    logger.info(
        "%-35s | acc: %.4f | f1: %.4f | auc: %s",
        name, metrics["accuracy"], metrics["macro_f1"],
        f"{roc_auc:.4f}" if roc_auc else "N/A"
    )

    # Classification report
    report = classification_report(y, y_pred, target_names=["negative", "positive"])
    print(f"\n── {name} ── {split} set ──")
    print(report)

    # Confusion matrix plot
    cm_path = _plot_confusion_matrix(name, y, y_pred, split, save_dir)

    # ROC curve plot
    roc_path = None
    if y_prob is not None:
        roc_path = _plot_roc_curve(name, y, y_prob, split, save_dir)

    # Inference speed
    inference_ms = _benchmark_inference(pipeline, X, n_speed)
    metrics["inference_ms_per_sample"] = inference_ms

    return {
        "name":           name,
        "split":          split,
        "metrics":        metrics,
        "cm_path":        cm_path,
        "roc_path":       roc_path,
        "y_pred":         y_pred,
        "y_prob":         y_prob,
        "report":         report,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Confusion matrix
# ─────────────────────────────────────────────────────────────────────────────

def _plot_confusion_matrix(name, y_true, y_pred, split, save_dir) -> str:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
        ax=ax, linewidths=0.5,
    )
    ax.set_title(f"{name}\nConfusion Matrix ({split})", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    path = f"{save_dir}/cm_{name}_{split}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved confusion matrix → %s", path)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# ROC curve
# ─────────────────────────────────────────────────────────────────────────────

def _plot_roc_curve(name, y_true, y_prob, split, save_dir) -> str:
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_true, y_prob, ax=ax, name=name)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_title(f"ROC Curve — {name} ({split})", fontsize=11, fontweight="bold")
    plt.tight_layout()
    path = f"{save_dir}/roc_{name}_{split}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved ROC curve → %s", path)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Inference speed benchmark
# ─────────────────────────────────────────────────────────────────────────────

def _benchmark_inference(pipeline, X, n: int = 200) -> float:
    """
    Returns average inference time in milliseconds per sample.
    Runs on a random subset of n samples to keep it fast.
    """
    idx   = np.random.choice(len(X), min(n, len(X)), replace=False)
    batch = X[idx]
    t0    = time.perf_counter()
    pipeline.predict(batch)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    ms_per_sample = round(elapsed_ms / len(batch), 3)
    logger.info("Inference speed: %.3f ms/sample (n=%d)", ms_per_sample, len(batch))
    return ms_per_sample


# ─────────────────────────────────────────────────────────────────────────────
# Error analysis
# ─────────────────────────────────────────────────────────────────────────────

def error_analysis(
    name:       str,
    pipeline,
    X:          np.ndarray,
    y:          np.ndarray,
    raw_texts:  np.ndarray = None,
    top_n:      int = 20,
    save_dir:   str = "outputs",
) -> pd.DataFrame:
    """
    Identify and analyse misclassified samples.

    Groups errors into:
      - False Positives: predicted positive, actually negative
      - False Negatives: predicted negative, actually positive

    For each error, logs: text length, confidence (if available), raw text.

    Args:
        raw_texts : original (uncleaned) reviews for readable output.
                    Falls back to cleaned texts if not provided.
        top_n     : number of errors to show per error type.

    Returns:
        DataFrame of misclassified samples with columns:
        idx, true_label, pred_label, confidence, word_count, text
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    y_pred = pipeline.predict(X)
    texts  = raw_texts if raw_texts is not None else X

    # Confidence scores
    if hasattr(pipeline, "predict_proba"):
        y_prob = pipeline.predict_proba(X)[:, 1]
    else:
        y_prob = np.full(len(X), np.nan)

    label_map = {0: "negative", 1: "positive"}
    errors = []
    for i, (yt, yp) in enumerate(zip(y, y_pred)):
        if yt != yp:
            errors.append({
                "idx":        i,
                "true_label": label_map[yt],
                "pred_label": label_map[yp],
                "confidence": round(float(y_prob[i]), 4) if not np.isnan(y_prob[i]) else None,
                "word_count": len(str(texts[i]).split()),
                "text":       str(texts[i])[:300],
            })

    df_errors = pd.DataFrame(errors)

    if df_errors.empty:
        logger.info("%s: No errors found (perfect classifier on this split)", name)
        return df_errors

    total_errors  = len(df_errors)
    error_rate    = round(total_errors / len(y) * 100, 2)
    false_pos     = df_errors[df_errors["pred_label"] == "positive"]
    false_neg     = df_errors[df_errors["pred_label"] == "negative"]

    print(f"\n── Error Analysis: {name} ──")
    print(f"Total errors : {total_errors}/{len(y)}  ({error_rate}%)")
    print(f"False Positives (predicted pos, actually neg): {len(false_pos)}")
    print(f"False Negatives (predicted neg, actually pos): {len(false_neg)}")

    # Short review errors (< 30 words) — model struggles with brevity
    short_errors = df_errors[df_errors["word_count"] < 30]
    print(f"Errors on short reviews (<30 words): {len(short_errors)} ({round(len(short_errors)/total_errors*100,1)}%)")

    print(f"\nTop {min(top_n, len(false_pos))} False Positives:")
    print(false_pos.head(top_n)[["true_label", "pred_label", "confidence", "word_count", "text"]].to_string(index=False))

    print(f"\nTop {min(top_n, len(false_neg))} False Negatives:")
    print(false_neg.head(top_n)[["true_label", "pred_label", "confidence", "word_count", "text"]].to_string(index=False))

    # Save error CSV
    path = f"{save_dir}/errors_{name}.csv"
    df_errors.to_csv(path, index=False)
    logger.info("Error analysis saved → %s", path)

    # Error length distribution plot
    _plot_error_lengths(name, df_errors, save_dir)

    return df_errors


def _plot_error_lengths(name, df_errors, save_dir):
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, grp in df_errors.groupby("pred_label"):
        ax.hist(grp["word_count"], bins=30, alpha=0.7, label=f"pred={label}")
    ax.set_xlabel("Word Count")
    ax.set_ylabel("Error Count")
    ax.set_title(f"Error Distribution by Review Length — {name}", fontsize=11, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    path = f"{save_dir}/error_lengths_{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Cross-model comparison table
# ─────────────────────────────────────────────────────────────────────────────

def compare_models(results: list, save_dir: str = "outputs") -> pd.DataFrame:
    """
    Build a comparison DataFrame from a list of evaluate_pipeline() results.

    Args:
        results  : list of dicts returned by evaluate_pipeline()
        save_dir : directory to save comparison plot

    Returns:
        pd.DataFrame sorted by macro_f1 descending.
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    rows = []
    for r in results:
        m = r["metrics"]
        rows.append({
            "model":          r["name"],
            "accuracy":       m["accuracy"],
            "macro_f1":       m["macro_f1"],
            "precision":      m["precision"],
            "recall":         m["recall"],
            "roc_auc":        m.get("roc_auc"),
            "inference_ms":   m.get("inference_ms_per_sample"),
        })

    df = pd.DataFrame(rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)
    df.index += 1  # rank from 1

    print("\n" + "═" * 85)
    print(f"{'RANK':<5} {'MODEL':<35} {'ACC':>7} {'F1':>7} {'AUC':>7} {'MS/SAMPLE':>10}")
    print("─" * 85)
    for rank, row in df.iterrows():
        auc = f"{row['roc_auc']:.4f}" if row["roc_auc"] else "  N/A "
        ms  = f"{row['inference_ms']:.3f}" if row["inference_ms"] else "  N/A "
        print(f"{rank:<5} {row['model']:<35} {row['accuracy']:>7.4f} {row['macro_f1']:>7.4f} {auc:>7} {ms:>10}")
    print("═" * 85)

    # Bar chart
    _plot_comparison(df, save_dir)

    path = f"{save_dir}/model_comparison.csv"
    df.to_csv(path)
    logger.info("Comparison table saved → %s", path)
    return df


def _plot_comparison(df, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(df)))[::-1]

    # F1 bar chart
    axes[0].barh(df["model"][::-1], df["macro_f1"][::-1], color=colors[::-1])
    axes[0].set_xlabel("Macro F1")
    axes[0].set_title("Model Comparison — Macro F1", fontweight="bold")
    axes[0].set_xlim(df["macro_f1"].min() - 0.02, 1.0)
    for i, (_, row) in enumerate(df[::-1].iterrows()):
        axes[0].text(row["macro_f1"] + 0.001, i, f"{row['macro_f1']:.4f}", va="center", fontsize=8)

    # Inference speed
    if df["inference_ms"].notna().any():
        axes[1].barh(df["model"][::-1], df["inference_ms"][::-1], color=colors[::-1])
        axes[1].set_xlabel("Inference Time (ms/sample)")
        axes[1].set_title("Inference Speed Comparison", fontweight="bold")

    plt.tight_layout()
    path = f"{save_dir}/model_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Comparison plot saved → %s", path)


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline

    texts = np.array([
        "loved this brilliant film great acting outstanding",
        "terrible movie waste of time avoid",
        "amazing story kept me engaged throughout",
        "awful dull boring predictable plot",
        "masterpiece of cinema wonderful experience",
        "horrible acting bad script dreadful",
    ] * 50)
    labels = np.array([1, 0, 1, 0, 1, 0] * 50)

    pipe = Pipeline([("tfidf", TfidfVectorizer()), ("clf", LogisticRegression(max_iter=200))])
    pipe.fit(texts[:200], labels[:200])

    result = evaluate_pipeline("test_tfidf_logreg", pipe, texts[200:], labels[200:],
                                split="test", save_dir="/tmp/eval_test")

    df_err = error_analysis("test_tfidf_logreg", pipe, texts[200:], labels[200:],
                             raw_texts=texts[200:], save_dir="/tmp/eval_test")

    compare_models([result], save_dir="/tmp/eval_test")
    print("\n✓ evaluate.py self-test passed")
