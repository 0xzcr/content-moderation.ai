"""
Evaluation: full classification report, confusion matrix, AUC-ROC, error analysis.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from datasets import Dataset, DatasetDict

logger = logging.getLogger(__name__)

LABELS = ["safe", "toxic"]
ID2LABEL = {0: "safe", 1: "toxic"}


def get_predictions(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    dataset: Dataset,
    batch_size: int = 64,
    device: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    for i in range(0, len(dataset), batch_size):
        batch = dataset[i : i + batch_size]
        # input_ids      = torch.tensor(batch["input_ids"]).to(device)
        # attention_mask = torch.tensor(batch["attention_mask"]).to(device) <----deprecated
        input_ids= batch["input_ids"].clone().detach().to(device) # new version
        attention_mask = batch["attention_mask"].clone().detach().to(device)
        
        labels = batch["labels"]

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
        all_labels.extend(labels if isinstance(labels, list) else labels.tolist())
        all_preds.extend(preds.tolist())
        all_probs.append(probs)

    return np.array(all_labels), np.array(all_preds), np.vstack(all_probs)


def full_report(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    pred_probs: np.ndarray,
    output_dir: Path,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    report_str  = classification_report(true_labels, pred_labels, target_names=LABELS, digits=4, zero_division=0)
    report_dict = classification_report(true_labels, pred_labels, target_names=LABELS, output_dict=True, zero_division=0)
    logger.info(f"\n{report_str}")
    (output_dir / "classification_report.txt").write_text(report_str)

    try:
        auc = roc_auc_score(true_labels, pred_probs[:, 1])
        report_dict["auc_roc"] = auc
        logger.info(f"AUC-ROC: {auc:.4f}")
    except ValueError as e:
        logger.warning(f"AUC-ROC failed: {e}")

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(report_dict, f, indent=2)

    _plot_confusion_matrix(true_labels, pred_labels, output_dir)
    _plot_roc_curve(true_labels, pred_probs, output_dir)
    return report_dict


def _plot_confusion_matrix(true_labels, pred_labels, output_dir):
    cm      = confusion_matrix(true_labels, pred_labels)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, matrix, fmt, title in [
        (axes[0], cm,      "d",    "Counts"),
        (axes[1], cm_norm, ".2f",  "Normalised (recall)"),
    ]:
        ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=LABELS).plot(
            ax=ax, cmap="Blues", values_format=fmt, colorbar=False
        )
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_roc_curve(true_labels, pred_probs, output_dir):
    fpr, tpr, _ = roc_curve(true_labels, pred_probs[:, 1])
    auc = roc_auc_score(true_labels, pred_probs[:, 1])
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color="crimson", lw=2, label=f"toxic (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close()


def error_analysis(
    texts: list[str],
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    pred_probs: np.ndarray,
    output_dir: Path,
    n_samples: int = 50,
) -> pd.DataFrame:
    mask = true_labels != pred_labels
    confidence = pred_probs.max(axis=1)
    errors_df = pd.DataFrame({
        "text":        np.array(texts)[mask],
        "true_label":  [ID2LABEL[i] for i in true_labels[mask]],
        "pred_label":  [ID2LABEL[i] for i in pred_labels[mask]],
        "confidence":  confidence[mask],
        "safe_prob":   pred_probs[mask, 0],
        "toxic_prob":  pred_probs[mask, 1],
    })
    errors_df = errors_df.sort_values("confidence", ascending=False).head(n_samples)
    errors_df.to_csv(output_dir / "error_analysis.csv", index=False)
    logger.info(f"{mask.sum()} total errors — top {n_samples} saved to {output_dir / 'error_analysis.csv'}")
    return errors_df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/best_model/")
    parser.add_argument("--dataset",    default="data/processed/dataset/")
    parser.add_argument("--output",     default="outputs/evaluation/")
    parser.add_argument("--split",      default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    model     = AutoModelForSequenceClassification.from_pretrained(args.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    dataset   = DatasetDict.load_from_disk(args.dataset)[args.split]

    true_labels, pred_labels, pred_probs = get_predictions(model, tokenizer, dataset)
    full_report(true_labels, pred_labels, pred_probs, Path(args.output))

    raw_df = pd.read_csv("data/processed/dataset.csv")
    texts  = raw_df["text"].tolist()[:len(true_labels)]
    error_analysis(texts, true_labels, pred_labels, pred_probs, Path(args.output))
