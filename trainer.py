"""
Fine-tuning DistilBERT for binary sequence classification (safe / toxic).
"""

from __future__ import annotations

import logging
from pathlib import Path

import mlflow
import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import WeightedRandomSampler
import inspect
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from datasets import DatasetDict

logger = logging.getLogger(__name__)

NUM_LABELS = 2
LABEL2ID = {"safe": 0, "toxic": 1}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


# ---------------------------------------------------------------------------
# Custom Trainer — injects WeightedRandomSampler
# ---------------------------------------------------------------------------


class WeightedTrainer(Trainer):
    def __init__(self, *args, train_sampler: WeightedRandomSampler, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._train_sampler = train_sampler

    def _get_train_sampler(self, dataset) -> WeightedRandomSampler: #takes two args not just 'self'
        return self._train_sampler


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(eval_pred: tuple) -> dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    probs = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=1).numpy()

    macro_f1  = f1_score(labels, preds, average="macro", zero_division=0)
    precision = precision_score(labels, preds, average="macro", zero_division=0)
    recall    = recall_score(labels, preds, average="macro", zero_division=0)
    toxic_f1  = f1_score(labels, preds, pos_label=1, average="binary", zero_division=0)
    toxic_recall = recall_score(labels, preds, pos_label=1, average="binary", zero_division=0)

    try:
        auc = roc_auc_score(labels, probs[:, 1])
    except ValueError:
        auc = 0.0

    return {
        "eval_macro_f1":     macro_f1,
        "eval_precision":    precision,
        "eval_recall":       recall,
        "eval_toxic_f1":     toxic_f1,
        "eval_toxic_recall": toxic_recall,
        "eval_auc_roc":      auc,
    }


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------


def load_model(
    model_name: str = "distilbert-base-uncased",
    num_labels: int = NUM_LABELS,
    dropout: float = 0.1,
) -> AutoModelForSequenceClassification:
    # model = AutoModelForSequenceClassification.from_pretrained(   #gave error    
    #     num_labels=num_labels,
    #     id2label=ID2LABEL,
    #     label2id=LABEL2ID,
    #     hidden_dropout_prob=dropout,
    #     attention_probs_dropout_prob=dropout,
    #     ignore_mismatched_sizes=True,
    # )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model loaded: {trainable/1e6:.1f}M trainable params")
    return model


# ---------------------------------------------------------------------------
# Training arguments
# ---------------------------------------------------------------------------


def build_training_args(output_dir: str | Path, **kwargs) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=kwargs.get("num_epochs", 5),
        per_device_train_batch_size=kwargs.get("train_batch_size", 32),
        per_device_eval_batch_size=kwargs.get("eval_batch_size", 64),
        learning_rate=kwargs.get("learning_rate", 2e-5),
        warmup_ratio=kwargs.get("warmup_ratio", 0.06),
        weight_decay=kwargs.get("weight_decay", 0.01),
        max_grad_norm=kwargs.get("gradient_clip", 1.0),
        fp16=kwargs.get("fp16", False),
        eval_strategy="epoch",
        #eval_steps=kwargs.get("eval_steps", 500),
        save_strategy="epoch",
        #save_steps=kwargs.get("save_steps", 500),
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="mlflow",
        logging_steps=100,
        seed=42,
    )


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------


def train(
    dataset_dict: DatasetDict,
    train_sampler: WeightedRandomSampler,
    tokenizer: PreTrainedTokenizerBase,
    model_name: str = "distilbert-base-uncased",
    output_dir: str | Path = "outputs/",
    mlflow_experiment: str = "content-moderation",
    mlflow_tracking_uri: str = "mlruns/",
    **training_kwargs,
) -> tuple[WeightedTrainer, AutoModelForSequenceClassification]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_experiment)

    model = load_model(model_name)
    args  = build_training_args(output_dir, **training_kwargs)

    with mlflow.start_run():
        trainer = WeightedTrainer(
            model=model,
            args=args,
            train_dataset=dataset_dict["train"],
            eval_dataset=dataset_dict["val"],
            processing_class=tokenizer,
            compute_metrics=compute_metrics,
            train_sampler=train_sampler,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=training_kwargs.get("early_stopping_patience", 3))],
        )

        logger.info("Starting training...")
        trainer.train()

        logger.info("Evaluating on test set...")
        test_results = trainer.evaluate(dataset_dict["test"], metric_key_prefix="test")
        mlflow.log_metrics(test_results)
        logger.info(f"Test results: {test_results}")

        best_path = output_dir / "best_model"
        trainer.save_model(str(best_path))
        tokenizer.save_pretrained(str(best_path))
        mlflow.log_artifacts(str(best_path), artifact_path="best_model")
        logger.info(f"Best model saved to {best_path}")

    return trainer, model

#print(inspect.signature(TrainingArguments.__init__))