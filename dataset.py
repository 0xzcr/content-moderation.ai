"""
Data pipeline: raw CSV → cleaned → tokenized → HuggingFace Dataset splits.

Design decisions:
- Stratified split preserves class ratios across train/val/test.
- WeightedRandomSampler so the model sees balanced mini-batches.
- Text cleaning is minimal: strip HTML/URLs but keep punctuation and casing.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import WeightedRandomSampler
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from datasets import Dataset, DatasetDict

logger = logging.getLogger(__name__)

LABEL2ID: dict[str, int] = {"safe": 0, "toxic": 1}
ID2LABEL: dict[int, str] = {v: k for k, v in LABEL2ID.items()}


class SplitSizes(TypedDict):
    train: int
    val: int
    test: int


# ---------------------------------------------------------------------------
# Step 1: Text cleaning
# ---------------------------------------------------------------------------

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_HTML_RE = re.compile(r"<[^>]+>")
_MULTI_SPACE = re.compile(r"\s{2,}")
_USER_MENTION = re.compile(r"@\w+")


def clean_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    text = _HTML_RE.sub(" ", text)
    text = _USER_MENTION.sub("[USER]", text)
    text = _URL_RE.sub("[URL]", text)
    text = _MULTI_SPACE.sub(" ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Step 2: Label encoding
# ---------------------------------------------------------------------------


def encode_labels(df: pd.DataFrame, label_col: str, label2id: dict[str, int]) -> pd.DataFrame:
    unknown = set(df[label_col].unique()) - set(label2id)
    if unknown:
        raise ValueError(f"Unknown labels in dataset: {unknown}. Expected: {set(label2id)}")
    df = df.copy()
    df["label"] = df[label_col].map(label2id).astype(int)
    label_counts = df["label"].value_counts().sort_index()
    for idx, count in label_counts.items():
        logger.info(f"  {ID2LABEL.get(int(idx), str(idx)):<8} (id={idx}): {count:>7,} samples ({count/len(df)*100:.1f}%)")
    return df


# ---------------------------------------------------------------------------
# Step 3: Stratified train / val / test split
# ---------------------------------------------------------------------------


def stratified_split(
    df: pd.DataFrame,
    train_frac: float = 0.80,
    val_frac: float = 0.10,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    test_frac = round(1.0 - train_frac - val_frac, 6)
    assert test_frac > 0, "train_frac + val_frac must be < 1.0"

    trainval_df, test_df = train_test_split(
        df, test_size=test_frac, random_state=random_seed, stratify=df["label"]
    )
    val_of_trainval = val_frac / (train_frac + val_frac)
    train_df, val_df = train_test_split(
        trainval_df, test_size=val_of_trainval, random_state=random_seed, stratify=trainval_df["label"]
    )
    logger.info(f"Split → train: {len(train_df):,} | val: {len(val_df):,} | test: {len(test_df):,}")
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 4: Tokenization
# ---------------------------------------------------------------------------


def tokenize_dataset(
    df: pd.DataFrame,
    tokenizer: PreTrainedTokenizerBase,
    text_col: str = "text",
    max_length: int = 128,
) -> Dataset:
    dataset = Dataset.from_pandas(df[[text_col, "label"]], preserve_index=False)

    def _tokenize(batch: dict) -> dict:
        return tokenizer(
            batch[text_col],
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
        )

    tokenized = dataset.map(_tokenize, batched=True, batch_size=1000, remove_columns=[text_col], desc="Tokenizing")
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized


# ---------------------------------------------------------------------------
# Step 5: Class imbalance — WeightedRandomSampler
# ---------------------------------------------------------------------------


def make_weighted_sampler(labels: list[int] | np.ndarray) -> WeightedRandomSampler:
    labels_arr = np.array(labels)
    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(labels_arr), y=labels_arr
    )
    logger.info("Class weights for sampler:")
    for i, w in enumerate(class_weights):
        logger.info(f"  {ID2LABEL.get(i, str(i))}: {w:.4f}")
    sample_weights = class_weights[labels_arr]
    return WeightedRandomSampler(weights=sample_weights.tolist(), num_samples=len(sample_weights), replacement=True)


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------


def build_dataset(
    csv_path: str | Path,
    tokenizer_name: str = "distilbert-base-uncased",
    text_col: str = "text",
    label_col: str = "label",
    label2id: dict[str, int] | None = None,
    max_length: int = 128,
    train_frac: float = 0.80,
    val_frac: float = 0.10,
    random_seed: int = 42,
) -> tuple[DatasetDict, WeightedRandomSampler, AutoTokenizer]:
    if label2id is None:
        label2id = LABEL2ID

    logger.info(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df):,} rows")

    logger.info("Cleaning text...")
    df["text"] = df[text_col].map(clean_text)
    before = len(df)
    df = df[df["text"].str.len() > 0].copy()
    logger.info(f"Dropped {before - len(df):,} empty rows after cleaning")

    logger.info("Encoding labels...")
    df = encode_labels(df, label_col, label2id)

    logger.info("Splitting dataset...")
    train_df, val_df, test_df = stratified_split(df, train_frac, val_frac, random_seed)

    logger.info(f"Tokenizing (max_length={max_length})...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    train_tok = tokenize_dataset(train_df, tokenizer, max_length=max_length)
    val_tok   = tokenize_dataset(val_df,   tokenizer, max_length=max_length)
    test_tok  = tokenize_dataset(test_df,  tokenizer, max_length=max_length)

    dataset_dict = DatasetDict({"train": train_tok, "val": val_tok, "test": test_tok})

    logger.info("Building weighted sampler...")
    sampler = make_weighted_sampler(train_df["label"].tolist())

    logger.info("Data pipeline complete.")
    return dataset_dict, sampler, tokenizer
