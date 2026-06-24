"""
Tests for data pipeline.
Run: pytest test_data_pipeline.py -v
"""

import numpy as np
import pandas as pd
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from dataset import LABEL2ID, clean_text, encode_labels, make_weighted_sampler, stratified_split


def test_clean_text_strips_html():
    assert clean_text("<b>Hello</b>") == "Hello"

def test_clean_text_replaces_urls():
    assert "[URL]" in clean_text("Visit https://example.com for more")

def test_clean_text_replaces_mentions():
    assert "[USER]" in clean_text("@john you are great")

def test_clean_text_handles_empty():
    assert clean_text("") == ""
    assert clean_text("   ") == ""

def test_clean_text_preserves_punctuation():
    result = clean_text("Wow!!! Really???")
    assert "!" in result and "?" in result

def test_clean_text_handles_non_string():
    assert clean_text(None) == ""  # type: ignore


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "text":  ["post1", "post2", "post3", "post4"],
        "label": ["safe", "toxic", "safe", "toxic"],
    })

def test_encode_labels_maps_correctly(sample_df):
    result = encode_labels(sample_df, "label", LABEL2ID)
    assert result["label"].tolist() == [0, 1, 0, 1]

def test_encode_labels_raises_on_unknown(sample_df):
    bad = sample_df.copy()
    bad.loc[0, "label"] = "spam"
    with pytest.raises(ValueError, match="Unknown labels"):
        encode_labels(bad, "label", LABEL2ID)

def test_encode_labels_dtype_is_int(sample_df):
    result = encode_labels(sample_df, "label", LABEL2ID)
    assert result["label"].dtype == int


@pytest.fixture
def large_df():
    n = 1000
    labels = ["safe"] * 800 + ["toxic"] * 200
    df = pd.DataFrame({"text": [f"post {i}" for i in range(n)], "label": labels})
    df["label"] = df["label"].map(LABEL2ID)
    return df.sample(frac=1, random_state=42).reset_index(drop=True)

def test_split_sizes(large_df):
    train, val, test = stratified_split(large_df, 0.80, 0.10)
    total = len(train) + len(val) + len(test)
    assert total == len(large_df)
    assert abs(len(train) / total - 0.80) < 0.02

def test_split_is_stratified(large_df):
    train, val, test = stratified_split(large_df, 0.80, 0.10)
    orig = large_df["label"].value_counts(normalize=True).sort_index()
    for split in [train, val, test]:
        dist = split["label"].value_counts(normalize=True).sort_index()
        for idx in orig.index:
            assert abs(dist.get(idx, 0) - orig[idx]) < 0.05

def test_split_no_overlap(large_df):
    train, val, test = stratified_split(large_df)
    assert set(train.index).isdisjoint(set(val.index))
    assert set(train.index).isdisjoint(set(test.index))

def test_weighted_sampler_length():
    labels = [0] * 800 + [1] * 200
    sampler = make_weighted_sampler(labels)
    assert len(sampler) == 1000

def test_weighted_sampler_minority_upweighted():
    labels = [0] * 800 + [1] * 200
    sampler = make_weighted_sampler(labels)
    weights = list(sampler.weights)
    assert weights[800] > weights[0]  # toxic weight > safe weight
