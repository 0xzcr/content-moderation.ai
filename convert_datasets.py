"""
Converts raw Jigsaw train.csv → binary dataset (safe / toxic).

Usage:
    python convert_datasets.py --input data/train.csv --output data/processed/dataset.csv
"""

import argparse
import pandas as pd
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",        required=True,       help="Path to Jigsaw train.csv")
    parser.add_argument("--output",       required=True,       help="Output path for dataset.csv")
    parser.add_argument("--target-safe",  type=int, default=100_000)
    parser.add_argument("--target-toxic", type=int, default=15_000)
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.input}...")
    df = pd.read_csv(args.input)

    toxic_cols   = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    df["label"]  = df[toxic_cols].max(axis=1).map({1: "toxic", 0: "safe"})
    df           = df.rename(columns={"comment_text": "text"})[["text", "label"]]
    df           = df[df["text"].str.strip().str.len() > 5].dropna()

    toxic = df[df["label"] == "toxic"]
    safe  = df[df["label"] == "safe"]

    toxic_sample = toxic.sample(n=min(args.target_toxic, len(toxic)), random_state=42, replace=len(toxic) < args.target_toxic)
    safe_sample  = safe.sample(n=min(args.target_safe,  len(safe)),  random_state=42)

    final = (
        pd.concat([safe_sample, toxic_sample], ignore_index=True)
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )
    final.to_csv(args.output, index=False)

    print(f"\nSaved {len(final):,} rows → {args.output}")
    print("\nClass distribution:")
    for label, count in final["label"].value_counts().items():
        print(f"  {label:<8}: {count:>7,}  ({count/len(final)*100:.1f}%)")


if __name__ == "__main__":
    main()
