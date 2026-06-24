"""
Tokenizes and splits the processed dataset.

Usage:
    python prepare_data.py --input data/processed/dataset.csv --output data/processed/
"""

import argparse
import logging
import sys

from dataset import build_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      required=True,            help="Path to dataset.csv")
    parser.add_argument("--output",     default="data/processed/", help="Output directory")
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    from pathlib import Path
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    dataset_dict, sampler, tokenizer = build_dataset(
        csv_path=args.input,
        tokenizer_name=args.model_name,
        max_length=args.max_length,
    )

    dataset_dict.save_to_disk(str(out / "dataset"))
    tokenizer.save_pretrained(str(out / "tokenizer"))
    logger.info(f"Tokenized dataset → {out / 'dataset'}")
    logger.info(f"Tokenizer         → {out / 'tokenizer'}")


if __name__ == "__main__":
    main()
