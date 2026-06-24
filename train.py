"""
Fine-tunes DistilBERT on the processed dataset.

Usage:
    python train.py
    python train.py --epochs 3 --lr 3e-5 --batch-size 16
"""

import argparse
import logging
from pathlib import Path

from datasets import DatasetDict
from dataset import make_weighted_sampler
from trainer import train
import os

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true" #needed to run MLflow locally
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    default="data/processed/dataset/")
    parser.add_argument("--tokenizer",  default="data/processed/tokenizer/")
    parser.add_argument("--output",     default="outputs/")
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--epochs",     type=int,   default=5)
    parser.add_argument("--lr",         type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int,   default=32)
    parser.add_argument("--fp16",       action="store_true")
    args = parser.parse_args()

    from transformers import AutoTokenizer
    logger.info(f"Loading dataset from {args.dataset}")
    dataset_dict = DatasetDict.load_from_disk(args.dataset)
    tokenizer    = AutoTokenizer.from_pretrained(args.tokenizer)
    sampler      = make_weighted_sampler(dataset_dict["train"]["labels"])

    train(
        dataset_dict=dataset_dict,
        train_sampler=sampler,
        tokenizer=tokenizer,
        model_name=args.model_name,
        output_dir=args.output,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        train_batch_size=args.batch_size,
        fp16=args.fp16,
    )
    
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
