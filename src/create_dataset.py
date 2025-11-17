import sys
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import os
from rdkit import Chem
import networkx as nx
from sklearn.model_selection import train_test_split


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.utils import (
    setup_logging,
    get_logger,
    read_csv_parquet_torch,
    save_csv_parquet_torch,
)

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create dataset")
    parser.add_argument(
        "--log_fn",
        type=Path,
        default=None,
        help="Path to log file (optional)",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )
    parser.add_argument(
        "--data_fn",
        type=Path,
        required=True,
        help="Path to data file",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Path to output directory",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="dataset",
        help="Name of the dataset",
    )
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    setup_logging(Path(args.log_fn), args.log_level)
    try:
        # Log configuration
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Logging to: {args.log_fn}")
        logger.info(f"Data file: {args.data_fn}")
        data_fn = args.data_fn.resolve()
        logger.info(f"Output directory: {args.output_dir}")
        logger.info(f"Dataset name: {args.dataset_name}")

        # Load data
        df = read_csv_parquet_torch(data_fn)
        logger.info(f"Loaded {len(df)} samples")

        process_data(df, args.dataset_name, Path(args.output_dir))

    except Exception as e:
        logger.exception("Script failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
