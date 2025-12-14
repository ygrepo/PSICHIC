import argparse
import sys
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils import (
    setup_logging,
    get_logger,
    read_csv_parquet_torch,
    save_csv_parquet_torch,
)

logger = get_logger(__name__)


def load_data(path: str) -> pd.DataFrame:
    df = joblib.load(path).reset_index(drop=True)
    logger.info(f"Columns: {df.columns}")
    logger.info(f"Head: {df.head()}")
    df.rename(
        columns={
            "ec": "ID",
            "log10_value": "regression_label",
            "Sequence": "Protein",
            "Smiles": "Ligand",
        },
        inplace=True,
    )
    cols = [
        "ID",
        "Ligand",
        "Protein",
        "regression_label",
    ]
    df = df[cols]
    return df


# Function to perform cold split
def cold_split(
    unique_items: list,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state=42,
) -> tuple[list, list, list]:
    """
    Perform cold split on unique items.
    Returns train, val, test items.
    """
    train_items, test_items = train_test_split(
        unique_items, test_size=test_size, random_state=random_state
    )
    # Adjust val_size to account for test_size
    val_ratio = val_size / (1 - test_size)
    logger.info(f"val_ratio: {val_ratio}")
    train_items, val_items = train_test_split(
        train_items, test_size=val_ratio, random_state=random_state
    )
    return train_items, val_items, test_items


def split_data(
    df: pd.DataFrame,
    split_mode: str,
    drug_col: str,
    protein_col: str,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    logger.info(f"Splitting data...{split_mode}")
    # Get unique drugs and proteins
    unique_drugs = df[drug_col].unique()
    unique_proteins = df[protein_col].unique()

    logger.info(f"Total samples: {len(df)}")
    logger.info(f"Unique drugs: {len(unique_drugs)}")
    logger.info(f"Unique proteins: {len(unique_proteins)}")
    if split_mode == "drug":
        train_drugs, val_drugs, test_drugs = cold_split(
            unique_drugs, test_size, val_size, random_state
        )
        train_drugs_set = set(train_drugs)
        val_drugs_set = set(val_drugs)
        test_drugs_set = set(test_drugs)
        assert (
            len(train_drugs_set & val_drugs_set) == 0
        ), "Overlap between train and val drugs!"
        assert (
            len(train_drugs_set & test_drugs_set) == 0
        ), "Overlap between train and test drugs!"
        assert (
            len(val_drugs_set & test_drugs_set) == 0
        ), "Overlap between val and test drugs!"
        train_df = df[df[drug_col].isin(train_drugs)].copy()
        val_df = df[df[drug_col].isin(val_drugs)].copy()
        test_df = df[df[drug_col].isin(test_drugs)].copy()
    elif split_mode == "protein":
        train_proteins, val_proteins, test_proteins = cold_split(
            unique_proteins, test_size, val_size, random_state
        )
        train_proteins_set = set(train_proteins)
        val_proteins_set = set(val_proteins)
        test_proteins_set = set(test_proteins)
        assert (
            len(train_proteins_set & val_proteins_set) == 0
        ), "Overlap between train and val proteins!"
        assert (
            len(train_proteins_set & test_proteins_set) == 0
        ), "Overlap between train and test proteins!"
        assert (
            len(val_proteins_set & test_proteins_set) == 0
        ), "Overlap between val and test proteins!"
        train_df = df[df[protein_col].isin(train_proteins)].copy()
        val_df = df[df[protein_col].isin(val_proteins)].copy()
        test_df = df[df[protein_col].isin(test_proteins)].copy()
    else:
        raise ValueError(f"Unknown split mode: {split_mode}")
    # Reset indices
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Print split statistics
    logger.info(
        f"    Train: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)"
    )
    logger.info(f"    Val: {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
    logger.info(f"    Test: {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")

    return train_df, val_df, test_df


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Create train, val, test data")
    parser.add_argument(
        "--N",
        type=int,
        default=0,
        help="Number of rows to process (for testing)",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Path to the dataset directory",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="dataset",
        help="Name of the dataset",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Label to use for regression",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Path to output directory",
    )
    parser.add_argument(
        "--split_mode",
        type=str,
        default="drug",
        help="Split mode (drug or protein)",
    )
    parser.add_argument(
        "--log_fn",
        type=Path,
        default=None,
        help="Path to save log file",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="Logging level (e.g., 'INFO', 'DEBUG')",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def process_data(
    df: pd.DataFrame,
    split_mode: str,
    dataset_name: str,
    output_dir: Path,
    seed: int,
):
    """Process the data and save it to the output directory."""
    train_df, val_df, test_df = split_data(
        df, split_mode, "Ligand", "Protein", random_state=seed
    )
    output_dir = output_dir.resolve()
    output_file = output_dir / f"{dataset_name}_{split_mode}" / "train.csv"
    logger.info(f"Saving train to: {output_file}")
    save_csv_parquet_torch(train_df, output_file)
    output_file = output_dir / f"{dataset_name}_{split_mode}" / "val.csv"
    logger.info(f"Saving val to: {output_file}")
    save_csv_parquet_torch(val_df, output_file)
    output_file = output_dir / f"{dataset_name}_{split_mode}" / "test.csv"
    logger.info(f"Saving test to: {output_file}")
    save_csv_parquet_torch(test_df, output_file)


def main():
    args = parse_args()
    setup_logging(Path(args.log_fn), args.log_level)
    try:
        # Log configuration
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Logging to: {args.log_fn}")
        data_dir = args.data_dir.resolve()
        logger.info(f"Data directory: {args.data_dir}")
        logger.info(f"Output directory: {args.output_dir}")
        logger.info(f"Dataset name: {args.dataset_name}")
        logger.info(f"Label: {args.label}")
        logger.info(f"Split mode: {args.split_mode}")

        # Load data
        data_fn = data_dir / f"dataset_{args.dataset_name}_{args.label}"
        data_fn = data_fn / "A01_dataset"
        data_fn = data_fn / f"{args.label}_with_features.joblib"
        logger.info(f"Data file: {data_fn}")
        df = load_data(data_fn)
        logger.info(f"Loaded {len(df)} samples")
        if args.N > 0:
            df = df.head(n=args.N)
            logger.info(f"Selected {len(df)} samples")
        process_data(
            df, args.split_mode, args.dataset_name, Path(args.output_dir), args.seed
        )
        logger.info("Data processing completed successfully.")

    except Exception as e:
        logger.exception("Script failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
