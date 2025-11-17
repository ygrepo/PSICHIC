import sys
from pathlib import Path
from tqdm import tqdm

import logging
import pandas as pd
import torch


import numpy as np

from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree

# ---- One base for everything ----
BASE_LOGGER = "psichic"
_BASE = logging.getLogger(BASE_LOGGER)  # the only logger we configure here


def setup_logging(log_path: str | Path | None, level: str = "INFO") -> logging.Logger:
    """Configure the base logger once (file + console)."""
    if getattr(_BASE, "_configured", False):
        return _BASE

    _BASE.handlers.clear()
    _BASE.setLevel(getattr(logging, level.upper(), logging.INFO))

    fmt = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Optional file handler
    if log_path:
        fh = logging.FileHandler(str(log_path), encoding="utf-8")
        fh.setFormatter(fmt)
        _BASE.addHandler(fh)

    # Console handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    _BASE.addHandler(sh)

    # Do not bubble to the *root* logger
    _BASE.propagate = False
    _BASE._configured = True
    return _BASE


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a child logger that inherits the base handlers (no child handlers)."""
    full_name = BASE_LOGGER if not name else f"{BASE_LOGGER}.{name}"
    logger = logging.getLogger(full_name)
    # Ensure children don't keep their own handlers (which would double-log)
    if logger is not _BASE and logger.handlers:
        logger.handlers.clear()
    logger.propagate = True  # bubble to BASE only
    return logger


# Convenience logger for this module
logger = get_logger(__name__)


def save_csv_parquet_torch(df: pd.DataFrame, fn: Path) -> None:
    if fn.suffix == ".parquet":
        logger.info(f"Saving to parquet: {fn}")
        df.to_parquet(fn)
        return
    if fn.suffix == ".csv":
        logger.info(f"Saving to csv: {fn}")
        df.to_csv(fn, index=False)
        return

    if fn.suffix == ".pt":
        logger.info(f"Saving to torch: {fn}")
        torch.save(df, fn)
        return

    raise ValueError(f"Unsupported file format: {fn.suffix}")


def read_csv_parquet_torch(fn: Path) -> pd.DataFrame:
    if fn.suffix == ".parquet":
        return pd.read_parquet(fn)
    if fn.suffix == ".csv":
        return pd.read_csv(fn)
    if fn.suffix == ".pt":
        return torch.load(fn)
    raise ValueError(f"Unsupported file format: {fn.suffix}")


def compute_pna_degrees(
    train_loader: DataLoader,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mol_max_degree = -1
    clique_max_degree = -1
    prot_max_degree = -1

    for data in tqdm(train_loader):
        # mol
        mol_d = degree(
            data.mol_edge_index[1], num_nodes=data.mol_x.shape[0], dtype=torch.long
        )
        mol_max_degree = max(mol_max_degree, int(mol_d.max()))
        # clique
        try:
            clique_d = degree(
                data.clique_edge_index[1],
                num_nodes=data.clique_x.shape[0],
                dtype=torch.long,
            )
        except RuntimeError:
            logger.info(f"clique edge index {data.clique_edge_index[1]}")
            logger.info(f"clique x {data.clique_x}")
            logger.info(f"clique shape {data.clique_x.shape}")
            logger.info(f"atom shape {data.mol_x.shape[0]}")
            break
        clique_max_degree = max(clique_max_degree, int(clique_d.max()))
        # protein
        prot_d = degree(
            data.prot_edge_index[1],
            num_nodes=data.prot_node_aa.shape[0],
            dtype=torch.long,
        )
        prot_max_degree = max(prot_max_degree, int(prot_d.max()))

    # Compute the in-degree histogram tensor
    mol_deg = torch.zeros(mol_max_degree + 1, dtype=torch.long)
    clique_deg = torch.zeros(clique_max_degree + 1, dtype=torch.long)
    prot_deg = torch.zeros(prot_max_degree + 1, dtype=torch.long)

    for data in tqdm(train_loader):
        # mol
        mol_d = degree(
            data.mol_edge_index[1], num_nodes=data.mol_x.shape[0], dtype=torch.long
        )
        mol_deg += torch.bincount(mol_d, minlength=mol_deg.numel())

        # clique
        clique_d = degree(
            data.clique_edge_index[1],
            num_nodes=data.clique_x.shape[0],
            dtype=torch.long,
        )
        clique_deg += torch.bincount(clique_d, minlength=clique_deg.numel())

        # Protein
        prot_d = degree(
            data.prot_edge_index[1],
            num_nodes=data.prot_node_aa.shape[0],
            dtype=torch.long,
        )
        prot_deg += torch.bincount(prot_d, minlength=prot_deg.numel())

    return mol_deg, clique_deg, prot_deg


class CustomWeightedRandomSampler(torch.utils.data.WeightedRandomSampler):
    """WeightedRandomSampler except allows for more than 2^24 samples to be sampled"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __iter__(self):
        rand_tensor = np.random.choice(
            range(0, len(self.weights)),
            size=self.num_samples,
            p=self.weights.numpy() / torch.sum(self.weights).numpy(),
            replace=self.replacement,
        )
        rand_tensor = torch.from_numpy(rand_tensor)
        return iter(rand_tensor.tolist())
