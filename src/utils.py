import sys
import argparse
from pathlib import Path

import logging
import pandas as pd
import torch


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


def str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("yes", "true", "t", "y", "1"):
        return True
    if v in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got '{v}'.")
