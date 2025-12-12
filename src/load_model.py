# scripts/load_mutaplm_model.py
import sys
from pathlib import Path
import argparse
import os

# Make imports robust regardless of CWD (repo layout: <repo>/{model,scripts,configs,...})

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.models.model_factory import (
    load_model_factory,
    ModelType,
)
from src.utils import setup_logging


def parse_args():
    p = argparse.ArgumentParser(description="Create and load PLM model")
    p.add_argument("--log_fn", type=str, default="")
    p.add_argument("--log_level", type=str, default="INFO")
    p.add_argument("--model_type", type=str, default="")
    p.add_argument("--model_fn", type=str, default="")
    p.add_argument(
        "--config",
        type=str,
        default=str(REPO_ROOT / "configs" / "mutaplm_inference.yaml"),
    )
    return p.parse_args()


def main():

    args = parse_args()
    logger = setup_logging(Path(args.log_fn), args.log_level)
    try:
        # Log configuration
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Logging to: {args.log_fn}")
        logger.info(f"Config: {args.config}")
        logger.info("Loading model...")
        logger.info(f"Model type: {args.model_type}")
        logger.info(f"Model fn: {args.model_fn}")
        mt = ModelType.from_str(args.model_type)
        load_model_factory(mt, model_ref=args.model_fn, config_path=Path(args.config))
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.exception("Script failed: %s", e)  # or this
        sys.exit(1)


if __name__ == "__main__":
    main()
