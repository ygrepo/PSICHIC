from __future__ import annotations

from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

import os
import numpy as np
import torch
import torch.nn as nn
from torch.serialization import add_safe_globals
import torch.nn.functional as F
import yaml
import re

from enum import Enum
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from typing import Optional, Union, List, Literal
import warnings
from esm import pretrained  # FAIR’s original library

MODEL_DIR = Path(__file__).parent.parent / "pretrained"

# map ESM2 layer counts to HF repos (used when local path isn't available)
_ESM2_REPO = {
    6: "facebook/esm2_t6_8M_UR50D",
    12: "facebook/esm2_t12_35M_UR50D",
    30: "facebook/esm2_t30_150M_UR50D",
    33: "facebook/esm2_t33_650M_UR50D",
    36: "facebook/esm2_t36_3B_UR50D",
}

from src.utils import read_csv_parquet_torch, save_csv_parquet_torch, get_logger

logger = get_logger(__name__)


def _detect_max_len(model, model_type) -> int:
    """
    Return the maximum supported sequence length for this model.
    The value is conservative (does not include BOS/EOS padding budget).
    """
    # Import lazily or use your existing ModelType
    MT = (
        ModelType if isinstance(model_type, ModelType) else ModelType
    )  # no-op, just clarity

    if model_type == MT.ESMV1:
        # FAIR ESMv1 commonly 1022
        return int(getattr(model, "max_positions", 1022))

    if model_type in (MT.ESM2, MT.PROTEINCLIP):
        # HF ESM2 exposes .config.max_position_embeddings (commonly 1024)
        return int(
            getattr(getattr(model, "config", None), "max_position_embeddings", 1024)
        )

    if model_type == MT.MUTAPLM:
        # Prefer explicit attribute if your model defines it
        if hasattr(model, "max_len"):
            return int(model.max_len)
        if hasattr(model, "sequence_length"):
            return int(model.sequence_length)
        # Last resort: conservative default; warn once
        warnings.warn(
            "[MutaPLM] max_len not found on model; defaulting to 1024. "
            "Attach `model.max_len = <int>` after constructing your model to silence this.",
            RuntimeWarning,
        )
        return 1024

    raise ValueError(f"Unknown model type for max_len detection: {model_type}")


def _attach_max_len(model, model_type) -> int:
    """
    Detect and attach `model.max_len` if missing. Return the value.
    """
    if hasattr(model, "max_len"):
        try:
            val = int(model.max_len)
            return val
        except Exception:
            pass  # fall through to re-detect

    val = _detect_max_len(model, model_type)
    try:
        setattr(model, "max_len", int(val))
    except Exception:
        # If model is a torchscript or restricts setattr, just ignore
        pass
    return int(val)


# Enum for model types
class ModelType(Enum):
    ESMV1 = "ESMv1"
    ESM2 = "ESM2"
    MUTAPLM = "MUTAPLM"
    PROTEINCLIP = "ProteinCLIP"
    LLAMA = "LLAMA"

    @property
    def path(self) -> Path:
        """Local default path for this model type."""
        base = Path(
            os.getenv(
                "MODEL_BASE", "/sc/arion/projects/DiseaseGeneCell/Huang_lab_data/models"
            )
        )
        mapping = {
            # ESMv1: return hub alias (cleaner)
            ModelType.ESMV1: "/sc/arion/projects/DiseaseGeneCell/Huang_lab_data/.torch_hub/checkpoints/esm1v_t33_650M_UR90S_5.pt",
            # ModelType.ESMV1: "esm1v_t33_650M_UR90S_5",
            # ModelType.ESMV1: "/sc/arion/projects/DiseaseGeneCell/Huang_lab_data/models/esm1v_t33_650M_UR90S_5",
            # ESM2 can be an HF repo id or a local dir
            # ModelType.ESMV1: Path(
            #     "/sc/arion/projects/DiseaseGeneCell/Huang_lab_project/drug_discovery/output/esm1v_local"
            # ),
            ModelType.ESM2: Path(
                os.getenv("ESM2_PATH", str(base / "esm2_t33_650M_UR50D_safe"))
            ),
            ModelType.MUTAPLM: base / "mutaplm.pth",
            ModelType.PROTEINCLIP: base / "proteinclip",
            ModelType.LLAMA: "meta-llama/Meta-Llama-3-8B-Instruct",
        }
        return mapping[self]

    def __str__(self) -> str:
        return self.value

    @classmethod
    def from_str(cls, s: str) -> "ModelType":
        """Case-insensitive, accepts value or name."""
        s_norm = s.strip().lower()
        for m in cls:
            if m.value.lower() == s_norm or m.name.lower() == s_norm:
                return m
        raise ValueError(f"Unknown model type: {s}")


PLM_MODEL = [ModelType.ESMV1, ModelType.ESM2, ModelType.MUTAPLM, ModelType.PROTEINCLIP]
MODEL_TYPE = list(ModelType)


def select_device(pref: str) -> torch.device:
    pref = (pref or "auto").lower()
    if pref.startswith("cuda"):
        return torch.device(pref) if torch.cuda.is_available() else torch.device("cpu")
    if pref == "mps":
        return (
            torch.device("mps")
            if getattr(torch.backends, "mps", None)
            and torch.backends.mps.is_available()
            else torch.device("cpu")
        )
    if pref == "cpu":
        return torch.device("cpu")
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _device_or_default(device: Optional[Union[str, torch.device]]) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_HF_model(model_name: str) -> AutoModel:
    """
    Load an ESM model safely.

    - Prefers safetensors if available (no torch.load / pickle)
    - Works offline with local paths
    - Enforces HF_HOME for caching on HPC
    """
    logger.info(f"HF_HOME: {os.environ['HF_HOME']}")
    logger.info(f"Loading model: {model_name}")

    model = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
    max_len = getattr(model.config, "max_position_embeddings", 1024)
    logger.info(f"Model max token length (from config): {max_len}")
    return model


def load_HF_AutoModel(model_name: str) -> AutoModelForCausalLM:
    """
    Load an AutoModelForCausalLM model.
    - If `model_name` is a local path that exists: load from disk.
    - If it's a local path that does not exist: fallback to HF Hub.
    - If it's a Hub repo ID: use local cache or download if not cached.
    """
    logger.info(f"HF_HOME: {os.environ.get('HF_HOME')}")
    logger.info(f"Loading model: {model_name}")

    # --- optional auth token
    HF_TOKEN_PATH = os.environ.get("HF_TOKEN_PATH")
    HF_TOKEN = None
    if HF_TOKEN_PATH is not None and os.path.exists(HF_TOKEN_PATH):
        with open(HF_TOKEN_PATH, "r") as f:
            HF_TOKEN = f.read().strip()

    if os.path.isdir(model_name):
        # Local path exists → use it
        logger.info(f"Loading model from local path: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(model_name).eval()
    else:
        # Treat as Hub repo (or fallback if local path is missing)
        logger.info(f"Loading model from HF Hub or cache: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=HF_TOKEN,
        ).eval()

    device = _device_or_default(None)
    model.to(device)
    return model


def load_HF_tokenizer(
    model_name: str,
    *,
    HF_TOKEN: str | None = None,
    CACHE_DIR: str | None = None,
) -> AutoTokenizer:
    """Load ESM tokenizer."""

    if HF_TOKEN is not None:
        logger.info(f"Using HF_TOKEN: {HF_TOKEN}, CACHE_DIR: {CACHE_DIR}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=True, token=HF_TOKEN, cache_dir=CACHE_DIR
        )
    elif CACHE_DIR is not None:
        logger.info(f"Using CACHE_DIR: {CACHE_DIR}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)
    else:
        logger.info("Using HF_TOKEN: None, CACHE_DIR: None")
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    return tokenizer


def _resolve_parent_esm2_path(layers: int) -> str:
    """
    Prefer your local mirror for 33-layer if present, otherwise fall back to HF repo.
    Extend here if you have local mirrors for 6/12/30/36.
    """
    if layers == 33:
        local = ModelType.ESM2.path  # your /.../esm2_t33_650M_UR50D_safe
        if Path(local).exists():
            return str(local)
    # fall back to HF repo id
    return _ESM2_REPO[layers]


def _ensure_torch_home() -> Path:
    hub_dir = (
        os.environ.get("TORCH_HOME")
        or "/sc/arion/projects/DiseaseGeneCell/Huang_lab_data/.torch_hub"
    )
    hub_dir = Path(hub_dir)
    (hub_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    try:
        torch.hub.set_dir(str(hub_dir))
    except Exception:
        pass
    return hub_dir


def _hub_name_from_ref(model_ref: str | Path) -> str:
    """
    Accepts either a hub alias (esm1v_t33_650M_UR90S_5) or a filesystem path.
    - If it's a .pt path: return its basename without directory (used for cache file name).
    - If it's a path-like string without .pt: use basename as the hub alias.
    - Else, return the string as-is.
    """
    ref = str(model_ref)
    if ref.endswith(".pt"):
        return Path(ref).name.replace(".pt", "")
    if os.path.sep in ref:
        return Path(ref).name
    return ref


def load_fair_esm_v1_cached(model_ref: str | Path, *, device: torch.device):
    """
    ESMv1 loader with cache detection:
    - If model_ref is a local .pt path -> load local.
    - Else treat it as a hub alias and check $TORCH_HOME/checkpoints/<alias>.pt:
        - If exists -> load local from cache.
        - Else -> download from hub (to cache) and load.
    Also handles corrupted cache files by removing and re-downloading once.
    """
    hub_dir = _ensure_torch_home()
    ref = str(model_ref)
    is_local = ref.endswith(".pt") and Path(ref).is_file()
    logger.info(f"Loading ESMv1 model: {ref} (is_local={is_local})")
    if is_local:
        # Local .pt explicitly provided
        model, alphabet = pretrained.load_model_and_alphabet_local(ref)
        model = model.to(device).eval()
        return model, alphabet, f"local:{ref}"

    # Hub path: resolve expected cache file
    hub_name = _hub_name_from_ref(ref)  # e.g., "esm1v_t33_650M_UR90S_5"
    cache_ckpt = hub_dir / "checkpoints" / f"{hub_name}.pt"

    # 1) Try cached file if present
    if cache_ckpt.is_file():
        try:
            model, alphabet = pretrained.load_model_and_alphabet_local(str(cache_ckpt))
            model = model.to(device).eval()
            return model, alphabet, f"cache:{cache_ckpt}"
        except Exception as e:
            # Corrupted/incompatible cache -> delete and re-download
            try:
                cache_ckpt.unlink(missing_ok=True)
            except Exception:
                pass  # best effort
            # fall through to hub download

    # 2) Download from hub (this writes into $TORCH_HOME/checkpoints/)
    model, alphabet = pretrained.load_model_and_alphabet(hub_name)
    model = model.to(device).eval()
    return model, alphabet, f"hub:{hub_name}"


# Load Model Factory Function
def load_model_factory(
    model_type: ModelType,
    *,
    config_path: Path = Path("configs/mutaplm_inference.yaml"),
):
    """
    Returns:
      (model, tokenizer) for HF models and PROTEINCLIP (parent PLM + tokenizer).
      (model, None)      for MutaPLM.
    For PROTEINCLIP: loads the matching parent ESM2 (by layers) and attaches ONNX head at `model.proteinclip`.
    """
    device = _device_or_default(None)
    logger.info("Using device: %s", device)

    if model_type == ModelType.ESMV1:
        model_ref = model_type.path  # can be a hub name *or* a local .pt path
        model, alphabet, src = load_fair_esm_v1_cached(model_ref, device=device)
        _attach_max_len(model, model_type)
        logger.info("Loaded FAIR ESMv1 model and Alphabet (%s)", src)
        return model, alphabet

    if model_type == ModelType.ESM2:
        # (unchanged HF path)
        model_path = str(model_type.path)
        model = load_HF_model(model_path).to(device).eval()
        CACHE_DIR = os.environ.get("HF_CACHE_DIR")
        logger.info(f"Loading Tokenizer from {CACHE_DIR}")
        tokenizer = load_HF_tokenizer(model_path, HF_TOKEN=None, CACHE_DIR=CACHE_DIR)
        _attach_max_len(model, model_type)
        logger.info("Loaded HF ESM2: %s", model_path)
        return model, tokenizer

    if model_type == ModelType.MUTAPLM:
        model = create_mutaplm_model(config_path, device)
        model_path = model_type.path
        model = load_mutaplm_model(model, model_path)
        _attach_max_len(model, model_type)
        logger.info("Loaded model: %s", model_path)
        return model, None

    if model_type == ModelType.LLAMA:
        model_path = model_type.path
        model = load_HF_AutoModel(model_path)
        tokenizer = load_HF_tokenizer(model_path)
        _attach_max_len(model, model_type)
        logger.info("Loaded model: %s", model_path)
        return model, tokenizer

    if model_type == ModelType.PROTEINCLIP:
        # 1) Decide which ESM2 depth to use (default 33); allow override via env
        layers = int(os.getenv("PROTEINCLIP_ESM_LAYERS", "33"))
        parent_ref = _resolve_parent_esm2_path(layers)

        # 2) Load the matching parent PLM + tokenizer
        model = load_HF_model(parent_ref)
        tokenizer = load_HF_tokenizer(parent_ref)
        _attach_max_len(model, model_type)
        logger.info("Loaded parent PLM (ESM2-%d layers): %s", layers, parent_ref)

        # 3) Sanity check: confirm actual num_hidden_layers matches requested
        actual_layers = int(getattr(model.config, "num_hidden_layers", layers))
        if actual_layers != layers:
            logger.warning(
                "Requested ESM2 layers=%d, but loaded model has %d layers.",
                layers,
                actual_layers,
            )
            layers = actual_layers  # keep them in sync for the head

        # 4) Load the ProteinCLIP head for that depth and validate input dim
        hidden_size = int(getattr(model.config, "hidden_size", 1280))
        clip_head = load_proteinclip(
            "esm",
            layers,
            model_dir=ModelType.PROTEINCLIP.path,
            expected_in_dim=hidden_size,
        )

        # 5) Attach to model
        setattr(model, "proteinclip", clip_head)
        logger.info(
            "Attached ProteinCLIP head (input_dim=%s, output_dim=%s)",
            clip_head.input_dim,
            clip_head.output_dim,
        )
        return model, tokenizer

    raise ValueError(f"Unknown model type: {model_type}")


def create_mutaplm_model(cfg_path: Path, device):
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    with cfg_path.open() as f:
        model_cfg = yaml.safe_load(f)

    model_cfg["device"] = device
    model = MutaPLM(**model_cfg).to(device).eval()

    # Keep CPU in float32 (your class defaults to bf16 for from_pretrained)
    if device.type != "cuda":
        model.float()

    logger.info("Model loaded successfully.")
    return model
