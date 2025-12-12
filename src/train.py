import json
import pandas as pd
import torch
import numpy as np
import os
import random
import argparse
import ast
from pathlib import Path
import sys
from datetime import datetime
from torch_geometric.loader import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils import (
    setup_logging,
    get_logger,
    str2bool,
)

from src.model_utils import (
    compute_pna_degrees,
    CustomWeightedRandomSampler,
)

from src.load_model import load_model_factory
from src.models.model_factory import ModelType

logger = get_logger(__name__)

# Utils
from src.model_utils import virtual_screening


from src.dataset import ProteinMoleculeDataset
from src.trainer import Trainer
from src.metrics import evaluate_reg, evaluate_cls, evaluate_mcls

# Preprocessing
from src.protein_init import protein_init
from src.ligand_init import ligand_init

# Model
from src.models.net import net

# --- Argument Parsing Helpers ---


def tuple_type(s):
    try:
        value = ast.literal_eval(s)
        if not isinstance(value, tuple):
            raise ValueError
    except (ValueError, SyntaxError):
        raise argparse.ArgumentTypeError(f"Invalid tuple value: {s}")
    return value


def list_type(s):
    try:
        value = ast.literal_eval(s)
        if not isinstance(value, list):
            raise ValueError
    except (ValueError, SyntaxError):
        raise argparse.ArgumentTypeError(f"Invalid list value: {s}")
    return value


def parse_args() -> argparse.Namespace:
    """Configures and parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Model Training and Evaluation Script")

    ### Seed and device
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--device", type=str, default="cuda", help='e.g., "cuda" or "cpu"'
    )
    parser.add_argument("--config_path", type=Path, default="config")

    parser.add_argument(
        "--model_plm_type", type=str, default="ESM2", help="Type of model to use"
    )
    parser.add_argument(
        "--model_plm_fn", type=str, default="", help="Path to model checkpoint"
    )
    ### Data and Pre-processing
    parser.add_argument(
        "--datafolder", type=Path, default="./dataset/pdb2020", help="Protein data path"
    )
    parser.add_argument(
        "--result_path",
        type=Path,
        default="",
        help="Path to save results",
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        default="",
        help="Path to save model",
    )
    parser.add_argument(
        "--interpret_path",
        type=Path,
        default=None,
        help="Path to save interpretation results",
    )
    parser.add_argument(
        "--save_interpret",
        type=str2bool,
        default=True,
        help="Save interpretation results",
    )
    ### Task Type
    parser.add_argument(
        "--regression_task",
        type=str2bool,
        default=True,
        help="True if regression else False",
    )
    parser.add_argument(
        "--classification_task",
        type=str2bool,
        default=False,
        help="True if classification else False",
    )
    parser.add_argument(
        "--mclassification_task",
        type=int,
        default=0,
        help="Number of multiclassification, 0 if no multiclass task",
    )
    ### Training Schedule
    parser.add_argument(
        "--epochs", type=int, default=30, help="Number of training epochs"
    )
    parser.add_argument(
        "--evaluate_epoch", type=int, default=1, help="Evaluate every N epochs"
    )
    parser.add_argument(
        "--total_iters",
        type=int,
        default=None,
        help="Total training iterations (overrides epochs)",
    )
    parser.add_argument(
        "--evaluate_step", type=int, default=500, help="Evaluate every N iterations"
    )
    ### Optimizer Params
    parser.add_argument("--lrate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--eps", type=float, default=1e-8, help="AdamW epsilon")
    parser.add_argument(
        "--betas", type=tuple_type, default="(0.9,0.999)", help="AdamW betas"
    )
    ### Batching and Sampling
    parser.add_argument("--n", type=int, default=0, help="Number of rows to load")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--sampling_col", type=str, default="", help="Column for weighted sampling"
    )

    ### Finetuning / Loading
    parser.add_argument(
        "--trained_model_path",
        type=Path,
        default=None,
        help="Path to a pretrained model directory",
    )
    parser.add_argument(
        "--finetune_modules",
        type=list_type,
        default=None,
        help="List of modules to finetune",
    )

    parser.add_argument("--nb_mode", type=str2bool, default=False)

    parser.add_argument("--log_fn", type=str, default="train.log")
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )
    args = parser.parse_args()

    # if args.epochs is not None and args.total_iters is not None:
    #     logger.info(
    #         "If epochs and total iters are both not None, then we only use iters."
    #     )
    #     args.epochs = None

    return args


# --- Setup Functions ---


def load_and_merge_config(
    trained_model_path: Path | None,
    config_path: Path,
    lrate: float,
    eps: float,
    betas: tuple,
    regression_task: bool,
    classification_task: bool,
    mclassification_task: int,
) -> dict:
    """Loads the base JSON config and overwrites it with args."""
    if trained_model_path is not None and trained_model_path.exists():
        trained_config_file = trained_model_path / "config.json"
        logger.info(f"Loading config from trained_model_path: {trained_model_path}")
        config_file = trained_config_file
    else:
        logger.info(f"Loading config from config_path: {config_path}")
        config_file = config_path / "config.json"

    logger.info(f"Loading configuration from: {config_file}")

    # Check if config file exists before trying to open it
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_file, "r") as f:
        config = json.load(f)

    # Overwrite config with command-line arguments
    config["optimizer"]["lrate"] = lrate
    config["optimizer"]["eps"] = eps
    config["optimizer"]["betas"] = betas
    config["tasks"]["regression_task"] = regression_task
    config["tasks"]["classification_task"] = classification_task
    config["tasks"]["mclassification_task"] = mclassification_task

    return config


def setup_environment(
    seed: int,
    device: str,
) -> torch.device:
    """Sets random seeds, and sets device."""

    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Set device
    # Validate and set device
    try:
        device = torch.device(device)
    except Exception:
        logger.error(f"Invalid device '{device}', falling back to CPU")
        device = torch.device("cpu")

    logger.info(f"Set seed to {seed}")
    logger.info(f"Using device: {device}")

    return device


# --- Data Loading Functions ---


def load_dataframes(
    datafolder: Path,
    n: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads the train, validation, and test dataframes."""
    logger.info("Loading dataframes...")
    train_path = datafolder / "train.csv"
    test_path = datafolder / "test.csv"
    valid_path = datafolder / "valid.csv"
    if n > 0:
        train_df = pd.read_csv(train_path, nrows=n)
        test_df = pd.read_csv(test_path, nrows=n)
        valid_df = pd.read_csv(valid_path, nrows=n)
    else:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        valid_df = pd.read_csv(valid_path)

    return train_df, test_df, valid_df


def load_or_init_graphs(
    datafolder: Path,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    model_plm_type: str,
    model_plm_fn: str,
) -> tuple[dict, dict]:
    """Loads pre-computed graph data or initializes it if not found."""

    # Get unique proteins and ligands
    if valid_df is not None:
        protein_seqs = list(
            set(
                train_df["Protein"].tolist()
                + test_df["Protein"].tolist()
                + valid_df["Protein"].tolist()
            )
        )
        ligand_smiles = list(
            set(
                train_df["Ligand"].tolist()
                + test_df["Ligand"].tolist()
                + valid_df["Ligand"].tolist()
            )
        )
    else:
        protein_seqs = list(
            set(train_df["Protein"].tolist() + test_df["Protein"].tolist())
        )
        ligand_smiles = list(
            set(train_df["Ligand"].tolist() + test_df["Ligand"].tolist())
        )

    # Load or initialize protein graphs
    protein_path = datafolder / "protein.pt"
    if protein_path.exists():
        logger.info("Loading Protein Graph data...")
        protein_dict = torch.load(protein_path)
    else:
        mt = ModelType.from_str(model_plm_type)
        model, alphabet = load_model_factory(model_type=mt, model_ref=model_plm_fn)
        model.eval()
        if torch.cuda.is_available():
            logger.info("Using CUDA")
            model = model.cuda()
        logger.info("Initialising Protein Sequence to Protein Graph...")
        logger.info(f"Number of unique proteins: {len(protein_seqs)}")
        protein_dict = protein_init(model, alphabet, protein_seqs)
        logger.info(f"Saving Protein Graph data to: {protein_path}")
        torch.save(protein_dict, protein_path)

    # Load or initialize ligand graphs
    ligand_path = datafolder / "ligand.pt"
    if ligand_path.exists():
        logger.info("Loading Ligand Graph data...")
        ligand_dict = torch.load(ligand_path)
    else:
        logger.info("Initialising Ligand SMILES to Ligand Graph...")
        logger.info(f"Number of unique ligands: {len(ligand_smiles)}")
        ligand_dict = ligand_init(ligand_smiles)
        logger.info(f"Saving Ligand Graph data to: {ligand_path}")
        torch.save(ligand_dict, ligand_path)

    torch.cuda.empty_cache()
    logger.info("Graph data loaded/initialised.")
    return protein_dict, ligand_dict


def prepare_dataloaders(
    sampling_col: str,
    protein_dict: dict[str, dict],
    ligand_dict: dict[str, dict],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    device: str,
    batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Creates and returns train, validation, and test DataLoaders."""

    # Setup training sampler
    train_shuffle = True
    train_sampler = None
    if sampling_col:
        train_weights = torch.from_numpy(train_df[sampling_col].values)
        train_sampler = CustomWeightedRandomSampler(
            train_weights, len(train_weights), replacement=True
        )
        train_shuffle = False
        logger.info(
            f"Using CustomWeightedRandomSampler on column '{sampling_col}'. Shuffle is False."
        )

    # Create datasets
    train_dataset = ProteinMoleculeDataset(
        train_df, ligand_dict, protein_dict, device=device
    )
    test_dataset = ProteinMoleculeDataset(
        test_df, ligand_dict, protein_dict, device=device
    )

    # Create loaders
    follow_batch_keys = ["mol_x", "clique_x", "prot_node_aa"]
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        follow_batch=follow_batch_keys,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        follow_batch=follow_batch_keys,
    )

    valid_loader = None
    if valid_df is not None:
        valid_dataset = ProteinMoleculeDataset(
            valid_df, ligand_dict, protein_dict, device=device
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            follow_batch=follow_batch_keys,
        )

    logger.info("DataLoaders created.")
    return train_loader, valid_loader, test_loader


# --- Model and Trainer Functions ---


def get_pna_degrees(
    trained_model_path: Path | None,
    datafolder: Path,
    train_loader: DataLoader,
    model_path: Path,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute or load PNA (Principal Neighborhood Aggregation) degree statistics.

    PNA (Principal Neighborhood Aggregation) requires pre-computed node degrees
    for each graph domain (ligand graph, clique graph, protein graph). These
    degree histograms are used by PNAConv / PNA layers to apply:
        • degree-scaled aggregation
        • degree-based normalization
        • aggregator corrections based on graph sparsity/density

    This function provides the following logic:

    1. If `trained_model_path` does NOT exist:
         We are running a new training job.
         Attempt to load cached degree statistics from `datafolder/degree.pt`.
         If no cached file exists, compute degrees from the training set
           (via `compute_pna_degrees(train_loader)`), then save them.
         Copy the resulting degree file into the new model directory
           (`model_path/degree.pt`) so future runs can reuse them.

    2. If `trained_model_path` DOES exist:
         Load existing degree statistics from the previously trained model.
         This allows evaluation or fine-tuning without recomputing degrees.

    Returns
    -------
    tuple(torch.Tensor, torch.Tensor, torch.Tensor)
        A tuple containing:
            ligand_deg : degree histogram for ligand molecular graphs
            clique_deg : degree histogram for clique/fragment graphs
            protein_deg: degree histogram for protein residue graphs

    Notes
    -----
    • This function ensures degree statistics are computed exactly once per dataset.
    • The degrees are saved to disk so repeated training is deterministic and fast.
    • All downstream PNAConv layers rely on these tensors for correct scaling.
    """
    if trained_model_path is None or not trained_model_path.exists():
        # Compute degrees from scratch
        degree_path = datafolder / "degree.pt"
        if not degree_path.exists():
            logger.info("Computing training data degrees for PNA...")
            mol_deg, clique_deg, prot_deg = compute_pna_degrees(train_loader)
            degree_dict = {
                "ligand_deg": mol_deg,
                "clique_deg": clique_deg,
                "protein_deg": prot_deg,
            }
            torch.save(degree_dict, degree_path)
        else:
            logger.info("Loading pre-computed PNA degrees...")
            degree_dict = torch.load(degree_path)

        # Save degrees to model result directory
        torch.save(degree_dict, model_path / "degree.pt")

    else:
        # Load degrees from the trained model directory
        logger.info(
            f"Loading PNA degrees from trained model path: {trained_model_path}"
        )
        degree_dict = torch.load(trained_model_path / "degree.pt")

    return (
        degree_dict["ligand_deg"],
        degree_dict["clique_deg"],
        degree_dict["protein_deg"],
    )


def initialize_model(
    config: dict,
    mol_deg: torch.Tensor,
    prot_deg: torch.Tensor,
    device: str,
    trained_model_path: Path | None,
) -> net:
    """Initializes the network model and loads pretrained weights if specified."""

    model = net(
        mol_deg,
        prot_deg,
        # MOLECULE
        mol_in_channels=config["params"]["mol_in_channels"],
        prot_in_channels=config["params"]["prot_in_channels"],
        prot_evo_channels=config["params"]["prot_evo_channels"],
        hidden_channels=config["params"]["hidden_channels"],
        pre_layers=config["params"]["pre_layers"],
        post_layers=config["params"]["post_layers"],
        aggregators=config["params"]["aggregators"],
        scalers=config["params"]["scalers"],
        total_layer=config["params"]["total_layer"],
        K=config["params"]["K"],
        heads=config["params"]["heads"],
        dropout=config["params"]["dropout"],
        dropout_attn_score=config["params"]["dropout_attn_score"],
        # output
        regression_head=config["tasks"]["regression_task"],
        classification_head=config["tasks"]["classification_task"],
        multiclassification_head=config["tasks"]["mclassification_task"],
        device=device,
    ).to(device)

    model.reset_parameters()

    if trained_model_path is not None and trained_model_path.exists():
        param_dict = trained_model_path / "model.pt"
        model.load_state_dict(torch.load(param_dict, map_location=device), strict=False)
        logger.info(f"Pretrained model loaded from: {param_dict}")

    nParams = sum([p.nelement() for p in model.parameters()])
    logger.info(f"Model loaded with {nParams} parameters.")

    return model


def initialize_trainer(
    model: net,
    config: dict,
    train_loader: DataLoader,
    total_iters: int,
    epochs: int,
    device: str,
    model_path: Path,
    seed: int,
    finetune_modules: list[str] | None,
    regression_weight: float = 1,
    classification_weight: float = 1,
    multiclassification_weight: float = 1,
) -> Trainer:
    """Initializes the training engine."""

    # Determine evaluation metric
    if config["tasks"]["regression_task"]:
        evaluation_metric = "rmse"
    elif config["tasks"]["classification_task"]:
        evaluation_metric = "roc"
    elif config["tasks"]["mclassification_task"]:
        evaluation_metric = "macro_f1"
    else:
        raise Exception(
            "No valid interaction property prediction task specified in config."
        )

    engine = Trainer(
        model=model,
        lrate=config["optimizer"]["lrate"],
        min_lrate=config["optimizer"]["min_lrate"],
        wdecay=config["optimizer"]["weight_decay"],
        betas=config["optimizer"]["betas"],
        eps=config["optimizer"]["eps"],
        amsgrad=config["optimizer"]["amsgrad"],
        clip=config["optimizer"]["clip"],
        steps_per_epoch=len(train_loader),
        num_epochs=epochs,
        total_iters=total_iters,
        warmup_iters=config["optimizer"]["warmup_iters"],
        lr_decay_iters=config["optimizer"]["lr_decay_iters"],
        schedule_lr=config["optimizer"]["schedule_lr"],
        regression_weight=regression_weight,
        classification_weight=classification_weight,
        multiclassification_weight=multiclassification_weight,
        evaluate_metric=evaluation_metric,
        result_path=model_path,
        runid=seed,
        finetune_modules=finetune_modules,
        device=device,
    )

    # Save the config to the model directory
    with open(model_path / "config.json", "w") as f:
        json.dump(config, f, indent=4)

    return engine


# --- Main Execution ---


def run_training(
    engine: Trainer,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    evaluate_epoch: int,
    evaluate_step: int,
):
    """Runs the main training loop."""
    logger.info("-" * 50)
    logger.info("Start training model")

    if epochs is not None and epochs:
        engine.train_epoch(
            train_loader,
            val_loader=valid_loader,
            test_loader=test_loader,
            evaluate_epoch=evaluate_epoch,
        )
    else:
        engine.train_step(
            train_loader,
            val_loader=valid_loader,
            test_loader=test_loader,
            evaluate_step=evaluate_step,
        )

    logger.info("Finished training model")
    logger.info("-" * 50)


def run_evaluation(
    model: net,
    model_path: Path,
    test_df: pd.DataFrame,
    test_loader: DataLoader,
    interpret_path: Path,
    device: str,
    save_interpret: bool,
    ligand_dict: dict,
):
    """Loads the best model and runs final evaluation/screening."""
    logger.info("Loading best checkpoint and predicting test data")
    logger.info("-" * 50)

    timestamp = datetime.now()
    year = timestamp.year
    month = timestamp.month  # Month (1-12)
    day = timestamp.day  # Day of the month (1-31)
    fn = model_path / f"{year}_{month:02d}_{day:02d}_model.pt"
    model.load_state_dict(torch.load(fn, map_location=device))

    screen_df = virtual_screening(
        test_df,
        model,
        test_loader,
        result_path=interpret_path,
        save_interpret=save_interpret,
        ligand_dict=ligand_dict,
        device=device,
    )
    timestamp = datetime.now()
    year = timestamp.year
    month = timestamp.month  # Month (1-12)
    day = timestamp.day  # Day of the month (1-31)
    fn = model_path / f"{year}_{month:02d}_{day:02d}_test_prediction.csv"
    screen_df.to_csv(fn, index=False)
    logger.info(f"Test predictions saved to: {fn}")


def main():
    """Main execution flow."""
    args = parse_args()
    try:
        setup_logging(args.log_fn, args.log_level)

        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Data folder: {args.datafolder}")
        logger.info(f"Model plm type: {args.model_plm_type}")
        logger.info(f"Model plm fn: {args.model_plm_fn}")
        logger.info(f"Result path: {args.result_path}")
        logger.info(f"Config path: {args.config_path}")
        trained_model_path = args.trained_model_path
        if trained_model_path is None or str(trained_model_path) == ".":
            trained_model_path = None
        else:
            trained_model_path = trained_model_path.resolve()
        logger.info(f"Trained model path: {trained_model_path}")
        model_path = args.model_path.resolve()
        logger.info(f"Model path: {model_path}")
        if args.interpret_path is None:
            interpret_path = model_path
        else:
            interpret_path = args.interpret_path.resolve()
        logger.info(f"Interpret path: {interpret_path}")
        logger.info(f"Learning rate: {args.lrate}")
        logger.info(f"Number of rows to load: {args.n}")
        logger.info(f"Batch size: {args.batch_size}")
        total_iters = args.total_iters
        epochs = args.epochs
        if epochs is not None and total_iters is not None:
            logger.info(
                "If epochs and total iters are both not None, then we only use iters."
            )
            epochs = None
        logger.info(f"Total iters: {total_iters}")
        logger.info(f"Epochs: {epochs}")
        logger.info(f"Seed: {args.seed}")
        logger.info(f"Regression task: {args.regression_task}")
        logger.info(f"Classification task: {args.classification_task}")
        logger.info(f"Multiclassification task: {args.mclassification_task}")
        logger.info(f"Finetune modules: {args.finetune_modules}")
        logger.info(f"Notebook mode: {args.nb_mode}")
        logger.info(f"Device: {args.device}")
        config_path = args.config_path.resolve()
        config = load_and_merge_config(
            trained_model_path,
            config_path,
            args.lrate,
            args.eps,
            args.betas,
            args.regression_task,
            args.classification_task,
            args.mclassification_task,
        )
        device = setup_environment(args.seed, args.device)

        # Load Data
        datafolder = args.datafolder.resolve()
        train_df, test_df, valid_df = load_dataframes(datafolder, args.n)
        logger.info(
            f"Data loaded. Train: {len(train_df)}-{len(test_df)}-{len(valid_df)}"
        )
        protein_dict, ligand_dict = load_or_init_graphs(
            datafolder,
            train_df,
            test_df,
            valid_df,
            args.model_plm_type,
            args.model_plm_fn,
        )
        train_loader, valid_loader, test_loader = prepare_dataloaders(
            args.sampling_col,
            protein_dict,
            ligand_dict,
            train_df,
            test_df,
            valid_df,
            args.device,
            args.batch_size,
        )
        logger.info(
            f"DataLoaders created. Train: {len(train_loader.dataset)}-{len(valid_loader.dataset)}-{len(test_loader.dataset)}"
        )
        # Initialize Model and Trainer
        mol_deg, _, prot_deg = get_pna_degrees(
            trained_model_path, datafolder, train_loader, model_path
        )
        model = initialize_model(config, mol_deg, prot_deg, device, trained_model_path)
        engine = initialize_trainer(
            model,
            config,
            train_loader,
            total_iters,
            epochs,
            device,
            model_path,
            args.seed,
            args.finetune_modules,
        )

        # Run Training
        run_training(
            engine,
            train_loader,
            valid_loader,
            test_loader,
            epochs,
            args.evaluate_epoch,
            args.evaluate_step,
        )

        # Run Final Evaluation
        run_evaluation(
            model,
            model_path,
            test_df,
            test_loader,
            interpret_path,
            device,
            args.save_interpret,
            ligand_dict,
        )

    except Exception as e:
        logger.exception("Script failed: %s", e)
        sys.exit(1)

    logger.info("Script finished.")


if __name__ == "__main__":
    main()
