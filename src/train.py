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
from torch_geometric.loader import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils import (
    setup_logging,
    get_logger,
    compute_pna_degrees,
    CustomWeightedRandomSampler,
)

logger = get_logger(__name__)

# Utils
from utils.utils import virtual_screening


from src.dataset import ProteinMoleculeDataset
from utils.trainer import Trainer
from utils.metrics import *

# Preprocessing
from src.protein_init import protein_init
from src.ligand_init import ligand_init

# Model
from models.net import net

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
    parser.add_argument("--config_path", type=Path, default="config/config.json")

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
        default="",
        help="Path to save interpretation results",
    )
    parser.add_argument(
        "--save_interpret", type=bool, default=True, help="Save interpretation results"
    )
    ### Task Type
    parser.add_argument(
        "--regression_task",
        type=bool,
        default=True,
        help="True if regression else False",
    )
    parser.add_argument(
        "--classification_task",
        type=bool,
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
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--sampling_col", type=str, default="", help="Column for weighted sampling"
    )

    ### Finetuning / Loading
    parser.add_argument(
        "--trained_model_path",
        type=Path,
        default="",
        help="Path to a pretrained model directory",
    )
    parser.add_argument(
        "--finetune_modules",
        type=list_type,
        default=None,
        help="List of modules to finetune",
    )

    parser.add_argument("--nb_mode", type=bool, default=False)

    parser.add_argument("--log_fn", type=str, default="train.log")
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )
    args = parser.parse_args()

    if args.epochs is not None and args.total_iters is not None:
        logger.info(
            "If epochs and total iters are both not None, then we only use iters."
        )
        args.epochs = None

    return args


# --- Setup Functions ---


def load_and_merge_config(
    trained_model_path: Path,
    config_path: Path,
    lrate: float,
    eps: float,
    betas: tuple,
    regression_task: bool,
    classification_task: bool,
    mclassification_task: int,
) -> dict:
    """Loads the base JSON config and overwrites it with args."""

    # Logic fix: If finetuning, load config from model dir. Else, load from config_path.
    if trained_model_path.exists():
        logger.info(f"Loading config from: {trained_model_path}")
        config_file = trained_model_path / "config.json"
    else:
        logger.info(f"Loading config from: {config_path}")
        config_file = config_path

    logger.info(f"Loading configuration from: {config_file}")
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
    model_path: Path,
    interpret_path: Path,
    device: str,
) -> tuple[torch.device, Path, Path]:
    """Sets random seeds, creates directories, and sets device."""

    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Set device
    device = torch.device(device)

    logger.info(f"Set seed to {seed}")
    logger.info(f"Using device: {device}")

    return device, model_path, interpret_path


# --- Data Loading Functions ---


def load_dataframes(
    datafolder: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads the train, validation, and test dataframes."""
    train_df = pd.read_csv(datafolder / "train.csv")
    test_df = pd.read_csv(datafolder / "test.csv")

    valid_path = datafolder / "valid.csv"
    valid_df = pd.read_csv(valid_path)

    return train_df, test_df, valid_df


def load_or_init_graphs(
    datafolder: Path,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    valid_df: pd.DataFrame,
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
        logger.info("Initialising Protein Sequence to Protein Graph...")
        protein_dict = protein_init(protein_seqs)
        torch.save(protein_dict, protein_path)

    # Load or initialize ligand graphs
    ligand_path = datafolder / "ligand.pt"
    if ligand_path.exists():
        logger.info("Loading Ligand Graph data...")
        ligand_dict = torch.load(ligand_path)
    else:
        logger.info("Initialising Ligand SMILES to Ligand Graph...")
        ligand_dict = ligand_init(ligand_smiles)
        torch.save(ligand_dict, ligand_path)

    torch.cuda.empty_cache()
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

    return train_loader, valid_loader, test_loader


# --- Model and Trainer Functions ---


def get_pna_degrees(
    trained_model_path: Path,
    datafolder: Path,
    train_loader: DataLoader,
    model_path: Path,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Loads or computes PNA degrees."""

    if not trained_model_path.exists():
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
    trained_model_path: Path,
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

    if trained_model_path.exists():
        param_dict = trained_model_path / "model.pt"
        model.load_state_dict(torch.load(param_dict, map_location=device), strict=False)
        logger.info(f"Pretrained model loaded from: {param_dict}")

    nParams = sum([p.nelement() for p in model.parameters()])
    logger.info(f"Model loaded with {nParams} parameters.")

    return model


def initialize_trainer(model, config, train_loader, args, device, model_path):
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
        num_epochs=args.epochs,
        total_iters=args.total_iters,
        warmup_iters=config["optimizer"]["warmup_iters"],
        lr_decay_iters=config["optimizer"]["lr_decay_iters"],
        schedule_lr=config["optimizer"]["schedule_lr"],
        regression_weight=1,
        classification_weight=1,
        evaluate_metric=evaluation_metric,
        result_path=args.result_path,
        runid=args.seed,
        finetune_modules=args.finetune_modules,
        device=device,
    )

    # Save the config to the model directory
    with open(os.path.join(model_path, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    return engine


# --- Main Execution ---


def run_training(engine, train_loader, valid_loader, test_loader, args):
    """Runs the main training loop."""
    logger.info("-" * 50)
    logger.info("Start training model")

    if args.epochs:
        engine.train_epoch(
            train_loader,
            val_loader=valid_loader,
            test_loader=test_loader,
            evaluate_epoch=args.evaluate_epoch,
        )
    else:
        engine.train_step(
            train_loader,
            val_loader=valid_loader,
            test_loader=test_loader,
            evaluate_step=args.evaluate_step,
        )

    logger.info("Finished training model")
    logger.info("-" * 50)


def run_evaluation(
    model, model_path, test_df, test_loader, interpret_path, ligand_dict, args
):
    """Loads the best model and runs final evaluation/screening."""
    logger.info("Loading best checkpoint and predicting test data")
    logger.info("-" * 50)

    best_model_file = os.path.join(model_path, "model.pt")
    model.load_state_dict(torch.load(best_model_file, map_location=args.device))

    screen_df = virtual_screening(
        test_df,
        model,
        test_loader,
        result_path=interpret_path,
        save_interpret=args.save_interpret,
        ligand_dict=ligand_dict,
        device=args.device,
    )

    pred_file = os.path.join(args.result_path, f"test_prediction_seed{args.seed}.csv")
    screen_df.to_csv(pred_file, index=False)
    logger.info(f"Test predictions saved to: {pred_file}")


def main():
    """Main execution flow."""
    # 1. Setup
    args = parse_args()
    try:
        setup_logging(args.log_fn, args.log_level)

        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Data folder: {args.datafolder}")
        logger.info(f"Result path: {args.result_path}")
        logger.info(f"Config path: {args.config_path}")
        logger.info(f"Trained model path: {args.trained_model_path}")
        logger.info(f"Learning rate: {args.lrate}")
        logger.info(f"Weight decay: {args.wdecay}")
        logger.info(f"Batch size: {args.batch_size}")
        logger.info(f"Total iters: {args.total_iters}")
        logger.info(f"Epochs: {args.epochs}")
        logger.info(f"Seed: {args.seed}")
        logger.info(f"Regression task: {args.regression_task}")
        logger.info(f"Classification task: {args.classification_task}")
        logger.info(f"Multiclassification task: {args.mclassification_task}")
        logger.info(f"Finetune modules: {args.finetune_modules}")
        logger.info(f"Notebook mode: {args.nb_mode}")
        logger.info(f"Device: {args.device}")

        trained_model_path = args.trained_model_path.resolve()
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
        model_path = args.model_path.resolve()
        interpret_path = args.interpret_path.resolve()
        device, model_path, interpret_path = setup_environment(
            args.seed, model_path, interpret_path, args.device
        )

        # 2. Load Data
        datafolder = args.datafolder.resolve()
        train_df, test_df, valid_df = load_dataframes(datafolder)
        protein_dict, ligand_dict = load_or_init_graphs(
            datafolder, train_df, test_df, valid_df
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

        # 3. Initialize Model and Trainer
        mol_deg, clique_deg, prot_deg = get_pna_degrees(
            trained_model_path, datafolder, train_loader, model_path
        )
        model = initialize_model(config, mol_deg, prot_deg, device, args)
        engine = initialize_trainer(
            model, config, train_loader, args, device, model_path
        )

        # 4. Run Training
        run_training(engine, train_loader, valid_loader, test_loader, args)

        # 5. Run Final Evaluation
        run_evaluation(
            model, model_path, test_df, test_loader, interpret_path, ligand_dict, args
        )

    except Exception as e:
        logger.exception("Script failed: %s", e)
        sys.exit(1)

    logger.info("Script finished.")


if __name__ == "__main__":
    main()
